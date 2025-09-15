import numpy as np
import torch
import torch.optim as optim
import argparse
import sys
import os
import time
import json
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F
from models.video_tokenizer import VideoTokenizer
from models.latent_actions import LatentActionModel
from models.dynamics import DynamicsModel
from datasets.data_utils import visualize_reconstruction, load_data_and_data_loaders
from tqdm import tqdm
import json
from einops import rearrange
from utils.wandb_utils import (
    init_wandb, log_training_metrics, log_learning_rate, log_system_metrics, finish_wandb
)
from utils.scheduler_utils import create_cosine_scheduler
from utils.utils import (
    readable_timestamp,
    save_training_state,
    load_videotokenizer_from_checkpoint,
    load_latent_actions_from_checkpoint,
    prepare_pipeline_run_root,
    prepare_stage_dirs,
)
from utils.config import DynamicsConfig, load_stage_config_merged
import wandb
from dataclasses import asdict
from utils.distributed import init_distributed_from_env, wrap_ddp_if_needed, unwrap_model, get_dataloader_distributed_kwargs, print_param_count_if_main, cleanup_distributed


def main():
    args: DynamicsConfig = load_stage_config_merged(DynamicsConfig, default_config_path=os.path.join(os.getcwd(), 'configs', 'dynamics.yaml'))
    # Minimal DDP setup
    ddp = init_distributed_from_env()
    device = ddp['device']

    is_main = ddp['is_main']
    if args.save and is_main:
        run_root = os.environ.get('NG_RUN_ROOT_DIR')
        if not run_root:
            run_root, _ = prepare_pipeline_run_root(base_cwd=os.getcwd())
        stage_dir, checkpoints_dir, visualizations_dir = prepare_stage_dirs(run_root, 'dynamics')
        print(f"Results will be saved in {stage_dir}")

    if os.path.isfile(args.video_tokenizer_path):
        video_tokenizer, vq_ckpt = load_videotokenizer_from_checkpoint(args.video_tokenizer_path, device)
        video_tokenizer.eval()
        for p in video_tokenizer.parameters():
            p.requires_grad = False
    else:
        raise FileNotFoundError(f"Video tokenizer checkpoint not found at {args.video_tokenizer_path}")

    if os.path.isfile(args.latent_actions_path):
        latent_action_model, latent_action_ckpt = load_latent_actions_from_checkpoint(args.latent_actions_path, device)
        latent_action_model.eval()
        for p in latent_action_model.parameters():
            p.requires_grad = False
    else:
        raise FileNotFoundError(f"Latent Action Model checkpoint not found at {args.latent_actions_path}")

    dynamics_model = DynamicsModel(
        frame_size=(args.frame_size, args.frame_size),
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        conditioning_dim=latent_action_model.action_dim,
        latent_dim=args.latent_dim,
        num_bins=args.num_bins,
    ).to(device)

    if args.checkpoint:
        model, ckpt = load_dynamics_from_checkpoint(args.checkpoint, device, model)

    print_param_count_if_main(dynamics_model, "DynamicsModel", is_main)

    if args.compile:
        dynamics_model = torch.compile(dynamics_model, mode="reduce-overhead", fullgraph=False, dynamic=True)

    dynamics_model = wrap_ddp_if_needed(dynamics_model, ddp['is_distributed'], ddp['local_rank'])

    # Create parameter groups to avoid weight decay on biases and norm layers
    decay = []
    no_decay = []
    for name, param in dynamics_model.named_parameters():
        if param.requires_grad:
            if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name:
                no_decay.append(param)
            else:
                decay.append(param)

    optimizer = optim.AdamW([
        {'params': decay, 'weight_decay': 0.01},
        {'params': no_decay, 'weight_decay': 0}
    ], lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, fused=True)

    # Create cosine scheduler with warmup
    scheduler = create_cosine_scheduler(optimizer, args.n_updates)

    # AMP scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=bool(args.amp))

    results = {
        'n_updates': 0,
        'dynamics_losses': [],
        'loss_vals': [],
    }

    # Initialize W&B if enabled and available
    if args.use_wandb and is_main:
        run_name = f"dynamics_{readable_timestamp()}"
        init_wandb(args.wandb_project, asdict(args), run_name)
        wandb.watch(unwrap_model(dynamics_model), log="all", log_freq=args.log_interval)

    unwrap_model(dynamics_model).train()

    _, _, training_loader, _, _ = load_data_and_data_loaders(
        dataset=args.dataset, 
        batch_size=args.batch_size, 
        num_frames=args.context_length,
        **get_dataloader_distributed_kwargs(ddp)
    )
    train_iter = iter(training_loader)

    for i in tqdm(range(0, args.n_updates)):
        try:
            x, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(training_loader)  # Reset iterator when epoch ends
            x, _ = next(train_iter)

        x = x.to(device, non_blocking=True)  # [batch_size, seq_len, channels, height, width]
        optimizer.zero_grad(set_to_none=True)

        # Get video tokenizer latents for current frames
        video_tokens = video_tokenizer.tokenize(x) # [B, S, P]
        video_latents = video_tokenizer.quantizer.get_latents_from_indices(video_tokens, dim=-1) # [B, S, P, L]
        if args.use_actions:
            quantized_actions = latent_action_model.encode(x)  # [B, S - 1, A]
        else:
            quantized_actions = None

        target_next_tokens = video_tokens # [B, S, P]

        # Predict next frame latents using dynamics model under autocast
        with torch.amp.autocast('cuda', enabled=bool(args.amp), dtype=torch.bfloat16 if args.amp else None):
            predicted_next_logits, mask_positions, loss = unwrap_model(dynamics_model)(
                video_latents, training=True, conditioning=quantized_actions, targets=target_next_tokens
            )  # logits, mask, loss

        # Backward + clip with scaler
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unwrap_model(dynamics_model).parameters(), max_norm=1.0)
        optimizer.step()
        scaler.update()
        scheduler.step()  # Step the learning rate scheduler

        results["loss_vals"].append(loss.cpu().detach())
        results["n_updates"] = i

        # Log to W&B if enabled
        if args.use_wandb and is_main:
            metrics = {
                'loss': loss.item(),
            }
            log_training_metrics(i, metrics, prefix="train")
            log_system_metrics(i)
            log_learning_rate(optimizer, i)

        if i % args.log_interval == 0 and is_main:
            predicted_next_indices = torch.argmax(predicted_next_logits, dim=-1)
            predicted_next_latents = video_tokenizer.quantizer.get_latents_from_indices(predicted_next_indices, dim=-1)

            hyperparameters = args.__dict__
            save_training_state(unwrap_model(dynamics_model), optimizer, scheduler, hyperparameters, checkpoints_dir, prefix='dynamics', step=i)

            # Decode predicted latents
            with torch.no_grad():
                predicted_frames = video_tokenizer.decoder(predicted_next_latents[:16]) # [B, S, C, H, W]

            # Ground truth frames
            target_frames_full = x[:, 1:] # [B, S - 1, C, H, W]

            # Display the masked patches in the ground truth frames as black
            masked_target_frames_full = target_frames_full.clone()

            # Convert mask_positions to patch-level mask for visualization
            if mask_positions is not None:
                B, S, N = mask_positions.shape
                patch_size = args.patch_size
                H, W = args.frame_size, args.frame_size
                mask_for_viz = mask_positions[:, 1:]
                B_viz, S_viz, N_viz = mask_for_viz.shape
                pixel_mask = torch.zeros(B_viz, S_viz, H, W, device=mask_positions.device)
                for b in range(B_viz):
                    for s in range(S_viz):
                        patch_mask = mask_for_viz[b, s]
                        for patch_idx in range(N_viz):
                            if patch_mask[patch_idx]:
                                patch_row = (patch_idx // (W // patch_size)) * patch_size
                                patch_col = (patch_idx % (W // patch_size)) * patch_size
                                pixel_mask[b, s, patch_row:patch_row+patch_size, patch_col:patch_col+patch_size] = 1
                pixel_mask_expanded = pixel_mask.unsqueeze(2).expand(-1, -1, 3, -1, -1)
                masked_target_frames_full = masked_target_frames_full * (1 - pixel_mask_expanded)

            if args.save:
                save_path = os.path.join(visualizations_dir, f'dynamics_prediction_step_{i}.png')

            visualize_reconstruction(masked_target_frames_full[:16].cpu(), predicted_frames[:16].cpu(), save_path)

            print('Step', i, 'Loss:', torch.mean(torch.stack(results["loss_vals"][-args.log_interval:])).item())

    # Finish W&B run
    if args.use_wandb and is_main:
        finish_wandb()
    cleanup_distributed(ddp['is_distributed'])

if __name__ == "__main__":
    main()
