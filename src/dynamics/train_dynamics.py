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
from src.video_tokenizer.models.video_tokenizer import Video_Tokenizer
from src.latent_action_model.models.lam import LAM
from src.dynamics.models.dynamics import DynamicsModel
from datasets.data_utils import visualize_reconstruction, load_data_and_data_loaders
from tqdm import tqdm
import json
from einops import rearrange
from src.utils.wandb_utils import (
    init_wandb, log_training_metrics, log_learning_rate, log_system_metrics, finish_wandb
)
from src.utils.scheduler_utils import create_cosine_scheduler
from src.utils.utils import readable_timestamp
from src.utils.utils import save_training_state, load_videotokenizer_from_checkpoint, load_lam_from_checkpoint
from src.utils.utils import prepare_run_dirs

from src.utils.config import DynamicsConfig, load_config
import wandb
from dataclasses import asdict


def main():
    args: DynamicsConfig = load_config(DynamicsConfig, default_config_path=os.path.join(os.getcwd(), 'configs', 'dynamics.yaml'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.save:
        run_dir, checkpoints_dir, visualizations_dir, run_name = prepare_run_dirs('dynamics', args.filename, base_cwd=os.getcwd())

    if os.path.isfile(args.video_tokenizer_path):
        video_tokenizer, vq_ckpt = load_videotokenizer_from_checkpoint(args.video_tokenizer_path, device)
        video_tokenizer.eval()
        for p in video_tokenizer.parameters():
            p.requires_grad = False
    else:
        raise FileNotFoundError(f"Video tokenizer checkpoint not found at {args.video_tokenizer_path}")

    if os.path.isfile(args.lam_path):
        lam, lam_ckpt = load_lam_from_checkpoint(args.lam_path, device)
        lam.eval()
        for p in lam.parameters():
            p.requires_grad = False
    else:
        raise FileNotFoundError(f"LAM checkpoint not found at {args.lam_path}")

    dynamics_model = DynamicsModel(
        frame_size=(args.frame_size, args.frame_size),
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        conditioning_dim=lam.action_dim,
        latent_dim=args.latent_dim,
        num_bins=args.num_bins,
    ).to(device)

    if args.compile:
        dynamics_model = torch.compile(dynamics_model, mode="reduce-overhead", fullgraph=False, dynamic=True)

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
    if args.use_wandb:
        run_name = args.wandb_run_name or f"dynamics_{readable_timestamp()}"
        init_wandb(args.wandb_project, asdict(args), run_name)
        wandb.watch(dynamics_model, log="all", log_freq=args.log_interval)

    dynamics_model.train()

    _, _, training_loader, _, _ = load_data_and_data_loaders(
        dataset=args.dataset, 
        batch_size=args.batch_size, 
        num_frames=args.context_length
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
        quantized_video_latents = video_tokenizer.tokenize(x) # [B, S, P, L]
        if args.use_actions:
            actions = lam.encoder(x)  # [B, S - 1, A]
            quantized_actions = lam.quantizer(actions) # [B, S - 1, A]
        else:
            quantized_actions = None

        target_next_tokens = video_tokenizer.quantizer.get_indices_from_latents(quantized_video_latents, dim=-1) # [B, S - 1, P]

        # Predict next frame latents using dynamics model under autocast
        with torch.amp.autocast('cuda', enabled=bool(args.amp), dtype=torch.bfloat16 if args.amp else None):
            predicted_next_logits, mask_positions = dynamics_model(quantized_video_latents, training=True, conditioning=quantized_actions)  # [B, S, P, L^D]

        num_masked = mask_positions.sum().item() # Scalar
        total_positions = mask_positions.numel() # Scalar
        masking_rate = num_masked / total_positions # Scalar

        # Compute loss only on masked tokens (MaskGit-style)
        mask_for_loss = mask_positions
        masked_logits = predicted_next_logits.reshape(-1, predicted_next_logits.shape[-1])  # [P, L^D]
        masked_targets = target_next_tokens.reshape(-1)  # [P]
        masked_mask = mask_for_loss.reshape(-1)  # [P]

        masked_logits = masked_logits[masked_mask]  # [num_masked, L^D]
        masked_targets = masked_targets[masked_mask]  # [num_masked]
        loss = F.cross_entropy(masked_logits, masked_targets)

        # Backward + clip with scaler
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(dynamics_model.parameters(), max_norm=1.0)
        optimizer.step()
        scaler.update()
        scheduler.step()  # Step the learning rate scheduler

        results["loss_vals"].append(loss.cpu().detach())
        results["n_updates"] = i

        predicted_next_indices = torch.argmax(predicted_next_logits, dim=-1) # [B, S - 1, P]
        predicted_next_latents = video_tokenizer.quantizer.get_latents_from_indices(predicted_next_indices, dim=-1) # [B, S - 1, P, L]

        # Log to W&B if enabled
        if args.use_wandb:
            metrics = {
                'loss': loss.item(),
                'masking_rate': masking_rate if mask_positions is not None else 0.0,
            }
            log_training_metrics(i, metrics, prefix="train")
            log_system_metrics(i)
            log_learning_rate(optimizer, i)

        if i % args.log_interval == 0:
            if args.save:
                hyperparameters = args.__dict__
                save_training_state(dynamics_model, optimizer, scheduler, hyperparameters, checkpoints_dir, prefix='dynamics', step=i)

                # Decode predicted latents (predicted_next_latents: [B, S - 1, ...])
                with torch.no_grad():
                    predicted_frames = video_tokenizer.decoder(predicted_next_latents[:16])  # [B, S - 1, ...]
 
                # Ground truth frames
                target_frames_full = x[:, 1:]  # [B, S - 1, ...]

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

                save_path = os.path.join(visualizations_dir, f'dynamics_prediction_step_{i}_{args.filename}.png')
                visualize_reconstruction(masked_target_frames_full[:16].cpu(), predicted_frames[:16].cpu(), save_path)

            print('Step', i, 'Loss:', torch.mean(torch.stack(results["loss_vals"][-args.log_interval:])).item())

    # Finish W&B run
    if args.use_wandb:
        finish_wandb()

if __name__ == "__main__":
    main()
