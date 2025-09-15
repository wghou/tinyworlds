import torch
import torch.optim as optim
import sys
import os
import torch.nn.functional as F
from models.latent_actions import LatentActionModel
from tqdm import tqdm
from datasets.data_utils import visualize_reconstruction, load_data_and_data_loaders
from utils.utils import (
    readable_timestamp, 
    save_training_state, 
    prepare_stage_dirs, 
    prepare_pipeline_run_root, 
    load_latent_actions_from_checkpoint
)
import json
import wandb
from utils.config import LatentActionsConfig, load_stage_config_merged
from utils.wandb_utils import init_wandb, log_system_metrics, finish_wandb
from dataclasses import asdict
from utils.distributed import init_distributed_from_env, wrap_ddp_if_needed, unwrap_model, get_dataloader_distributed_kwargs, print_param_count_if_main, cleanup_distributed


def main():
    # Minimal DDP setup
    ddp = init_distributed_from_env()
    device = ddp['device']

    # Load stage config merged with training_config.yaml (training takes priority), plus CLI overrides
    args: LatentActionsConfig = load_stage_config_merged(LatentActionsConfig, default_config_path=os.path.join(os.getcwd(), 'configs', 'latent_actions.yaml'))

    # Create organized save directory structure under a shared run root
    run_root = os.environ.get('NG_RUN_ROOT_DIR')
    if not run_root:
        run_root, _ = prepare_pipeline_run_root(base_cwd=os.getcwd())
    is_main = ddp['is_main']
    if is_main:
        stage_dir, checkpoints_dir, visualizations_dir = prepare_stage_dirs(run_root, 'latent_actions')
        print(f'Results will be saved in {stage_dir}')

    # Load sequence data for training
    _, _, training_loader, validation_loader, _ = load_data_and_data_loaders(
        dataset=args.dataset, 
        batch_size=args.batch_size, 
        num_frames=args.context_length,
        **get_dataloader_distributed_kwargs(ddp)
    )

    # Initialize model
    model = LatentActionModel(
        frame_size=(args.frame_size, args.frame_size),
        n_actions=args.n_actions,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
    ).to(device)

    if args.checkpoint:
        model, _ = load_latent_actions_from_checkpoint(args.checkpoint, device, model)

    print_param_count_if_main(model, "LatentActionModel", is_main)

    # Optionally compile
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True)

    model = wrap_ddp_if_needed(model, ddp['is_distributed'], ddp['local_rank'])

    # AdamW as optimizer
    param_groups = [{"params": unwrap_model(model).parameters(), "weight_decay": 0.01}]
    optimizer = optim.AdamW(param_groups, lr=args.learning_rate)

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda', enabled=bool(args.amp))

    # Initialize results tracking
    results = {
        'n_updates': 0,
        'total_losses': [],
        'recon_losses': [],
    }

    # Initialize W&B if enabled and available
    if args.use_wandb and is_main:
        run_name = f"latent_actions_{readable_timestamp()}"
        init_wandb(args.wandb_project, asdict(args), run_name)

    # Training loop
    train_iter = iter(training_loader)
    for i in tqdm(range(args.n_updates)):
        try:
            frame_sequences, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(training_loader)
            frame_sequences, _ = next(train_iter)
        
        frame_sequences = frame_sequences.to(device, non_blocking=True)

        # Forward pass with current step for warmup under autocast
        with torch.amp.autocast('cuda', enabled=bool(args.amp), dtype=torch.bfloat16 if args.amp else None):
            loss, pred_frames = unwrap_model(model)(frame_sequences)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Track results
        results["total_losses"].append(loss.cpu().detach())
        results["n_updates"] = i

        # Log to W&B if enabled and available
        if args.use_wandb and is_main:
            wandb.log({
                'train/loss': loss.item(),
            }, step=i)
            log_system_metrics(i)
  
        # Save model and visualize results periodically
        if i % args.log_interval == 0 and is_main:
            with torch.no_grad():
                actions = unwrap_model(model).encoder(frame_sequences)
                actions_quantized = unwrap_model(model).quantizer(actions)
                idx = unwrap_model(model).quantizer.get_indices_from_latents(actions_quantized)
                codebook_usage = idx.unique().numel() / unwrap_model(model).quantizer.codebook_size
                z_e_var = actions.var(dim=0, unbiased=False).mean().item()
                pred_frames_var = pred_frames.var(dim=0, unbiased=False).mean().item()

            if args.use_wandb:
                wandb.log({
                    "latent_actions/codebook_usage": codebook_usage,
                    "latent_actions/encoder_variance": z_e_var,
                    "latent_actions/decoder_variance": pred_frames_var,
                }, step=i)
                log_action_distribution(actions_quantized, i, args.n_actions)

            hyperparameters = vars(args)
            checkpoint_path = save_training_state(unwrap_model(model), optimizer, None, hyperparameters, checkpoints_dir, prefix='latent_actions', step=i)
            save_path = os.path.join(visualizations_dir, f'reconstructions_latent_actions_step_{i}.png')
            visualize_reconstruction(frame_sequences, pred_frames, save_path)
            
            print('Step', i, 'Loss:', loss.item(), 'Codebook Usage:', codebook_usage, 'Encoder Variance:', z_e_var, 'Decoder Variance:', pred_frames_var)

    # Finish W&B run
    if args.use_wandb and is_main:
        finish_wandb()
    cleanup_distributed(ddp['is_distributed'])

if __name__ == "__main__":
    main()
