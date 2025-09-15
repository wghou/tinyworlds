import torch
import torch.optim as optim
import sys
import os
import torch.nn.functional as F
from models.latent_actions import LatentActionModel
from tqdm import tqdm
from datasets.data_utils import visualize_reconstruction, load_data_and_data_loaders
from utils.utils import readable_timestamp, save_training_state, prepare_stage_dirs, prepare_pipeline_run_root
import json
import wandb
from utils.config import LatentActionsConfig, load_stage_config_merged
from utils.wandb_utils import init_wandb, log_system_metrics, finish_wandb
from dataclasses import asdict


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load stage config merged with training_config.yaml (training takes priority), plus CLI overrides
    args: LatentActionsConfig = load_stage_config_merged(LatentActionsConfig, default_config_path=os.path.join(os.getcwd(), 'configs', 'latent_actions.yaml'))

    # Create organized save directory structure under a shared run root
    run_root = os.environ.get('NG_RUN_ROOT_DIR')
    if not run_root:
        run_root, _ = prepare_pipeline_run_root(base_cwd=os.getcwd())
    if args.save:
        stage_dir, checkpoints_dir, visualizations_dir = prepare_stage_dirs(run_root, 'latent_actions')
        print(f'Results will be saved in {stage_dir}')

    # Load sequence data for training
    _, _, training_loader, validation_loader, _ = load_data_and_data_loaders(
        dataset=args.dataset, 
        batch_size=args.batch_size, 
        num_frames=args.context_length
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

    # Print parameter count
    try:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"LatentActionModel parameters: {num_params/1e6:.2f}M ({num_params})")
    except Exception:
        pass

    # Optionally compile
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True)

    # AdamW as optimizer
    param_groups = [{"params": model.parameters(), "weight_decay": 0.01}]
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
    if args.use_wandb:
        run_name = args.wandb_run_name or f"latent_actions_{readable_timestamp()}"
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
            loss, pred_frames = model(frame_sequences)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Track results
        results["total_losses"].append(loss.cpu().detach())
        results["n_updates"] = i

        # Log to W&B if enabled and available
        if args.use_wandb:
            wandb.log({
                'train/total_loss': loss.item(),
                'step': i
            })
            log_system_metrics(i)
  
        # Print progress every 50 steps
        if (i - 1) % 50 == 0:
            with torch.no_grad():
                actions = model.encoder(frame_sequences)
                # Quantize actions, compute joint-code usage via indices
                actions_quantized = model.quantizer(actions)
                idx = model.quantizer.get_indices_from_latents(actions_quantized)
                codebook_usage = idx.unique().numel() / model.quantizer.codebook_size
                # Per-dimension mean variance for clearer signal
                z_e_var = actions.var(dim=0, unbiased=False).mean().item()

                # compute decoder variance
                pred_frames_var = pred_frames.var(dim=0, unbiased=False).mean().item()
                print(f"Step {i}: loss={loss.item():.6f}, codebook_usage: {codebook_usage}, z_e_var: {z_e_var}, pred_frames_var: {pred_frames_var}")

                # Log codebook and action statistics to W&B
                if args.use_wandb:
                    wandb.log({
                        "latent_actions/codebook_usage": codebook_usage,
                        "latent_actions/encoder_variance": z_e_var,
                        "latent_actions/decoder_variance": pred_frames_var,
                        "step": i
                    })

        # Save model and visualize results periodically
        if i % args.log_interval == 0 and args.save:
            hyperparameters = vars(args)
            checkpoint_path = save_training_state(model, optimizer, None, hyperparameters, checkpoints_dir, prefix='latent_actions', step=i)
            save_path = os.path.join(visualizations_dir, f'reconstructions_latent_actions_step_{i}.png')
            visualize_reconstruction(frame_sequences, pred_frames, save_path)

    # Finish W&B run
    if args.use_wandb:
        finish_wandb()

if __name__ == "__main__":
    main()
