import numpy as np
import torch
import torch.optim as optim
import sys
import os
from models.video_tokenizer import VideoTokenizer
from datasets.data_utils import visualize_reconstruction, load_data_and_data_loaders
from utils.scheduler_utils import create_cosine_scheduler
from tqdm import tqdm
import json
import wandb
import torch.nn.functional as F
from utils.utils import readable_timestamp, save_training_state, prepare_stage_dirs, prepare_pipeline_run_root
from utils.config import VQVAEConfig, load_config
from utils.utils import save_training_state
from utils.wandb_utils import init_wandb, log_training_metrics, log_system_metrics, finish_wandb
from dataclasses import asdict

def main():
    print(f"Video Tokenizer Training")
    # Load config (YAML + dotlist overrides)
    args: VQVAEConfig = load_config(VQVAEConfig, default_config_path=os.path.join(os.getcwd(), 'configs', 'video_tokenizer.yaml'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Always define a timestamp-like name
    timestamp = args.filename or readable_timestamp()

    # Create organized save directory structure under a shared run root
    run_root = os.environ.get('NG_RUN_ROOT_DIR')
    if not run_root:
        run_root, _ = prepare_pipeline_run_root(base_cwd=os.getcwd())
    if args.save:
        stage_dir, checkpoints_dir, visualizations_dir = prepare_stage_dirs(run_root, 'video_tokenizer')
        print(f'Results will be saved in {stage_dir}')

    training_data, validation_data, training_loader, validation_loader, x_train_var = load_data_and_data_loaders(
        dataset=args.dataset, 
        batch_size=args.batch_size, 
        num_frames=args.context_length
    )

    model = VideoTokenizer(
        frame_size=(args.frame_size, args.frame_size), 
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        latent_dim=args.latent_dim,
        num_bins=args.num_bins,
    ).to(device)

    # Optionally compile the model
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True)

    # Create parameter groups to avoid weight decay on biases and norm layers
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name:
                no_decay.append(param)
            else:
                decay.append(param)

    # Try to enable fused AdamW for better throughput
    try:
        optimizer = optim.AdamW([
            {'params': decay, 'weight_decay': 0.01},
            {'params': no_decay, 'weight_decay': 0}
        ], lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, fused=True)
    except TypeError:
        optimizer = optim.AdamW([
            {'params': decay, 'weight_decay': 0.01},
            {'params': no_decay, 'weight_decay': 0}
        ], lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)

    # Create cosine scheduler with warmup
    scheduler = create_cosine_scheduler(optimizer, args.n_updates)

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda', enabled=bool(args.amp))

    results = {
        'n_updates': 0,
        'recon_errors': [],
        'loss_vals': [],
        'perplexities': [],
    }

    if args.checkpoint:
        # TODO: use util for loading checkpoint and optimizer/scheduler state dicts
        if os.path.isfile(args.checkpoint):
            print(f"Loading checkpoint from {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['model'])
            results = checkpoint['results']
            # Load hyperparameters but don't override current args
            saved_hyperparameters = checkpoint['hyperparameters']
            print(f"Resuming from update {results['n_updates']}")

            # Restore optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Restore scheduler state if available
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print(f"No checkpoint found at {args.checkpoint}")

    # Initialize W&B if enabled and available
    if args.use_wandb:
        cfg = asdict(args)
        cfg.update({'timestamp': timestamp})
        run_name = args.wandb_run_name or f"video_tokenizer_{timestamp}"
        init_wandb(args.wandb_project, cfg, run_name)
        wandb.watch(model, log="all", log_freq=args.log_interval)

    model.train()

    start_iter = max(args.start_iteration, results['n_updates'])

    train_iter = iter(training_loader)
    for i in tqdm(range(start_iter, args.n_updates)):
        try:
            (x, _) = next(train_iter)
        except StopIteration:
            train_iter = iter(training_loader)  # Reset iterator when epoch ends
            (x, _) = next(train_iter)

        x = x.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Forward + loss under autocast
        with torch.amp.autocast('cuda', enabled=bool(args.amp), dtype=torch.bfloat16 if args.amp else None):
            x_hat = model(x)
            recon_loss = F.smooth_l1_loss(x_hat, x)

        # Backward with scaler
        scaler.scale(recon_loss).backward()

        # Clip gradients before stepping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # Step the learning rate scheduler

        results["recon_errors"].append(recon_loss.cpu().detach())
        results["loss_vals"].append(recon_loss.cpu().detach())
        results["n_updates"] = i

        # Log to W&B if enabled and available
        if args.use_wandb:
            metrics = {
                'loss': recon_loss.item(),
                'learning_rate': scheduler.get_last_lr()[0],
            }
            log_training_metrics(i, metrics, prefix='train')

            if i % args.log_interval == 0:
                with torch.no_grad():
                    indices = model.tokenize(x)
                    unique_codes = torch.unique(indices).numel()
                    wandb.log({'train/codebook_usage': unique_codes / model.codebook_size, 'step': i})

            log_system_metrics(i)

        if i % args.log_interval == 0:
            if args.save:
                hyperparameters = args.__dict__
                save_training_state(model, optimizer, scheduler, hyperparameters, checkpoints_dir, prefix='video_tokenizer', step=i)
                # Visualizations
                x_hat_vis = x_hat.detach().cpu()
                x_vis = x.detach().cpu()
                save_path = os.path.join(visualizations_dir, f'video_tokenizer_recon_step_{i}_{args.filename}.png')
                visualize_reconstruction(x_vis[:16], x_hat_vis[:16], save_path)

        print('Update #', i, 'Recon Loss:', torch.mean(torch.stack(results["recon_errors"][-args.log_interval:])).item())

    # Finish W&B run
    if args.use_wandb:
        finish_wandb()

if __name__ == "__main__":
    main()