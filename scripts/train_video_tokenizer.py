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
from utils.config import VideoTokenizerConfig, load_stage_config_merged
from utils.utils import save_training_state, load_videotokenizer_from_checkpoint
from utils.wandb_utils import init_wandb, log_training_metrics, log_system_metrics, finish_wandb
from dataclasses import asdict
from utils.distributed import init_distributed_from_env, wrap_ddp_if_needed, unwrap_model, get_dataloader_distributed_kwargs, print_param_count_if_main, cleanup_distributed


def main():
    print(f"Video Tokenizer Training")
    # Load stage config merged with training_config.yaml (training takes priority), plus CLI overrides
    args: VideoTokenizerConfig = load_stage_config_merged(VideoTokenizerConfig, default_config_path=os.path.join(os.getcwd(), 'configs', 'video_tokenizer.yaml'))

    print(f"args.compile: {args.compile}")

    # Minimal DDP setup
    ddp = init_distributed_from_env()
    device = ddp['device']

    # Always define a timestamp-like name
    timestamp = readable_timestamp()

    # Create organized save directory structure under a shared run root
    run_root = os.environ.get('NG_RUN_ROOT_DIR')
    if not run_root:
        run_root, _ = prepare_pipeline_run_root(base_cwd=os.getcwd())
    is_main = ddp['is_main']
    if is_main:
        stage_dir, checkpoints_dir, visualizations_dir = prepare_stage_dirs(run_root, 'video_tokenizer')
        print(f'Results will be saved in {stage_dir}')

    training_data, validation_data, training_loader, validation_loader, x_train_var = load_data_and_data_loaders(
        dataset=args.dataset, 
        batch_size=args.batch_size, 
        num_frames=args.context_length,
        **get_dataloader_distributed_kwargs(ddp)
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

    if args.checkpoint:
        model, _ = load_videotokenizer_from_checkpoint(args.checkpoint, device, model)

    print_param_count_if_main(model, "VideoTokenizer", is_main)

    # Optionally compile the model
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True)

    # Wrap with DDP
    model = wrap_ddp_if_needed(model, ddp['is_distributed'], ddp['local_rank'])

    # Create parameter groups to avoid weight decay on biases and norm layers
    decay = []
    no_decay = []
    for name, param in unwrap_model(model).named_parameters():
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

    # Initialize W&B if enabled and available
    if args.use_wandb and is_main:
        cfg = asdict(args)
        cfg.update({'timestamp': timestamp})
        run_name = f"video_tokenizer_{timestamp}"
        init_wandb(args.wandb_project, cfg, run_name)
        wandb.watch(unwrap_model(model), log="all", log_freq=args.log_interval)

    unwrap_model(model).train()

    train_iter = iter(training_loader)
    for i in tqdm(range(args.n_updates)):
        try:
            (x, _) = next(train_iter)
        except StopIteration:
            train_iter = iter(training_loader)  # Reset iterator when epoch ends
            (x, _) = next(train_iter)

        x = x.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Forward + loss under autocast
        with torch.amp.autocast('cuda', enabled=bool(args.amp), dtype=torch.bfloat16 if args.amp else None):
            loss, x_hat = unwrap_model(model)(x)

        # Backward with scaler
        scaler.scale(loss).backward()

        # Clip gradients before stepping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # Step the learning rate scheduler

        results["recon_errors"].append(loss.cpu().detach())
        results["loss_vals"].append(loss.cpu().detach())
        results["n_updates"] = i

        # Log to W&B if enabled and available
        if args.use_wandb and is_main:
            metrics = {
                'loss': loss.item(),
                'learning_rate': scheduler.get_last_lr()[0],
            }
            log_training_metrics(i, metrics, prefix='train')
            log_system_metrics(i)

        if i % args.log_interval == 0 and is_main:
            if args.use_wandb:
                with torch.no_grad():
                    indices = unwrap_model(model).tokenize(x)
                    unique_codes = torch.unique(indices).numel()
                    wandb.log({'train/codebook_usage': unique_codes / model.codebook_size}, step=i)

            hyperparameters = args.__dict__
            save_training_state(unwrap_model(model), optimizer, scheduler, hyperparameters, checkpoints_dir, prefix='video_tokenizer', step=i)
            x_hat_vis = x_hat.detach().cpu()
            x_vis = x.detach().cpu()
            save_path = os.path.join(visualizations_dir, f'video_tokenizer_recon_step_{i}.png')
            visualize_reconstruction(x_vis[:16], x_hat_vis[:16], save_path)
            
            print('Step', i, 'Loss:', torch.mean(torch.stack(results["recon_errors"][-args.log_interval:])).item())

    # Finish W&B run
    if args.use_wandb and is_main:
        finish_wandb()
    cleanup_distributed(ddp['is_distributed'])

if __name__ == "__main__":
    main()