import torch
import torch.optim as optim
import sys
import os
import torch.nn.functional as F

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.lam import LAM
from tqdm import tqdm
from src.latent_action_model.utils import visualize_reconstructions
from datasets.data_utils import load_data_and_data_loaders
from src.utils.utils import readable_timestamp
import json
import wandb
from src.utils.config import LAMConfig, load_config
from src.utils.utils import save_training_state, prepare_run_dirs
from src.utils.wandb_utils import init_wandb, log_system_metrics, finish_wandb
from dataclasses import asdict

# Load config (YAML + dotlist overrides)
args: LAMConfig = load_config(LAMConfig, default_config_path=os.path.join(os.getcwd(), 'configs', 'lam.yaml'))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create organized save directory structure
    if args.save:
        run_dir, checkpoints_dir, visualizations_dir, run_name = prepare_run_dirs('latent_action_model', args.filename, base_cwd=os.getcwd())
        print(f'Results will be saved in {run_dir}')

    # Load sequence data for training
    _, _, training_loader, validation_loader, _ = load_data_and_data_loaders(
        dataset=args.dataset, 
        batch_size=args.batch_size, 
        num_frames=args.context_length
    )

    # Initialize model
    model = LAM(
        frame_size=(args.frame_size, args.frame_size),
        n_actions=args.n_actions,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
    ).to(device)

    # Optionally compile
    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True)
            print("✅ LAM compiled with torch.compile")
        except Exception as e:
            print(f"⚠️ torch.compile not available or failed: {e}")

    # Optimizer (AdamW with no-decay for codebook and higher LR)
    base_lr = args.learning_rate
    weight_decay = 0.01
    try:
        optimizer = optim.AdamW([
            {"params": model.encoder.parameters()},
            {"params": model.decoder.parameters()},
            {"params": model.quantizer.embedding.parameters(), "weight_decay": 0.0, "lr": base_lr * 2.0},
        ], lr=base_lr, weight_decay=weight_decay)
        print("✅ Using AdamW with no-decay codebook and 2x LR")
    except Exception:
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

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
        run_name = args.wandb_run_name or f"lam_{readable_timestamp()}"
        init_wandb(args.wandb_project, asdict(args), run_name)

    # Training loop
    train_iter = iter(training_loader)
    for epoch in tqdm(range(args.n_updates)):
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
        results["n_updates"] = epoch
        
        # Log to W&B if enabled and available
        if args.use_wandb:
            wandb.log({
                'train/total_loss': loss.item(),
                'step': epoch
            })
            log_system_metrics(epoch)
            
        # Print progress every 50 steps
        if (epoch - 1) % 50 == 0:
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
                print(f"Step {epoch}: loss={loss.item():.6f}, codebook_usage: {codebook_usage}, z_e_var: {z_e_var}, pred_frames_var: {pred_frames_var}")
                
                # Log codebook and action statistics to W&B
                if args.use_wandb:
                    wandb.log({
                        "lam/codebook_usage": codebook_usage,
                        "lam/encoder_variance": z_e_var,
                        "lam/decoder_variance": pred_frames_var,
                        "step": epoch
                    })
                
        # Save model and visualize results periodically
        if epoch % args.log_interval == 0 and args.save:
            hyperparameters = vars(args)
            checkpoint_path = save_training_state(model, optimizer, None, hyperparameters, checkpoints_dir, prefix='lam', step=epoch)
            
            visualize_reconstructions(
                frame_sequences[:, 0], 
                frame_sequences[:, -1], 
                pred_frames[:, -1],
                os.path.join(visualizations_dir, f'reconstructions_lam_epoch_{epoch}_{args.filename or "run"}.png')
            )
    
    # Finish W&B run
    if args.use_wandb:
        finish_wandb()

if __name__ == "__main__":
    main()
