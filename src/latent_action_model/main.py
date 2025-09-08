import torch
import torch.optim as optim
import sys
import os
import torch.nn.functional as F

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.lam import LAM
import argparse
from tqdm import tqdm
from src.latent_action_model.utils import visualize_reconstructions
from src.vqvae.utils import load_data_and_data_loaders
import multiprocessing
import time
import json
import wandb

# Performance flags via CLI
# We place parser creation early to apply backend flags ASAP

def readable_timestamp():
    """Generate a readable timestamp for filenames"""
    return time.strftime("%a_%b_%d_%H_%M_%S_%Y")


def save_run_configuration(args, run_dir, timestamp, device):
    """Save all configuration parameters and run information to a file"""
    config = {
        'timestamp': timestamp,
        'device': str(device),
        'model_architecture': {
            'frame_size': (args.frame_size, args.frame_size),
            'n_actions': args.n_actions,
            'patch_size': args.patch_size,
            'embed_dim': args.embed_dim,
            'num_heads': args.num_heads,
            'hidden_dim': args.hidden_dim,
            'num_blocks': args.num_blocks,
            'action_dim': args.action_dim,
            'beta': args.beta,
            'quantization_method': 'Vector Quantization (VQ)'
        },
        'training_parameters': {
            'batch_size': args.batch_size,
            'n_updates': args.n_updates,
            'learning_rate': args.learning_rate,
            'log_interval': args.log_interval,
            'seq_length': args.seq_length,
            'dataset': args.dataset
        },
        'directories': {
            'run_dir': run_dir,
            'checkpoints_dir': os.path.join(run_dir, 'checkpoints') if args.save else None,
            'visualizations_dir': os.path.join(run_dir, 'visualizations') if args.save else None
        }
    }
    
    config_path = os.path.join(run_dir, 'run_config.json') if args.save else None
    if config_path:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f'Configuration saved to: {config_path}')


def save_lam_model_and_results(model, optimizer, results, hyperparameters, timestamp, checkpoints_dir):
    """Save LAM model checkpoint including model state, optimizer state, results and hyperparameters"""
    results_to_save = {
        'model': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    checkpoint_path = os.path.join(checkpoints_dir, f'lam_checkpoint_{timestamp}.pth')
    torch.save(results_to_save, checkpoint_path)
    return checkpoint_path


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_updates", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--frame_size", type=int, default=64)
    parser.add_argument("--n_actions", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size for ST-Transformer")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension for ST-Transformer")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension for feed-forward")
    parser.add_argument("--num_blocks", type=int, default=2, help="Number of ST-Transformer blocks")
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--dataset", type=str, default="SONIC")
    parser.add_argument("--seq_length", type=int, default=8, help="Length of frame sequences")
    parser.add_argument("--beta", type=float, default=0.05, help="VQ loss weight")
    parser.add_argument("-save", action="store_true", default=True)
    parser.add_argument("--filename", type=str, default=readable_timestamp())
    parser.add_argument("--log_interval", type=int, default=50, help="Interval for saving model and logging")
    
    # W&B arguments
    parser.add_argument("--use_wandb", action="store_true", default=True, help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="nano-genie", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb_media", action="store_true", default=False, help="Log images/videos to W&B (off by default)")

    parser.add_argument("--vq_reinit_check_interval", type=int, default=100, help="Steps between dead-code usage-window checks")
    # Performance flags
    parser.add_argument("--amp", action="store_true", default=False, help="Enable mixed precision (bfloat16)")
    parser.add_argument("--tf32", action="store_true", default=False, help="Enable TF32 on Ampere+")
    parser.add_argument("--compile", action="store_true", default=False, help="Compile the model with torch.compile")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Speed-related backend settings
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
            torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('high' if args.tf32 else 'medium')
        except Exception:
            pass

    # Try to relax inductor settings to avoid fused-attention rank errors
    try:
        import torch._inductor.config as inductor_config
        inductor_config.fuse_attention = False
    except Exception:
        pass
    
    # Create organized save directory structure
    if args.save:
        run_dir = os.path.join(os.getcwd(), 'src', 'latent_action_model', 'results', f'lam_{readable_timestamp()}')
        os.makedirs(run_dir, exist_ok=True)
        checkpoints_dir = os.path.join(run_dir, 'checkpoints')
        visualizations_dir = os.path.join(run_dir, 'visualizations')
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(visualizations_dir, exist_ok=True)
        print(f'Results will be saved in {run_dir}')

    # Load sequence data for training
    _, _, training_loader, validation_loader, _ = load_data_and_data_loaders(
        dataset=args.dataset, 
        batch_size=args.batch_size, 
        num_frames=args.seq_length
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
        action_dim=args.action_dim,
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

    if args.save:
        save_run_configuration(args, run_dir, args.filename, device)

    # Initialize W&B if enabled and available
    if args.use_wandb:
        run_name = args.wandb_run_name or f"lam_{args.filename}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

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
            # Log training metrics
            wandb.log({
                'train/total_loss': loss.item(),
                'step': epoch
            })
            
            # Log system metrics
            if torch.cuda.is_available():
                wandb.log({
                    'system/gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                    'system/gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
                    'step': epoch
                })
            
            # Log learning rate
            for j, param_group in enumerate(optimizer.param_groups):
                wandb.log({
                    f'learning_rate/group_{j}': param_group['lr'],
                    'step': epoch
                })

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
                print(f"Step {epoch}: loss={loss.item():.6f},codebook_usage: {codebook_usage}, z_e_var: {z_e_var}")
                
                # Log codebook and action statistics to W&B
                if args.use_wandb:
                    wandb.log({
                        "lam/codebook_usage": codebook_usage,
                        "lam/encoder_variance": z_e_var,
                        "step": epoch
                    })
                
        # Save model and visualize results periodically
        if epoch % args.log_interval == 0 and args.save:
            hyperparameters = args.__dict__
            checkpoint_path = save_lam_model_and_results(model, optimizer, results, hyperparameters, args.filename, checkpoints_dir)
            
            visualize_reconstructions(
                frame_sequences[:, 0], 
                frame_sequences[:, -1], 
                pred_frames[:, -1],
                os.path.join(visualizations_dir, f'reconstructions_lam_epoch_{epoch}_{args.filename}.png')
            )
    
    # Finish W&B run
    if args.use_wandb:
        wandb.finish()
        print("✅ W&B run finished")

    # Verification: Test if similar transitions map to same actions
    print("\nVerifying model behavior...")
    test_frames, _ = next(iter(validation_loader))
    test_frames = test_frames.to(device)

    # Test Case 1: Similar frame transitions
    prev_frame = test_frames[0:1, 0]
    next_frame_similar = test_frames[0:1, 1]
    action1 = model.encode(prev_frame, next_frame_similar)

    next_frame_similar2 = test_frames[0:1, 2]
    action2 = model.encode(prev_frame, next_frame_similar2)

    print("Actions for similar transitions:", action1.item(), action2.item())

    # Test Case 2: Different frame transition
    next_frame_different = test_frames[0:1, -1]
    action3 = model.encode(prev_frame, next_frame_different)
    print("Action for different transition:", action3.item())

if __name__ == "__main__":
    main()
