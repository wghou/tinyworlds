import numpy as np
import torch
import torch.optim as optim
import argparse
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import utils
from src.vqvae.models.video_tokenizer import Video_Tokenizer
from utils import visualize_reconstruction
from src.utils.scheduler_utils import create_cosine_scheduler
from tqdm import tqdm
import json
import wandb
import math

parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
timestamp = utils.readable_timestamp()

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--n_updates", type=int, default=2000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=8)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--learning_rate", type=float, default=4e-4)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--dataset",  type=str, default='SONIC')
parser.add_argument("--context_length", type=int, default=4)

# Model architecture parameters
parser.add_argument("--patch_size", type=int, default=4, help="Patch size for ST-Transformer")
parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension for ST-Transformer")
parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension for feed-forward")
parser.add_argument("--num_blocks", type=int, default=2, help="Number of ST-Transformer blocks")
parser.add_argument("--latent_dim", type=int, default=6, help="Latent dimension")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument("--num_bins", type=int, default=4, help="Number of bins per dimension for finite scalar quantization")

# whether or not to save model
parser.add_argument("-save", action="store_true", default=True)
parser.add_argument("--filename",  type=str, default=timestamp)

# Add checkpoint arguments
parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file to resume from")
parser.add_argument("--start_iteration", type=int, default=0, help="Iteration to start from")

parser.add_argument("--use_wandb", action="store_true", default=False, help="Enable Weights & Biases logging")
parser.add_argument("--wandb_project", type=str, default="nano-genie", help="W&B project name")
parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")

# Learning rate scheduler parameters
parser.add_argument("--lr_step_size", type=int, default=2000, help="Step size for learning rate decay")
parser.add_argument("--lr_gamma", type=float, default=0.5, help="Gamma for learning rate decay")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create organized save directory structure
if args.save:
    # Create main results directory for this run
    run_dir = os.path.join(os.getcwd(), 'src', 'vqvae', 'results', f'videotokenizer_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    visualizations_dir = os.path.join(run_dir, 'visualizations')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)
    
    print(f'Results will be saved in {run_dir}')
    print(f'Checkpoints: {checkpoints_dir}')
    print(f'Visualizations: {visualizations_dir}')

def save_run_configuration(args, run_dir, timestamp, device):
    """
    Save all configuration parameters and run information to a file
    
    Args:
        args: Parsed arguments
        run_dir: Directory to save configuration
        timestamp: Timestamp for the run
        device: Device being used
    """
    config = {
        'timestamp': timestamp,
        'device': str(device),
        'model_architecture': {
            'frame_size': (64, 64),
            'patch_size': args.patch_size,
            'embed_dim': args.embed_dim,
            'num_heads': args.num_heads,
            'hidden_dim': args.hidden_dim,
            'num_blocks': args.num_blocks,
            'latent_dim': args.latent_dim,
            'dropout': args.dropout,
            'num_bins': args.num_bins,
            'quantization_method': 'Finite Scalar Quantization (FSQ)'
        },
        'training_parameters': {
            'batch_size': args.batch_size,
            'n_updates': args.n_updates,
            'learning_rate': args.learning_rate,
            'log_interval': args.log_interval,
            'context_length': args.context_length,
            'dataset': args.dataset,
            'lr_step_size': args.lr_step_size,
            'lr_gamma': args.lr_gamma
        },
        'checkpoint_info': {
            'checkpoint_path': args.checkpoint,
            'start_iteration': args.start_iteration
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

def save_videotokenizer_model_and_results(model, optimizer, results, hyperparameters, timestamp, checkpoints_dir):
    """
    Save video tokenizer checkpoint including model state, optimizer state, results and hyperparameters
    
    Args:
        model: The PyTorch model
        optimizer: The optimizer
        results: Dictionary containing training results
        hyperparameters: Dictionary of hyperparameters
        timestamp: String timestamp for filename
        checkpoints_dir: Directory to save checkpoints
    """
    results_to_save = {
        'model': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    checkpoint_path = os.path.join(checkpoints_dir, f'videotokenizer_checkpoint_{timestamp}.pth')
    torch.save(results_to_save, checkpoint_path)
    return checkpoint_path

"""
Load data and define batch data loaders
"""

training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
    dataset=args.dataset, 
    batch_size=args.batch_size, 
    num_frames=args.context_length
)
"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

model = Video_Tokenizer(
    frame_size=(64, 64), 
    patch_size=args.patch_size,
    embed_dim=args.embed_dim,
    num_heads=args.num_heads,
    hidden_dim=args.hidden_dim,
    num_blocks=args.num_blocks,
    latent_dim=args.latent_dim,
    num_bins=args.num_bins,
).to(device)

"""
Set up optimizer and training loop
"""
# Create parameter groups to avoid weight decay on biases and norm layers
decay = []
no_decay = []
for name, param in model.named_parameters():
    if param.requires_grad:
        if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)

optimizer = optim.AdamW([
    {'params': decay, 'weight_decay': 0.01},
    {'params': no_decay, 'weight_decay': 0}
], lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)

# Create cosine scheduler with warmup
scheduler = create_cosine_scheduler(optimizer, args.n_updates)

"""
Load checkpoint if specified
"""
results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}

if args.checkpoint:
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

# Save configuration after model setup
if args.save:
    save_run_configuration(args, run_dir, timestamp, device)

# Initialize W&B if enabled and available
if args.use_wandb:
    # Create model configuration
    model_config = {
        'frame_size': (64, 64),
        'patch_size': args.patch_size,
        'embed_dim': args.embed_dim,
        'num_heads': args.num_heads,
        'hidden_dim': args.hidden_dim,
        'num_blocks': args.num_blocks,
        'latent_dim': args.latent_dim,
        'dropout': args.dropout,
        'num_bins': args.num_bins,
    }
    
    # Create W&B config
    wandb_config = {
        'batch_size': args.batch_size,
        'n_updates': args.n_updates,
        'learning_rate': args.learning_rate,
        'log_interval': args.log_interval,
        'dataset': args.dataset,
        'context_length': args.context_length,
        'model_architecture': model_config,
        'device': str(device),
        'timestamp': timestamp
    }
    
    # Initialize W&B
    run_name = args.wandb_run_name or f"video_tokenizer_{timestamp}"
    wandb.init(
        project=args.wandb_project,
        config=wandb_config,
        name=run_name,
        tags=["video-tokenizer", "training"]
    )
    
    # Watch model for gradients and parameters
    wandb.watch(model, log="all", log_freq=args.log_interval)
    
    print(f"🚀 W&B run initialized: {wandb.run.name}")
    print(f"📊 Project: {args.wandb_project}")
    print(f"🔗 View at: {wandb.run.get_url()}")

model.train()

def train():
    start_iter = max(args.start_iteration, results['n_updates'])
    
    train_iter = iter(training_loader)
    for i in tqdm(range(start_iter, args.n_updates)):
        try:
            (x, _) = next(train_iter)
        except StopIteration:
            train_iter = iter(training_loader)  # Reset iterator when epoch ends
            (x, _) = next(train_iter)
            
        x = x.to(device)
        optimizer.zero_grad()

        x_hat = model(x)

        recon_loss = torch.mean((x_hat - x)**2)
        recon_loss.backward()
        
        # Clip gradients to prevent instability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()  # Step the learning rate scheduler

        results["recon_errors"].append(recon_loss.cpu().detach())
        results["loss_vals"].append(recon_loss.cpu().detach())
        results["n_updates"] = i

        # Log to W&B if enabled and available
        if args.use_wandb:
            # Log basic metrics every step
            metrics = {
                'train/loss': recon_loss.item(),
                'train/learning_rate': scheduler.get_last_lr()[0],
                'train/x_hat_variance': torch.var(x_hat, dim=0).mean().item(),
                'train/x_variance': torch.var(x, dim=0).mean().item(),
                'step': i
            }

            # Calculate codebook usage only during log intervals
            if i % args.log_interval == 0:
                with torch.no_grad():
                    indices = model.tokenize(x)
                    unique_codes = torch.unique(indices).numel()
                    metrics['train/codebook_usage'] = unique_codes / model.codebook_size

            wandb.log(metrics)
            
            # Log system metrics
            if torch.cuda.is_available():
                wandb.log({
                    'system/gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                    'system/gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
                    'step': i
                })
            
            # Log learning rate
            for j, param_group in enumerate(optimizer.param_groups):
                wandb.log({
                    f'learning_rate/group_{j}': param_group['lr'],
                    'step': i
                })

        if i % args.log_interval == 0:
            """
            save model and print values
            """
            if args.save:
                hyperparameters = args.__dict__
                save_videotokenizer_model_and_results(
                    model, optimizer, results, hyperparameters, args.filename, checkpoints_dir)
                
                # Add visualization
                save_path = os.path.join(visualizations_dir, f'reconstruction_step_{i}_{args.filename}.png')
                visualize_reconstruction(x[:16], x_hat[:16], save_path)  # Visualize first 16 images
                
                # Log reconstruction comparison to W&B
                if args.use_wandb:
                    # Ensure tensors are on CPU and in the right format
                    original = x[:16].detach().cpu()
                    reconstructed = x_hat[:16].detach().cpu()
                    
                    # Denormalize from [-1, 1] to [0, 1] if needed
                    if original.min() < 0:
                        original = (original + 1) / 2
                        reconstructed = (reconstructed + 1) / 2
                    
                    # Clamp to valid range
                    original = torch.clamp(original, 0, 1)
                    reconstructed = torch.clamp(reconstructed, 0, 1)
                    
                    # Create comparison images
                    comparison_images = []
                    for k in range(min(16, original.shape[0])):
                        # Stack original and reconstructed side by side
                        comparison = torch.cat([original[k], reconstructed[k]], dim=2)  # Concatenate horizontally
                        comparison_images.append(comparison)
                    
            print('Update #', i, 'Recon Error:',
                  torch.mean(torch.stack(results["recon_errors"][-args.log_interval:])).item(),
                  'Loss', torch.mean(torch.stack(results["loss_vals"][-args.log_interval:])).item())
    
    # Finish W&B run
    if args.use_wandb:
        wandb.finish()
        print("✅ W&B run finished")

if __name__ == "__main__":
    train()