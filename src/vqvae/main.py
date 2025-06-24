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
from tqdm import tqdm
import time
import json

parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
timestamp = utils.readable_timestamp()

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=10000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=8)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--log_interval", type=int, default=250)
parser.add_argument("--dataset",  type=str, default='SONIC')
parser.add_argument("--context_length", type=int, default=4)

# Model architecture parameters
parser.add_argument("--patch_size", type=int, default=8, help="Patch size for ST-Transformer")
parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension for ST-Transformer")
parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension for feed-forward")
parser.add_argument("--num_blocks", type=int, default=2, help="Number of ST-Transformer blocks")
parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument("--codebook_size", type=int, default=256, help="Codebook size for vector quantization")

# whether or not to save model
parser.add_argument("-save", action="store_true", default=True)
parser.add_argument("--filename",  type=str, default=timestamp)

# Add checkpoint arguments
parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file to resume from")
parser.add_argument("--start_iteration", type=int, default=0, help="Iteration to start from")

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
            'codebook_size': args.codebook_size,
            'beta': args.beta
        },
        'training_parameters': {
            'batch_size': args.batch_size,
            'n_updates': args.n_updates,
            'learning_rate': args.learning_rate,
            'log_interval': args.log_interval,
            'context_length': args.context_length,
            'dataset': args.dataset
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
    dropout=args.dropout, 
    codebook_size=args.codebook_size,
    beta=args.beta
).to(device)

"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

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
    else:
        print(f"No checkpoint found at {args.checkpoint}")

# Save configuration after model setup
if args.save:
    save_run_configuration(args, run_dir, timestamp, device)

model.train()

def train():
    start_iter = max(args.start_iteration, results['n_updates'])

    print(f"Starting training")
    
    for i in tqdm(range(start_iter, args.n_updates)):
        (x, _) = next(iter(training_loader))
        x = x.to(device)
        optimizer.zero_grad()

        x_hat, vq_loss = model(x)

        recon_loss = torch.mean((x_hat - x)**2) / x_train_var
        
        # Weight the losses to prevent mode collapse
        loss = recon_loss  # Temporarily disable VQ loss to fix mode collapse
        # loss = 100.0 * recon_loss + 0.01 * vq_loss  # Much more aggressive weighting
        loss.backward()
        
        # Clip gradients to prevent instability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        # Debug loss balance
        if i % 10 == 0:  # Print every 10 iterations
            print(f"Iteration {i}:")
            print(f"  Recon Loss: {recon_loss.item():.6f}")
            print(f"  VQ Loss: {vq_loss.item():.6f}")
            print(f"  Total Loss: {loss.item():.6f}")
            print(f"  VQ/Recon Ratio: {vq_loss.item()/recon_loss.item():.2f}")
            print(f"  x_hat variance: {torch.var(x_hat, dim=0).mean().item():.6f}")
            print(f"  x variance: {torch.var(x, dim=0).mean().item():.6f}")
            print("---")

        # print variance of x_hat across the batch dimension
        print(f"Variance of x_hat: {torch.var(x_hat, dim=0).mean().item()}")

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

            print('Update #', i, 'Recon Error:',
                  np.mean(results["recon_errors"][-args.log_interval:]),
                  'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
                  'VQ Loss:', vq_loss.cpu().detach().numpy())


if __name__ == "__main__":
    train()