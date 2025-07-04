import numpy as np
import torch
import torch.optim as optim
import argparse
import sys
import os
import time

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import utils
from src.vqvae.models.video_tokenizer import Video_Tokenizer
from src.latent_action_model.models.lam import LAM
from src.dynamics.models.dynamics_model import DynamicsModel
from src.vqvae.utils import visualize_reconstruction, load_data_and_data_loaders
from tqdm import tqdm
import json
from einops import rearrange

parser = argparse.ArgumentParser()

def readable_timestamp():
    """Generate a readable timestamp for filenames"""
    return time.strftime("%a_%b_%d_%H_%M_%S_%Y")

"""
Hyperparameters
"""
timestamp = readable_timestamp()

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--n_updates", type=int, default=2000)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--dataset",  type=str, default='SONIC')
parser.add_argument("--context_length", type=int, default=4)

# Model architecture parameters - must match the video tokenizer parameters
parser.add_argument("--patch_size", type=int, default=4)  # Match video tokenizer
parser.add_argument("--embed_dim", type=int, default=128)  # Match video tokenizer
parser.add_argument("--num_heads", type=int, default=4)   # Match video tokenizer
parser.add_argument("--hidden_dim", type=int, default=512)  # Match video tokenizer
parser.add_argument("--num_blocks", type=int, default=2)  # Match video tokenizer
parser.add_argument("--latent_dim", type=int, default=32)  # Match video tokenizer latent_dim
parser.add_argument("--dropout", type=float, default=0.1)

# Paths to pre-trained models
parser.add_argument("--video_tokenizer_path", type=str, required=True, 
                   help="Path to pre-trained video tokenizer checkpoint")
parser.add_argument("--lam_path", type=str, required=True, 
                   help="Path to pre-trained latent action model checkpoint")

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
    run_dir = os.path.join(os.getcwd(), 'src', 'dynamics', 'results', f'dynamics_{timestamp}')
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
            'height': 64,
            'width': 64,
            'channels': 3
        },
        'training_parameters': {
            'batch_size': args.batch_size,
            'n_updates': args.n_updates,
            'learning_rate': args.learning_rate,
            'log_interval': args.log_interval,
            'context_length': args.context_length,
            'dataset': args.dataset
        },
        'pretrained_models': {
            'video_tokenizer_path': args.video_tokenizer_path,
            'lam_path': args.lam_path
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

def save_dynamics_model_and_results(model, optimizer, results, hyperparameters, timestamp, checkpoints_dir):
    """
    Save dynamics model checkpoint including model state, optimizer state, results and hyperparameters
    
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
    checkpoint_path = os.path.join(checkpoints_dir, f'dynamics_checkpoint_{timestamp}.pth')
    torch.save(results_to_save, checkpoint_path)
    return checkpoint_path

"""
Load data and define batch data loaders
"""
training_data, validation_data, training_loader, validation_loader, x_train_var = load_data_and_data_loaders(
    dataset=args.dataset, 
    batch_size=args.batch_size, 
    num_frames=args.context_length
)

"""
Load pre-trained models
"""
print("Loading pre-trained video tokenizer...")
video_tokenizer = Video_Tokenizer(
    frame_size=(64, 64), 
    patch_size=args.patch_size,
    embed_dim=args.embed_dim,
    num_heads=args.num_heads,
    hidden_dim=args.hidden_dim,
    num_blocks=args.num_blocks,
    latent_dim=args.latent_dim,
    dropout=args.dropout, 
    codebook_size=64,
    beta=0.01
).to(device)

# Load video tokenizer checkpoint
if os.path.isfile(args.video_tokenizer_path):
    print(f"Loading video tokenizer from {args.video_tokenizer_path}")
    checkpoint = torch.load(args.video_tokenizer_path, map_location=device)
    video_tokenizer.load_state_dict(checkpoint['model'])
    video_tokenizer.eval()  # Set to evaluation mode
else:
    raise FileNotFoundError(f"Video tokenizer checkpoint not found at {args.video_tokenizer_path}")

print("Loading pre-trained latent action model...")
lam = LAM(
    frame_size=(64, 64),
    n_actions=8,  # Match full pipeline
    patch_size=8,  # Match full pipeline
    embed_dim=args.embed_dim,
    num_heads=args.num_heads,
    hidden_dim=args.hidden_dim,
    num_blocks=args.num_blocks,
    action_dim=32,  # Match full pipeline
    dropout=args.dropout
).to(device)

# Load LAM checkpoint
if os.path.isfile(args.lam_path):
    print(f"Loading LAM from {args.lam_path}")
    checkpoint = torch.load(args.lam_path, map_location=device)
    lam.load_state_dict(checkpoint['model'])
    lam.eval()  # Set to evaluation mode
else:
    raise FileNotFoundError(f"LAM checkpoint not found at {args.lam_path}")

"""
Set up dynamics model
"""
print("Initializing dynamics model...")
dynamics_model = DynamicsModel(
    frame_size=(64, 64),
    patch_size=args.patch_size,
    embed_dim=args.embed_dim,
    num_heads=args.num_heads,
    hidden_dim=args.hidden_dim,
    num_blocks=args.num_blocks,
    latent_dim=args.latent_dim,
    dropout=args.dropout,
    height=64,
    width=64,
    channels=3
).to(device)

"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(dynamics_model.parameters(), lr=args.learning_rate, amsgrad=True)

"""
Load checkpoint if specified
"""
results = {
    'n_updates': 0,
    'dynamics_losses': [],
    'loss_vals': [],
}

if args.checkpoint:
    if os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        dynamics_model.load_state_dict(checkpoint['model'])
        results = checkpoint['results']
        print(f"Resuming from update {results['n_updates']}")
        
        # Restore optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print(f"No checkpoint found at {args.checkpoint}")

dynamics_model.train()

# Save configuration after model setup
if args.save:
    save_run_configuration(args, run_dir, timestamp, device)

def train():
    start_iter = max(args.start_iteration, results['n_updates'])

    print(f"Starting dynamics model training")
    
    for i in tqdm(range(start_iter, args.n_updates)):
        (x, _) = next(iter(training_loader))
        x = x.to(device)  # [batch_size, seq_len, channels, height, width]
        optimizer.zero_grad()

        # Get video tokenizer latents for current frames
        with torch.no_grad():
            video_latents = video_tokenizer.encoder(x)  # [batch_size, seq_len, num_patches, latent_dim]
            # Apply vector quantization to get discrete latents
            _, quantized_video_latents, _ = video_tokenizer.vq(video_latents) # [batch_size, seq_len, num_patches, latent_dim]
        
        # Get action latents for frame transitions
        with torch.no_grad():
            # Encode frame sequences to get actions
            actions, _ = lam.encoder(x)  # [batch_size, seq_len-1, action_dim]
            # Quantize actions
            actions_flat = actions.reshape(-1, actions.size(-1)) # [batch_size * seq_len-1, action_dim]
            _, quantized_actions_flat, _ = lam.quantizer(actions_flat) # [batch_size * seq_len-1, action_dim]
            quantized_actions = quantized_actions_flat.reshape(actions.shape)  # [batch_size, seq_len-1, action_dim]
            
            # Pad the quantized actions at the end to match the sequence length
            batch_size, seq_len, num_patches, latent_dim = quantized_video_latents.shape # [batch_size, seq_len, num_patches, latent_dim]
            zero_action = torch.zeros(batch_size, 1, 32, device=device)  # [batch_size, 1, action_dim]
            quantized_actions_padded = torch.cat([quantized_actions, zero_action], dim=1) # [batch_size, seq_len, action_dim]
            
            # Expand action latents to match patch dimension
            quantized_actions_padded = rearrange(quantized_actions_padded, 'b s a -> b s 1 a')  # [batch_size, seq_len, 1, action_dim]

        # Combine video latents and action latents
        combined_latents = quantized_video_latents + quantized_actions_padded  # [batch_size, seq_len, num_patches, latent_dim]
        
        # Predict next frame latents using dynamics model
        predicted_next_latents = dynamics_model(combined_latents, training=True)  # [batch_size, seq_len, num_patches, latent_dim]
        
        # Get target next frame latents (shift by 1)
        target_next_latents = quantized_video_latents[:, 1:]  # [batch_size, seq_len-1, num_patches, latent_dim]
        predicted_next_latents = predicted_next_latents[:, :-1]  # [batch_size, seq_len-1, num_patches, latent_dim]
        
        # Compute dynamics loss
        dynamics_loss = torch.mean((predicted_next_latents - target_next_latents)**2)
        
        # Total loss
        loss = dynamics_loss
        loss.backward()
        
        # Clip gradients to prevent instability
        torch.nn.utils.clip_grad_norm_(dynamics_model.parameters(), max_norm=1.0)
        
        optimizer.step()

        results["dynamics_losses"].append(dynamics_loss.cpu().detach())
        results["loss_vals"].append(loss.cpu().detach())
        results["n_updates"] = i

        # Debug prints
        if i % 10 == 0:  # Print every 10 iterations
            print(f"Iteration {i}:")
            print(f"  Dynamics Loss: {dynamics_loss.item():.6f}")
            print(f"  Video latents variance: {torch.var(quantized_video_latents).item():.6f}")
            print(f"  Action latents variance: {torch.var(quantized_actions_padded).item():.6f}")
            print(f"  Combined latents variance: {torch.var(combined_latents).item():.6f}")
            print(f"  Predicted next latents variance: {torch.var(predicted_next_latents).item():.6f}")
            print("---")

        if i % args.log_interval == 0:
            """
            save model and print values
            """
            if args.save:
                hyperparameters = args.__dict__
                checkpoint_path = save_dynamics_model_and_results(
                    dynamics_model, optimizer, results, hyperparameters, args.filename, checkpoints_dir)
                
                # Add visualization - decode predicted latents back to frames
                with torch.no_grad():
                    # Use video tokenizer decoder to convert predicted latents back to frames
                    predicted_frames = video_tokenizer.decoder(predicted_next_latents[:16])  # First 16 samples
                    target_frames = x[:16, 1:]  # Target frames (shifted by 1)
                    
                    save_path = os.path.join(visualizations_dir, f'dynamics_prediction_step_{i}_{args.filename}.png')
                    visualize_reconstruction(target_frames[:16], predicted_frames[:16], save_path)

            print('Update #', i, 'Dynamics Loss:',
                  torch.mean(torch.stack(results["dynamics_losses"][-args.log_interval:])).item(),
                  'Total Loss', torch.mean(torch.stack(results["loss_vals"][-args.log_interval:])).item())


if __name__ == "__main__":
    train()