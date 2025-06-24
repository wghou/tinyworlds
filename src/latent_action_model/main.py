import torch
import torch.optim as optim
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.lam import LAM
import argparse
from tqdm import tqdm
import os
from utils import visualize_reconstructions, load_sequence_dataset
import multiprocessing
import time
import json

def readable_timestamp():
    """Generate a readable timestamp for filenames"""
    return time.strftime("%a_%b_%d_%H_%M_%S_%Y")

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
            'frame_size': (args.frame_size, args.frame_size),
            'n_actions': args.n_actions,
            'patch_size': args.patch_size,
            'embed_dim': args.embed_dim,
            'num_heads': args.num_heads,
            'hidden_dim': args.hidden_dim,
            'num_blocks': args.num_blocks,
            'action_dim': args.action_dim,
            'dropout': args.dropout
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
    """
    Save LAM model checkpoint including model state, optimizer state, results and hyperparameters
    
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
    checkpoint_path = os.path.join(checkpoints_dir, f'lam_checkpoint_{timestamp}.pth')
    torch.save(results_to_save, checkpoint_path)
    return checkpoint_path

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_updates", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--frame_size", type=int, default=64)
    parser.add_argument("--n_actions", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size for ST-Transformer")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension for ST-Transformer")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension for feed-forward")
    parser.add_argument("--num_blocks", type=int, default=2, help="Number of ST-Transformer blocks")
    parser.add_argument("--action_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--dataset", type=str, default="PONG")
    parser.add_argument("--seq_length", type=int, default=8, help="Length of frame sequences")
    
    # Add save arguments
    parser.add_argument("-save", action="store_true", default=True)
    parser.add_argument("--filename", type=str, default=readable_timestamp())
    parser.add_argument("--log_interval", type=int, default=100, help="Interval for saving model and logging")
    
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create organized save directory structure
    if args.save:
        # Create main results directory for this run
        run_dir = os.path.join(os.getcwd(), 'src', 'latent_action_model', 'results', f'lam_{readable_timestamp()}')
        os.makedirs(run_dir, exist_ok=True)
        
        # Create subdirectories
        checkpoints_dir = os.path.join(run_dir, 'checkpoints')
        visualizations_dir = os.path.join(run_dir, 'visualizations')
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(visualizations_dir, exist_ok=True)
        
        print(f'Results will be saved in {run_dir}')
        print(f'Checkpoints: {checkpoints_dir}')
        print(f'Visualizations: {visualizations_dir}')

    # Load sequence data for training
    sequence_loader = load_sequence_dataset(args.batch_size, args.seq_length)

    # Initialize model with ST-Transformer architecture
    model = LAM(
        frame_size=(args.frame_size, args.frame_size),
        n_actions=args.n_actions,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        action_dim=args.action_dim,
        dropout=args.dropout
    ).to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Initialize results tracking
    results = {
        'n_updates': 0,
        'total_losses': [],
        'recon_losses': [],
        'vq_losses': [],
        'diversity_losses': [],
    }

    # After model init
    print("\nInitial VQ codebook variance:", 
          torch.var(model.quantizer.embedding.weight).item())

    # Save configuration after model setup
    if args.save:
        save_run_configuration(args, run_dir, args.filename, device)

    # Use sequence_loader for training
    train_iter = iter(sequence_loader)
    for epoch in tqdm(range(args.n_updates)):
        try:
            # Get next batch of sequences
            frame_sequences, _ = next(train_iter)
        except StopIteration:
            # Restart iterator if we run out of batches
            train_iter = iter(sequence_loader)
            frame_sequences, _ = next(train_iter)
        
        frame_sequences = frame_sequences.to(device)  # [batch_size, seq_len, channels, height, width]
        
        # Forward pass with full sequences
        loss, pred_frames, action_indices, loss_dict = model(frame_sequences)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track results
        results["total_losses"].append(loss.cpu().detach().numpy())
        results["recon_losses"].append(loss_dict['recon_loss'].cpu().detach().numpy())
        results["vq_losses"].append(loss_dict['vq_loss'].cpu().detach().numpy())
        results["diversity_losses"].append(loss_dict['diversity_loss'].cpu().detach().numpy())
        results["n_updates"] = epoch
        
        # Print progress every 10 steps
        if (epoch + 1) % 50 == 0:
            # print(f"\nStep {epoch+1}")
            # print(f"Total Loss: {loss.item():.4f}")
            # print(f"Reconstruction Loss: {loss_dict['recon_loss'].item():.4f}")
            # print(f"VQ Loss: {loss_dict['vq_loss'].item():.4f}")
            # print(f"Diversity Loss: {loss_dict['diversity_loss'].item():.4f}")
            
            # Check VQ codebook usage
            with torch.no_grad():
                # Get encoded actions for batch
                actions, _ = model.encoder(frame_sequences)
                actions_flat = actions.reshape(-1, actions.size(-1))
                
                # Get quantized indices
                _, _, action_indices = model.quantizer(actions_flat)
                
                # Print both continuous and discrete counts
                unique_continuous = torch.unique(actions_flat, dim=0).shape[0]
                unique_discrete = torch.unique(action_indices).shape[0]
                
                # Check frame differences
                frame_diff = torch.mean((frame_sequences[:, -1] - frame_sequences[:, 0])**2).item()
                # print(f"Avg frame difference: {frame_diff:.4f}")
                
                # print(f"Unique continuous actions: {unique_continuous}")
                # print(f"Unique discrete actions: {unique_discrete}/{args.n_actions}")
                # print(f"Discrete actions: {torch.unique(action_indices).cpu().numpy()}")
                
                # Check encoder output variance
                z_e_var = torch.var(actions_flat).item()
                # print(f"Encoder output variance: {z_e_var:.4f}")
                
                # Check VQ codebook variance
                vq_var = torch.var(model.quantizer.embedding.weight).item()
                # print(f"VQ codebook variance: {vq_var:.4f}")
        
        # Save model and visualize results periodically
        if epoch % args.log_interval == 0:
            if args.save:
                # Save model checkpoint
                hyperparameters = args.__dict__
                checkpoint_path = save_lam_model_and_results(model, optimizer, results, hyperparameters, args.filename, checkpoints_dir)
                
                # Save reconstructions from current batch
                visualize_reconstructions(
                    frame_sequences[:, 0], 
                    frame_sequences[:, -1], 
                    pred_frames[:, -1],  # Only use last prediction
                    os.path.join(visualizations_dir, f'reconstructions_lam_epoch_{epoch}_{args.filename}.png')
                )
                
                print(f'Epoch {epoch}: Total Loss: {loss.item():.4f}, '
                      f'Recon Loss: {loss_dict["recon_loss"].item():.4f}, '
                      f'VQ Loss: {loss_dict["vq_loss"].item():.4f}, '
                      f'Diversity Loss: {loss_dict["diversity_loss"].item():.4f}')

    # Verification: Test if similar transitions map to same actions
    print("\nVerifying model behavior...")

    # Get some test frames from the sequence loader
    test_frames, _ = next(iter(sequence_loader))
    test_frames = test_frames.to(device)

    # Test Case 1: Similar frame transitions
    prev_frame = test_frames[0:1, 0]  # First frame of first sequence
    next_frame_similar = test_frames[0:1, 1]  # Second frame of first sequence
    action1 = model.encode(prev_frame, next_frame_similar)

    # Slightly different but similar transition
    next_frame_similar2 = test_frames[0:1, 2]  # Third frame of first sequence
    action2 = model.encode(prev_frame, next_frame_similar2)

    print("Actions for similar transitions:", action1.item(), action2.item())

    # Test Case 2: Different frame transition
    next_frame_different = test_frames[0:1, -1]  # Last frame of first sequence
    action3 = model.encode(prev_frame, next_frame_different)

    print("Action for different transition:", action3.item())

if __name__ == '__main__':
    # This is required for multiprocessing on macOS
    multiprocessing.set_start_method('spawn', force=True)
    main()
