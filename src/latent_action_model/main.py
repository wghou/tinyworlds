import torch
import torch.optim as optim
from models.lam import LAM
import argparse
from tqdm import tqdm
import os
from utils import (visualize_action_clusters, visualize_action_interpolation, 
                  plot_action_distribution, visualize_reconstructions, 
                  load_sequence_dataset)
import multiprocessing

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_updates", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--frame_size", type=int, default=64)
    parser.add_argument("--n_actions", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for ST-Transformer")
    parser.add_argument("--embed_dim", type=int, default=512, help="Embedding dimension for ST-Transformer")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--hidden_dim", type=int, default=2048, help="Hidden dimension for feed-forward")
    parser.add_argument("--num_blocks", type=int, default=6, help="Number of ST-Transformer blocks")
    parser.add_argument("--action_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--dataset", type=str, default="PONG")
    parser.add_argument("--seq_length", type=int, default=8, help="Length of frame sequences")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # After model init
    print("\nInitial VQ codebook variance:", 
          torch.var(model.quantizer.embedding.weight).item())

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
        
        # Print progress every 10 steps
        if (epoch + 1) % 10 == 0:
            print(f"\nStep {epoch+1}")
            print(f"Total Loss: {loss.item():.4f}")
            print(f"Reconstruction Loss: {loss_dict['recon_loss'].item():.4f}")
            print(f"VQ Loss: {loss_dict['vq_loss'].item():.4f}")
            print(f"Diversity Loss: {loss_dict['diversity_loss'].item():.4f}")
            
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
                print(f"Avg frame difference: {frame_diff:.4f}")
                
                print(f"Unique continuous actions: {unique_continuous}")
                print(f"Unique discrete actions: {unique_discrete}/{args.n_actions}")
                print(f"Discrete actions: {torch.unique(action_indices).cpu().numpy()}")
                
                # Check encoder output variance
                z_e_var = torch.var(actions_flat).item()
                print(f"Encoder output variance: {z_e_var:.4f}")
                
                # Check VQ codebook variance
                vq_var = torch.var(model.quantizer.embedding.weight).item()
                print(f"VQ codebook variance: {vq_var:.4f}")
        
        # Visualize results periodically
        vis_interval = 10
        if epoch % vis_interval == 0:
            # Create results directory if it doesn't exist
            os.makedirs('results', exist_ok=True)
            
            # Save reconstructions from current batch
            visualize_reconstructions(
                frame_sequences[:, 0], 
                frame_sequences[:, -1], 
                pred_frames[:, -1],  # Only use last prediction
                f'src/latent_action_model/training_viz/reconstructions_epoch_{epoch}.png'
            )

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
