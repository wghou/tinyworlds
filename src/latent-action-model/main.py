import torch
import torch.optim as optim
from models.lam import LAM
import argparse
from tqdm import tqdm
from utils import visualize_action_clusters, visualize_action_interpolation, plot_action_distribution

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=3e-3)
parser.add_argument("--frame_size", type=int, default=64)
parser.add_argument("--n_actions", type=int, default=8)
parser.add_argument("--h_dim", type=int, default=128)
parser.add_argument("--res_h_dim", type=int, default=64)
parser.add_argument("--n_res_layers", type=int, default=2)
parser.add_argument("--action_dim", type=int, default=64)
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = LAM(
    frame_size=(args.frame_size, args.frame_size),
    n_actions=args.n_actions,
    h_dim=args.h_dim,
    res_h_dim=args.res_h_dim, 
    n_res_layers=args.n_res_layers,
    action_dim=args.action_dim
).to(device)

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# After model init
print("\nInitial VQ codebook variance:", 
      torch.var(model.quantizer.embedding.weight).item())

# Training loop
for epoch in tqdm(range(args.n_updates)):
    # Generate synthetic data with controlled differences
    prev_frame = torch.randn(args.batch_size, 3, args.frame_size, args.frame_size).to(device)

    # Create different types of transitions
    batch_splits = args.batch_size // 4
    next_frame = torch.zeros_like(prev_frame)
    
    # Large random changes
    next_frame[:batch_splits] = torch.randn_like(prev_frame[:batch_splits])
    
    # Small random changes
    next_frame[batch_splits:2*batch_splits] = prev_frame[batch_splits:2*batch_splits] + \
        0.1 * torch.randn_like(prev_frame[batch_splits:2*batch_splits])
    
    # Horizontal shifts
    next_frame[2*batch_splits:3*batch_splits] = torch.roll(
        prev_frame[2*batch_splits:3*batch_splits], 
        shifts=10, dims=3
    )
    
    # Vertical shifts
    next_frame[3*batch_splits:] = torch.roll(
        prev_frame[3*batch_splits:], 
        shifts=10, dims=2
    )
    
    # Add noise to prevent exact patterns
    next_frame = next_frame + 0.01 * torch.randn_like(next_frame)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    loss, pred_next, actions = model(prev_frame, next_frame)
    
    # Add diversity loss
    action_probs = torch.bincount(actions, minlength=args.n_actions).float()
    action_probs = action_probs / action_probs.sum()
    diversity_loss = -torch.sum(action_probs * torch.log(action_probs + 1e-10))
    loss = loss - 0 * diversity_loss
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Print progress every 100 steps
    if (epoch + 1) % 100 == 0:
        print(f"\nStep {epoch+1}")
        print(f"Loss: {loss.item():.4f}")
        
        # Check VQ codebook usage
        with torch.no_grad():
            # Get encoded actions for batch
            z_e = model.encoder(prev_frame, next_frame)
            _, _, indices = model.quantizer(z_e)
            unique_actions = torch.unique(indices)
            print(f"Unique actions used: {len(unique_actions)}/{args.n_actions}")
            print(f"Actions: {unique_actions.cpu().numpy()}")
            
            # Check frame differences
            frame_diff = torch.mean((next_frame - prev_frame)**2).item()
            print(f"Avg frame difference: {frame_diff:.4f}")
            
            # Check encoder output variance
            z_e_var = torch.var(z_e).item()
            print(f"Encoder output variance: {z_e_var:.4f}")
            
            # Check VQ codebook variance
            vq_var = torch.var(model.quantizer.embedding.weight).item()
            print(f"VQ codebook variance: {vq_var:.4f}")
    
    # Visualize results periodically
    vis_interval = 100
    test_frames = torch.randn(10, 3, args.frame_size, args.frame_size).to(device)  # Remove clamp
    test_frame = test_frames[0:1]
    if epoch % vis_interval == 0:
        visualize_action_clusters(model, test_frames, f'results/clusters_epoch_{epoch}.png')
        visualize_action_interpolation(model, test_frame, f'results/actions_epoch_{epoch}.png')
        plot_action_distribution(model, test_frames, f'results/dist_epoch_{epoch}.png')

# Verification: Test if similar transitions map to same actions
print("\nVerifying model behavior...")

# Test Case 1: Similar frame transitions
prev_frame = torch.randn(1, 3, args.frame_size, args.frame_size).to(device)
next_frame_similar = prev_frame + 0.1 * torch.randn_like(prev_frame)
action1 = model.encode(prev_frame, next_frame_similar)

# Slightly different but similar transition
next_frame_similar2 = prev_frame + 0.1 * torch.randn_like(prev_frame)
action2 = model.encode(prev_frame, next_frame_similar2)

print("Actions for similar transitions:", action1.item(), action2.item())

# Test Case 2: Different frame transition
next_frame_different = torch.randn_like(prev_frame)  # Completely different frame
action3 = model.encode(prev_frame, next_frame_different)

print("Action for different transition:", action3.item())
