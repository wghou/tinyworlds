import torch
import matplotlib.pyplot as plt
import os

# TODO: clean up and merge with total utils
def normalize_for_display(tensor):
    """Scale tensor values to [0,1] range for display"""
    b, c, h, w = tensor.shape
    tensor = tensor.view(b, c, -1)
    min_vals = tensor.min(dim=2, keepdim=True)[0]
    max_vals = tensor.max(dim=2, keepdim=True)[0]
    tensor = (tensor - min_vals) / (max_vals - min_vals + 1e-6)
    return tensor.view(b, c, h, w)

def visualize_action_clusters(model, test_frames, save_path=None):
    """
    Visualize how different frame transitions map to different actions.
    Shows clusters of frame pairs that map to the same action.
    
    Args:
        model: Trained LAM model
        test_frames: Tensor of sequential frames (B, T, C, H, W)
        save_path: Optional path to save visualization
    """
    with torch.no_grad():
        # Get actions for consecutive frames
        actions = []
        frame_pairs = []
        for i in range(len(test_frames)-1):
            prev_frame = test_frames[i:i+1]
            next_frame = test_frames[i+1:i+2]
            action = model.encode(prev_frame, next_frame)
            actions.append(action.item())
            frame_pairs.append((prev_frame, next_frame))
        
        # Create subplot for each unique action
        unique_actions = sorted(set(actions))
        n_unique = len(unique_actions)
        
        # Handle case with single row
        if n_unique == 1:
            fig, axs = plt.subplots(1, 4, figsize=(12, 3))
            axs = [axs]  # Make it 2D for consistent indexing
        else:
            fig, axs = plt.subplots(n_unique, 4, figsize=(12, 3*n_unique))
        
        for i, action in enumerate(unique_actions):
            # Get first 2 examples of this action
            examples = [(p, n) for (p, n), a in zip(frame_pairs, actions) if a == action][:2]
            
            if examples:
                for j, (prev, next) in enumerate(examples):
                    # Show prev frame
                    prev_norm = normalize_for_display(prev.detach().to('cpu', dtype=torch.float32))
                    axs[i][j*2].imshow(prev_norm[0].permute(1,2,0).contiguous().numpy())
                    axs[i][j*2].set_title(f'Before (A{action})')
                    axs[i][j*2].axis('off')
                    
                    # Show next frame
                    next_norm = normalize_for_display(next.detach().to('cpu', dtype=torch.float32))
                    axs[i][j*2+1].imshow(next_norm[0].permute(1,2,0).contiguous().numpy())
                    axs[i][j*2+1].set_title(f'After (A{action})')
                    axs[i][j*2+1].axis('off')
        
        plt.tight_layout()
        if save_path:
            # Create results directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.close()

def visualize_action_interpolation(model, frame, save_path=None):
    """
    Visualize how different actions affect the next frame prediction.
    Shows what each discrete action does to the input frame.
    
    Args:
        model: Trained LAM model
        frame: Single frame tensor (1, C, H, W)
        save_path: Optional path to save visualization
    """
    with torch.no_grad():
        # Get all possible actions
        n_actions = model.quantizer.n_e
        actions = torch.arange(n_actions, device=frame.device)
        
        # Generate predictions for each action
        predictions = []
        for action in actions:
            action_tensor = action.unsqueeze(0).expand(frame.shape[0])
            pred = model.decode(frame, action_tensor)
            predictions.append(pred)
        
        predictions = torch.cat(predictions, dim=0)
        
        # Create visualization grid
        n_cols = min(4, n_actions)
        n_rows = (n_actions + n_cols - 1) // n_cols
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
        if n_rows == 1:
            axs = [axs]  # Make 2D for consistent indexing
            
        for i, pred in enumerate(predictions):
            row, col = i // n_cols, i % n_cols
            pred_norm = normalize_for_display(pred.detach().unsqueeze(0).to('cpu', dtype=torch.float32))
            axs[row][col].imshow(pred_norm[0].permute(1,2,0).contiguous().numpy())
            axs[row][col].set_title(f'Action {i}')
            axs[row][col].axis('off')
            
        # Hide empty subplots
        for i in range(n_actions, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axs[row][col].axis('off')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

def plot_action_distribution(model, test_frames, save_path=None):
    """
    Plot distribution of inferred actions across a test set.
    Shows if model is using all available actions or collapsing to few.
    
    Args:
        model: Trained LAM model
        test_frames: Tensor of sequential frames (B, T, C, H, W) 
        save_path: Optional path to save visualization
    """
    with torch.no_grad():
        actions = []
        for i in range(len(test_frames)-1):
            prev_frame = test_frames[i:i+1]
            next_frame = test_frames[i+1:i+2]
            action = model.encode(prev_frame, next_frame)
            actions.append(action.item())
            
        # Plot histogram
        plt.figure(figsize=(10, 5))
        plt.hist(actions, bins=range(model.quantizer.n_e + 1), 
                align='left', rwidth=0.8)
        plt.title('Distribution of Inferred Actions')
        plt.xlabel('Action Index')
        plt.ylabel('Count')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

def visualize_reconstructions(prev_frames, next_frames, pred_next, save_path=None):
    """
    Visualize ground truth frames and reconstructed frames side by side.
    
    Args:
        prev_frames: Previous frames tensor (B, C, H, W)
        next_frames: Ground truth next frames tensor (B, C, H, W)
        pred_next: Predicted next frames tensor (B, C, H, W)
        save_path: Path to save visualization
    """
    with torch.no_grad():  # Add this to prevent gradient tracking
        # Use actual batch size, but limit to 4 for visualization
        batch_size = prev_frames.shape[0]
        n_examples = min(4, batch_size)
        
        prev_frames = prev_frames[:n_examples].detach().to('cpu', dtype=torch.float32)
        next_frames = next_frames[:n_examples].detach().to('cpu', dtype=torch.float32)
        pred_next = pred_next[:n_examples].detach().to('cpu', dtype=torch.float32)

        # Normalize for display
        prev_frames = normalize_for_display(prev_frames)
        next_frames = normalize_for_display(next_frames)
        pred_next = normalize_for_display(pred_next)
        
        # Create figure with 3 rows (prev, next, pred) and n_examples columns
        fig, axs = plt.subplots(3, n_examples, figsize=(3*n_examples, 9))
        
        # Handle single subplot case
        if n_examples == 1:
            axs = axs.reshape(3, 1)
        
        for i in range(n_examples):
            # Show previous frame
            axs[0,i].imshow(prev_frames[i].permute(1,2,0).contiguous().numpy())
            axs[0,i].set_title('Previous' if i == 0 else '')
            axs[0,i].axis('off')
            
            # Show ground truth next frame
            axs[1,i].imshow(next_frames[i].permute(1,2,0).contiguous().numpy())
            axs[1,i].set_title('Ground Truth' if i == 0 else '')
            axs[1,i].axis('off')
            
            # Show predicted next frame
            axs[2,i].imshow(pred_next[i].permute(1,2,0).contiguous().numpy())
            axs[2,i].set_title('Predicted' if i == 0 else '')
            axs[2,i].axis('off')
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.close()
