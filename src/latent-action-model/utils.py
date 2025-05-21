import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import os

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
    Visualize how different frame transitions map to discrete actions.
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
                    prev_norm = normalize_for_display(prev)
                    axs[i][j*2].imshow(prev_norm[0].permute(1,2,0).cpu())
                    axs[i][j*2].set_title(f'Before (A{action})')
                    axs[i][j*2].axis('off')
                    
                    # Show next frame
                    next_norm = normalize_for_display(next)
                    axs[i][j*2+1].imshow(next_norm[0].permute(1,2,0).cpu())
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
    Visualize how each discrete action affects a given frame.
    Shows predicted next frame for each possible action.
    
    Args:
        model: Trained LAM model
        frame: Single frame tensor (1, C, H, W)
        save_path: Optional path to save visualization
    """
    with torch.no_grad():
        # Try all possible actions
        actions = torch.arange(model.quantizer.embedding.num_embeddings)
        predictions = model.decode(frame.repeat(len(actions), 1, 1, 1), actions)
        
        # Calculate required subplot dimensions
        n_total = len(actions) + 1  # +1 for original frame
        n_cols = 3
        n_rows = (n_total + n_cols - 1) // n_cols  # Ceiling division
        
        # Create subplot grid
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        axs = axs.flatten()
        
        # Show original frame
        frame_norm = normalize_for_display(frame)
        axs[0].imshow(frame_norm[0].permute(1,2,0).cpu())
        axs[0].set_title('Original')
        axs[0].axis('off')
        
        # Show prediction for each action
        predictions_norm = normalize_for_display(predictions)
        for i, pred in enumerate(predictions_norm):
            axs[i+1].imshow(pred.permute(1,2,0).cpu())
            axs[i+1].set_title(f'Action {i}')
            axs[i+1].axis('off')
        
        # Turn off any unused subplots
        for i in range(len(predictions) + 1, len(axs)):
            axs[i].axis('off')
            
        plt.tight_layout()
        if save_path:
            # Create results directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
        plt.hist(actions, bins=range(model.quantizer.embedding.num_embeddings + 1), 
                align='left', rwidth=0.8)
        plt.title('Distribution of Inferred Actions')
        plt.xlabel('Action Index')
        plt.ylabel('Count')
        
        if save_path:
            plt.savefig(save_path)
        plt.close() 