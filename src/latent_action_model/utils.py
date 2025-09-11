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
