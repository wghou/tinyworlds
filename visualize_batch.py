#!/usr/bin/env python3
"""
Visualize a batch of video sequences as a grid
Each row = batch element, each column = timestep
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.vqvae.utils import load_data_and_data_loaders

def visualize_batch(frames, save_path=None, title="Video Sequences Batch", max_batch_size=8, max_seq_length=8):
    """
    Visualize a batch of video sequences as a grid
    
    Args:
        frames: Tensor of shape [batch_size, seq_len, C, H, W] with values in [-1, 1]
        save_path: Path to save the visualization (optional)
        title: Title for the plot
        max_batch_size: Maximum number of batch elements to show
        max_seq_length: Maximum number of timesteps to show
    """
    # Move to CPU and get dimensions
    frames = frames.detach().cpu()
    batch_size, seq_len, C, H, W = frames.shape
    
    # Limit batch size and sequence length for visualization
    batch_size = min(batch_size, max_batch_size)
    seq_len = min(seq_len, max_seq_length)
    frames = frames[:batch_size, :seq_len]
    
    # Denormalize from [-1, 1] to [0, 1]
    frames = (frames + 1) / 2
    frames = torch.clamp(frames, 0, 1)
    
    # Create figure
    fig, axes = plt.subplots(batch_size, seq_len, figsize=(2 * seq_len, 2 * batch_size))
    
    # Handle single row/column case
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    if seq_len == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each frame
    for i in range(batch_size):
        for j in range(seq_len):
            frame = frames[i, j].permute(1, 2, 0).numpy()  # [H, W, C]
            
            axes[i, j].imshow(frame)
            axes[i, j].set_title(f'B{i}, T{j}', fontsize=10)
            axes[i, j].axis('off')
    
    # Add row and column labels
    for i in range(batch_size):
        axes[i, 0].set_ylabel(f'Batch {i}', fontsize=12, fontweight='bold')
    
    for j in range(seq_len):
        axes[0, j].set_title(f'Timestep {j}', fontsize=12, fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def visualize_batch_with_stats(frames, save_path=None, title="Video Sequences Batch with Statistics"):
    """
    Visualize a batch with additional statistics
    """
    # Move to CPU and get dimensions
    frames = frames.detach().cpu()
    batch_size, seq_len, C, H, W = frames.shape
    
    # Calculate statistics
    frame_mean = frames.mean().item()
    frame_std = frames.std().item()
    frame_min = frames.min().item()
    frame_max = frames.max().item()
    
    print(f"Batch Statistics:")
    print(f"  Shape: {frames.shape}")
    print(f"  Mean: {frame_mean:.4f}")
    print(f"  Std: {frame_std:.4f}")
    print(f"  Range: [{frame_min:.4f}, {frame_max:.4f}]")
    
    # Create visualization
    visualize_batch(frames, save_path, title)
    
    return {
        'mean': frame_mean,
        'std': frame_std,
        'min': frame_min,
        'max': frame_max,
        'shape': frames.shape
    }

def visualize_multiple_batches(dataloader, num_batches=3, save_dir="batch_visualizations"):
    """
    Visualize multiple batches from a dataloader
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for batch_idx, (frames, _) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        print(f"\nBatch {batch_idx + 1}:")
        stats = visualize_batch_with_stats(
            frames, 
            save_path=os.path.join(save_dir, f"batch_{batch_idx + 1}.png"),
            title=f"Batch {batch_idx + 1} - Video Sequences"
        )

def main():
    parser = argparse.ArgumentParser(description="Visualize video sequence batches")
    parser.add_argument("--dataset", type=str, default="SONIC", help="Dataset to use")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=8, help="Sequence length")
    parser.add_argument("--num_batches", type=int, default=1, help="Number of batches to visualize")
    parser.add_argument("--save_dir", type=str, default="batch_visualizations", help="Directory to save visualizations")
    parser.add_argument("--max_batch_display", type=int, default=8, help="Maximum batch elements to display")
    parser.add_argument("--max_seq_display", type=int, default=8, help="Maximum timesteps to display")
    args = parser.parse_args()
    
    print(f"Loading {args.dataset} dataset...")
    
    # Load data
    _, _, validation_loader, _, _ = load_data_and_data_loaders(
        dataset=args.dataset, 
        batch_size=args.batch_size, 
        num_frames=args.seq_length
    )
    
    print(f"Dataset loaded. Visualizing {args.num_batches} batch(es)...")
    
    # Visualize batches
    for batch_idx, (frames, _) in enumerate(validation_loader):
        if batch_idx >= args.num_batches:
            break
        
        print(f"\n{'='*60}")
        print(f"Batch {batch_idx + 1}")
        print(f"{'='*60}")
        
        # Calculate statistics
        frames_cpu = frames.detach().cpu()
        batch_size, seq_len, C, H, W = frames_cpu.shape
        
        print(f"Batch shape: {frames_cpu.shape}")
        print(f"Frame range: [{frames_cpu.min().item():.4f}, {frames_cpu.max().item():.4f}]")
        print(f"Frame mean: {frames_cpu.mean().item():.4f}")
        print(f"Frame std: {frames_cpu.std().item():.4f}")
        
        # Create visualization
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, f"batch_{batch_idx + 1}.png")
        visualize_batch(
            frames, 
            save_path=save_path,
            title=f"Batch {batch_idx + 1} - {args.dataset} Sequences",
            max_batch_size=args.max_batch_display,
            max_seq_length=args.max_seq_display
        )
        
        print(f"Visualization saved to: {save_path}")
    
    print(f"\n✅ Visualization complete! Check {args.save_dir}/ for saved images.")

if __name__ == "__main__":
    main() 