#!/usr/bin/env python3
"""
Video Tokenizer Inference Visualization

This script loads a trained video tokenizer and visualizes:
1. Ground truth sequences from the dataloader
2. Reconstructed sequences after encoding/decoding
3. Side-by-side comparisons
4. Reconstruction quality metrics

Usage:
    python visualize_video_tokenizer.py --video_tokenizer_path /path/to/checkpoint.pth
"""

import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import random
from tqdm import tqdm
import sys

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.vqvae.models.video_tokenizer import Video_Tokenizer
from datasets.utils import load_data_and_data_loaders


def load_video_tokenizer(checkpoint_path, device):
    """Load trained video tokenizer from checkpoint"""
    print(f"Loading video tokenizer from {checkpoint_path}")
    
    # Initialize model with same architecture as training
    model = Video_Tokenizer(
        frame_size=(64, 64),
        patch_size=4,
        embed_dim=128,
        num_heads=4,
        hidden_dim=512,
        num_blocks=2,
        latent_dim=6,
        num_bins=4,
        beta=0.01
    ).to(device)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"weights_only=True failed, trying weights_only=False: {e}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print("✅ Video tokenizer loaded successfully")
    
    return model


def encode_decode_sequence(video_tokenizer, sequence, device):
    """Encode and decode a video sequence"""
    with torch.no_grad():
        # Move to device
        sequence = sequence.to(device)
        
        # Encode to latent representation
        latents = video_tokenizer.encoder(sequence)  # [batch, seq_len, num_patches, latent_dim]
        
        # Quantize to get discrete tokens
        quantized_latents = video_tokenizer.vq(latents)  # [batch, seq_len, num_patches, latent_dim]
        
        # Decode back to video frames
        reconstructed = video_tokenizer.decoder(quantized_latents)  # [batch, seq_len, C, H, W]
        
        return reconstructed, latents, quantized_latents


def calculate_reconstruction_metrics(original, reconstructed):
    """Calculate reconstruction quality metrics"""
    with torch.no_grad():
        # MSE Loss
        mse_loss = torch.mean((original - reconstructed) ** 2).item()
        
        # PSNR (Peak Signal-to-Noise Ratio)
        max_val = original.max().item()
        psnr = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(torch.tensor(mse_loss))
        psnr = psnr.item()
        
        # SSIM approximation (simplified)
        # For simplicity, we'll use a basic structural similarity measure
        mu_x = original.mean()
        mu_y = reconstructed.mean()
        sigma_x = original.std()
        sigma_y = reconstructed.std()
        sigma_xy = ((original - mu_x) * (reconstructed - mu_y)).mean()
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
        ssim = ssim.item()
        
        return {
            'mse': mse_loss,
            'psnr': psnr,
            'ssim': ssim
        }


def visualize_all_sequences_comparison(sequences_data, save_path=None, title="Video Tokenizer Reconstruction - All Sequences"):
    """Create a comprehensive visualization of all sequences in a single image"""
    
    num_sequences = len(sequences_data)
    seq_len = sequences_data[0]['original'].shape[1]  # Get sequence length from first sequence
    
    # Create a large figure to accommodate all sequences
    fig, axes = plt.subplots(num_sequences * 2, seq_len, figsize=(3 * seq_len, 4 * num_sequences))
    
    # Handle single subplot case
    if num_sequences == 1 and seq_len == 1:
        axes = axes.reshape(2, 1)
    elif num_sequences == 1:
        axes = axes.reshape(2, seq_len)
    elif seq_len == 1:
        axes = axes.reshape(num_sequences * 2, 1)
    
    # Process each sequence
    for seq_idx, seq_data in enumerate(sequences_data):
        original = seq_data['original'].detach().cpu()
        reconstructed = seq_data['reconstructed'].detach().cpu()
        metrics = seq_data['metrics']
        
        # Denormalize from [-1, 1] to [0, 1] if needed
        if original.min() < 0:
            original = (original + 1) / 2
            reconstructed = (reconstructed + 1) / 2
        
        # Clamp to valid range
        original = torch.clamp(original, 0, 1)
        reconstructed = torch.clamp(reconstructed, 0, 1)
        
        # Plot original sequence (top row for each sequence)
        for i in range(seq_len):
            frame = original[0, i].permute(1, 2, 0).numpy()  # [H, W, C]
            row_idx = seq_idx * 2
            axes[row_idx, i].imshow(frame)
            axes[row_idx, i].set_title(f'Seq {seq_idx+1} - Original Frame {i+1}', fontsize=8)
            axes[row_idx, i].axis('off')
        
        # Plot reconstructed sequence (bottom row for each sequence)
        for i in range(seq_len):
            frame = reconstructed[0, i].permute(1, 2, 0).numpy()  # [H, W, C]
            row_idx = seq_idx * 2 + 1
            axes[row_idx, i].imshow(frame)
            axes[row_idx, i].set_title(f'Seq {seq_idx+1} - Reconstructed Frame {i+1}', fontsize=8)
            axes[row_idx, i].axis('off')
        
        # Add metrics text for each sequence
        metrics_text = f"MSE: {metrics['mse']:.4f}\nPSNR: {metrics['psnr']:.2f} dB\nSSIM: {metrics['ssim']:.4f}"
        fig.text(0.02, 0.98 - (seq_idx * 0.1), f"Sequence {seq_idx+1}: {metrics_text}", 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comprehensive visualization saved to: {save_path}")
    
    plt.close()  # Close the figure to prevent popup


def visualize_latent_space_summary(all_latents_data, save_path=None):
    """Create a comprehensive latent space visualization for all sequences"""
    
    # Combine all latent data
    all_latents = []
    all_quantized = []
    
    for seq_data in all_latents_data:
        all_latents.append(seq_data['latents'].detach().cpu())
        all_quantized.append(seq_data['quantized_latents'].detach().cpu())
    
    # Concatenate all sequences
    latents = torch.cat(all_latents, dim=0)  # [total_patches, latent_dim]
    quantized = torch.cat(all_quantized, dim=0)  # [total_patches, latent_dim]
    
    # Flatten spatial dimensions
    latents_flat = latents.reshape(-1, latents.shape[-1])  # [N, latent_dim]
    quantized_flat = quantized.reshape(-1, quantized.shape[-1])  # [N, latent_dim]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Latent space distribution (first 2 dimensions)
    axes[0, 0].scatter(latents_flat[:, 0], latents_flat[:, 1], alpha=0.6, s=1, label='Original')
    axes[0, 0].scatter(quantized_flat[:, 0], quantized_flat[:, 1], alpha=0.6, s=1, label='Quantized')
    axes[0, 0].set_xlabel('Latent Dim 1')
    axes[0, 0].set_ylabel('Latent Dim 2')
    axes[0, 0].set_title('Latent Space Distribution (Dim 1 vs Dim 2)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Latent space distribution (dimensions 3 vs 4)
    if latents_flat.shape[1] >= 4:
        axes[0, 1].scatter(latents_flat[:, 2], latents_flat[:, 3], alpha=0.6, s=1, label='Original')
        axes[0, 1].scatter(quantized_flat[:, 2], quantized_flat[:, 3], alpha=0.6, s=1, label='Quantized')
        axes[0, 1].set_xlabel('Latent Dim 3')
        axes[0, 1].set_ylabel('Latent Dim 4')
        axes[0, 1].set_title('Latent Space Distribution (Dim 3 vs Dim 4)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Latent dimension variance
    latent_vars = torch.var(latents_flat, dim=0)
    quantized_vars = torch.var(quantized_flat, dim=0)
    
    x = range(len(latent_vars))
    axes[1, 0].bar([i-0.2 for i in x], latent_vars, width=0.4, label='Original', alpha=0.7)
    axes[1, 0].bar([i+0.2 for i in x], quantized_vars, width=0.4, label='Quantized', alpha=0.7)
    axes[1, 0].set_xlabel('Latent Dimension')
    axes[1, 0].set_ylabel('Variance')
    axes[1, 0].set_title('Latent Dimension Variance')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Quantization error per dimension
    quantization_error = torch.mean((latents_flat - quantized_flat) ** 2, dim=0)
    axes[1, 1].bar(x, quantization_error)
    axes[1, 1].set_xlabel('Latent Dimension')
    axes[1, 1].set_ylabel('Mean Squared Error')
    axes[1, 1].set_title('Quantization Error per Dimension')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Latent space summary saved to: {save_path}")
    
    plt.close()  # Close the figure to prevent popup


def main(args):
    device = torch.device(args.device)
    
    # Load video tokenizer
    video_tokenizer = load_video_tokenizer(args.video_tokenizer_path, device)
    
    # Load data
    print("Loading dataset...")
    _, _, data_loader, _, _ = load_data_and_data_loaders(
        dataset=args.dataset, 
        batch_size=1, 
        num_frames=args.sequence_length
    )
    
    # Create output directory in inference_results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"inference_results/video_tokenizer_visualizations_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Collect data for all sequences
    all_sequences_data = []
    all_latents_data = []
    
    # Process multiple sequences
    for i in tqdm(range(args.num_sequences), desc="Processing sequences"):
        # Get random sequence from dataloader
        random_idx = random.randint(0, len(data_loader.dataset) - 1)
        sequence_data = data_loader.dataset[random_idx]  # Could be tuple or tensor
        
        # Handle different dataset return formats
        if isinstance(sequence_data, tuple):
            sequence = sequence_data[0]  # Assume first element is the sequence
        else:
            sequence = sequence_data
        
        sequence = sequence.unsqueeze(0)[:, -2:-1] # Add batch dimension [1, seq_len, C, H, W]
        
        # TODO: replace last 2 frames with black frames
        # sequence[:, -2:] = 0

        # Encode and decode
        reconstructed, latents, quantized_latents = encode_decode_sequence(
            video_tokenizer, sequence, device
        )
        
        # Calculate metrics
        metrics = calculate_reconstruction_metrics(sequence, reconstructed)
        
        # Store data for comprehensive visualization
        all_sequences_data.append({
            'original': sequence,
            'reconstructed': reconstructed,
            'metrics': metrics
        })
        
        all_latents_data.append({
            'latents': latents,
            'quantized_latents': quantized_latents
        })
        
        # Print metrics
        print(f"\nSequence {i+1} Metrics:")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")
    
    # Create comprehensive visualization of all sequences
    all_sequences_save_path = os.path.join(output_dir, "all_sequences_comparison.png")
    visualize_all_sequences_comparison(all_sequences_data, all_sequences_save_path)
    
    # Create latent space summary visualization
    latent_summary_save_path = os.path.join(output_dir, "latent_space_summary.png")
    visualize_latent_space_summary(all_latents_data, latent_summary_save_path)
    
    # Calculate and display average metrics
    all_metrics = [seq_data['metrics'] for seq_data in all_sequences_data]
    avg_metrics = {
        'mse': np.mean([m['mse'] for m in all_metrics]),
        'psnr': np.mean([m['psnr'] for m in all_metrics]),
        'ssim': np.mean([m['ssim'] for m in all_metrics])
    }
    
    print(f"\n{'='*60}")
    print("AVERAGE METRICS ACROSS ALL SEQUENCES:")
    print(f"  MSE: {avg_metrics['mse']:.4f}")
    print(f"  PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"  SSIM: {avg_metrics['ssim']:.4f}")
    print(f"{'='*60}")
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, "metrics_summary.txt")
    with open(metrics_file, 'w') as f:
        f.write("Video Tokenizer Reconstruction Metrics\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Number of sequences: {args.num_sequences}\n")
        f.write(f"Sequence length: {args.sequence_length}\n")
        f.write(f"Dataset: {args.dataset}\n\n")
        
        f.write("Average Metrics:\n")
        f.write(f"  MSE: {avg_metrics['mse']:.4f}\n")
        f.write(f"  PSNR: {avg_metrics['psnr']:.2f} dB\n")
        f.write(f"  SSIM: {avg_metrics['ssim']:.4f}\n\n")
        
        f.write("Per-Sequence Metrics:\n")
        for i, metrics in enumerate(all_metrics):
            f.write(f"Sequence {i+1}: MSE={metrics['mse']:.4f}, PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.4f}\n")
    
    print(f"\nMetrics summary saved to: {metrics_file}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"📊 Main visualization: {all_sequences_save_path}")
    print(f"🔍 Latent space analysis: {latent_summary_save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Video Tokenizer Inference Visualization")
    parser.add_argument("--video_tokenizer_path", type=str, default="/Users/almondgod/Repositories/nano-genie/src/vqvae/results/videotokenizer_sun_jul_20_18_16_29_2025/checkpoints/videotokenizer_checkpoint_sun_jul_20_18_16_29_2025.pth",
                       help="Path to video tokenizer checkpoint")
    parser.add_argument("--dataset", type=str, default="SONIC",
                       help="Dataset to use for visualization")
    parser.add_argument("--sequence_length", type=int, default=4,
                       help="Length of video sequences to visualize")
    parser.add_argument("--num_sequences", type=int, default=5,
                       help="Number of sequences to visualize")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu/cuda)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args) 