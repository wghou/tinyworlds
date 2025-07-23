import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import random
import sys

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.vqvae.models.video_tokenizer import Video_Tokenizer
from src.dynamics.models.dynamics_model import DynamicsModel
from src.vqvae.utils import load_data_and_data_loaders

def load_video_tokenizer(checkpoint_path, device):
    print(f"Loading video tokenizer from {checkpoint_path}")
    model = Video_Tokenizer(
        frame_size=(64, 64),
        patch_size=4,
        embed_dim=128,
        num_heads=4,
        hidden_dim=512,
        num_blocks=2,
        latent_dim=6,
        dropout=0.1,
        num_bins=4,
        beta=0.01
    ).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"weights_only=True failed, trying weights_only=False: {e}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print("✅ Video tokenizer loaded successfully")
    return model

def load_dynamics_model(checkpoint_path, device):
    print(f"Loading dynamics model from {checkpoint_path}")
    model = DynamicsModel(
        frame_size=(64, 64),
        patch_size=4,
        embed_dim=128,
        num_heads=4,
        hidden_dim=512,
        num_blocks=2,
        latent_dim=6,
        num_bins=4,
        dropout=0.1
    ).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"weights_only=True failed, trying weights_only=False: {e}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print("✅ Dynamics model loaded successfully")
    return model

def visualize_three_sequences(gt, tokenizer_recon, dynamics_recon, save_path):
    # gt, tokenizer_recon, dynamics_recon: [1, seq_len, C, H, W]
    seq_len = gt.shape[1]
    fig, axes = plt.subplots(3, seq_len, figsize=(3*seq_len, 9))
    row_labels = ["Ground Truth", "Detokenized Tokens (Tokenizer Recon)", "Detokenized Dynamics Tokens (Dynamics Recon)"]
    for row, data in enumerate([gt, tokenizer_recon, dynamics_recon]):
        # Denormalize if needed
        d = data.detach().cpu()
        if d.min() < 0:
            d = (d + 1) / 2
        d = torch.clamp(d, 0, 1)
        for col in range(seq_len):
            axes[row, col].imshow(d[0, col].permute(1, 2, 0).numpy())
            if row == 0:
                axes[row, col].set_title(f"Frame {col+1}")
            axes[row, col].axis('off')
        # Add row label on the left, vertically centered
        axes[row, 0].set_ylabel(row_labels[row], fontsize=14, rotation=0, labelpad=80, va='center')
    plt.suptitle("Ground Truth vs Detokenized Tokens vs Detokenized Dynamics Tokens", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {save_path}")

def main(args):
    device = torch.device(args.device)
    # Load models
    video_tokenizer = load_video_tokenizer(args.video_tokenizer_path, device)
    dynamics_model = load_dynamics_model(args.dynamics_model_path, device)
    # Load data
    _, _, data_loader, _, _ = load_data_and_data_loaders(
        dataset=args.dataset,
        batch_size=1,
        num_frames=args.sequence_length
    )
    # Sample a sequence
    random_idx = random.randint(0, len(data_loader.dataset) - 1)
    sequence_data = data_loader.dataset[random_idx]
    if isinstance(sequence_data, tuple):
        sequence = sequence_data[0]
    else:
        sequence = sequence_data
    sequence = sequence.unsqueeze(0).to(device)  # [1, seq_len, C, H, W]
    print(f"Sampled sequence shape: {sequence.shape}")
    with torch.no_grad():
        # Tokenize (encode + quantize)
        latents = video_tokenizer.encoder(sequence)  # [1, seq_len, num_patches, latent_dim]
        quantized_latents = video_tokenizer.vq(latents)
        print(f"Quantized latents shape: {quantized_latents.shape}")
        # Detokenize (decode quantized latents)
        tokenizer_recon = video_tokenizer.decoder(quantized_latents, training=False)  # [1, seq_len, C, H, W]
        print(f"Tokenizer recon shape: {tokenizer_recon.shape}")
        # Run dynamics model on quantized latents
        predicted_latents = dynamics_model(quantized_latents, training=False)  # [1, seq_len, num_patches, latent_dim]
        print(f"Dynamics model predicted latents shape: {predicted_latents.shape}")
        # Detokenize (decode predicted latents)
        dynamics_recon = video_tokenizer.decoder(predicted_latents, training=False)  # [1, seq_len, C, H, W]
        print(f"Dynamics recon shape: {dynamics_recon.shape}")
    # Visualization
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = "inference_results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"tokenizer_vs_dynamics_{timestamp}.png")
    visualize_three_sequences(sequence, tokenizer_recon, dynamics_recon, save_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Video Tokenizer vs Dynamics Model")
    parser.add_argument("--video_tokenizer_path", type=str, default="/Users/almondgod/Repositories/nano-genie/src/vqvae/results/videotokenizer_sun_jul_20_18_16_29_2025/checkpoints/videotokenizer_checkpoint_sun_jul_20_18_16_29_2025.pth")
    parser.add_argument("--dynamics_model_path", type=str, default="/Users/almondgod/Repositories/nano-genie/src/dynamics/results/dynamics_Sun_Jul_20_18_39_32_2025/checkpoints/dynamics_checkpoint_Sun_Jul_20_18_39_32_2025.pth")
    parser.add_argument("--dataset", type=str, default="SONIC", help="Dataset to use")
    parser.add_argument("--sequence_length", type=int, default=4, help="Sequence length to sample and visualize")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args) 