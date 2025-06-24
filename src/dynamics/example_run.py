#!/usr/bin/env python3
"""
Example script to run the dynamics model trainer.

This script demonstrates how to train a dynamics model using pre-trained
video tokenizer and latent action model checkpoints.

Usage:
    python example_run.py

Make sure you have:
1. A trained video tokenizer checkpoint (e.g., from src/vqvae/main.py)
2. A trained latent action model checkpoint (e.g., from src/latent_action_model/main.py)
3. The Pong dataset in data/pong_frames.h5
"""

import subprocess
import sys
import os

def main():
    # Example paths - update these to your actual checkpoint paths
    # These should point to the checkpoints subdirectory within each model's results folder
    video_tokenizer_path = "src/vqvae/results/videotokenizer_sun_jun_22_23_50_36_2025/checkpoints/videotokenizer_checkpoint_sun_jun_22_23_50_36_2025.pth"
    lam_path = "src/latent_action_model/results/lam_sun_jun_22_23_50_36_2025/checkpoints/lam_checkpoint_sun_jun_22_23_50_36_2025.pth"
    
    # Check if checkpoint files exist
    if not os.path.exists(video_tokenizer_path):
        print(f"Error: Video tokenizer checkpoint not found at {video_tokenizer_path}")
        print("Please train a video tokenizer first using src/vqvae/main.py")
        return
    
    if not os.path.exists(lam_path):
        print(f"Error: LAM checkpoint not found at {lam_path}")
        print("Please train a latent action model first using src/latent_action_model/main.py")
        return
    
    # Command to run the dynamics model trainer
    cmd = [
        sys.executable, "src/dynamics/main.py",
        "--video_tokenizer_path", video_tokenizer_path,
        "--lam_path", lam_path,
        "--batch_size", "16",  # Smaller batch size for memory efficiency
        "--n_updates", "5000",  # Number of training iterations
        "--learning_rate", "1e-4",
        "--log_interval", "100",  # Log every 100 iterations
        "--context_length", "4",  # Sequence length
        "--patch_size", "8",      # Match video tokenizer
        "--embed_dim", "128",     # Match video tokenizer
        "--num_heads", "4",       # Match video tokenizer
        "--hidden_dim", "512",    # Match video tokenizer
        "--num_blocks", "2",      # Match video tokenizer
        "--latent_dim", "16",     # Match video tokenizer latent_dim
        "--dropout", "0.1"
    ]
    
    print("Starting dynamics model training...")
    print(f"Video tokenizer: {video_tokenizer_path}")
    print(f"LAM: {lam_path}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        # Run the training
        subprocess.run(cmd, check=True)
        print("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return

if __name__ == "__main__":
    main() 