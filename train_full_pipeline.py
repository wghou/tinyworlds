#!/usr/bin/env python3
"""
Full Pipeline Training Script for SONIC Dataset

This script trains all three models in sequence:
1. Video Tokenizer on SONIC
2. LAM on SONIC  
3. Dynamics Model using the trained checkpoints

Prerequisites:
- SONIC dataset preprocessed and saved as data/sonic_frames.h5
- All dependencies installed (torch, einops, etc.)

Usage:
    python train_full_pipeline.py

The script will:
1. Train the video tokenizer on SONIC frames
2. Train the latent action model (LAM) on SONIC frame sequences
3. Find the latest checkpoints from steps 1 and 2
4. Train the dynamics model using the pre-trained checkpoints
5. Save all results in organized timestamped directories

Each model will save:
- Checkpoints in results/[model]/checkpoints/
- Visualizations in results/[model]/visualizations/
- Configuration in results/[model]/run_config.json
"""

import subprocess
import sys
import os
import time
import glob
from pathlib import Path
import argparse

def readable_timestamp():
    """Generate a readable timestamp for filenames"""
    return time.strftime("%a_%b_%d_%H_%M_%S_%Y")

def find_latest_checkpoint(base_dir, model_name):
    """
    Find the latest checkpoint for a given model
    
    Args:
        base_dir: Base directory to search in
        model_name: Name of the model (videotokenizer, lam, dynamics)
    
    Returns:
        Path to the latest checkpoint file
    """
    # Pattern to match the model's results directory
    pattern = os.path.join(base_dir, f"src/*/results/{model_name}_*")
    results_dirs = glob.glob(pattern)
    
    if not results_dirs:
        print(f"No checkpoints found for {model_name}")
        return None
    
    # Get the most recent directory (latest timestamp)
    latest_dir = max(results_dirs, key=os.path.getctime)
    
    # Look for checkpoint file
    checkpoint_pattern = os.path.join(latest_dir, "checkpoints", f"{model_name}_checkpoint_*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        print(f"No checkpoint files found for {model_name}")
        return None
    
    # Return the most recent checkpoint
    return max(checkpoint_files, key=os.path.getctime)

def run_command(cmd, description):
    """
    Run a command and handle errors
    
    Args:
        cmd: List of command arguments
        description: Description of what the command does
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Starting: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ {description} interrupted by user")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Full Pipeline Training Script")
    parser.add_argument("--use_wandb", action="store_true", default=False, 
                       help="Enable Weights & Biases logging for all models")
    parser.add_argument("--wandb_project", type=str, default="nano-genie-pipeline",
                       help="Base project name for W&B (each model will have its own project)")
    args = parser.parse_args()
    
    print("🚀 Starting Full Pipeline Training on SONIC Dataset")
    print(f"Timestamp: {readable_timestamp()}")
    
    if args.use_wandb:
        print("📊 W&B logging enabled")
        print(f"📈 Base project: {args.wandb_project}")
    
    # Check if we're in the right directory
    if not os.path.exists("src"):
        print("❌ Error: Please run this script from the nano-genie root directory")
        return
    
    # Check if SONIC dataset exists
    sonic_data_path = "data/sonic_frames.h5"
    if not os.path.exists(sonic_data_path):
        print(f"❌ Error: SONIC dataset not found at {sonic_data_path}")
        print("Please ensure the SONIC dataset is preprocessed and available")
        print("Note: The pipeline expects a preprocessed HDF5 file with frames")
        return
    
    print(f"✅ Found SONIC dataset at {sonic_data_path}")
    
    # Step 1: Train Video Tokenizer
    print("\n" + "="*60)
    print("STEP 1: Training Video Tokenizer on SONIC")
    print("="*60)
    
    video_tokenizer_cmd = [
        sys.executable, "src/vqvae/main.py",
        "--dataset", "SONIC",
        "--batch_size", "16",
        "--n_updates", "2000",  # Reduced for faster training
        "--learning_rate", "1e-4",  # Increased from 1e-4 for better convergence
        "--log_interval", "100",
        "--context_length", "4",
        "--patch_size", "4",
        "--embed_dim", "128",
        "--num_heads", "4",
        "--hidden_dim", "512",
        "--num_blocks", "2",
        "--latent_dim", "6",
        "--dropout", "0.1",
        "--num_bins", "4",  # Number of bins per dimension for FSQ
        "--use_wandb"
    ]
    
    # # Add W&B arguments if enabled
    if args.use_wandb:
        video_tokenizer_cmd.extend([
            "--use_wandb",
            "--wandb_project", f"{args.wandb_project}"
        ])
    
    if not run_command(video_tokenizer_cmd, "Video Tokenizer Training"):
        print("❌ Video tokenizer training failed. Stopping pipeline.")
        return
    
    # Step 2: Train LAM
    # print("\n" + "="*60)
    # print("STEP 2: Training LAM on SONIC")
    # print("="*60)
    
    # lam_cmd = [
    #     sys.executable, "src/latent_action_model/main.py",
    #     "--dataset", "SONIC",
    #     "--batch_size", "16",
    #     "--n_updates", "1000",  # Reduced for faster training
    #     "--learning_rate", "3e-3",
    #     "--log_interval", "50",
    #     "--seq_length", "8",
    #     "--patch_size", "4",  # Match video tokenizer patch_size
    #     "--embed_dim", "128",
    #     "--num_heads", "4",
    #     "--hidden_dim", "512",
    #     "--num_blocks", "2",
    #     "--action_dim", "6",
    #     "--dropout", "0.1",
    #     "--n_actions", "8",  # Exactly 8 actions for SONIC (up, down, left, right, up-left, up-right, down-left, down-right)
    #     "--beta", "1.0"  # VQ loss weight
    # ]
    
    # # Add W&B arguments if enabled
    # if args.use_wandb:
    #     lam_cmd.extend([
    #         "--use_wandb",
    #         "--wandb_project", f"{args.wandb_project}-lam"
    #     ])
    
    # if not run_command(lam_cmd, "LAM Training"):
    #     print("❌ LAM training failed. Stopping pipeline.")
    #     return
    
    # Step 3: Find the latest checkpoints
    print("\n" + "="*60)
    print("STEP 3: Finding Latest Checkpoints")
    print("="*60)
    
    video_tokenizer_checkpoint = find_latest_checkpoint(".", "videotokenizer")
    # lam_checkpoint = find_latest_checkpoint(".", "lam")
    
    if not video_tokenizer_checkpoint:
        print("❌ Could not find video tokenizer checkpoint")
        return
    
    # if not lam_checkpoint:
    #     print("❌ Could not find LAM checkpoint")
    #     return
    
    print(f"✅ Found video tokenizer checkpoint: {video_tokenizer_checkpoint}")
    # print(f"✅ Found LAM checkpoint: {lam_checkpoint}")
    
    # Step 4: Train Dynamics Model
    print("\n" + "="*60)
    print("STEP 4: Training Dynamics Model")
    print("="*60)
    
    dynamics_cmd = [
        sys.executable, "src/dynamics/main.py",
        "--video_tokenizer_path", video_tokenizer_checkpoint,
        # "--lam_path", lam_checkpoint,
        "--dataset", "SONIC",
        "--batch_size", "16",
        "--n_updates", "2000",
        "--learning_rate", "1e-4",
        "--log_interval", "100",
        "--context_length", "4",
        "--patch_size", "4",  # Match video tokenizer patch_size
        "--embed_dim", "128",
        "--num_heads", "4",
        "--hidden_dim", "512",
        "--num_blocks", "2",
        "--latent_dim", "6",
        "--num_bins", "4",
        "--dropout", "0.1",
        "--use_wandb",
    ]
    
    # Add W&B arguments if enabled
    if args.use_wandb:
        dynamics_cmd.extend([
            "--use_wandb",
            "--wandb_project", f"{args.wandb_project}-dynamics"
        ])
    
    if not run_command(dynamics_cmd, "Dynamics Model Training"):
        print("❌ Dynamics model training failed.")
        return
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 FULL PIPELINE TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Find all results
    dynamics_checkpoint = find_latest_checkpoint(".", "dynamics")
    
    print("\n📁 Results Summary:")
    print(f"Video Tokenizer: {video_tokenizer_checkpoint}")
    # print(f"LAM: {lam_checkpoint}")
    print(f"Dynamics Model: {dynamics_checkpoint}")
    
    print("\n📊 Directory Structure Created:")
    print("src/vqvae/results/videotokenizer_*/")
    print("src/latent_action_model/results/lam_*/")
    print("src/dynamics/results/dynamics_*/")
    
    if args.use_wandb:
        print(f"\n📈 W&B Projects:")
        print(f"Video Tokenizer: {args.wandb_project}-video-tokenizer")
        print(f"LAM: {args.wandb_project}-lam")
        print(f"Dynamics: {args.wandb_project}-dynamics")
    
    print("\n🎯 Next Steps:")
    print("1. Check the visualizations in each model's visualizations/ directory")
    print("2. Use the trained models for inference or further training")
    print("3. The checkpoints can be used for the full video generation pipeline")
    
    print(f"\n⏱️ Total training completed at: {readable_timestamp()}")

if __name__ == "__main__":
    main() 