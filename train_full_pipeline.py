#!/usr/bin/env python3
"""
Full Pipeline Training Script for SONIC Dataset

This script trains all three models in sequence:
1. Video Tokenizer
2. LAM
3. Dynamics Model using the trained checkpoints

Prerequisites:
- chosen dataset preprocessed and saved
- All dependencies installed (torch, einops, etc.)

Usage:
    python train_full_pipeline.py

The script will:
1. Train the video tokenizer on chosen dataset frames
2. Train the latent action model (LAM) on chosen dataset frame sequences
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
from src.utils.utils import readable_timestamp, run_command, find_latest_checkpoint

def parse_args():
    # TODO: change to configargs class, and add parsable yaml config files
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Full Pipeline Training Script")
    parser.add_argument("--use_wandb", action="store_true", default=True, 
                       help="Enable Weights & Biases logging for all models")
    parser.add_argument("--wandb_project", type=str, default="nano-genie-pipeline",
                       help="Base project name for W&B (each model will have its own project)")
    parser.add_argument("--dataset", type=str, default="ZELDA",
                       help="Dataset to use for training")
    parser.add_argument("--patch_size", type=int, default=8,
                       help="Patch size for video tokenizer")
    parser.add_argument("--embed_dim", type=int, default=128,
                       help="Embed dimension")
    parser.add_argument("--num_heads", type=int, default=8,
                       help="Number of heads")
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="Hidden dimension")
    parser.add_argument("--num_blocks", type=int, default=4,
                       help="Number of blocks")
    parser.add_argument("--latent_dim", type=int, default=5,
                       help="Latent dimension")
    parser.add_argument("--num_bins", type=int, default=4,
                       help="Number of bins per dimension for video tokenizer FSQ")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--log_interval", type=int, default=100,
                       help="Log interval")
    parser.add_argument("--context_length", type=int, default=4,
                       help="Context length")
    parser.add_argument("--batch_size", type=int, default=384,
                       help="Batch size")
    parser.add_argument("--n_actions", type=int, default=8,
                       help="Number of actions")
    parser.add_argument("--frame_size", type=int, default=128,
                       help="Frame size")
    # Performance flags applied to all subcommands
    parser.add_argument("--amp", action="store_true", default=True, help="Enable mixed precision (bfloat16)")
    parser.add_argument("--tf32", action="store_true", default=True, help="Enable TF32 on Ampere+")
    parser.add_argument("--compile", action="store_true", default=True, help="Compile models with torch.compile")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    print(f"🚀 Starting Full Pipeline Training on {args.dataset} Dataset")
    print(f"Timestamp: {readable_timestamp()}")
    
    if args.use_wandb:
        print("📊 W&B logging enabled")
        print(f"📈 Base project: {args.wandb_project}")
    
    # Check if we're in the right directory
    if not os.path.exists("src"):
        print("❌ Error: Please run this script from the nano-genie root directory")
        return
    
    # # Step 1: Train Video Tokenizer
    # print("\n" + "="*60)
    # print("STEP 1: Training Video Tokenizer on SONIC")
    # print("="*60)
    
    # video_tokenizer_cmd = [
    #     sys.executable, "src/vqvae/main.py",
    #     "--dataset", args.dataset,
    #     "--batch_size", str(args.batch_size),
    #     "--n_updates", "1",
    #     "--learning_rate", str(args.learning_rate),
    #     "--log_interval", str(args.log_interval),
    #     "--context_length", str(args.context_length),
    #     "--patch_size", str(args.patch_size),
    #     "--embed_dim", str(args.embed_dim),
    #     "--num_heads", str(args.num_heads),
    #     "--hidden_dim", str(args.hidden_dim),
    #     "--num_blocks", str(args.num_blocks),
    #     "--latent_dim", str(args.latent_dim),
    #     "--num_bins", str(args.num_bins),
    #     "--frame_size", str(args.frame_size),
    # ]
    # if args.amp:
    #     video_tokenizer_cmd.append("--amp")
    # if args.tf32:
    #     video_tokenizer_cmd.append("--tf32")
    # if args.compile:
    #     video_tokenizer_cmd.append("--compile")
    
    # # Add W&B arguments if enabled
    # if args.use_wandb:
    #     video_tokenizer_cmd.extend([
    #         "--use_wandb",
    #         "--wandb_project", f"{args.wandb_project}"
    #     ])
    
    # if not run_command(video_tokenizer_cmd, "Video Tokenizer Training"):
    #     print("❌ Video tokenizer training failed. Stopping pipeline.")
    #     return
    
    # # Step 2: Train LAM
    # print("\n" + "="*60)
    # print("STEP 2: Training LAM on SONIC")
    # print("="*60)
    
    # lam_cmd = [
    #     sys.executable, "src/latent_action_model/main.py",
    #     "--dataset", args.dataset,
    #     "--batch_size", str(args.batch_size),
    #     "--n_updates", "1",
    #     "--learning_rate", str(args.learning_rate),
    #     "--log_interval", str(args.log_interval),
    #     "--context_length", str(args.context_length),
    #     "--patch_size", str(args.patch_size),
    #     "--embed_dim", str(args.embed_dim),
    #     "--num_heads", str(args.num_heads),
    #     "--hidden_dim", str(args.hidden_dim),
    #     "--num_blocks", str(args.num_blocks),
    #     "--n_actions", str(args.n_actions),
    #     "--frame_size", str(args.frame_size),
    # ]
    # if args.amp:
    #     lam_cmd.append("--amp")
    # if args.tf32:
    #     lam_cmd.append("--tf32")
    # if args.compile:
    #     lam_cmd.append("--compile")
    
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
    lam_checkpoint = find_latest_checkpoint(".", "lam")
    
    if not video_tokenizer_checkpoint:
        print("❌ Could not find video tokenizer checkpoint")
        return
    
    if not lam_checkpoint:
        print("❌ Could not find LAM checkpoint")
        return
    
    print(f"✅ Found video tokenizer checkpoint: {video_tokenizer_checkpoint}")
    print(f"✅ Found LAM checkpoint: {lam_checkpoint}")
    
    # Step 4: Train Dynamics Model
    print("\n" + "="*60)
    print("STEP 4: Training Dynamics Model")
    print("="*60)
    
    dynamics_cmd = [
        sys.executable, "src/dynamics/main.py",
        "--video_tokenizer_path", video_tokenizer_checkpoint,
        "--lam_path", lam_checkpoint,
        "--dataset", args.dataset,
        "--batch_size", str(args.batch_size),
        "--n_updates", "1",
        "--learning_rate", str(args.learning_rate),
        "--log_interval", str(args.log_interval),
        "--context_length", str(args.context_length),
        "--patch_size", str(args.patch_size),
        "--embed_dim", str(args.embed_dim),
        "--num_heads", str(args.num_heads),
        "--hidden_dim", str(args.hidden_dim),
        "--num_blocks", str(args.num_blocks),
        "--latent_dim", str(args.latent_dim),
        "--num_bins", str(args.num_bins),
        "--use_actions",
        "--frame_size", str(args.frame_size),
        "--n_actions", str(args.n_actions),
    ]
    if args.amp:
        dynamics_cmd.append("--amp")
    if args.tf32:
        dynamics_cmd.append("--tf32")
    if args.compile:
        dynamics_cmd.append("--compile")
    
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
    print(f"LAM: {lam_checkpoint}")
    print(f"Dynamics Model: {dynamics_checkpoint}")
    
    if args.use_wandb:
        print(f"\n📈 W&B Projects:")
        print(f"Video Tokenizer: {args.wandb_project}-video-tokenizer")
        print(f"LAM: {args.wandb_project}-lam")
        print(f"Dynamics: {args.wandb_project}-dynamics")
    
    print(f"\n⏱️ Total training completed at: {readable_timestamp()}")

if __name__ == "__main__":
    main() 