#!/usr/bin/env python3
import subprocess
import sys
import os
from src.utils.utils import readable_timestamp, run_command, find_latest_checkpoint
from src.utils.config import PipelineConfig, load_config

# Load pipeline config (YAML + dotlist)
pipe: PipelineConfig = load_config(PipelineConfig, default_config_path=os.path.join(os.getcwd(), 'configs', 'pipeline.yaml'))


def main():
    print(f"🚀 Starting Full Pipeline Training on {pipe.dataset} Dataset")
    print(f"Timestamp: {readable_timestamp()}")
    
    if pipe.use_wandb:
        print("📊 W&B logging enabled")
        print(f"📈 Base project: {pipe.wandb_project}")
    
    if not os.path.exists("src"):
        print("❌ Error: Please run this script from the nano-genie root directory")
        return

    # Step 1: Train Video Tokenizer (optional)
    if pipe.run_vqvae:
        print("\n" + "="*60)
        print("STEP 1: Training Video Tokenizer")
        print("="*60)
        v_cmd = [
            sys.executable, "src/vqvae/main.py",
            "--config", pipe.vqvae_config,
        ]
        if not run_command(v_cmd, "Video Tokenizer Training"):
            print("❌ Video tokenizer training failed. Stopping pipeline.")
            return

    # Step 2: Train LAM
    if pipe.run_lam:
        print("\n" + "="*60)
        print("STEP 2: Training LAM")
        print("="*60)
        lam_cmd = [
            sys.executable, "src/latent_action_model/main.py",
            "--config", pipe.lam_config,
        ]
        if not run_command(lam_cmd, "LAM Training"):
            print("❌ LAM training failed. Stopping pipeline.")
            return

    # Step 3: Find the latest checkpoints
    print("\n" + "="*60)
    print("STEP 3: Finding Latest Checkpoints")
    print("="*60)

    video_tokenizer_checkpoint = find_latest_checkpoint(".", "videotokenizer")
    lam_checkpoint = find_latest_checkpoint(".", "lam")

    if pipe.run_dynamics:
        if not video_tokenizer_checkpoint:
            print("❌ Could not find video tokenizer checkpoint")
            return
        if not lam_checkpoint:
            print("❌ Could not find LAM checkpoint")
            return
        print(f"✅ Found video tokenizer checkpoint: {video_tokenizer_checkpoint}")
        print(f"✅ Found LAM checkpoint: {lam_checkpoint}")

    # Step 4: Train Dynamics Model
    if pipe.run_dynamics:
        print("\n" + "="*60)
        print("STEP 4: Training Dynamics Model")
        print("="*60)
        # Inject paths into dynamics config via dotlist override
        dyn_cmd = [
            sys.executable, "src/dynamics/main.py",
            "--config", pipe.dynamics_config,
            f"video_tokenizer_path={video_tokenizer_checkpoint}",
            f"lam_path={lam_checkpoint}",
        ]
        if not run_command(dyn_cmd, "Dynamics Model Training"):
            print("❌ Dynamics model training failed.")
            return

    print("\n" + "="*60)
    print("🎉 FULL PIPELINE TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)

    dynamics_checkpoint = find_latest_checkpoint(".", "dynamics")
    print("\n📁 Results Summary:")
    print(f"Video Tokenizer: {video_tokenizer_checkpoint}")
    print(f"LAM: {lam_checkpoint}")
    print(f"Dynamics Model: {dynamics_checkpoint}")


if __name__ == "__main__":
    main() 