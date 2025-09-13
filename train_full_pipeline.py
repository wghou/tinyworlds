#!/usr/bin/env python3
import sys
import os
from src.utils.utils import run_command, find_latest_checkpoint
from src.utils.config import PipelineConfig, load_config

# Load pipeline config (YAML + dotlist)
pipe: PipelineConfig = load_config(PipelineConfig, default_config_path=os.path.join(os.getcwd(), 'configs', 'pipeline.yaml'))


def main():
    if pipe.run_vqvae:
        v_cmd = [
            sys.executable, "src/vqvae/main.py",
            "--config", pipe.vqvae_config,
        ]
        if not run_command(v_cmd, "Video Tokenizer Training"):
            return

    if pipe.run_lam:
        lam_cmd = [
            sys.executable, "src/latent_action_model/main.py",
            "--config", pipe.lam_config,
        ]
        if not run_command(lam_cmd, "LAM Training"):
            return

    video_tokenizer_checkpoint = find_latest_checkpoint(".", "videotokenizer")
    lam_checkpoint = find_latest_checkpoint(".", "lam")

    if pipe.run_dynamics:
        dyn_cmd = [
            sys.executable, "src/dynamics/main.py",
            "--config", pipe.dynamics_config,
            f"video_tokenizer_path={video_tokenizer_checkpoint}",
            f"lam_path={lam_checkpoint}",
        ]
        if not run_command(dyn_cmd, "Dynamics Model Training"):
            return

    dynamics_checkpoint = find_latest_checkpoint(".", "dynamics")
    print("\n📁 Results Summary:")
    print(f"Video Tokenizer: {video_tokenizer_checkpoint}")
    print(f"LAM: {lam_checkpoint}")
    print(f"Dynamics Model: {dynamics_checkpoint}")


if __name__ == "__main__":
    main() 