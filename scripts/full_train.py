#!/usr/bin/env python3
import sys
import os
from utils.utils import run_command, find_latest_checkpoint, prepare_pipeline_run_root
from utils.config import TrainingConfig, load_config


def main():
    # Load training config (YAML + dotlist)
    train_config: TrainingConfig = load_config(TrainingConfig, default_config_path=os.path.join(os.getcwd(), 'configs', 'training.yaml'))

    print(f"training config settings: {train_config}")

    # Create top-level run root and export so child processes can use it
    run_root, run_name = prepare_pipeline_run_root(base_cwd=os.getcwd())
    os.environ['NG_RUN_ROOT_DIR'] = run_root

    if train_config.run_video_tokenizer:
        v_cmd = [
            sys.executable, "scripts/train_video_tokenizer.py",
            "--config", train_config.video_tokenizer_config,
        ]
        if not run_command(v_cmd, "Video Tokenizer Training"):
            return

    if train_config.run_latent_actions:
        latent_actions_cmd = [
            sys.executable, "scripts/train_latent_actions.py",
            "--config", train_config.latent_actions_config,
        ]
        if not run_command(latent_actions_cmd, "Latent Actions Training"):
            return

    video_tokenizer_checkpoint = find_latest_checkpoint(".", "video_tokenizer")
    latent_actions_checkpoint = find_latest_checkpoint(".", "latent_actions")

    if train_config.run_dynamics:
        dyn_cmd = [
            sys.executable, "scripts/train_dynamics.py",
            "--config", train_config.dynamics_config,
            f"video_tokenizer_path={video_tokenizer_checkpoint}",
            f"latent_actions_path={latent_actions_checkpoint}",
        ]
        if not run_command(dyn_cmd, "Dynamics Model Training"):
            return

    dynamics_checkpoint = find_latest_checkpoint(".", "dynamics")
    print("\n📁 Results Summary:")
    print(f"Video Tokenizer: {video_tokenizer_checkpoint}")
    print(f"Latent Actions: {latent_actions_checkpoint}")
    print(f"Dynamics Model: {dynamics_checkpoint}")


if __name__ == "__main__":
    main() 