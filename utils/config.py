from dataclasses import dataclass, field
from typing import Optional
import argparse
import os
from omegaconf import OmegaConf


@dataclass
class VQVAEConfig:
    # Training
    batch_size: int = 256
    n_updates: int = 2000
    learning_rate: float = 0.0004
    log_interval: int = 100
    dataset: str = "SONIC"
    context_length: int = 4
    frame_size: int = 128
    # Model
    patch_size: int = 8
    embed_dim: int = 128
    num_heads: int = 8
    hidden_dim: int = 256
    num_blocks: int = 4
    latent_dim: int = 6
    num_bins: int = 4
    # Save/IO
    save: bool = True
    filename: Optional[str] = None
    checkpoint: Optional[str] = None
    # Perf
    amp: bool = True
    tf32: bool = True
    compile: bool = False
    # W&B
    use_wandb: bool = False
    wandb_project: str = "nano-genie"
    wandb_run_name: Optional[str] = None


@dataclass
class LatentActionsConfig:
    # Training
    batch_size: int = 256
    n_updates: int = 1000
    learning_rate: float = 0.001
    log_interval: int = 50
    dataset: str = "SONIC"
    context_length: int = 4
    frame_size: int = 128
    # Model
    n_actions: int = 8
    patch_size: int = 8
    embed_dim: int = 128
    num_heads: int = 8
    hidden_dim: int = 256
    num_blocks: int = 4
    # Perf
    amp: bool = True
    tf32: bool = True
    compile: bool = False
    # W&B
    use_wandb: bool = True
    wandb_project: str = "nano-genie"
    wandb_run_name: Optional[str] = None
    # Save
    save: bool = True
    filename: Optional[str] = None
    checkpoint: Optional[str] = None


@dataclass
class DynamicsConfig:
    # Training
    batch_size: int = 256
    n_updates: int = 2000
    learning_rate: float = 1e-4
    log_interval: int = 10
    dataset: str = "SONIC"
    context_length: int = 4
    frame_size: int = 128
    # Model (must match tokenizer)
    patch_size: int = 8
    embed_dim: int = 128
    num_heads: int = 8
    hidden_dim: int = 256
    num_blocks: int = 4
    latent_dim: int = 5
    num_bins: int = 4
    n_actions: int = 8
    use_actions: bool = False
    # Paths
    video_tokenizer_path: Optional[str] = None
    latent_actions_path: Optional[str] = None
    # Perf
    amp: bool = True
    tf32: bool = True
    compile: bool = False
    # W&B
    use_wandb: bool = False
    wandb_project: str = "nano-genie"
    wandb_run_name: Optional[str] = None
    # Save
    save: bool = True
    filename: Optional[str] = None
    checkpoint: Optional[str] = None


@dataclass
class TrainingConfig:
    use_wandb: bool = True
    wandb_project: str = "nano-genie"
    dataset: str = "ZELDA"
    # Config paths for stages
    video_tokenizer_config: str = "configs/video_tokenizer.yaml"
    latent_actions_config: str = "configs/latent_actions.yaml"
    dynamics_config: str = "configs/dynamics.yaml"
    # Which stages to run
    run_video_tokenizer: bool = False
    run_latent_actions: bool = True
    run_dynamics: bool = True
    # Shared model params
    patch_size: int = 4
    context_length: int = 4
    frame_size: int = 64
    latent_dim: int = 6 # for video tokenizer
    num_bins: int = 4 # for video tokenizer
    n_actions: int = 16


def load_config(config_cls, default_config_path: Optional[str] = None):
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", type=str, default=default_config_path)
    # Accept dotlist overrides like key=value
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    base = OmegaConf.structured(config_cls())
    cfg = base

    if args.config is not None:
        if not os.path.isfile(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config} (cwd: {os.getcwd()})")
        file_cfg = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(cfg, file_cfg)

    # Merge dotlist overrides if any (ignore leading '--')
    dot_overrides = [s.lstrip('-') for s in (args.overrides or []) if '=' in s]
    if dot_overrides:
        cli_cfg = OmegaConf.from_dotlist(dot_overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    return OmegaConf.to_object(cfg)  # dataclass instance


def load_stage_config_merged(config_cls, default_config_path: Optional[str] = None, training_config_path: Optional[str] = None):
    """Load a stage config YAML, then overlay values from training_config.yaml (priority),
    restricted to keys that exist in the stage schema. CLI dotlist overrides still have highest priority.
    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", type=str, default=default_config_path)
    parser.add_argument("--training_config", type=str, default=training_config_path or os.path.join(os.getcwd(), 'configs', 'training_config.yaml'))
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    base = OmegaConf.structured(config_cls())
    cfg = base

    # Load stage file
    if args.config is not None:
        if not os.path.isfile(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config} (cwd: {os.getcwd()})")
        stage_file_cfg = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(cfg, stage_file_cfg)

    # Load training config and filter to known keys
    if args.training_config and os.path.isfile(args.training_config):
        training_file_cfg = OmegaConf.load(args.training_config)
        # Convert to plain dict to filter
        training_container = OmegaConf.to_container(training_file_cfg, resolve=True) or {}
        allowed_keys = set(cfg.keys())
        filtered_training = {k: v for k, v in training_container.items() if k in allowed_keys}
        if filtered_training:
            training_cfg_filtered = OmegaConf.create(filtered_training)
            # Training config takes priority over stage
            cfg = OmegaConf.merge(cfg, training_cfg_filtered)

    # Merge CLI overrides if any (ignore leading '--')
    dot_overrides = [s.lstrip('-') for s in (args.overrides or []) if '=' in s]
    if dot_overrides:
        cli_cfg = OmegaConf.from_dotlist(dot_overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    return OmegaConf.to_object(cfg) 