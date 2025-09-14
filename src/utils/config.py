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
    start_iteration: int = 0
    # Perf
    amp: bool = True
    tf32: bool = True
    compile: bool = False
    # W&B
    use_wandb: bool = False
    wandb_project: str = "nano-genie"
    wandb_run_name: Optional[str] = None
    # Debug
    debug_stats: bool = True


@dataclass
class LAMConfig:
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
    wandb_media: bool = False
    # Save
    save: bool = True
    filename: Optional[str] = None


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
    lam_path: Optional[str] = None
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


@dataclass
class PipelineConfig:
    use_wandb: bool = True
    wandb_project: str = "nano-genie-pipeline"
    dataset: str = "ZELDA"
    # Config paths for stages
    video_tokenizer_config: str = "configs/video_tokenizer.yaml"
    lam_config: str = "configs/lam.yaml"
    dynamics_config: str = "configs/dynamics.yaml"
    # Which stages to run
    run_video_tokenizer: bool = False
    run_lam: bool = True
    run_dynamics: bool = True


def load_config(config_cls, default_config_path: Optional[str] = None):
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", type=str, default=default_config_path)
    # Accept dotlist overrides like key=value
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    base = OmegaConf.structured(config_cls())
    cfg = base

    if args.config is not None and os.path.isfile(args.config):
        file_cfg = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(cfg, file_cfg)

    # Merge dotlist overrides if any (ignore leading '--')
    dot_overrides = [s.lstrip('-') for s in (args.overrides or []) if '=' in s]
    if dot_overrides:
        cli_cfg = OmegaConf.from_dotlist(dot_overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    return OmegaConf.to_object(cfg)  # dataclass instance 