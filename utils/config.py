from dataclasses import dataclass, field
from typing import Optional
import argparse
import os
from omegaconf import OmegaConf


@dataclass
class VideoTokenizerConfig:
	# Training
	batch_size: int
	n_updates: int
	learning_rate: float
	log_interval: int
	dataset: str
	context_length: int
	frame_size: int
	# Model
	patch_size: int
	embed_dim: int
	num_heads: int
	hidden_dim: int
	num_blocks: int
	latent_dim: int
	num_bins: int
	# Perf
	amp: bool
	tf32: bool
	compile: bool
	# W&B
	use_wandb: bool
	wandb_project: str
	# resume from checkpoint
	checkpoint: Optional[str]
	fps: Optional[int] = None
	preload_ratio: Optional[float] = None


@dataclass
class LatentActionsConfig:
	# Training
	batch_size: int
	n_updates: int
	learning_rate: float
	log_interval: int
	dataset: str
	context_length: int
	frame_size: int
	# Model
	n_actions: int
	patch_size: int
	embed_dim: int
	num_heads: int
	hidden_dim: int
	num_blocks: int
	# Perf
	amp: bool
	tf32: bool
	compile: bool
	# W&B
	use_wandb: bool
	wandb_project: str
	# resume from checkpoint
	checkpoint: Optional[str]
	fps: Optional[int] = None
	preload_ratio: Optional[float] = None


@dataclass
class DynamicsConfig:
	# Training
	batch_size: int
	n_updates: int
	learning_rate: float
	log_interval: int
	dataset: str
	context_length: int
	frame_size: int
	# Model (must match tokenizer)
	patch_size: int
	embed_dim: int
	num_heads: int
	hidden_dim: int
	num_blocks: int
	latent_dim: int
	num_bins: int
	n_actions: int
	use_actions: bool
	# Paths
	video_tokenizer_path: Optional[str]
	latent_actions_path: Optional[str]
	# Perf
	amp: bool
	tf32: bool
	compile: bool
	# W&B
	use_wandb: bool
	wandb_project: str
	# resume from checkpoint
	checkpoint: Optional[str]
	fps: Optional[int] = None
	preload_ratio: Optional[float] = None


@dataclass
class TrainingConfig:
	# WandB
	use_wandb: bool
	wandb_project: str
	# Dataset
	dataset: str
	# Config paths for stages
	video_tokenizer_config: str
	latent_actions_config: str
	dynamics_config: str
	# Which stages to run
	run_video_tokenizer: bool
	run_latent_actions: bool
	run_dynamics: bool
	# Shared model params
	patch_size: int
	context_length: int
	frame_size: int
	latent_dim: int
	num_bins: int
	n_actions: int
	# Performance
	amp: bool
	tf32: bool
	compile: bool
	# Distributed launch options
	distributed: bool
	nproc_per_node: int
	standalone: bool
	# These can vary per model
	embed_dim: Optional[int] = None
	num_heads: Optional[int] = None
	hidden_dim: Optional[int] = None
	num_blocks: Optional[int] = None
	learning_rate: Optional[float] = None
	batch_size: Optional[int] = None
	log_interval: Optional[int] = None
	n_updates: Optional[int] = None
	fps: Optional[int] = None
	preload_ratio: Optional[float] = None


@dataclass
class InferenceConfig:
	video_tokenizer_path: Optional[str]
	latent_actions_path: Optional[str]
	dynamics_path: Optional[str]
	device: str
	generation_steps: int
	context_window: int
	fps: int
	temperature: float
	use_actions: bool
	teacher_forced: bool
	use_latest_checkpoints: bool
	prediction_horizon: int
	dataset: str
	use_gt_actions: bool
	# Inference performance options
	amp: bool
	tf32: bool
	compile: bool
	# Interactive mode (user enters action ids)
	use_interactive_mode: bool
	preload_ratio: Optional[float] = None


def load_config(config_cls, default_config_path: Optional[str] = None):
	parser = argparse.ArgumentParser(add_help=True)
	parser.add_argument("--config", type=str, default=default_config_path)
	# Accept dotlist overrides like key=value
	parser.add_argument("overrides", nargs=argparse.REMAINDER)
	args = parser.parse_args()

	# Build a structured schema from the dataclass TYPE, not an instance.
	# This allows Python-side defaults to be omitted and provided solely via YAML.
	base = OmegaConf.structured(config_cls)
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

	# Return a typed dataclass instance
	return OmegaConf.to_object(cfg)


def load_stage_config_merged(config_cls, default_config_path: Optional[str] = None, training_config_path: Optional[str] = None):
	"""Load a stage config YAML, then overlay values from training_config.yaml (priority),
	restricted to keys that exist in the stage schema. CLI dotlist overrides still have highest priority.
	"""
	parser = argparse.ArgumentParser(add_help=True)
	parser.add_argument("--config", type=str, default=default_config_path)
	parser.add_argument("--training_config", type=str, default=training_config_path or os.path.join(os.getcwd(), 'configs', 'training.yaml'))
	parser.add_argument("overrides", nargs=argparse.REMAINDER)
	args = parser.parse_args()

	# Build a structured schema from the dataclass TYPE, not an instance.
	base = OmegaConf.structured(config_cls)
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
		filtered_training = {k: v for k, v in training_container.items() if k in allowed_keys and v is not None}
		if filtered_training:
			training_cfg_filtered = OmegaConf.create(filtered_training)
			# Training config takes priority over stage
			cfg = OmegaConf.merge(cfg, training_cfg_filtered)

	# Merge CLI overrides if any (ignore leading '--')
	dot_overrides = [s.lstrip('-') for s in (args.overrides or []) if '=' in s]
	if dot_overrides:
		cli_cfg = OmegaConf.from_dotlist(dot_overrides)
		cfg = OmegaConf.merge(cfg, cli_cfg)

	# Return a typed dataclass instance
	return OmegaConf.to_object(cfg) 