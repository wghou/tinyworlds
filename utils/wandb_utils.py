"""
Weights & Biases utilities for experiment tracking
"""

import wandb
import torch
import os
import time
from typing import Dict, Any, Optional


def init_wandb(project_name: str, config: Dict[str, Any], run_name: Optional[str] = None) -> wandb.run:

    # Generate run name if not provided
    if run_name is None:
        run_name = f"{project_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize wandb
    run = wandb.init(
        project=project_name,
        config=config,
        name=run_name,
        tags=[project_name, "training"]
    )
    
    print(f"🚀 W&B run initialized: {run.name}")
    print(f"📊 Project: {project_name}")
    print(f"🔗 View at: {run.get_url()}")
    
    return run


def log_training_metrics(step: int, metrics: Dict[str, float], prefix: str = "train"):
    # Add prefix to metric names
    prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    prefixed_metrics["step"] = step
    
    wandb.log(prefixed_metrics)


def log_model_gradients(model: torch.nn.Module, step: int):
    # Log gradients for each parameter
    for name, param in model.named_parameters():
        if param.grad is not None:
            wandb.log({
                f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy()),
                "step": step
            })


def log_model_parameters(model: torch.nn.Module, step: int):
    # Log parameters for each layer
    for name, param in model.named_parameters():
        wandb.log({
            f"parameters/{name}": wandb.Histogram(param.cpu().numpy()),
            "step": step
        })


def log_learning_rate(optimizer: torch.optim.Optimizer, step: int):
    for i, param_group in enumerate(optimizer.param_groups):
        wandb.log({
            f"learning_rate/group_{i}": param_group['lr'],
            "step": step
        })


def log_reconstruction_comparison(original: torch.Tensor, reconstructed: torch.Tensor, 
                                step: int, max_images: int = 16):

    # Ensure tensors are on CPU and in the right format
    original = original.detach().cpu()
    reconstructed = reconstructed.detach().cpu()
    
    # Denormalize from [-1, 1] to [0, 1] if needed
    if original.min() < 0:
        original = (original + 1) / 2
        reconstructed = (reconstructed + 1) / 2
    
    # Clamp to valid range
    original = torch.clamp(original, 0, 1)
    reconstructed = torch.clamp(reconstructed, 0, 1)
    
    # Take first max_images
    n_images = min(original.shape[0], max_images)
    original = original[:n_images]
    reconstructed = reconstructed[:n_images]
    
    # Create comparison images
    comparison_images = []
    for i in range(n_images):
        # Stack original and reconstructed side by side
        comparison = torch.cat([original[i], reconstructed[i]], dim=2)  # Concatenate horizontally
        comparison_images.append(comparison)

def log_video_sequence(frames: torch.Tensor, step: int, caption: str = "Generated Video"):
    # Ensure tensor is on CPU and in the right format
    frames = frames.detach().cpu()
    
    # Denormalize from [-1, 1] to [0, 1] if needed
    if frames.min() < 0:
        frames = (frames + 1) / 2
    
    # Clamp to valid range
    frames = torch.clamp(frames, 0, 1)
    
    # Take first batch item
    video = frames[0]  # [T, C, H, W]
    
    # Log to wandb
    wandb.log({
        "video_sequence": wandb.Video(
            video.numpy(),
            fps=2,
            caption=f"{caption} - Step {step}"
        ),
        "step": step
    })

def log_codebook_usage(codebook_usage: float, step: int, model_name: str = "model"):
    wandb.log({
        f"{model_name}/codebook_usage": codebook_usage,
        "step": step
    })

def log_action_distribution(action_indices: torch.Tensor, step: int, n_actions: int):
    # Convert to CPU and numpy
    action_indices = action_indices.detach().cpu().numpy()
    
    # Compute distribution
    action_counts = torch.bincount(torch.tensor(action_indices.flatten()), minlength=n_actions)
    action_probs = action_counts.float() / action_counts.sum()
    
    # Log distribution
    wandb.log({
        "action_distribution": wandb.Histogram(action_indices.flatten()),
        "action_entropy": -torch.sum(action_probs * torch.log(action_probs + 1e-8)),
        "unique_actions": len(torch.unique(torch.tensor(action_indices))),
        "step": step
    })

def log_system_metrics(step: int):
    if torch.cuda.is_available():
        wandb.log({
            "system/gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "system/gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
            "step": step
        })

def finish_wandb():
    """Finish the W&B run"""
    if wandb.run is not None:
        wandb.finish()

def create_wandb_config(args, model_config: Dict[str, Any]) -> Dict[str, Any]:
    config = {
        # Training parameters
        "batch_size": args.batch_size,
        "n_updates": args.n_updates,
        "learning_rate": args.learning_rate,
        "log_interval": getattr(args, 'log_interval', 100),
        
        # Dataset parameters
        "dataset": getattr(args, 'dataset', 'SONIC'),
        "context_length": getattr(args, 'context_length', 4),
        
        # Model architecture
        "model_architecture": model_config,
        
        # System parameters
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    
    # Add any additional args that might exist
    for attr in dir(args):
        if not attr.startswith('_') and not callable(getattr(args, attr)):
            if attr not in config:
                config[attr] = getattr(args, attr)
    
    return config
