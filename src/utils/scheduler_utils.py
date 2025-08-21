import math
import torch.optim as optim

def cosine_with_warmup(step, *, warmup_steps, total_steps, min_lr=0.0):
    """
    Creates a schedule with linear warmup and cosine decay.
    
    Args:
        step: Current step number
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate to decay to (as a fraction of max lr)
    
    Returns:
        Multiplicative factor for learning rate (between min_lr and 1.0)
    """
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (1 - min_lr) * cosine

def create_cosine_scheduler(optimizer, total_steps, warmup_fraction=0.05, min_lr_fraction=0.01):
    """
    Creates a cosine scheduler with warmup.
    
    Args:
        optimizer: The optimizer to schedule
        total_steps: Total number of training steps
        warmup_fraction: Fraction of total steps to use for warmup (default: 5%)
        min_lr_fraction: Minimum LR as a fraction of initial LR (default: 1%)
    
    Returns:
        LambdaLR scheduler
    """
    warmup_steps = int(total_steps * warmup_fraction)
    return optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_with_warmup(
            s,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr_fraction
        )
    ) 