import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict


def init_distributed_from_env(backend: str = 'nccl') -> Dict[str, object]:
    """Initialize torch.distributed from torchrun env vars.
    Returns a context dict with is_distributed, rank, local_rank, world_size, device, is_main.
    """
    world_size_env = int(os.environ.get('WORLD_SIZE', '1'))
    is_distributed = world_size_env > 1 and torch.cuda.is_available()
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    if is_distributed:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method='env://')

    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = (rank == 0)

    return {
        'is_distributed': is_distributed,
        'rank': rank,
        'local_rank': local_rank,
        'world_size': world_size,
        'device': device,
        'is_main': is_main,
    }


def wrap_ddp_if_needed(model: torch.nn.Module, is_distributed: bool, local_rank: int) -> torch.nn.Module:
    if is_distributed:
        return DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    return model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, 'module') else model


def get_dataloader_distributed_kwargs(ctx: Dict[str, object]) -> Dict[str, object]:
    return {
        'distributed': bool(ctx.get('is_distributed', False)),
        'rank': int(ctx.get('rank', 0)),
        'world_size': int(ctx.get('world_size', 1)),
    }


def print_param_count_if_main(model: torch.nn.Module, model_name: str, is_main: bool) -> None:
    if not is_main:
        return
    try:
        params = sum(p.numel() for p in model.parameters())
        print(f"{model_name} parameters: {params/1e6:.2f}M ({params})")
    except Exception:
        pass


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group() 