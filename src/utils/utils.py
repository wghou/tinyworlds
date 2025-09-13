import time
import glob
import subprocess
import os
import re


def readable_timestamp():
    """Generate a readable timestamp for filenames"""
    return time.strftime("%a_%b_%d_%H_%M_%S_%Y")

def find_latest_checkpoint(base_dir, model_name):
    """Find the most recent checkpoint for a given model prefix.
    Searches recursively under src/*/results/**/checkpoints/ for files named
    {model_name}_step_*.pt or .pth and returns the newest by step (tiebreak by ctime).
    """
    # Search recursively for checkpoints regardless of module/run folder names
    pattern = os.path.join(base_dir, "src", "**", "results", "**", "checkpoints", f"{model_name}_step_*.*")
    checkpoint_files = glob.glob(pattern, recursive=True)

    # Accept common torch extensions only
    checkpoint_files = [p for p in checkpoint_files if os.path.splitext(p)[1] in (".pt", ".pth")]

    if not checkpoint_files:
        raise Exception(f"No checkpoint files found for {model_name}")

    def extract_step(path: str) -> int:
        fname = os.path.basename(path)
        m = re.search(rf"{re.escape(model_name)}_step_(\d+)", fname)
        return int(m.group(1)) if m else -1

    checkpoint_files.sort(key=lambda p: (extract_step(p), os.path.getctime(p)))
    return checkpoint_files[-1]

def run_command(cmd, description):
    # Recommend environment tweaks for DataLoader throughput TODO: test
    env = os.environ.copy()
    env.setdefault("NG_NUM_WORKERS", str(max(2, (os.cpu_count() or 4) - 2)))
    env.setdefault("NG_PREFETCH_FACTOR", "4")
    env.setdefault("NG_PIN_MEMORY", "1")
    # Disable persistent workers in subprocess to ensure clean exit
    env["NG_PERSISTENT_WORKERS"] = "0"
    # Prefer TF32 globally
    env.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False
    except KeyboardInterrupt:
        return False

def save_training_state(model, optimizer, scheduler, config, checkpoints_dir, prefix, step):
    """Save a checkpoint with model/optimizer/scheduler and the exact config.
    The filename includes the global step and a timestamp for uniqueness.
    """
    import torch
    ts = readable_timestamp()
    state = {
        'model': (model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'config': config,
        'step': int(step) if step is not None else None,
        'timestamp': ts,
    }
    os.makedirs(checkpoints_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoints_dir, f"{prefix}_step_{int(step) if step is not None else 0}_{ts}.pth")
    torch.save(state, ckpt_path)
    return ckpt_path


def load_videotokenizer_from_checkpoint(checkpoint_path, device):
    """Instantiate Video_Tokenizer from a checkpoint's saved config and load weights."""
    import torch
    from src.vqvae.models.video_tokenizer import Video_Tokenizer
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get('config', {}) or {}
    # Build kwargs from saved config with sensible fallbacks
    frame_size = cfg.get('frame_size', 128)
    kwargs = {
        'frame_size': (frame_size, frame_size),
        'patch_size': cfg.get('patch_size', 8),
        'embed_dim': cfg.get('embed_dim', 128),
        'num_heads': cfg.get('num_heads', 8),
        'hidden_dim': cfg.get('hidden_dim', 256),
        'num_blocks': cfg.get('num_blocks', 4),
        'latent_dim': cfg.get('latent_dim', 6),
        'num_bins': cfg.get('num_bins', 4),
    }
    model = Video_Tokenizer(**kwargs).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    return model, ckpt


def load_lam_from_checkpoint(checkpoint_path, device):
    """Instantiate LAM from a checkpoint's saved config and load weights."""
    import torch
    from src.latent_action_model.models.lam import LAM
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get('config', {}) or {}
    frame_size = cfg.get('frame_size', 128)
    kwargs = {
        'frame_size': (frame_size, frame_size),
        'n_actions': cfg.get('n_actions', 8),
        'patch_size': cfg.get('patch_size', 8),
        'embed_dim': cfg.get('embed_dim', 128),
        'num_heads': cfg.get('num_heads', 8),
        'hidden_dim': cfg.get('hidden_dim', 256),
        'num_blocks': cfg.get('num_blocks', 4),
    }
    model = LAM(**kwargs).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    return model, ckpt


def load_dynamics_from_checkpoint(checkpoint_path, device):
    """Instantiate DynamicsModel from a checkpoint's saved config and load weights."""
    import torch
    from src.dynamics.models.dynamics_model import DynamicsModel
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get('config', {}) or {}
    frame_size = cfg.get('frame_size', 128)
    kwargs = {
        'frame_size': (frame_size, frame_size),
        'patch_size': cfg.get('patch_size', 8),
        'embed_dim': cfg.get('embed_dim', 128),
        'num_heads': cfg.get('num_heads', 8),
        'hidden_dim': cfg.get('hidden_dim', 256),
        'num_blocks': cfg.get('num_blocks', 4),
        'conditioning_dim': cfg.get('conditioning_dim', 3),
        'latent_dim': cfg.get('latent_dim', 6),
        'num_bins': cfg.get('num_bins', 4),
    }
    model = DynamicsModel(**kwargs).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    return model, ckpt


def prepare_run_dirs(module: str, filename: str | None, base_cwd: str | None = None):
    """
    Create an organized directory structure for a training run.

    Args:
        module: submodule name under src (e.g., 'vqvae', 'latent_action_model', 'dynamics')
        filename: optional custom run name; if None, use a timestamp
        base_cwd: optional base working directory; defaults to current working directory

    Returns:
        run_dir, checkpoints_dir, visualizations_dir, run_name
    """
    cwd = base_cwd or os.getcwd()
    ts = readable_timestamp()
    run_name = filename or ts
    run_dir = os.path.join(cwd, 'src', module, 'results', f"{module.split('/')[-1]}_{run_name}")
    os.makedirs(run_dir, exist_ok=True)
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    visualizations_dir = os.path.join(run_dir, 'visualizations')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)
    return run_dir, checkpoints_dir, visualizations_dir, run_name
