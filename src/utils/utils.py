import time
import glob
import subprocess
import os


def readable_timestamp():
    """Generate a readable timestamp for filenames"""
    return time.strftime("%a_%b_%d_%H_%M_%S_%Y")

def find_latest_checkpoint(base_dir, model_name):
    pattern = os.path.join(base_dir, f"src/*/results/{model_name}_*")
    results_dirs = glob.glob(pattern)

    if not results_dirs:
        raise Exception(f"No checkpoints found for {model_name}")

    latest_dir = max(results_dirs, key=os.path.getctime)
    checkpoint_pattern = os.path.join(latest_dir, "checkpoints", f"{model_name}_checkpoint_*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)

    if not checkpoint_files:
        raise Exception(f"No checkpoint files found for {model_name}")

    return max(checkpoint_files, key=os.path.getctime)

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

    print(f"Running command: {cmd}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, env=env)
        print(f"debug got here in run_command")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False
    except KeyboardInterrupt:
        print("DEBUG: KeyboardInterrupt")
        return False

# -----------------------------
# Unified checkpoint utilities
# -----------------------------

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
