import time
import glob
import subprocess
import os
import re


def readable_timestamp():
    """Generate a sortable timestamp for filenames (no weekday)."""
    return time.strftime("%Y_%m_%d_%H_%M_%S")

def find_latest_checkpoint(base_dir, model_name):
    """Find latest checkpoint by newest run dir, then highest step within it.
    Looks under a module-specific root when model_name is known.
    """
    # Choose module/results root by model name
    module_map = {
        'videotokenizer': 'video_tokenizer',
        'video_tokenizer': 'video_tokenizer',
        'vqvae': 'video_tokenizer',
        'lam': 'latent_action_model',
        'dynamics': 'dynamics',
    }
    module = module_map.get(model_name, '**')

    # Gather matching checkpoint files recursively
    if module == '**':
        roots = [os.path.join(base_dir, 'src/results', '**', 'checkpoints')]
    else:
        roots = [os.path.join(base_dir, 'src', module, 'results', '**', 'checkpoints')]

    files = []
    for root in roots:
        files.extend(glob.glob(os.path.join(root, f"{model_name}_step_*.*"), recursive=True))
        files.extend(glob.glob(os.path.join(root, f"{model_name}_checkpoint_*.*"), recursive=True))
    files = [p for p in files if os.path.splitext(p)[1] in ('.pt', '.pth')]
    if not files:
        raise Exception(f"No checkpoint files found for {model_name}")

    def run_dir_of(path: str) -> str:
        checkpoints_dir = os.path.dirname(path)
        return os.path.dirname(checkpoints_dir)

    run_dir_to_files = {}
    for p in files:
        rd = run_dir_of(p)
        run_dir_to_files.setdefault(rd, []).append(p)

    newest_run_dir = max(run_dir_to_files.keys(), key=lambda d: os.path.getctime(d))
    candidate_files = run_dir_to_files[newest_run_dir]

    def extract_step(path: str) -> int:
        fname = os.path.basename(path)
        m = re.search(r"_step_(\d+)", fname)
        return int(m.group(1)) if m else -1

    candidate_files.sort(key=lambda p: (extract_step(p), os.path.getctime(p)))
    return candidate_files[-1]

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
    from src.video_tokenizer.models.video_tokenizer import Video_Tokenizer
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
    from src.dynamics.models.dynamics import DynamicsModel
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get('config', {}) or {}
    frame_size = cfg.get('frame_size', 128)
    # Infer conditioning_dim from checkpoint if missing
    conditioning_dim = cfg.get('conditioning_dim', None)
    if conditioning_dim is None:
        cond_inferred = None
        for k, v in ckpt.get('model', {}).items():
            # Linear weight shape: [out_features, in_features]; in_features is conditioning dim
            if k.endswith('to_gamma_beta.1.weight'):
                cond_inferred = int(v.shape[1])
                break
        conditioning_dim = cond_inferred if cond_inferred is not None else 3
    kwargs = {
        'frame_size': (frame_size, frame_size),
        'patch_size': cfg.get('patch_size', 8),
        'embed_dim': cfg.get('embed_dim', 128),
        'num_heads': cfg.get('num_heads', 8),
        'hidden_dim': cfg.get('hidden_dim', 256),
        'num_blocks': cfg.get('num_blocks', 4),
        'conditioning_dim': conditioning_dim,
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
        module: submodule name under src (e.g., 'video_tokenizer', 'latent_action_model', 'dynamics')
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
