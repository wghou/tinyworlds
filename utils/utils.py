import time
import glob
import subprocess
import os
import re
from typing import Optional


def readable_timestamp():
    """Generate a sortable timestamp for filenames (no weekday)."""
    return time.strftime("%Y_%m_%d_%H_%M_%S")

def find_latest_checkpoint(base_dir, model_name, run_root_dir: Optional[str] = None, stage_name: Optional[str] = None):
    """Find latest checkpoint.
    If run_root_dir (and optional stage_name) are provided, search only under
    <run_root_dir>/<stage_name>/checkpoints (or <run_root_dir>/**/checkpoints if stage_name None).
    Otherwise, fall back to project-wide model type-based search.
    Newest run dir first, then highest step within it.
    If the newest run root has none for this model, keep searching older runs.
    """
    def collect_files_from_roots(roots, model_name):
        files = []
        # Accept files whether they start with the dataset name or the model name
        # Also include common aliases per model
        alias_map = {
            'video_tokenizer': ['video_tokenizer'],
            'latent_actions': ['latent_actions', 'lam', 'actions', 'action_tokenizer'],
            'dynamics': ['dynamics'],
        }
        aliases = alias_map.get(model_name, [model_name])
        step_patterns = [f"*{a}_step_*.*" for a in aliases]
        ckpt_patterns = [f"*{a}_checkpoint_*.*" for a in aliases]
        for root in roots:
            # Search recursively beneath checkpoints to support nested layouts like
            # <stage>/checkpoints/<dataset>/<file>.pth or Hugging Face download trees.
            for pat in step_patterns + ckpt_patterns:
                files.extend(glob.glob(os.path.join(root, "**", pat), recursive=True))
        return [p for p in files if os.path.splitext(p)[1] in ('.pt', '.pth')]

    def run_dir_of(path: str) -> str:
        # Walk up until we reach the 'checkpoints' directory, then return its parent (the stage dir)
        d = os.path.dirname(path)
        while d and os.path.basename(d) != 'checkpoints':
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        # If we found 'checkpoints', return its parent; otherwise, fallback to two-level up
        if d and os.path.basename(d) == 'checkpoints':
            return os.path.dirname(d)
        return os.path.dirname(os.path.dirname(os.path.dirname(path)))

    def project_wide_search():
        # Fallback to model_type/results search in repository
        # Allow multiple possible directory names per model
        model_type_dirs = {
            'video_tokenizer': ['video_tokenizer'],
            'latent_actions': ['latent_actions', 'lam', 'actions', 'action_tokenizer'],
            'dynamics': ['dynamics',],
        }
        dirs = model_type_dirs.get(model_name)
        if not dirs:
            roots = [os.path.join(base_dir, 'results', '**', 'checkpoints')]
        else:
            roots = [os.path.join(base_dir, 'results', '**', d, 'checkpoints') for d in dirs]
        files = collect_files_from_roots(roots, model_name)
        if not files:
            # Generic fallback: search all checkpoints regardless of stage dir name
            generic_roots = [os.path.join(base_dir, 'results', '**', 'checkpoints')]
            files = collect_files_from_roots(generic_roots, model_name)
        if not files:
            raise Exception(f"No checkpoint files found for {model_name}")
        run_dir_to_files = {}
        for p in files:
            rd = run_dir_of(p)
            run_dir_to_files.setdefault(rd, []).append(p)
        newest_run_dir = max(run_dir_to_files.keys(), key=lambda d: os.path.getctime(d))
        candidate_files = run_dir_to_files[newest_run_dir]
        return candidate_files

    if run_root_dir is not None:
        if stage_name:
            roots = [os.path.join(run_root_dir, stage_name, 'checkpoints')]
        else:
            roots = [os.path.join(run_root_dir, '**', 'checkpoints')]
        files = collect_files_from_roots(roots, model_name)
        if not files:
            # Fallback: search project-wide older runs until found
            candidate_files = project_wide_search()
        else:
            # Group by run dir within the provided root and choose newest run dir
            run_dir_to_files = {}
            for p in files:
                rd = run_dir_of(p)
                run_dir_to_files.setdefault(rd, []).append(p)
            newest_run_dir = max(run_dir_to_files.keys(), key=lambda d: os.path.getctime(d))
            candidate_files = run_dir_to_files[newest_run_dir]
    else:
        candidate_files = project_wide_search()

    def extract_step(path: str) -> int:
        fname = os.path.basename(path)
        m = re.search(r"_step_(\d+)", fname)
        return int(m.group(1)) if m else -1

    candidate_files.sort(key=lambda p: (extract_step(p), os.path.getctime(p)))
    return candidate_files[-1]

def run_command(cmd, description):
    # empirical max dataLoader throughput settings (I used 1-6 H100s)
    env = os.environ.copy()
    env.setdefault("NG_NUM_WORKERS", str(max(2, (os.cpu_count() or 4) - 2)))
    env.setdefault("NG_PREFETCH_FACTOR", "4")
    env.setdefault("NG_PIN_MEMORY", "1")
    env["NG_PERSISTENT_WORKERS"] = "0"
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


def load_videotokenizer_from_checkpoint(checkpoint_path, device, model = None):
    """Instantiate VideoTokenizer from a checkpoint's saved config and load weights."""
    import torch
    from models.video_tokenizer import VideoTokenizer
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    cfg = ckpt.get('config', {}) or {}
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
    if model is None:
        model = VideoTokenizer(**kwargs)
    model.load_state_dict(ckpt['model'], strict=True)
    model = model.to(device)
    return model, ckpt


def load_latent_actions_from_checkpoint(checkpoint_path, device, model = None):
    """Instantiate LatentActionModel from a checkpoint's saved config and load weights."""
    import torch
    from models.latent_actions import LatentActionModel
    ckpt = torch.load(checkpoint_path, map_location='cpu')
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
    if model is None:
        model = LatentActionModel(**kwargs)
    model.load_state_dict(ckpt['model'], strict=True)
    model = model.to(device)
    return model, ckpt


def load_dynamics_from_checkpoint(checkpoint_path, device, model = None):
    """Instantiate DynamicsModel from a checkpoint's saved config and load weights."""
    import torch
    from models.dynamics import DynamicsModel
    ckpt = torch.load(checkpoint_path, map_location='cpu')
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
    if model is None:
        model = DynamicsModel(**kwargs)
    model.load_state_dict(ckpt['model'], strict=True)
    model = model.to(device)
    return model, ckpt

def prepare_pipeline_run_root(run_name: Optional[str] = None, base_cwd: Optional[str] = None):
    """Create a top-level run root directory results/<timestamp_or_name>"""
    cwd = base_cwd or os.getcwd()
    ts = readable_timestamp()
    name = run_name or ts
    run_root = os.path.join(cwd, 'results', name)
    os.makedirs(run_root, exist_ok=True)
    return run_root, name


def prepare_stage_dirs(run_root_dir: str, stage_name: str):
    """Create stage subdirectories under the given run root.

    Structure:
      <run_root_dir>/<stage_name>/checkpoints
      <run_root_dir>/<stage_name>/visualizations

    Returns (stage_dir, checkpoints_dir, visualizations_dir).
    """
    stage_dir = os.path.join(run_root_dir, stage_name)
    checkpoints_dir = os.path.join(stage_dir, 'checkpoints')
    visualizations_dir = os.path.join(stage_dir, 'visualizations')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)
    return stage_dir, checkpoints_dir, visualizations_dir
