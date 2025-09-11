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
    env.setdefault("NG_PERSISTENT_WORKERS", "1")
    # Prefer TF32 globally
    env.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, env=env)
        return True
    except subprocess.CalledProcessError as e:
        raise Exception(f"\n{description} failed with error code {e.returncode}")
    except KeyboardInterrupt:
        raise Exception(f"\n{description} interrupted by user")
