import numpy as np
import torch
import torch.optim as optim
import argparse
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.vqvae.models.video_tokenizer import Video_Tokenizer
from datasets.utils import visualize_reconstruction, load_data_and_data_loaders
from src.utils.scheduler_utils import create_cosine_scheduler
from tqdm import tqdm
import json
import wandb
import torch.nn.functional as F
from src.utils.utils import readable_timestamp

parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
timestamp = readable_timestamp()

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--n_updates", type=int, default=2000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=8)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--learning_rate", type=float, default=4e-4)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--dataset",  type=str, default='SONIC')
parser.add_argument("--context_length", type=int, default=4)
parser.add_argument("--frame_size", type=int, default=128)

# Model architecture parameters
parser.add_argument("--patch_size", type=int, default=4, help="Patch size for ST-Transformer")
parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension for ST-Transformer")
parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension for feed-forward")
parser.add_argument("--num_blocks", type=int, default=2, help="Number of ST-Transformer blocks")
parser.add_argument("--latent_dim", type=int, default=6, help="Latent dimension")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument("--num_bins", type=int, default=4, help="Number of bins per dimension for finite scalar quantization")

# whether or not to save model
parser.add_argument("-save", action="store_true", default=True)
parser.add_argument("--filename",  type=str, default=timestamp)

# Add checkpoint arguments
parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file to resume from")
parser.add_argument("--start_iteration", type=int, default=0, help="Iteration to start from")

parser.add_argument("--use_wandb", action="store_true", default=False, help="Enable Weights & Biases logging")
parser.add_argument("--wandb_project", type=str, default="nano-genie", help="W&B project name")
parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")

# Learning rate scheduler parameters
parser.add_argument("--lr_step_size", type=int, default=2000, help="Step size for learning rate decay")
parser.add_argument("--lr_gamma", type=float, default=0.5, help="Gamma for learning rate decay")

# Performance flags
parser.add_argument("--amp", action="store_true", default=True, help="Enable mixed precision (bfloat16)")
parser.add_argument("--tf32", action="store_true", default=True, help="Enable TF32 on Ampere+")
parser.add_argument("--compile", action="store_true", default=False, help="Compile the model with torch.compile")

# Debug flags
parser.add_argument("--debug_stats", action="store_true", default=True, help="Print extra debug stats at log intervals")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Speed-related backend settings
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high' if args.tf32 else 'medium')
    except Exception:
        pass

# Try to relax inductor settings before compile to avoid fused-attention shape issues
try:
    import torch._inductor.config as inductor_config
    inductor_config.fuse_attention = False
except Exception:
    pass

# Create organized save directory structure
if args.save:
    # Create main results directory for this run
    run_dir = os.path.join(os.getcwd(), 'src', 'vqvae', 'results', f'videotokenizer_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    visualizations_dir = os.path.join(run_dir, 'visualizations')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)
    
    print(f'Results will be saved in {run_dir}')
    print(f'Checkpoints: {checkpoints_dir}')
    print(f'Visualizations: {visualizations_dir}')

def save_run_configuration(args, run_dir, timestamp, device):
    config = {
        'timestamp': timestamp,
        'device': str(device),
        'model_architecture': {
            'frame_size': (args.frame_size, args.frame_size),
            'patch_size': args.patch_size,
            'embed_dim': args.embed_dim,
            'num_heads': args.num_heads,
            'hidden_dim': args.hidden_dim,
            'num_blocks': args.num_blocks,
            'latent_dim': args.latent_dim,
            'dropout': args.dropout,
            'num_bins': args.num_bins,
            'quantization_method': 'Finite Scalar Quantization (FSQ)'
        },
        'training_parameters': {
            'batch_size': args.batch_size,
            'n_updates': args.n_updates,
            'learning_rate': args.learning_rate,
            'log_interval': args.log_interval,
            'context_length': args.context_length,
            'dataset': args.dataset,
            'lr_step_size': args.lr_step_size,
            'lr_gamma': args.lr_gamma
        },
        'checkpoint_info': {
            'checkpoint_path': args.checkpoint,
            'start_iteration': args.start_iteration
        },
        'directories': {
            'run_dir': run_dir,
            'checkpoints_dir': os.path.join(run_dir, 'checkpoints') if args.save else None,
            'visualizations_dir': os.path.join(run_dir, 'visualizations') if args.save else None
        }
    }
    
    config_path = os.path.join(run_dir, 'run_config.json') if args.save else None
    if config_path:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f'Configuration saved to: {config_path}')

def save_videotokenizer_model_and_results(model, optimizer, results, hyperparameters, timestamp, checkpoints_dir):
    results_to_save = {
        'model': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    checkpoint_path = os.path.join(checkpoints_dir, f'videotokenizer_checkpoint_{timestamp}.pth')
    torch.save(results_to_save, checkpoint_path)
    return checkpoint_path

training_data, validation_data, training_loader, validation_loader, x_train_var = load_data_and_data_loaders(
    dataset=args.dataset, 
    batch_size=args.batch_size, 
    num_frames=args.context_length
)

model = Video_Tokenizer(
    frame_size=(args.frame_size, args.frame_size), 
    patch_size=args.patch_size,
    embed_dim=args.embed_dim,
    num_heads=args.num_heads,
    hidden_dim=args.hidden_dim,
    num_blocks=args.num_blocks,
    latent_dim=args.latent_dim,
    num_bins=args.num_bins,
).to(device)

# Optionally compile the model
if args.compile:
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True)
        print("✅ Model compiled with torch.compile")
    except Exception as e:
        print(f"⚠️ torch.compile not available or failed: {e}")

# Create parameter groups to avoid weight decay on biases and norm layers
decay = []
no_decay = []
for name, param in model.named_parameters():
    if param.requires_grad:
        if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)

# Try to enable fused AdamW for better throughput
try:
    optimizer = optim.AdamW([
        {'params': decay, 'weight_decay': 0.01},
        {'params': no_decay, 'weight_decay': 0}
    ], lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, fused=True)
    print("✅ Using fused AdamW")
except TypeError:
    optimizer = optim.AdamW([
        {'params': decay, 'weight_decay': 0.01},
        {'params': no_decay, 'weight_decay': 0}
    ], lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)

# Create cosine scheduler with warmup
scheduler = create_cosine_scheduler(optimizer, args.n_updates)

# AMP scaler
scaler = torch.amp.GradScaler('cuda', enabled=bool(args.amp))

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}

if args.checkpoint:
    if os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        results = checkpoint['results']
        # Load hyperparameters but don't override current args
        saved_hyperparameters = checkpoint['hyperparameters']
        print(f"Resuming from update {results['n_updates']}")
        
        # Restore optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        print(f"No checkpoint found at {args.checkpoint}")

# Save configuration after model setup
if args.save:
    save_run_configuration(args, run_dir, timestamp, device)

# Initialize W&B if enabled and available
if args.use_wandb:
    # Create model configuration
    model_config = {
        'frame_size': (64, 64),
        'patch_size': args.patch_size,
        'embed_dim': args.embed_dim,
        'num_heads': args.num_heads,
        'hidden_dim': args.hidden_dim,
        'num_blocks': args.num_blocks,
        'latent_dim': args.latent_dim,
        'dropout': args.dropout,
        'num_bins': args.num_bins,
    }
    
    # Create W&B config
    wandb_config = {
        'batch_size': args.batch_size,
        'n_updates': args.n_updates,
        'learning_rate': args.learning_rate,
        'log_interval': args.log_interval,
        'dataset': args.dataset,
        'context_length': args.context_length,
        'model_architecture': model_config,
        'device': str(device),
        'timestamp': timestamp
    }
    
    # Initialize W&B
    run_name = args.wandb_run_name or f"video_tokenizer_{timestamp}"
    wandb.init(
        project=args.wandb_project,
        config=wandb_config,
        name=run_name,
        tags=["video-tokenizer", "training"]
    )
    
    # Watch model for gradients and parameters
    wandb.watch(model, log="all", log_freq=args.log_interval)
    
    print(f"🚀 W&B run initialized: {wandb.run.name}")
    print(f"📊 Project: {args.wandb_project}")
    print(f"🔗 View at: {wandb.run.get_url()}")

model.train()

# ------------- DEBUG HELPERS -------------
def _compute_grad_norm(parameters) -> float:
    total_sq = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.float().norm(2)
        total_sq += float(param_norm.item() ** 2)
    return float(total_sq ** 0.5)

@torch.no_grad()
def _module_param_l2_norm(module) -> float:
    total_sq = 0.0
    for p in module.parameters():
        if p is None:
            continue
        n = p.data.float().norm(2)
        total_sq += float(n.item() ** 2)
    return float(total_sq ** 0.5)

def _module_grad_l2_norm(module) -> float:
    total_sq = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        n = p.grad.data.float().norm(2)
        total_sq += float(n.item() ** 2)
    return float(total_sq ** 0.5)

@torch.no_grad()
def _debug_fsq_and_embedding_stats(model, x, num_bins):
    stats = {}
    embeddings = model.encoder(x)
    stats["embed_mean"] = float(embeddings.mean().item())
    stats["embed_std"] = float(embeddings.std().item())
    stats["embed_min"] = float(embeddings.min().item())
    stats["embed_max"] = float(embeddings.max().item())

    tanh_z = torch.tanh(embeddings)
    stats["tanh_abs_gt_0.99_frac"] = float((tanh_z.abs() > 0.99).float().mean().item())

    bounded = 0.5 * (tanh_z + 1.0) * (num_bins - 1)
    eps = 1e-3
    edge_mask = (bounded <= eps) | (bounded >= (num_bins - 1 - eps))
    stats["bounded_edge_frac"] = float(edge_mask.float().mean().item())

    quantized = model.vq(embeddings)
    indices = model.vq.get_indices_from_latents(quantized, dim=-1)
    unique_vals, unique_counts = torch.unique(indices, return_counts=True)
    num_unique = int(unique_vals.numel())
    top_share = float((unique_counts.max().float() / indices.numel()).item()) if unique_counts.numel() > 0 else 0.0
    stats["codebook_unique_count"] = num_unique
    stats["codebook_size"] = int(model.codebook_size)
    stats["top_code_share"] = top_share
    stats["unique_ratio"] = float(num_unique / max(1, model.codebook_size))

    # Decoder pathway activations
    dec_in = model.decoder.latent_embed(quantized)
    dec_in = dec_in + model.decoder.pos_spatial_dec.to(dec_in.device, dec_in.dtype)
    stats["decoder_in_std"] = float(dec_in.std().item())
    dec_mid = model.decoder.transformer(dec_in)
    stats["decoder_mid_std"] = float(dec_mid.std().item())

    # Parameter norms (coarse: encoder vs decoder)
    stats["param_norm/encoder"] = _module_param_l2_norm(model.encoder)
    stats["param_norm/decoder"] = _module_param_l2_norm(model.decoder)
    stats["param_norm/total"] = stats["param_norm/encoder"] + stats["param_norm/decoder"]

    return stats
# ------------- END DEBUG HELPERS -------------

# Adaptive tau control state
_edge_high_streak = 0

def _maybe_adjust_tau(model, dbg, streak_steps=1000, edge_thresh=0.10, tau_increment=0.5):
    global _edge_high_streak
    edge_frac = dbg.get('bounded_edge_frac', 0.0)
    # Access FSQ gate
    fsq_gate = model.vq.fsq_gate if hasattr(model.vq, 'fsq_gate') else None
    if fsq_gate is None:
        return None
    if edge_frac > edge_thresh:
        _edge_high_streak += 1
    else:
        _edge_high_streak = 0
    adjusted = False
    if _edge_high_streak >= streak_steps:
        new_min = min(fsq_gate.max_tau, fsq_gate.min_tau + tau_increment)
        if new_min > fsq_gate.min_tau:
            fsq_gate.min_tau = new_min
            adjusted = True
        _edge_high_streak = 0
    # Report current tau (clamped view)
    with torch.no_grad():
        tau_now = fsq_gate.log_tau.exp().clamp(fsq_gate.min_tau, fsq_gate.max_tau).item()
    return {'adjusted': adjusted, 'tau': tau_now, 'min_tau': fsq_gate.min_tau, 'max_tau': fsq_gate.max_tau, 'streak': _edge_high_streak}


def train():
    start_iter = max(args.start_iteration, results['n_updates'])
    
    train_iter = iter(training_loader)
    for i in tqdm(range(start_iter, args.n_updates)):
        try:
            (x, _) = next(train_iter)
        except StopIteration:
            train_iter = iter(training_loader)  # Reset iterator when epoch ends
            (x, _) = next(train_iter)
            
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Forward + loss under autocast
        with torch.amp.autocast('cuda', enabled=bool(args.amp), dtype=torch.bfloat16 if args.amp else None):
            x_hat = model(x)
            recon_loss = F.smooth_l1_loss(x_hat, x)

        # Backward with scaler
        scaler.scale(recon_loss).backward()

        # Clip gradients before stepping
        scaler.unscale_(optimizer)
        grad_norm_before_clip = _compute_grad_norm([p for p in model.parameters() if p.requires_grad])
        # Per-module grad norms (after unscale)
        enc_grad_norm = _module_grad_l2_norm(model.encoder)
        dec_grad_norm = _module_grad_l2_norm(model.decoder)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # Step the learning rate scheduler

        results["recon_errors"].append(recon_loss.cpu().detach())
        results["loss_vals"].append(recon_loss.cpu().detach())
        results["n_updates"] = i

        # Log to W&B if enabled and available
        if args.use_wandb:
            # Log basic metrics every step
            metrics = {
                'train/loss': recon_loss.item(),
                'train/learning_rate': scheduler.get_last_lr()[0],
                'train/grad_norm_before_clip': grad_norm_before_clip,
                'train/grad_norm_encoder': enc_grad_norm,
                'train/grad_norm_decoder': dec_grad_norm,
                'train/grad_ratio_dec_over_enc': (dec_grad_norm / (enc_grad_norm + 1e-8)),
                'train/x_variance': torch.var(x, dim=0).mean().item(),
                'train/x_hat_variance': torch.var(x_hat.detach(), dim=0).mean().item(),
                'step': i
            }

            # Quick NaN/Inf guards
            metrics['debug/has_nan_loss'] = int(not torch.isfinite(recon_loss))
            metrics['debug/has_nan_xhat'] = int(not torch.isfinite(x_hat).all())

            # Calculate codebook usage only during log intervals
            if i % args.log_interval == 0:
                with torch.no_grad():
                    indices = model.tokenize(x)
                    unique_codes = torch.unique(indices).numel()
                    metrics['train/codebook_usage'] = unique_codes / model.codebook_size
                    # Extra FSQ/embedding stats
                    dbg = _debug_fsq_and_embedding_stats(model, x, args.num_bins)
                    for k, v in dbg.items():
                        metrics[f'debug/{k}'] = v

            wandb.log(metrics)
            
            # Log system metrics
            if torch.cuda.is_available():
                wandb.log({
                    'system/gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                    'system/gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
                    'step': i
                })
            
            # Log learning rate
            for j, param_group in enumerate(optimizer.param_groups):
                wandb.log({
                    f'learning_rate/group_{j}': param_group['lr'],
                    'step': i
                })

        if i % args.log_interval == 0:
            """
            save model and print values
            """
            if args.save:
                hyperparameters = args.__dict__
                save_videotokenizer_model_and_results(
                    model, optimizer, results, hyperparameters, args.filename, checkpoints_dir
                )
                # Visualizations
                x_hat_vis = x_hat.detach().cpu()
                x_vis = x.detach().cpu()
                save_path = os.path.join(visualizations_dir, f'vqvae_recon_step_{i}_{args.filename}.png')
                visualize_reconstruction(x_vis[:16], x_hat_vis[:16], save_path)

            # Print consolidated debug info
            if args.debug_stats:
                dbg = _debug_fsq_and_embedding_stats(model, x, args.num_bins)
                tau_info = _maybe_adjust_tau(model, dbg)
                lr_now = scheduler.get_last_lr()[0]
                tau_str = ""
                if tau_info is not None:
                    tau_str = f" tau={tau_info['tau']:.3f} min_tau={tau_info['min_tau']:.3f} max_tau={tau_info['max_tau']:.3f} streak={tau_info['streak']}"
                    if tau_info['adjusted']:
                        print("[ADAPT] Increased FSQ min_tau by +0.5 due to sustained edge saturation.")
                print(
                    f"[DBG] step={i} lr={lr_now:.6g} loss={recon_loss.item():.6g} "
                    f"grad_norm(pre-clip)={grad_norm_before_clip:.3f} enc_gn={enc_grad_norm:.3f} dec_gn={dec_grad_norm:.3f} "
                    f"embed(mean/std/min/max)={dbg['embed_mean']:.3f}/{dbg['embed_std']:.3f}/"
                    f"{dbg['embed_min']:.3f}/{dbg['embed_max']:.3f} tanh_sat>0.99={dbg['tanh_abs_gt_0.99_frac']:.3%} "
                    f"edge_frac={dbg['bounded_edge_frac']:.3%} unique_codes={dbg['codebook_unique_count']}/"
                    f"{dbg['codebook_size']} (ratio {dbg['unique_ratio']:.3%}) top_code_share={dbg['top_code_share']:.3%} "
                    f"dec_in_std={dbg['decoder_in_std']:.4f} dec_mid_std={dbg['decoder_mid_std']:.4f} "
                    f"||enc||={dbg['param_norm/encoder']:.2f} ||dec||={dbg['param_norm/decoder']:.2f}{tau_str}"
                )
                if dbg['top_code_share'] > 0.5 or dbg['unique_ratio'] < 0.01:
                    print("[WARN] Codebook collapse signal detected: high top_code_share or very low unique_ratio.")

            print('Update #', i, 'Recon Loss:',
                  torch.mean(torch.stack(results["recon_errors"][-args.log_interval:])).item())
    
    # Finish W&B run
    if args.use_wandb:
        wandb.finish()
        print("✅ W&B run finished")

if __name__ == "__main__":
    train()