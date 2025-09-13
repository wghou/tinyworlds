import numpy as np
import torch
import torch.optim as optim
import argparse
import sys
import os
import time
import json
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.vqvae.models.video_tokenizer import Video_Tokenizer
from src.latent_action_model.models.lam import LAM
from src.dynamics.models.dynamics_model import DynamicsModel
from datasets.utils import visualize_reconstruction, load_data_and_data_loaders
from tqdm import tqdm
import json
from einops import rearrange

# Import wandb utilities
from src.utils.wandb_utils import (
    init_wandb, log_training_metrics, log_model_gradients, log_model_parameters,
    log_learning_rate, log_reconstruction_comparison, log_video_sequence,
    log_system_metrics, finish_wandb, create_wandb_config
)
from src.utils.scheduler_utils import create_cosine_scheduler
from src.utils.utils import readable_timestamp
from src.utils.utils import save_training_state, load_videotokenizer_from_checkpoint, load_lam_from_checkpoint

from src.utils.config import DynamicsConfig, load_config
import wandb

# Load config (YAML + dotlist overrides)
args: DynamicsConfig = load_config(DynamicsConfig, default_config_path=os.path.join(os.getcwd(), 'configs', 'dynamics.yaml'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create organized save directory structure
if args.save:
    ts = readable_timestamp()
    # Create main results directory for this run
    run_dir = os.path.join(os.getcwd(), 'src', 'dynamics', 'results', f'dynamics_{args.filename or ts}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    visualizations_dir = os.path.join(run_dir, 'visualizations')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)

print("Loading pre-trained video tokenizer...")
if os.path.isfile(args.video_tokenizer_path):
    print(f"Loading video tokenizer from {args.video_tokenizer_path}")
    video_tokenizer, vq_ckpt = load_videotokenizer_from_checkpoint(args.video_tokenizer_path, device)
    video_tokenizer.eval()
    for p in video_tokenizer.parameters():
        p.requires_grad = False
    print("✅ Video tokenizer loaded from its saved config and frozen")
else:
    raise FileNotFoundError(f"Video tokenizer checkpoint not found at {args.video_tokenizer_path}")

print("Loading pre-trained latent action model...")
if os.path.isfile(args.lam_path):
    print(f"Loading LAM from {args.lam_path}")
    lam, lam_ckpt = load_lam_from_checkpoint(args.lam_path, device)
    lam.eval()
    for p in lam.parameters():
        p.requires_grad = False
    print("✅ LAM loaded from its saved config and frozen")
else:
    raise FileNotFoundError(f"LAM checkpoint not found at {args.lam_path}")

"""
Set up dynamics model
"""
print("Initializing dynamics model...")
dynamics_model = DynamicsModel(
    frame_size=(args.frame_size, args.frame_size),
    patch_size=args.patch_size,
    embed_dim=args.embed_dim,
    num_heads=args.num_heads,
    hidden_dim=args.hidden_dim,
    num_blocks=args.num_blocks,
    conditioning_dim=lam.action_dim,
    latent_dim=args.latent_dim,
    num_bins=args.num_bins,
).to(device)

# Optionally compile dynamics model
if args.compile:
    try:
        dynamics_model = torch.compile(dynamics_model, mode="reduce-overhead", fullgraph=False, dynamic=True)
        print("✅ Dynamics model compiled with torch.compile")
    except Exception as e:
        print(f"⚠️ torch.compile not available or failed: {e}")

"""
Set up optimizer and training loop
"""
# Create parameter groups to avoid weight decay on biases and norm layers
decay = []
no_decay = []
for name, param in dynamics_model.named_parameters():
    if param.requires_grad:
        if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)

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

# AMP scaler for mixed precision
scaler = torch.amp.GradScaler('cuda', enabled=bool(args.amp))

results = {
    'n_updates': 0,
    'dynamics_losses': [],
    'loss_vals': [],
}

# Initialize W&B if enabled and available
if args.use_wandb:
    # TODO: make one function util for this
    # Create model configuration
    model_config = {
        'frame_size': (args.frame_size, args.frame_size),
        'patch_size': args.patch_size,
        'embed_dim': args.embed_dim,
        'num_heads': args.num_heads,
        'hidden_dim': args.hidden_dim,
        'num_blocks': args.num_blocks,
        'latent_dim': args.latent_dim,
        'use_actions': args.use_actions
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
        'timestamp': ts
    }
    
    # Initialize W&B
    run_name = args.wandb_run_name or f"dynamics_{ts}"
    wandb.init(
        project=args.wandb_project,
        config=wandb_config,
        name=run_name,
        tags=["dynamics", "training"]
    )

    # Watch model for gradients and parameters
    wandb.watch(dynamics_model, log="all", log_freq=args.log_interval)
    
    print(f"🚀 W&B run initialized: {wandb.run.name}")
    print(f"📊 Project: {args.wandb_project}")
    print(f"🔗 View at: {wandb.run.get_url()}")

dynamics_model.train()

def train():
    _, _, training_loader, _, _ = load_data_and_data_loaders(
        dataset=args.dataset, 
        batch_size=args.batch_size, 
        num_frames=args.context_length
    )
    train_iter = iter(training_loader)
    
    for i in tqdm(range(0, args.n_updates)):
        try:
            x, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(training_loader)  # Reset iterator when epoch ends
            x, _ = next(train_iter)
            
        x = x.to(device, non_blocking=True)  # [batch_size, seq_len, channels, height, width]
        optimizer.zero_grad(set_to_none=True)

        # Get video tokenizer latents for current frames (frozen model)
        with torch.no_grad():
            # TODO: make video tokenizer forward pass for inference return quantized video latents
            # TODO: should I pass in indices or latents?
            video_latents = video_tokenizer.encoder(x)  # [batch_size, seq_len, num_patches, latent_dim]
            
            # Apply vector quantization to get discrete latents
            quantized_video_latents = video_tokenizer.quantizer(video_latents) # [batch_size, seq_len, num_patches, latent_dim]
            
            if args.use_actions:
                actions = lam.encoder(x)  # [batch_size, seq_len - 1, action_dim]
                quantized_actions = lam.quantizer(actions) # [batch_size, seq_len-1, action_dim]
            else:
                quantized_actions = None

        target_next_tokens = video_tokenizer.quantizer.get_indices_from_latents(quantized_video_latents, dim=-1) # [batch_size, seq_len-1, num_patches]

        # Predict next frame latents using dynamics model under autocast
        with torch.amp.autocast('cuda', enabled=bool(args.amp), dtype=torch.bfloat16 if args.amp else None):
            predicted_next_logits, mask_positions = dynamics_model(quantized_video_latents, training=True, conditioning=quantized_actions)  # [batch_size, seq_len, num_patches, codebook_size]

            if mask_positions is not None:
                num_masked = mask_positions.sum().item()
                total_positions = mask_positions.numel()
                masking_rate = num_masked / total_positions

            # Compute loss only on masked tokens (MaskGit-style)
            if mask_positions is not None:
                mask_for_loss = mask_positions
                masked_logits = predicted_next_logits.reshape(-1, predicted_next_logits.shape[-1])  # [N, codebook_size]
                masked_targets = target_next_tokens.reshape(-1)  # [N]
                masked_mask = mask_for_loss.reshape(-1)  # [N]
                
                if masked_mask.sum() > 0:  # If there are masked tokens
                    assert masked_logits.shape[0] == masked_mask.shape[0], f"Shape mismatch: {masked_logits.shape[0]} vs {masked_mask.shape[0]}"
                    assert masked_targets.shape[0] == masked_mask.shape[0], f"Shape mismatch: {masked_targets.shape[0]} vs {masked_mask.shape[0]}"
                    masked_logits = masked_logits[masked_mask]  # [num_masked, codebook_size]
                    masked_targets = masked_targets[masked_mask]  # [num_masked]
                    dynamics_loss = F.cross_entropy(masked_logits, masked_targets)
                else:
                    dynamics_loss = torch.tensor(0.0, device=predicted_next_logits.device, requires_grad=True)
            else:
                dynamics_loss = F.cross_entropy(
                    predicted_next_logits.reshape(-1, predicted_next_logits.shape[-1]),  # [N, codebook_size]
                    target_next_tokens.reshape(-1),  # [N]
                )

            # Total loss
            loss = dynamics_loss

        # Backward + clip with scaler
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(dynamics_model.parameters(), max_norm=1.0)
        optimizer.step()
        scaler.update()
        scheduler.step()  # Step the learning rate scheduler

        results["dynamics_losses"].append(dynamics_loss.cpu().detach())
        results["loss_vals"].append(loss.cpu().detach())
        results["n_updates"] = i

        predicted_next_indices = torch.argmax(predicted_next_logits, dim=-1) # [batch_size, seq_len-1, num_patches]
        predicted_next_latents = video_tokenizer.quantizer.get_latents_from_indices(predicted_next_indices, dim=-1) # [batch_size, seq_len-1, num_patches, latent_dim]
        
        # Log to W&B if enabled
        if args.use_wandb:
            # Log training metrics
            metrics = {
                'dynamics_loss': dynamics_loss.item(),
                'total_loss': loss.item(),
                'masking_rate': masking_rate
            }
            log_training_metrics(i, metrics, prefix="train")
            # Log system metrics
            log_system_metrics(i)
            # Log learning rate
            log_learning_rate(optimizer, i)

        # Action diversity metrics (minimal)
        if args.use_actions and args.use_wandb and quantized_actions is not None:
            qa = quantized_actions.squeeze(2)
            div = dynamics_model.compute_action_diversity(actions, qa, lam.quantizer)
            wandb.log({
                'actions/usage': float(div['action_usage']),
                'actions/entropy': float(div['action_entropy']),
                'actions/pre_quant_var': float(div['pre_quant_var']),
                'step': i
            })

        # Debug prints
        if i % 10 == 0:  # Print every 10 iterations
            print(f"Iteration {i}, Dynamics Loss: {dynamics_loss.item():.6f}")

        if i % args.log_interval == 0:
            """
            save model and print values
            """
            if args.save:
                hyperparameters = args.__dict__
                save_training_state(dynamics_model, optimizer, scheduler, hyperparameters, checkpoints_dir, prefix='dynamics', step=i)
                
                # Decode predicted latents (predicted_next_latents: [B, seq_len-1, ...])
                with torch.no_grad():
                    predicted_frames = video_tokenizer.decoder(predicted_next_latents[:16])  # [B, seq_len-1, ...]
 
                # Ground truth frames
                target_frames_full = x[:, 1:]  # [B, seq_len-1, ...]

                # Display the masked patches in the ground truth frames as black
                masked_target_frames_full = target_frames_full.clone()
                
                # Convert mask_positions to patch-level mask for visualization
                if mask_positions is not None:
                    # mask_positions: [B, S, N] where N is number of patches
                    # We need to convert this to pixel-level mask
                    B, S, N = mask_positions.shape
                    patch_size = args.patch_size
                    H, W = args.frame_size, args.frame_size  # Frame size
                    
                    # Create pixel-level mask for the target frames (which have seq_len-1)
                    # We need to slice mask_positions to match target_frames_full
                    mask_for_viz = mask_positions[:, 1:]  # Remove first timestep to match target_frames_full
                    B_viz, S_viz, N_viz = mask_for_viz.shape
                    
                    # Create pixel-level mask
                    pixel_mask = torch.zeros(B_viz, S_viz, H, W, device=mask_positions.device)
                    
                    # For each batch and sequence, convert patch mask to pixel mask
                    for b in range(B_viz):
                        for s in range(S_viz):
                            patch_mask = mask_for_viz[b, s]  # [N]
                            # Convert patch indices to pixel coordinates
                            for patch_idx in range(N_viz):
                                if patch_mask[patch_idx]:  # If patch is masked
                                    # Calculate patch position
                                    patch_row = (patch_idx // (W // patch_size)) * patch_size
                                    patch_col = (patch_idx % (W // patch_size)) * patch_size
                                    # Set patch pixels to black (0)
                                    pixel_mask[b, s, patch_row:patch_row+patch_size, patch_col:patch_col+patch_size] = 1
                    
                    # Apply mask to frames (set masked patches to black)
                    # pixel_mask: [B_viz, S_viz, H, W], masked_target_frames_full: [B_viz, S_viz, C, H, W]
                    # Need to expand pixel_mask to include channel dimension
                    pixel_mask_expanded = pixel_mask.unsqueeze(2).expand(-1, -1, 3, -1, -1)  # [B_viz, S_viz, C, H, W]
                    masked_target_frames_full = masked_target_frames_full * (1 - pixel_mask_expanded)

                save_path = os.path.join(visualizations_dir, f'dynamics_prediction_step_{i}_{args.filename}.png')
                visualize_reconstruction(masked_target_frames_full[:16].cpu(), predicted_frames[:16].cpu(), save_path)

            print('Update #', i, 'Dynamics Loss:',
                  torch.mean(torch.stack(results["dynamics_losses"][-args.log_interval:])).item(),
                  'Total Loss', torch.mean(torch.stack(results["loss_vals"][-args.log_interval:])).item())

    # Finish W&B run
    if args.use_wandb:
        finish_wandb()

if __name__ == "__main__":
    train()