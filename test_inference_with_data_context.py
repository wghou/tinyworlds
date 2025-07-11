#!/usr/bin/env python3
"""
Diagnostic script to test inference quality using real data context.
This helps determine if the quality degradation is due to:
1. Error accumulation in rollouts (autoregressive generation)
2. Training-inference distribution mismatch

The script runs inference but uses real frames from the dataset as context,
then only generates the next frame. This isolates the issue.
"""

import torch
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
import cv2

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(__file__))

from src.dynamics.models.dynamics_model import DynamicsModel
from src.vqvae.models.video_tokenizer import Video_Tokenizer
from src.latent_action_model.models.lam import LAM
from src.vqvae.utils import load_data_and_data_loaders

def load_models(video_tokenizer_path, lam_path, dynamics_path, device):
    """Load all three trained models"""
    print("Loading trained models...")
    
    # Load video tokenizer
    print(f"Loading video tokenizer from {video_tokenizer_path}")
    video_tokenizer = Video_Tokenizer(
        frame_size=(64, 64),
        patch_size=4,
        embed_dim=128,
        num_heads=4,
        hidden_dim=512,
        num_blocks=2,
        latent_dim=32,
        dropout=0.1,
        codebook_size=64,
        beta=0.01
    ).to(device)
    
    try:
        checkpoint = torch.load(video_tokenizer_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"weights_only=True failed, trying weights_only=False: {e}")
        checkpoint = torch.load(video_tokenizer_path, map_location=device, weights_only=False)
    
    video_tokenizer.load_state_dict(checkpoint['model'])
    video_tokenizer.eval()
    print("✅ Video tokenizer loaded")
    
    # Load LAM
    print(f"Loading LAM from {lam_path}")
    lam = LAM(
        frame_size=(64, 64),
        n_actions=8,
        patch_size=4,  # Match video tokenizer patch_size
        embed_dim=128,
        num_heads=4,
        hidden_dim=512,
        num_blocks=2,
        action_dim=32,
        dropout=0.1,
        beta=1.0
    ).to(device)
    
    try:
        checkpoint = torch.load(lam_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"weights_only=True failed, trying weights_only=False: {e}")
        checkpoint = torch.load(lam_path, map_location=device, weights_only=False)
    
    lam.load_state_dict(checkpoint['model'])
    lam.eval()
    print("✅ LAM loaded")
    
    # Load dynamics model
    print(f"Loading dynamics model from {dynamics_path}")
    dynamics_model = DynamicsModel(
        frame_size=(64, 64),
        patch_size=4,
        embed_dim=128,
        num_heads=4,
        hidden_dim=512,
        num_blocks=2,
        latent_dim=32,
        dropout=0.1
    ).to(device)
    
    try:
        checkpoint = torch.load(dynamics_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"weights_only=True failed, trying weights_only=False: {e}")
        checkpoint = torch.load(dynamics_path, map_location=device, weights_only=False)
    
    dynamics_model.load_state_dict(checkpoint['model'])
    dynamics_model.eval()
    print("✅ Dynamics model loaded")
    
    return video_tokenizer, lam, dynamics_model

def encode_frame_to_tokens(video_tokenizer, frame):
    """Encode frame to video tokens"""
    with torch.no_grad():
        # frame: [batch_size, seq_len, C, H, W]
        video_latents = video_tokenizer.encoder(frame)  # [batch_size, seq_len, num_patches, latent_dim]
        _, quantized_video_latents, _ = video_tokenizer.vq(video_latents)  # [batch_size, seq_len, num_patches, latent_dim]
        return quantized_video_latents, video_latents

def get_action_from_frame_transition(lam, frame1, frame2):
    """Get action latent from real frame transition"""
    with torch.no_grad():
        # Concatenate frames for action prediction
        frames = torch.cat([frame1, frame2], dim=1)  # [batch_size, 2, C, H, W]
        actions, _ = lam.encoder(frames)  # [batch_size, 1, action_dim]
        actions_flat = actions.reshape(-1, actions.size(-1))
        _, quantized_actions_flat, _ = lam.quantizer(actions_flat)
        return quantized_actions_flat

def predict_next_tokens(dynamics_model, video_latents, action_latent):
    """Use dynamics model to predict next video tokens"""
    with torch.no_grad():
        batch_size, seq_len, num_patches, latent_dim = video_latents.shape
        
        # Expand action to match video latents shape
        action_expanded = action_latent.unsqueeze(1).unsqueeze(1).expand(-1, seq_len, num_patches, -1)
        
        # Add action latents to video latents
        combined_latents = video_latents + action_expanded
        
        # Predict next video tokens
        next_video_latents = dynamics_model(combined_latents, training=False)
        
        # Return the last timestep prediction
        next_video_latents = next_video_latents[:, -1, :, :]
        
        return next_video_latents

def visualize_comparison(real_frames, predicted_frames, target_frames, actions, save_path):
    """Visualize comparison between real, predicted, and target frames"""
    # Move to CPU and denormalize
    real_frames = (real_frames.detach().cpu() + 1) / 2
    predicted_frames = (predicted_frames.detach().cpu() + 1) / 2
    target_frames = (target_frames.detach().cpu() + 1) / 2
    
    # Clamp to [0, 1]
    real_frames = torch.clamp(real_frames, 0, 1)
    predicted_frames = torch.clamp(predicted_frames, 0, 1)
    target_frames = torch.clamp(target_frames, 0, 1)
    
    num_frames = real_frames.shape[1]
    
    # Create figure with 3 rows: real context, predicted next, target next
    fig, axes = plt.subplots(3, num_frames, figsize=(4 * num_frames, 12))
    
    # Row 1: Real context frames
    for i in range(num_frames):
        frame = real_frames[0, i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(frame)
        axes[0, i].set_title(f'Real Frame {i+1}', fontsize=10)
        axes[0, i].axis('off')
    
    # Row 2: Predicted next frame
    predicted_frame = predicted_frames[0, 0].permute(1, 2, 0).numpy()
    axes[1, 0].imshow(predicted_frame)
    axes[1, 0].set_title(f'Predicted Next (Action {actions[0].item()})', fontsize=10)
    axes[1, 0].axis('off')
    
    # Row 3: Target next frame
    target_frame = target_frames[0, 0].permute(1, 2, 0).numpy()
    axes[2, 0].imshow(target_frame)
    axes[2, 0].set_title('Target Next', fontsize=10)
    axes[2, 0].axis('off')
    
    # Hide unused subplots
    for i in range(1, num_frames):
        axes[1, i].axis('off')
        axes[2, i].axis('off')
    
    plt.suptitle('Real Context vs Generated Next Frame', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")

def visualize_sequence_comparison(frame_results, save_path):
    """Visualize comparison for all frame predictions in a sequence."""
    if not frame_results:
        print("No frame results to visualize.")
        return
    
    num_frames = len(frame_results)
    
    # Create figure with 3 rows: context frames, predicted next, target next
    fig, axes = plt.subplots(3, num_frames, figsize=(4 * num_frames, 12))
    
    for frame_idx, frame_result in enumerate(frame_results):
        context_frames_data = frame_result['context_frames_data']
        predicted_frame = frame_result['predicted_frame']
        target_frame = frame_result['target_frame']
        
        # Move to CPU and denormalize
        context_frames_data = (context_frames_data.detach().cpu() + 1) / 2
        predicted_frame = (predicted_frame.detach().cpu() + 1) / 2
        target_frame = (target_frame.detach().cpu() + 1) / 2
        
        # Clamp to [0, 1]
        context_frames_data = torch.clamp(context_frames_data, 0, 1)
        predicted_frame = torch.clamp(predicted_frame, 0, 1)
        target_frame = torch.clamp(target_frame, 0, 1)
        
        # Row 1: Show all context frames (stacked horizontally)
        context_frames_np = context_frames_data[0].permute(0, 2, 3, 1).numpy()  # [num_context_frames, H, W, C]
        # Create a horizontal concatenation of context frames
        context_concat = np.concatenate([context_frames_np[i] for i in range(context_frames_np.shape[0])], axis=1)
        axes[0, frame_idx].imshow(context_concat)
        axes[0, frame_idx].set_title(f'Context (Frames 1-{frame_result["context_frames"]})', fontsize=10)
        axes[0, frame_idx].axis('off')
        
        # Row 2: Predicted next frame
        predicted_frame_np = predicted_frame[0, 0].permute(1, 2, 0).numpy()
        axes[1, frame_idx].imshow(predicted_frame_np)
        axes[1, frame_idx].set_title(f'Predicted Frame {frame_result["frame_idx"]}\nMSE: {frame_result["mse_loss"]:.4f}', fontsize=10)
        axes[1, frame_idx].axis('off')
        
        # Row 3: Target next frame
        target_frame_np = target_frame[0, 0].permute(1, 2, 0).numpy()
        axes[2, frame_idx].imshow(target_frame_np)
        axes[2, frame_idx].set_title(f'Target Frame {frame_result["frame_idx"]}\nPSNR: {frame_result["psnr"]:.1f}dB', fontsize=10)
        axes[2, frame_idx].axis('off')
    
    plt.suptitle('Sequence Prediction: Real Context vs Generated Next Frames', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sequence visualization saved to: {save_path}")

def test_with_real_context(video_tokenizer, lam, dynamics_model, dataloader, device, num_tests=10):
    """Test inference using real data context for all frames in sequence"""
    print(f"\n🧪 Testing inference with real data context ({num_tests} samples)...")
    
    results = {
        'frame_predictions': [],  # Store results for each frame transition
        'sequence_mse_losses': [],
        'sequence_psnr_values': [],
        'sequence_ssim_values': [],
        'latent_variances': [],
        'action_distribution': []
    }
    
    for test_idx in range(num_tests):
        print(f"\n--- Test {test_idx + 1}/{num_tests} ---")
        
        # Get a sequence from the dataloader
        try:
            batch = next(iter(dataloader))
            frames = batch[0].to(device)  # [batch_size, seq_len, C, H, W]
        except StopIteration:
            print("Dataloader exhausted, stopping tests")
            break
        
        batch_size, seq_len, C, H, W = frames.shape
        print(f"Frame sequence shape: {frames.shape}")
        
        # Test prediction for each frame transition in the sequence
        frame_results = []
        total_mse = 0.0
        total_psnr = 0.0
        
        for frame_idx in range(1, seq_len):  # Start from frame 1, predict frame 2, 3, 4...
            print(f"  Predicting frame {frame_idx + 1} from frames 1-{frame_idx}")
            
            # Use frames 1 to frame_idx as context
            context_frames = frames[:, :frame_idx, :, :, :]  # [batch_size, frame_idx, C, H, W]
            target_frame = frames[:, frame_idx:frame_idx+1, :, :, :]   # [batch_size, 1, C, H, W]
            
            # Get action from real frame transition (frame_idx -> frame_idx+1)
            action_latent = get_action_from_frame_transition(lam, context_frames[:, -1:], target_frame)
            
            # Encode context frames to video tokens
            video_latents, _ = encode_frame_to_tokens(video_tokenizer, context_frames)
            
            # Predict next frame using dynamics model
            predicted_next_latents = predict_next_tokens(dynamics_model, video_latents, action_latent)
            
            # Decode predicted latents to frames
            predicted_next_latents = predicted_next_latents.unsqueeze(1)  # Add sequence dimension
            predicted_frames = video_tokenizer.decoder(predicted_next_latents)
            
            # Calculate metrics for this frame prediction
            with torch.no_grad():
                # MSE loss
                mse_loss = torch.mean((predicted_frames - target_frame) ** 2).item()
                total_mse += mse_loss
                
                # PSNR
                max_val = 2.0  # Since frames are in [-1, 1]
                psnr = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(torch.mean((predicted_frames - target_frame) ** 2))
                total_psnr += psnr.item()
                
                # Latent variance
                latent_var = torch.var(predicted_next_latents).item()
                
                # Store frame-specific results
                frame_result = {
                    'frame_idx': frame_idx + 1,
                    'context_frames': frame_idx,
                    'mse_loss': mse_loss,
                    'psnr': psnr.item(),
                    'latent_var': latent_var,
                    'action_mean': action_latent.mean().item(),
                    'predicted_frame': predicted_frames.clone(),
                    'target_frame': target_frame.clone(),
                    'context_frames_data': context_frames.clone()
                }
                frame_results.append(frame_result)
                
                print(f"    Frame {frame_idx + 1}: MSE={mse_loss:.6f}, PSNR={psnr.item():.2f}dB")
        
        # Store sequence results
        results['frame_predictions'].append(frame_results)
        results['sequence_mse_losses'].append(total_mse / (seq_len - 1))  # Average MSE per frame
        results['sequence_psnr_values'].append(total_psnr / (seq_len - 1))  # Average PSNR per frame
        results['latent_variances'].extend([fr['latent_var'] for fr in frame_results])
        results['action_distribution'].extend([fr['action_mean'] for fr in frame_results])
        
        print(f"  Sequence Average MSE: {total_mse / (seq_len - 1):.6f}")
        print(f"  Sequence Average PSNR: {total_psnr / (seq_len - 1):.2f} dB")
        
        # Save visualization for first few tests
        if test_idx < 3:
            save_dir = "inference_results"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"real_context_test_{test_idx+1}_all_frames.png")
            visualize_sequence_comparison(frame_results, save_path)
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics ({len(results['sequence_mse_losses'])} tests):")
    print(f"Average Sequence MSE Loss: {np.mean(results['sequence_mse_losses']):.6f} ± {np.std(results['sequence_mse_losses']):.6f}")
    print(f"Average Sequence PSNR: {np.mean(results['sequence_psnr_values']):.2f} ± {np.std(results['sequence_psnr_values']):.2f} dB")
    print(f"Average Latent Variance: {np.mean(results['latent_variances']):.6f} ± {np.std(results['latent_variances']):.6f}")
    print(f"Action Distribution Mean: {np.mean(results['action_distribution']):.6f} ± {np.std(results['action_distribution']):.6f}")
    
    # Print per-frame statistics
    print(f"\n📈 Per-Frame Statistics:")
    num_frames = len(results['frame_predictions'][0]) if results['frame_predictions'] else 0
    for frame_idx in range(num_frames):
        frame_mse_losses = [test[frame_idx]['mse_loss'] for test in results['frame_predictions']]
        frame_psnr_values = [test[frame_idx]['psnr'] for test in results['frame_predictions']]
        print(f"  Frame {frame_idx + 2}: MSE={np.mean(frame_mse_losses):.6f}±{np.std(frame_mse_losses):.6f}, "
              f"PSNR={np.mean(frame_psnr_values):.2f}±{np.std(frame_psnr_values):.2f} dB")
    
    return results

def test_with_masked_inference(video_tokenizer, lam, dynamics_model, dataloader, device, num_tests=5):
    """Test inference with masking applied (to test Theory 2)"""
    print(f"\n🎭 Testing inference with masking applied ({num_tests} samples)...")
    
    # Temporarily modify dynamics model to apply masking during inference
    original_forward = dynamics_model.decoder.forward
    
    def masked_forward(latents, training=True):
        """Forward pass with masking applied even during inference"""
        batch_size, seq_len, num_patches, latent_dim = latents.shape
        
        # Apply masking during inference (like training)
        masking_rate = 0.75  # Fixed masking rate
        mask = torch.bernoulli(torch.full((batch_size, seq_len), masking_rate, device=latents.device))
        mask = mask.unsqueeze(-1).unsqueeze(-1)
        latents = latents * mask
        
        # Call original forward
        return original_forward(latents, training=False)
    
    # Replace forward method
    dynamics_model.decoder.forward = masked_forward
    
    try:
        results = test_with_real_context(video_tokenizer, lam, dynamics_model, dataloader, device, num_tests)
        print(f"\n🎭 Masked Inference Results:")
        print(f"Average Sequence MSE Loss: {np.mean(results['sequence_mse_losses']):.6f} ± {np.std(results['sequence_mse_losses']):.6f}")
        print(f"Average Sequence PSNR: {np.mean(results['sequence_psnr_values']):.2f} ± {np.std(results['sequence_psnr_values']):.2f} dB")
        
        # Print per-frame statistics for masked inference
        print(f"\n🎭 Per-Frame Statistics (Masked):")
        num_frames = len(results['frame_predictions'][0]) if results['frame_predictions'] else 0
        for frame_idx in range(num_frames):
            frame_mse_losses = [test[frame_idx]['mse_loss'] for test in results['frame_predictions']]
            frame_psnr_values = [test[frame_idx]['psnr'] for test in results['frame_predictions']]
            print(f"  Frame {frame_idx + 2}: MSE={np.mean(frame_mse_losses):.6f}±{np.std(frame_mse_losses):.6f}, "
                  f"PSNR={np.mean(frame_psnr_values):.2f}±{np.std(frame_psnr_values):.2f} dB")
        
    finally:
        # Restore original forward method
        dynamics_model.decoder.forward = original_forward
    
    return results

def create_quality_progression_plot(results, save_path):
    """Create a plot showing how prediction quality changes over frame sequence"""
    if not results['frame_predictions']:
        print("No results to plot.")
        return
    
    # Extract per-frame statistics
    num_frames = len(results['frame_predictions'][0])
    frame_mse_means = []
    frame_mse_stds = []
    frame_psnr_means = []
    frame_psnr_stds = []
    
    for frame_idx in range(num_frames):
        frame_mse_losses = [test[frame_idx]['mse_loss'] for test in results['frame_predictions']]
        frame_psnr_values = [test[frame_idx]['psnr'] for test in results['frame_predictions']]
        
        frame_mse_means.append(np.mean(frame_mse_losses))
        frame_mse_stds.append(np.std(frame_mse_losses))
        frame_psnr_means.append(np.mean(frame_psnr_values))
        frame_psnr_stds.append(np.std(frame_psnr_values))
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    frame_numbers = list(range(2, num_frames + 2))  # Frame 2, 3, 4, ...
    
    # MSE plot
    ax1.errorbar(frame_numbers, frame_mse_means, yerr=frame_mse_stds, 
                marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Prediction Quality Progression: MSE Loss')
    ax1.grid(True, alpha=0.3)
    
    # PSNR plot
    ax2.errorbar(frame_numbers, frame_psnr_means, yerr=frame_psnr_stds, 
                marker='s', capsize=5, capthick=2, linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('Prediction Quality Progression: PSNR')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Quality progression plot saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Test inference with real data context")
    parser.add_argument("--video_tokenizer_path", type=str, default="/Users/almondgod/Repositories/nano-genie/src/vqvae/results/videotokenizer_sat_jun_28_00_44_40_2025/checkpoints/videotokenizer_checkpoint_sat_jun_28_00_44_40_2025.pth")
    parser.add_argument("--lam_path", type=str, default="/Users/almondgod/Repositories/nano-genie/src/latent_action_model/results/lam_Sat_Jun_28_12_51_06_2025/checkpoints/lam_checkpoint_Sat_Jun_28_12_51_06_2025.pth")
    parser.add_argument("--dynamics_path", type=str, default="/Users/almondgod/Repositories/nano-genie/src/dynamics/results/dynamics_Sat_Jun_28_13_05_43_2025/checkpoints/dynamics_checkpoint_Sat_Jun_28_13_05_43_2025.pth")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_tests", type=int, default=20)
    parser.add_argument("--test_masking", action="store_true", help="Test with masking applied")
    
    args = parser.parse_args()
    
    # Load models
    video_tokenizer, lam, dynamics_model = load_models(
        args.video_tokenizer_path, args.lam_path, args.dynamics_path, args.device
    )
    
    # Load data
    print("\n📊 Loading SONIC dataset...")
    _, _, _, validation_loader, _ = load_data_and_data_loaders(
        dataset='SONIC', batch_size=1, num_frames=4
    )
    
    # Test 1: Normal inference with real context
    normal_results = test_with_real_context(
        video_tokenizer, lam, dynamics_model, validation_loader, args.device, args.num_tests
    )
    
    # Create quality progression plot
    save_dir = "inference_results"
    os.makedirs(save_dir, exist_ok=True)
    progression_path = os.path.join(save_dir, "quality_progression_normal.png")
    create_quality_progression_plot(normal_results, progression_path)
    
    # Test 2: Inference with masking (if requested)
    if args.test_masking:
        masked_results = test_with_masked_inference(
            video_tokenizer, lam, dynamics_model, validation_loader, args.device, args.num_tests // 2
        )
        
        # Create quality progression plot for masked inference
        masked_progression_path = os.path.join(save_dir, "quality_progression_masked.png")
        create_quality_progression_plot(masked_results, masked_progression_path)
        
        # Compare results
        print(f"\n🔍 Comparison:")
        print(f"Normal MSE: {np.mean(normal_results['sequence_mse_losses']):.6f}")
        print(f"Masked MSE: {np.mean(masked_results['sequence_mse_losses']):.6f}")
        print(f"Normal PSNR: {np.mean(normal_results['sequence_psnr_values']):.2f} dB")
        print(f"Masked PSNR: {np.mean(masked_results['sequence_psnr_values']):.2f} dB")
    
    print(f"\n✅ Diagnostic tests completed!")
    print(f"Check 'inference_results/' directory for visualizations")

if __name__ == "__main__":
    main() 