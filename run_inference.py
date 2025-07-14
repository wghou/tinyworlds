import torch
from src.dynamics.models.dynamics_model import DynamicsModel
from src.vqvae.models.video_tokenizer import Video_Tokenizer
from src.latent_action_model.models.lam import LAM
from einops import rearrange
import argparse
from src.vqvae.utils import load_data_and_data_loaders
import matplotlib.pyplot as plt
import time
import os
import cv2
import numpy as np

def predict_next_tokens(dynamics_model, video_latents, action_latent=None, temperature=1.0, use_actions=True):
    """Use dynamics model to predict next video tokens with temperature sampling"""
    with torch.no_grad():
        # Prepare inputs for dynamics model
        # video_latents: [1, seq_len, num_patches, latent_dim]
        # action_latent: [1, action_dim] (optional)
        
        batch_size, seq_len, num_patches, latent_dim = video_latents.shape

        if use_actions and action_latent is not None:
            assert action_latent.shape[1] == latent_dim, "Action latent dimension must match video latent dimension"
            
            # Expand action to match video latents shape for each timestep
            action_expanded = action_latent.unsqueeze(1).unsqueeze(1).expand(-1, seq_len, num_patches, -1)  # [1, seq_len, num_patches, action_dim]
            
            # Add action latents to video latents (not concatenate)
            combined_latents = video_latents + action_expanded  # [1, seq_len, num_patches, latent_dim]
        else:
            # Use only video latents without action latents
            combined_latents = video_latents  # [1, seq_len, num_patches, latent_dim]
        
        # Predict next video tokens using the dynamics model
        next_video_latents = dynamics_model(combined_latents, training=False)  # [1, seq_len, num_patches, latent_dim]
        
        # Apply temperature sampling to reduce variance
        if temperature != 1.0:
            # Add noise scaled by temperature
            noise = torch.randn_like(next_video_latents) * (temperature - 1.0) * 0.1
            next_video_latents = next_video_latents + noise
        
        # Return the last timestep prediction (next frame)
        next_video_latents = next_video_latents[:, -1, :, :]  # [1, num_patches, latent_dim]
        
        return next_video_latents

def sample_action_with_diversity(previous_actions, n_actions, diversity_weight=0.1):
    """Sample action with diversity to avoid getting stuck in loops"""
    if len(previous_actions) < 2:
        # Just sample randomly for first few actions
        return torch.randint(0, n_actions, (1,))
    
    # Get recent action distribution
    recent_actions = previous_actions[-5:]  # Last 5 actions
    action_counts = torch.bincount(torch.tensor([a.item() for a in recent_actions]), minlength=n_actions)
    
    # Calculate diversity penalty (favor less frequent actions)
    diversity_penalty = action_counts.float() / len(recent_actions)
    
    # Sample with diversity
    if torch.rand(1).item() < diversity_weight:
        # Sample from less frequent actions
        min_count = action_counts.min()
        candidate_actions = torch.where(action_counts == min_count)[0]
        action_index = candidate_actions[torch.randint(0, len(candidate_actions), (1,))]
    else:
        # Sample randomly
        action_index = torch.randint(0, n_actions, (1,))
    
    return action_index

def load_models(video_tokenizer_path, lam_path, dynamics_path, device, use_actions=True):
    """Load trained models (LAM is optional if not using actions)"""
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
    
    # Try loading with weights_only=True first, fallback to False if it fails
    try:
        checkpoint = torch.load(video_tokenizer_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"weights_only=True failed, trying weights_only=False: {e}")
        checkpoint = torch.load(video_tokenizer_path, map_location=device, weights_only=False)
    
    video_tokenizer.load_state_dict(checkpoint['model'])
    video_tokenizer.eval()
    print("✅ Video tokenizer loaded")
    
    # Load LAM only if using actions
    lam = None
    if use_actions:
        print(f"Loading LAM from {lam_path}")
        lam = LAM(
            frame_size=(64, 64),
            n_actions=8,
            patch_size=4,  # Match video tokenizer patch_size
            embed_dim=128,
            num_heads=4,
            hidden_dim=512,
            num_blocks=2,
            action_dim=32,  # Match checkpoint (was 16)
            dropout=0.1,
            beta=1.0
        ).to(device)
        
        # Try loading with weights_only=True first, fallback to False if it fails
        try:
            checkpoint = torch.load(lam_path, map_location=device, weights_only=True)
        except Exception as e:
            print(f"weights_only=True failed, trying weights_only=False: {e}")
            checkpoint = torch.load(lam_path, map_location=device, weights_only=False)
        
        lam.load_state_dict(checkpoint['model'])
        lam.eval()
        print("✅ LAM loaded")
    else:
        print("⚠️ Skipping LAM loading (not using actions)")
    
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
    
    # Try loading with weights_only=True first, fallback to False if it fails
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
    """Encode a single frame to video tokens"""
    with torch.no_grad():
        # Add batch and sequence dimensions if needed
        if frame.dim() == 3:
            frame = frame.unsqueeze(0).unsqueeze(0)  # [C, H, W] -> [1, 1, C, H, W]
        elif frame.dim() == 4:
            frame = frame.unsqueeze(1)  # [1, C, H, W] -> [1, 1, C, H, W]
        
        # Encode frame to latent representation
        latent = video_tokenizer.encoder(frame)  # [1, 1, num_patches, latent_dim]

        print(f"latent shape: {latent.shape}")
        
        # Quantize to get video tokens
        quantized_latent = video_tokenizer.vq(latent)

        print(f"quantized_latent shape: {quantized_latent.shape}")
        
        return quantized_latent
    
def sample_first_frame_from_dataloader(dataloader):
    for batch in dataloader:
        frame = batch[0]
        break
    return frame
    
def sample_random_action(n_actions):
    random_action = torch.randint(0, n_actions, (1,))
    return random_action

def get_lam_latent_from_action_index(lam, action_index):
    with torch.no_grad():
        action_latent = lam.quantizer.embedding.weight[action_index] # get the latent action embedding in the codebook at action index
        return action_latent
    
def visualize_inference(frames, inferred_actions, fps, use_actions=True):
    """
    Visualize the inference results showing frames alternating with their inferred action numbers.
    Also save an MP4 file showing just the frames in order.
    
    Args:
        frames: Tensor of shape [batch_size, num_frames, C, H, W] 
        inferred_actions: List of action indices
        fps: Frames per second for the MP4 video
        use_actions: Whether actions were used in generation
    """
    # Move to CPU and convert to numpy
    frames = frames.detach().cpu()
    
    # Denormalize frames from [-1, 1] to [0, 1]
    frames = (frames + 1) / 2
    frames = torch.clamp(frames, 0, 1)
    
    # Get dimensions
    batch_size, num_frames, C, H, W = frames.shape
    
    if use_actions:
        # Create figure with alternating frames and action numbers
        fig, axes = plt.subplots(1, num_frames * 2 - 1, figsize=(4 * (num_frames * 2 - 1), 4))
        
        # Handle single subplot case
        if num_frames == 1:
            axes = [axes]
        
        for i in range(num_frames):
            # Plot frame
            frame_idx = i * 2
            frame = frames[0, i].permute(1, 2, 0).numpy()  # [H, W, C]
            axes[frame_idx].imshow(frame)
            axes[frame_idx].set_title(f'Frame {i+1}', fontsize=12)
            axes[frame_idx].axis('off')
            
            # Plot action number (except for the last frame)
            if i < len(inferred_actions):
                action_idx = frame_idx + 1
                if action_idx < len(axes):
                    axes[action_idx].text(0.5, 0.5, f'Action\n{inferred_actions[i].item()}', 
                                        ha='center', va='center', fontsize=20, fontweight='bold',
                                        transform=axes[action_idx].transAxes)
                    axes[action_idx].set_title(f'Action {i+1}', fontsize=12)
                    axes[action_idx].axis('off')
        
        plt.suptitle('Video Generation with Inferred Actions', fontsize=16, fontweight='bold')
    else:
        # Create figure with just frames (no actions)
        fig, axes = plt.subplots(1, num_frames, figsize=(4 * num_frames, 4))
        
        # Handle single subplot case
        if num_frames == 1:
            axes = [axes]
        
        for i in range(num_frames):
            frame = frames[0, i].permute(1, 2, 0).numpy()  # [H, W, C]
            axes[i].imshow(frame)
            axes[i].set_title(f'Frame {i+1}', fontsize=12)
            axes[i].axis('off')
        
        plt.suptitle('Video Generation (No Actions)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the visualization
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = "inference_results"
    os.makedirs(save_dir, exist_ok=True)
    
    if use_actions:
        save_path = os.path.join(save_dir, f"inference_results_{timestamp}.png")
        mp4_path = os.path.join(save_dir, f"inference_video_{timestamp}.mp4")
    else:
        save_path = os.path.join(save_dir, f"inference_results_no_actions_{timestamp}.png")
        mp4_path = os.path.join(save_dir, f"inference_video_no_actions_{timestamp}.mp4")
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")
    
    # Save MP4 video of just the frames
    save_frames_as_mp4(frames, mp4_path, fps)
    print(f"MP4 video saved to: {mp4_path}")
    
    # Also display some statistics
    print(f"\nInference Statistics:")
    print(f"Total frames generated: {num_frames}")
    if use_actions:
        print(f"Actions used: {[action.item() for action in inferred_actions]}")
        print(f"Action distribution: {torch.bincount(torch.tensor([action.item() for action in inferred_actions]))}")
    else:
        print(f"No actions used.")

def save_frames_as_mp4(frames, output_path, fps=2):
    """
    Save frames as an MP4 video file.
    
    Args:
        frames: Tensor of shape [batch_size, num_frames, C, H, W] with values in [0, 1]
        output_path: Path to save the MP4 file
        fps: Frames per second for the video
    """
    # Get dimensions
    batch_size, num_frames, C, H, W = frames.shape
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    # Convert each frame and write to video
    for i in range(num_frames):
        # Get frame and convert to numpy
        frame = frames[0, i].permute(1, 2, 0).numpy()  # [H, W, C]
        
        # Convert from RGB to BGR (OpenCV format)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Convert from float [0, 1] to uint8 [0, 255]
        frame_uint8 = (frame_bgr * 255).astype(np.uint8)
        
        # Write frame to video
        out.write(frame_uint8)
    
    # Release video writer
    out.release()

def get_model_context_sizes(video_tokenizer, dynamics_model):
    """Get the context window sizes for both models"""
    # Since models don't have explicit max_seq_len, use the default context lengths
    # Both video tokenizer and dynamics model are trained with context_length=4 by default
    video_context_size = 4  # Default from training
    dynamics_context_size = 4  # Default from training
    
    print(f"Video tokenizer context size: {video_context_size}")
    print(f"Dynamics model context size: {dynamics_context_size}")
    
    # Return the minimum of both
    context_window = min(video_context_size, dynamics_context_size)
    print(f"Using context window size: {context_window}")
    
    return context_window

def main(args):
    video_tokenizer, lam, dynamics_model = load_models(args.video_tokenizer_path, args.lam_path, args.dynamics_path, args.device, use_actions=args.use_actions)
    
    # Load data and get first frame
    _, _, _, validation_loader, _ = load_data_and_data_loaders(dataset='SONIC', batch_size=1, num_frames=1)
    frame = sample_first_frame_from_dataloader(validation_loader)
    frame = frame.to(args.device)
    frames = frame  # [1, 1, C, H, W] - add sequence dimension

    # Initialize action tracking
    if args.use_actions:
        n_actions = lam.quantizer.n_e  # Use n_e instead of codebook_size
        inferred_actions = []
    else:
        n_actions = 0
        inferred_actions = []
    
    # Get context window size from models
    context_window = get_model_context_sizes(video_tokenizer, dynamics_model)
    
    # Override with command line argument if provided
    if args.context_window != 4:  # Default value
        context_window = args.context_window
        print(f"Overriding with command line context window: {context_window}")

    for i in range(args.generation_steps):
        # Sample action only if using actions
        if args.use_actions:
            action_index = sample_action_with_diversity(inferred_actions, n_actions)
            inferred_actions.append(action_index)
            action_latent = get_lam_latent_from_action_index(lam, action_index)
        else:
            action_index = None
            action_latent = None

        print(f"frames shape: {frames.shape}")

        # encode all current frames to video tokens
        video_latents = encode_frame_to_tokens(video_tokenizer, frames)  # [1, seq_len, num_patches, latent_dim]

        print(f"video_latents shape: {video_latents.shape}")
        
        # Maintain sliding window: remove oldest frames if sequence is too long
        if video_latents.shape[0] > context_window:
            # Remove oldest frame from both frames and video_latents
            frames = frames[:, -context_window:, :, :, :]  # Keep last context_window frames
            video_latents = video_latents[:, -context_window:, :, :]  # Keep last context_window latents

        # predict next video tokens using all current video latents
        next_video_latents = predict_next_tokens(
            dynamics_model, video_latents, action_latent, 
            temperature=args.temperature, use_actions=args.use_actions
        )  # [1, num_patches, latent_dim]

        # decode next video tokens to frames
        next_video_latents = next_video_latents.unsqueeze(1)  # Add sequence dimension: [1, 1, num_patches, latent_dim]
        next_frames = video_tokenizer.decoder(next_video_latents)  # [1, 1, C, H, W]
        
        # add next frames to frames
        frames = torch.cat([frames, next_frames], dim=1)  # [1, seq_len+1, C, H, W]
        
        if args.use_actions:
            print(f"Step {i+1}: Generated frame with action {action_index.item()}, sequence length: {frames.shape[1]}")
        else:
            print(f"Step {i+1}: Generated frame (no actions), sequence length: {frames.shape[1]}")
    
    visualize_inference(frames, inferred_actions, args.fps, use_actions=args.use_actions)

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with the trained video generation pipeline")
    parser.add_argument("--video_tokenizer_path", type=str, default="/Users/almondgod/Repositories/nano-genie/src/vqvae/results/videotokenizer_thu_jul_10_22_28_46_2025/checkpoints/videotokenizer_checkpoint_thu_jul_10_22_28_46_2025.pth")
    parser.add_argument("--lam_path", type=str, default="/Users/almondgod/Repositories/nano-genie/src/latent_action_model/results/lam_Sat_Jul_12_15_59_55_2025/checkpoints/lam_checkpoint_Sat_Jul_12_15_59_55_2025.pth")
    parser.add_argument("--dynamics_path", type=str, default="/Users/almondgod/Repositories/nano-genie/src/dynamics/results/dynamics_Sun_Jul_13_17_19_55_2025/checkpoints/dynamics_checkpoint_Sun_Jul_13_17_19_55_2025.pth")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--generation_steps", type=int, default=10, help="Number of frames to generate")
    parser.add_argument("--context_window", type=int, default=4, help="Maximum sequence length for context window")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second for the MP4 video")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling (lower = more conservative)")
    parser.add_argument("--use_actions", action="store_true", default=False, help="Whether to use action latents in the dynamics model (default: False)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
