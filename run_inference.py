import torch
from src.dynamics.models.dynamics_model import DynamicsModel
from src.vqvae.models.video_tokenizer import Video_Tokenizer
from src.latent_action_model.models.lam import LAM
import argparse
from src.vqvae.utils import load_data_and_data_loaders
import matplotlib.pyplot as plt
import time
import os
import random
import glob

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
        # if temperature != 1.0:
        #     # Add noise scaled by temperature
        #     noise = torch.randn_like(next_video_latents) * (temperature - 1.0) * 0.1
        #     next_video_latents = next_video_latents + noise
        
        print(f"next_video_latents shape: {next_video_latents.shape}")
        
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
        latent_dim=6,
        dropout=0.1,
        num_bins=4
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
        latent_dim=6,
        num_bins=4,
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
    
# def sample_first_frame_from_dataloader(dataloader):
#     batch = next(iter(dataloader))
#     frame = batch[0]
#     return frame
    
def sample_random_action(n_actions):
    random_action = torch.randint(0, n_actions, (1,))
    return random_action

def get_lam_latent_from_action_index(lam, action_index):
    with torch.no_grad():
        action_latent = lam.quantizer.embedding.weight[action_index] # get the latent action embedding in the codebook at action index
        return action_latent
    
def visualize_inference(predicted_frames, ground_truth_frames, inferred_actions, fps, use_actions=True):
    """
    Visualize the inference results showing ground truth vs predicted frames side by side.
    Also save an MP4 file showing just the predicted frames in order.
    
    Args:
        predicted_frames: Tensor of shape [batch_size, num_frames, C, H, W] - predicted frames
        ground_truth_frames: Tensor of shape [batch_size, num_frames, C, H, W] - ground truth frames
        inferred_actions: List of action indices
        fps: Frames per second for the MP4 video
        use_actions: Whether actions were used in generation
    """
    # Move to CPU and convert to numpy
    predicted_frames = predicted_frames.detach().cpu()
    ground_truth_frames = ground_truth_frames.detach().cpu()
    
    # Denormalize frames from [-1, 1] to [0, 1]
    predicted_frames = (predicted_frames + 1) / 2
    predicted_frames = torch.clamp(predicted_frames, 0, 1)
    ground_truth_frames = (ground_truth_frames + 1) / 2
    ground_truth_frames = torch.clamp(ground_truth_frames, 0, 1)

    # predicted_frames = predicted_frames.unsqueeze(1) # This line was removed as per the edit hint
    
    # Get dimensions
    batch_size, num_frames, C, H, W = predicted_frames.shape

    _, num_gt_frames, _, _, _ = ground_truth_frames.shape
    
    # Create figure with ground truth and predictions side by side
    fig, axes = plt.subplots(2, num_frames, figsize=(4 * num_frames, 8))
    
    # Handle single subplot case
    if num_frames == 1:
        axes = axes.reshape(2, 1)

    print(f"ground_truth_frames shape: {ground_truth_frames.shape}")
    print(f"predicted_frames shape: {predicted_frames.shape}")
    
    # Plot ground truth frames (top row)
    for i in range(num_gt_frames):
        frame = ground_truth_frames[0, i].permute(1, 2, 0).numpy()  # [H, W, C]
        axes[0, i].imshow(frame)
        axes[0, i].set_title(f'Ground Truth {i+1}', fontsize=12, color='green')
        axes[0, i].axis('off')
    
    # Plot predicted frames (bottom row)
    for i in range(num_frames):
        frame = predicted_frames[0, i].permute(1, 2, 0).numpy()  # [H, W, C]
        axes[1, i].imshow(frame)
        title = f'Predicted {i+1}'
        if use_actions and i < len(inferred_actions):
            title += f'\nAction {inferred_actions[i].item()}' if i < len(inferred_actions) else ''
        axes[1, i].set_title(title, fontsize=12, color='red')
        axes[1, i].axis('off')
    
    plt.suptitle('Ground Truth vs Predicted Frames', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the visualization
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = "inference_results"
    os.makedirs(save_dir, exist_ok=True)
    
    if use_actions:
        save_path = os.path.join(save_dir, f"inference_results_gt_vs_pred_{timestamp}.png")
        mp4_path = os.path.join(save_dir, f"inference_video_{timestamp}.mp4")
    else:
        save_path = os.path.join(save_dir, f"inference_results_gt_vs_pred_no_actions_{timestamp}.png")
        mp4_path = os.path.join(save_dir, f"inference_video_no_actions_{timestamp}.mp4")
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")

    all_frames = torch.cat([ground_truth_frames, predicted_frames], dim=1)
    save_frames_as_mp4(all_frames, mp4_path, fps)
    
    # Calculate and display reconstruction error
    mse_error = torch.mean((predicted_frames - ground_truth_frames) ** 2).item()
    print(f"\nInference Statistics:")
    print(f"Total frames generated: {num_frames}")
    print(f"Mean Squared Error (GT vs Pred): {mse_error:.6f}")
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
    import cv2
    import numpy as np

    batch_size, num_frames, C, H, W = frames.shape

    # OpenCV expects (W, H)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for i in range(num_frames):
        frame = frames[0, i].detach().cpu().permute(1, 2, 0).numpy()  # [H, W, C]
        # Ensure float32
        frame = frame.astype(np.float32)
        # Clamp and scale
        frame = np.clip(frame, 0, 1)
        frame = (frame * 255).astype(np.uint8)
        # If grayscale, convert to 3 channels
        if frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"MP4 video saved to: {output_path}")

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
    
    # Load data and get ground truth sequence
    _, _, data_loader, _, _ = load_data_and_data_loaders(dataset='SONIC', batch_size=1, num_frames=4)  # +1 for initial frame
    random_idx = random.randint(0, len(data_loader.dataset) - 1)
    ground_truth_sequence = data_loader.dataset[random_idx][0]  # Get the full sequence
    ground_truth_sequence = ground_truth_sequence.unsqueeze(0).to(args.device)  # [1, seq_len, C, H, W]
    
    print(f"Loaded ground truth sequence with shape: {ground_truth_sequence.shape}")
    
    # Start with the first frame from ground truth
    context_frames = ground_truth_sequence
    generated_frames = context_frames  # List to store generated frames (excluding initial)

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

    # Only use last context_window frames for model input
    # if context_frames.shape[1] > context_window:
    #     model_input_frames = context_frames[:, -context_window:, :, :, :]
    # else:
    print(f"context_window: {context_window}")
    print(f"context_frames shape: {context_frames.shape}")
    model_input_frames = context_frames

    video_latent_sequence = encode_frame_to_tokens(video_tokenizer, model_input_frames)  # [1, seq_len, num_patches, latent_dim]
    print(f"video_latent_sequence shape: {video_latent_sequence.shape}")
    for i in range(args.generation_steps):
        # Sample action only if using actions
        if args.use_actions:
            action_index = sample_action_with_diversity(inferred_actions, n_actions)
            inferred_actions.append(action_index)
            action_latent = get_lam_latent_from_action_index(lam, action_index)
        else:
            action_index = None
            action_latent = None

        print(f"context_frames shape: {context_frames.shape}")

        # Only use last context_window frames for model input
        video_latents = video_latent_sequence  # [1, seq_len, num_patches, latent_dim]
        print(f"video_latents shape: {video_latents.shape}")

        # predict next video tokens using all current video latents
        next_video_logits = predict_next_tokens(
            dynamics_model, video_latents, action_latent, 
            temperature=args.temperature, use_actions=args.use_actions
        )  # [1, seq_len, num_patches, codebook_size]

        print(f"next_video_latents shape: {next_video_logits.shape}")

        # get indices o
        next_video_indices = torch.argmax(next_video_logits, dim=-1) # [1, seq_len, num_patches]

        next_video_latents = video_tokenizer.vq.get_latents_from_indices(next_video_indices, dim=-1)
        print(f"next_video_latents shape: {next_video_latents.shape}")
        # decode next video tokens to frames
        # next_video_latents = next_video_latents.unsqueeze(1)  # Add sequence dimension: [1, 1, num_patches, latent_dim]
        next_frames = video_tokenizer.decoder(next_video_latents)  # [1, seq_len, C, H, W]

        print(f"next_frames shape: {next_frames.shape}")
        print(f"ground_truth_sequence shape: {ground_truth_sequence.shape}")
        # visualize next frames and ground truth frames side by side
        visualize_decoded_frames(next_frames, generated_frames[:, -context_window:, :, :], step=i) # Pass ground_truth_sequence[:, i+1:i+2, :, :, :]
        
        generated_frames = torch.cat([generated_frames, next_frames[:, -1:, :, :]], dim=1)  # Store for visualization/MSE
        video_latent_sequence = next_video_latents
        
        if args.use_actions:
            print(f"Step {i+1}: Generated frame with action {action_index.item()}, sequence length: {context_frames.shape[1]}")
        else:
            print(f"Step {i+1}: Generated frame (no actions), sequence length: {context_frames.shape[1]}")
    
    # Stack all generated frames (excluding initial frame)
    predicted_frames = generated_frames[:, -args.generation_steps:, :, :, :]
    # predicted_frames shape: [1, generation_steps, C, H, W]
    ground_truth_frames = ground_truth_sequence  # [1, generation_steps, C, H, W]
    
    print(f"Ground truth frames shape: {ground_truth_frames.shape}")
    print(f"Predicted frames shape: {predicted_frames.shape}")
    
    visualize_inference(predicted_frames, ground_truth_frames, inferred_actions, args.fps, use_actions=args.use_actions)

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with the trained video generation pipeline")
    parser.add_argument("--video_tokenizer_path", type=str, default="/Users/almondgod/Repositories/nano-genie/src/vqvae/results/videotokenizer_sun_jul_20_21_50_32_2025/checkpoints/videotokenizer_checkpoint_sun_jul_20_21_50_32_2025.pth")
    parser.add_argument("--lam_path", type=str, default="/Users/almondgod/Repositories/nano-genie/src/latent_action_model/results/lam_Sat_Jul_12_15_59_55_2025/checkpoints/lam_checkpoint_Sat_Jul_12_15_59_55_2025.pth")
    parser.add_argument("--dynamics_path", type=str, default="/Users/almondgod/Repositories/nano-genie/src/dynamics/results/dynamics_Tue_Jul_22_20_08_24_2025/checkpoints/dynamics_checkpoint_Tue_Jul_22_20_08_24_2025.pth")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--generation_steps", type=int, default=4, help="Number of frames to generate")
    parser.add_argument("--context_window", type=int, default=4, help="Maximum sequence length for context window")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second for the MP4 video")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling (lower = more conservative)")
    parser.add_argument("--use_actions", action="store_true", default=False, help="Whether to use action latents in the dynamics model (default: False)")
    parser.add_argument("--use_latest_checkpoints", action="store_true", default=False, help="If set, automatically find and use the latest video tokenizer and dynamics checkpoints.")
    return parser.parse_args()

def visualize_decoded_frames(predicted_frames, ground_truth_frames, step=0):
    """
    Visualize predicted and ground truth sequences side by side.
    Args:
        predicted_frames: [1, seq_len, 3, H, W]
        ground_truth_frames: [1, seq_len, 3, H, W]
    """
    import matplotlib.pyplot as plt
    import os
    import time
    # Move to CPU and clamp
    predicted_frames = predicted_frames.detach().cpu()
    ground_truth_frames = ground_truth_frames.detach().cpu()
    if predicted_frames.min() < 0:
        predicted_frames = (predicted_frames + 1) / 2
    if ground_truth_frames.min() < 0:
        ground_truth_frames = (ground_truth_frames + 1) / 2
    predicted_frames = torch.clamp(predicted_frames, 0, 1)
    ground_truth_frames = torch.clamp(ground_truth_frames, 0, 1)
    # Get sequence length
    seq_len = predicted_frames.shape[1]
    fig, axes = plt.subplots(2, seq_len, figsize=(3*seq_len, 6))
    if seq_len == 1:
        axes = axes.reshape(2, 1)
    for i in range(seq_len):
        gt_img = ground_truth_frames[0, i].permute(1, 2, 0).numpy()
        pred_img = predicted_frames[0, i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(gt_img)
        axes[0, i].set_title(f"GT {i+1}")
        axes[0, i].axis('off')
        axes[1, i].imshow(pred_img)
        axes[1, i].set_title(f"Pred {i+1}")
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel("Ground Truth", fontsize=14)
    axes[1, 0].set_ylabel("Predicted", fontsize=14)
    plt.suptitle("Decoded Sequence: Ground Truth vs Predicted", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_dir = "inference_results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"decoded_comparison_{time.strftime('%Y%m%d_%H%M%S')}_step_{step}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Decoded sequence visualization saved to: {save_path}")

def find_latest_checkpoint(base_dir, model_name):
    pattern = os.path.join(base_dir, f"src/*/results/{model_name}_*/checkpoints/{model_name}_checkpoint_*.pth")
    checkpoint_files = glob.glob(pattern)
    if not checkpoint_files:
        print(f"No checkpoint files found for {model_name}")
        return None
    return max(checkpoint_files, key=os.path.getctime)

if __name__ == "__main__":
    args = parse_args()
    if args.use_latest_checkpoints:
        base_dir = os.path.abspath(os.path.dirname(__file__))
        vt_ckpt = find_latest_checkpoint(base_dir, "videotokenizer")
        dyn_ckpt = find_latest_checkpoint(base_dir, "dynamics")
        if vt_ckpt:
            print(f"[INFO] Using latest video tokenizer checkpoint: {vt_ckpt}")
            args.video_tokenizer_path = vt_ckpt
        else:
            print("[WARN] No video tokenizer checkpoint found, using default.")
        if dyn_ckpt:
            print(f"[INFO] Using latest dynamics checkpoint: {dyn_ckpt}")
            args.dynamics_path = dyn_ckpt
        else:
            print("[WARN] No dynamics checkpoint found, using default.")
    main(args)
