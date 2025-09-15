# python run_inference.py --config configs/inference.yaml

import torch
from datasets.data_utils import load_data_and_data_loaders
import matplotlib.pyplot as plt
import time
import os
import random
import glob
import re
from utils.utils import load_videotokenizer_from_checkpoint, load_latent_actions_from_checkpoint, load_dynamics_from_checkpoint
from utils.config import InferenceConfig, load_config
from einops import repeat


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_action_with_diversity(previous_actions, n_actions, diversity_weight=0.1, device=None):
    """Sample action with diversity to avoid getting stuck in loops"""
    if len(previous_actions) < 2:
        # Just sample randomly for first few actions
        return torch.randint(0, n_actions, (1,), device=device)
    
    # Get recent action distribution
    recent_actions = previous_actions[-5:]  # Last 5 actions
    action_counts = torch.bincount(torch.tensor([a.item() for a in recent_actions], device=device), minlength=n_actions)
    
    # Sample with diversity
    if torch.rand(1, device=device).item() < diversity_weight:
        # Sample from less frequent actions
        min_count = action_counts.min()
        candidate_actions = torch.where(action_counts == min_count)[0]
        sampled_action_index = candidate_actions[torch.randint(0, len(candidate_actions), (1,), device=device)]
    else:
        # Sample randomly
        sampled_action_index = torch.randint(0, n_actions, (1,), device=device)
    
    return sampled_action_index


def load_models(video_tokenizer_path, latent_actions_path, dynamics_path, device, use_actions=True):
    # Load tokenizer and dynamics, and Latent Actions if using actions
    video_tokenizer, _vt_ckpt = load_videotokenizer_from_checkpoint(video_tokenizer_path, device)
    video_tokenizer.eval()
    latent_action_model = None
    if use_actions:
        latent_action_model, _latent_action_ckpt = load_latent_actions_from_checkpoint(latent_actions_path, device)
        latent_action_model.eval()
    dynamics_model, _dyn_ckpt = load_dynamics_from_checkpoint(dynamics_path, device)
    dynamics_model.eval()
    return video_tokenizer, latent_action_model, dynamics_model


def sample_random_action(n_actions):
    random_action = torch.randint(0, n_actions, (1,))
    return random_action


def visualize_inference(predicted_frames, ground_truth_frames, inferred_actions, fps, use_actions=True):
    # Move to CPU and convert to numpy
    predicted_frames = predicted_frames.detach().cpu()
    ground_truth_frames = ground_truth_frames.detach().cpu()
    
    # Denormalize frames from [-1, 1] to [0, 1]
    predicted_frames = (predicted_frames + 1) / 2
    predicted_frames = torch.clamp(predicted_frames, 0, 1)
    ground_truth_frames = (ground_truth_frames + 1) / 2
    ground_truth_frames = torch.clamp(ground_truth_frames, 0, 1)

    # Get dimensions
    B, T, C, H, W = predicted_frames.shape

    _, num_gt_frames, _, _, _ = ground_truth_frames.shape
    
    # Create figure with ground truth and predictions side by side
    fig, axes = plt.subplots(2, T, figsize=(4 * T, 8))
    
    # Handle single subplot case
    if T == 1:
        axes = axes.reshape(2, 1)

    # Plot ground truth frames (top row)
    for i in range(num_gt_frames):
        frame = ground_truth_frames[0, i].permute(1, 2, 0).numpy()  # [H, W, C]
        axes[0, i].imshow(frame)
        axes[0, i].set_title(f'Ground Truth {i+1}', fontsize=12, color='green')
        axes[0, i].axis('off')

    # Plot predicted frames (bottom row)
    for i in range(T):
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
    print(f"Total frames generated: {T}")
    print(f"Mean Squared Error (GT vs Pred): {mse_error:.6f}")
    if use_actions:
        print(f"Actions used: {[action.item() for action in inferred_actions]}")
        print(f"Action distribution: {torch.bincount(torch.tensor([action.item() for action in inferred_actions]))}")
    else:
        print(f"No actions used.")


# TODO: get working mp4
def save_frames_as_mp4(frames, output_path, fps=2):
    import cv2
    import numpy as np

    B, T, C, H, W = frames.shape

    # OpenCV expects (W, H)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for i in range(T):
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


# TODO: name mskgit schedule
def exp_schedule_torch(t, T, P, k=5.0, device=None):
    x = t / T
    k_tensor = torch.tensor(k, device=device)
    result = P * torch.expm1(k_tensor * x) / torch.expm1(k_tensor)
    # Ensure we reach P by the final step
    if t == T - 1:
        return torch.tensor(P, dtype=result.dtype, device=device)
    return result


def main():
    args: InferenceConfig = load_config(InferenceConfig, default_config_path=os.path.join(os.getcwd(), 'configs', 'inference.yaml'))

    # Auto-resolve latest checkpoints if requested or any path missing
    def missing(path: str | None) -> bool:
        return (path is None) or (not os.path.isfile(path))

    if args.use_latest_checkpoints or missing(args.video_tokenizer_path) or (args.use_actions and missing(args.latent_actions_path)) or missing(args.dynamics_path):
        base_dir = os.getcwd()
        try:
            vt_ckpt = find_latest_checkpoint(base_dir, "video_tokenizer")
            args.video_tokenizer_path = vt_ckpt
        except Exception as e:
            pass
        if args.use_actions:
            try:
                lam_ckpt = find_latest_checkpoint(base_dir, "latent_actions")
                args.latent_actions_path = lam_ckpt
            except Exception as e:
                pass
        try:
            dyn_ckpt = find_latest_checkpoint(base_dir, "dynamics")
            args.dynamics_path = dyn_ckpt
        except Exception as e:
            pass

    # Validate required paths
    if missing(args.video_tokenizer_path):
        raise FileNotFoundError("video_tokenizer_path is not set or not a file. Set it in configs/inference.yaml or enable use_latest_checkpoints with available runs.")
    if args.use_actions and missing(args.latent_actions_path):
        raise FileNotFoundError("latent_actions_path is not set or not a file while use_actions is True. Set it in configs/inference.yaml or enable use_latest_checkpoints.")
    if missing(args.dynamics_path):
        raise FileNotFoundError("dynamics_path is not set or not a file. Set it in configs/inference.yaml or enable use_latest_checkpoints.")
 
    # Move models to the specified device
    video_tokenizer, latent_action_model, dynamics_model = load_models(args.video_tokenizer_path, args.latent_actions_path, args.dynamics_path, args.device, use_actions=args.use_actions)

    # Determine how many ground-truth frames we need: context + generation steps + prediction horizon
    frames_to_load = args.context_window + args.generation_steps * args.prediction_horizon

    # Load data and get ground truth sequence
    _, _, data_loader, _, _ = load_data_and_data_loaders(
        dataset=args.dataset, batch_size=1, num_frames=frames_to_load)

    random_idx = random.randint(0, len(data_loader.dataset) - 1)
    og_ground_truth_frames = data_loader.dataset[random_idx][0]  # full sequence
    og_ground_truth_frames = og_ground_truth_frames.unsqueeze(0).to(args.device)  # [1, frames_to_load, C, H, W]

    ground_truth_frames = og_ground_truth_frames[:, :frames_to_load, :, :, :]  # [1, frames_to_load, C, H, W]

    # Start with initial context (first context_window GT frames)
    context_frames = ground_truth_frames[:, :args.context_window, :, :, :]
    generated_frames = context_frames.clone()

    # Initialize action tracking
    if args.use_actions:
        n_actions = latent_action_model.quantizer.codebook_size
        print(f"n_actions: {n_actions}")
        inferred_actions = []
    else:
        n_actions = 0
        inferred_actions = []

    # Ensure we don’t exceed available GT frames in teacher-forced mode
    max_possible_steps = ground_truth_frames.shape[1] - args.context_window
    if args.teacher_forced and args.generation_steps > max_possible_steps:
        print(f"[WARN] Requested {args.generation_steps} generation steps but only {max_possible_steps} are possible with teacher-forced context. Clamping.")
    effective_steps = args.generation_steps if not args.teacher_forced else min(args.generation_steps, max_possible_steps)

    for i in range(effective_steps):
        # Select context depending on teacher-forced flag
        if args.teacher_forced:
            context_start = i  # shift window along ground truth
            context_frames = ground_truth_frames[:, context_start:context_start+args.context_window, :, :, :]
        else:
            # Autoregressive: last context_window frames from generated sequence
            context_frames = generated_frames[:, -args.context_window:, :, :, :]

        # Encode context frames each iteration
        video_indices = video_tokenizer.tokenize(context_frames)
        video_latents = video_tokenizer.quantizer.get_latents_from_indices(video_indices)

        # Sample action only if using actions
        if args.use_gt_actions:
            # pass last 2 frames through latent_action_model to get action latent
            gt_action_latents = latent_action_model.encode(context_frames) # [1, T - 1, A]
            sampled_action_index = sample_action_with_diversity(inferred_actions, n_actions, device=args.device) # [1]
            inferred_actions.append(sampled_action_index) # [i]
            sampled_action_index_tensor = repeat(torch.tensor(sampled_action_index, device=args.device), 'i -> 1 i') # [1, i]
            sampled_action_latent = latent_action_model.quantizer.get_latents_from_indices(sampled_action_index_tensor) # [1, i, A]
            action_latent = torch.cat([gt_action_latents, sampled_action_latent], dim=1) # [1, T, A]
        elif args.use_actions:
            sampled_action_index = sample_action_with_diversity(inferred_actions, n_actions, device=args.device) # [1]
            inferred_actions.append(sampled_action_index) # [i]
            recent_inferred_actions = inferred_actions[-args.context_window:] if len(inferred_actions) > args.context_window else inferred_actions # [i or T_ctx]
            recent_inferred_actions_tensor = repeat(torch.tensor(recent_inferred_actions, device=args.device), 'i -> 1 i') # [1, i or T_ctx]
            print(f"recent_inferred_actions_tensor shape: {recent_inferred_actions_tensor.shape}")
            action_latent = latent_action_model.quantizer.get_latents_from_indices(recent_inferred_actions_tensor) # [1, i or T_ctx, A]
            print(f"action_latent shape: {action_latent.shape}")
            if len(recent_inferred_actions) < args.context_window:
                # if we dont have enough inferred actions (in the beginning) add enough gt to fill the sequence
                gt_pad_actions = latent_action_model.encode(context_frames[:, :args.context_window - len(recent_inferred_actions) + 1])  # [1, context_window - len(inferred_actions), A]
                quantized_gt_pad_actions = latent_action_model.quantizer(gt_pad_actions) # [1, context_window - len(inferred_actions), A]
                action_latent = torch.cat([quantized_gt_pad_actions, action_latent], dim=1) # [1, S, A]
                print(f"action_latent shape: {action_latent.shape}")
        else:
            sampled_action_index = None
            action_latent = None

        # Use model's iterative inference helper
        def idx_to_latents(idx):
            return video_tokenizer.quantizer.get_latents_from_indices(idx, dim=-1)

        next_video_latents = dynamics_model.forward_inference(
            context_latents=video_latents,
            prediction_horizon=args.prediction_horizon,
            num_steps=10,
            index_to_latents_fn=idx_to_latents,
            conditioning=action_latent
        )

        # decode next video tokens to frames
        next_frames = video_tokenizer.detokenize(next_video_latents)  # [1, T, C, H, W]

        generated_frames = torch.cat([generated_frames, next_frames[:, -args.prediction_horizon:, :, :]], dim=1)

        if args.use_actions:
            print(f"Step {i+1}: Generated frame with action {sampled_action_index.item()}, sequence length: {context_frames.shape[1]}")

    pred_len = min(effective_steps, generated_frames.shape[1] - args.context_window)

    visualize_inference(generated_frames, ground_truth_frames, inferred_actions, args.fps, use_actions=args.use_actions)


# TODO: use utils fun
def visualize_decoded_frames(predicted_frames, ground_truth_frames, step=0):
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
    pattern = os.path.join(base_dir, f"results/{model_name}_*/checkpoints/{model_name}_step_*.pth")
    checkpoint_files = glob.glob(pattern)
    if not checkpoint_files:
        print(f"No checkpoint files found for {model_name}")
        return None
    return max(checkpoint_files, key=os.path.getctime)


if __name__ == "__main__":
    main()
