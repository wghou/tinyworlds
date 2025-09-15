# python run_inference.py --use_latest_checkpoints --use_actions

import torch
import argparse
from datasets.data_utils import load_data_and_data_loaders
import matplotlib.pyplot as plt
import time
import os
import random
import glob
import re
from utils.utils import load_videotokenizer_from_checkpoint, load_lam_from_checkpoint, load_dynamics_from_checkpoint

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

# TODO: use utils fun
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

def main(args):
    # Move models to the specified device
    video_tokenizer, latent_action_model, dynamics_model = load_models(args.video_tokenizer_path, args.latent_actions_path, args.dynamics_path, args.device, use_actions=args.use_actions)

    # Determine how many ground-truth frames we need: context + generation steps + prediction horizon
    frames_to_load = args.context_window + args.generation_steps * args.prediction_horizon

    # Load data and get ground truth sequence
    _, _, data_loader, _, _ = load_data_and_data_loaders(
        dataset=args.dataset, batch_size=1, num_frames=frames_to_load)

    random_idx = random.randint(0, len(data_loader.dataset) - 1)
    og_ground_truth_sequence = data_loader.dataset[random_idx][0]  # full sequence
    og_ground_truth_sequence = og_ground_truth_sequence.unsqueeze(0).to(args.device)  # [1, frames_to_load, C, H, W]

    ground_truth_sequence = og_ground_truth_sequence[:, :frames_to_load, :, :, :]  # [1, frames_to_load, C, H, W]

    # Start with initial context (first context_window GT frames)
    context_frames = ground_truth_sequence[:, :args.context_window, :, :, :]
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
    max_possible_steps = ground_truth_sequence.shape[1] - args.context_window
    if args.teacher_forced and args.generation_steps > max_possible_steps:
        print(f"[WARN] Requested {args.generation_steps} generation steps but only {max_possible_steps} are possible with teacher-forced context. Clamping.")
    effective_steps = args.generation_steps if not args.teacher_forced else min(args.generation_steps, max_possible_steps)

    for i in range(effective_steps):
        # Select context depending on teacher-forced flag
        if args.teacher_forced:
            context_start = i  # shift window along ground truth
            context_frames = ground_truth_sequence[:, context_start:context_start+args.context_window, :, :, :]
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
            sampled_action_index = sample_action_with_diversity(inferred_actions, n_actions, device=args.device) # [1, 1, A]
            inferred_actions.append(sampled_action_index)
            sampled_action_latent = latent_action_model.quantizer.get_latents_from_indices(sampled_action_index).unsqueeze(1) # [1, 1, A]
            action_latent = torch.cat([gt_action_latents, sampled_action_latent], dim=1) # [1, T, A]
        elif args.use_actions:
            sampled_action_index = sample_action_with_diversity(inferred_actions, n_actions, device=args.device)
            inferred_actions.append(sampled_action_index)
            recent_inferred_actions = inferred_actions[-args.context_window:] if len(inferred_actions) > args.context_window else inferred_actions
            action_latent = latent_action_model.quantizer.get_latents_from_indices(torch.tensor(recent_inferred_actions, device=args.device)).unsqueeze(0) # [1, seq_len, A]
            if len(recent_inferred_actions) < args.context_window:
                # if we dont have enough inferred actions (in the beginning) add enough gt to fill the sequence
                gt_pad_actions = latent_action_model.encode(context_frames[:, :args.context_window - len(recent_inferred_actions) + 1])  # [1, context_window - len(inferred_actions), A]
                quantized_gt_pad_actions = latent_action_model.quantizer(gt_pad_actions) # [1, context_window - len(inferred_actions), A]
                action_latent = torch.cat([quantized_gt_pad_actions, action_latent], dim=1) # [1, S, A]
        else:
            sampled_action_index = None
            action_latent = None

        # MaskGit-style iterative decoding
        # for timestep m in range M: 
        # 1. run inference with all tokens masked
        # 2. get argmax tokens and their corresponding probabilities
        # 3. choose top n tokens with highest probabilities and unmask them

        # scheduling: n = e^(t/M) * P, where M is num steps and P is num patches
        M = 10  # Number of decoding steps
        P = video_latents.shape[2]  # Number of patches

        # Start with all tokens masked
        current_latents = video_latents.clone() # [1, T, P, L]

        # append mask frame
        mask_token_value = dynamics_model.mask_token.data  # Extract the actual tensor value
        # Expand mask token to match the required shape: [1, 1, P, L]
        mask_latents = mask_token_value.expand(1, args.prediction_horizon, P, -1).to(args.device)  # [1, T, P, L]

        input_latents = torch.cat([current_latents, mask_latents], dim=1) # [1, T, P, L]

        mask = torch.full((1, args.prediction_horizon, P, 1), True, dtype=torch.bool, device=args.device)  # [1, T, P, 1]

        n_tokens = 0

        for m in range(M):
            prev_n_tokens = n_tokens
            n_tokens_raw = exp_schedule_torch(m, M, P, device=args.device)
            n_tokens = int(n_tokens_raw)
            tokens_to_unmask = n_tokens - prev_n_tokens
            print(f"DEBUG: m={m}, n_tokens_raw={n_tokens_raw:.2f}, n_tokens={n_tokens}, tokens_to_unmask={tokens_to_unmask}")

            # Run dynamics model to get predictions
            with torch.no_grad():
                if args.use_actions:
                    predicted_logits, _ = dynamics_model(input_latents, training=False, conditioning=action_latent)  # [B, T, P, L^D]
                else:
                    predicted_logits, _ = dynamics_model(input_latents, training=False)  # [B, T, P, L^D]

            # Get probabilities and top predictions
            probs = torch.softmax(predicted_logits, dim=-1)  # [B, T, P, L^D]

            max_probs, predicted_indices = torch.max(probs, dim=-1)  # [B, T, P]

            # among only the currently masked latents (check the mask tensor), 
            # 1. get the top n_tokens with highest probabilities
            # 2. unmask them in the mask tensor
            # 3. add them to the "input_latents" tensor instead of the mask embeddings in those positions        
            masked_probs = max_probs[:, -1, :]  # [B, P] - only last timestep
            masked_mask = mask[:, -1, :, 0]  # [B, P] - only last timestep

            # Ensure we unmask at least 1 token if there are still masked tokens
            if masked_mask.sum() > 0 and tokens_to_unmask == 0:
                tokens_to_unmask = 1

            if masked_mask.sum() > 0:
                # Get indices of masked positions
                masked_indices = torch.where(masked_mask[0])[0]  # [num_masked]
                masked_pos_probs = masked_probs[0, masked_indices]  # [num_masked]

                # Select top tokens_to_unmask tokens with highest probabilities
                if len(masked_pos_probs) > tokens_to_unmask:
                    top_indices = torch.topk(masked_pos_probs, tokens_to_unmask, largest=True).indices
                    tokens_to_unmask_indices = masked_indices[top_indices]
                else:
                    # Unmask all remaining tokens
                    tokens_to_unmask_indices = masked_indices

                # 1. Unmask selected positions in the mask tensor
                mask[0, -1, tokens_to_unmask_indices, 0] = False

                for idx in tokens_to_unmask_indices:
                    predicted_latent = video_tokenizer.quantizer.get_latents_from_indices(
                        predicted_indices[0:1, -1:, idx:idx+1], dim=-1
                    ).to(args.device)  # [1, 1, 1, L]

                    # Update input_latents at the last timestep, this position
                    input_latents[0, -1, idx] = predicted_latent[0, 0, 0]

                print(f"Step {m+1}/{M}: Unmasked {len(tokens_to_unmask_indices)} tokens (target: {tokens_to_unmask}, remaining: {masked_mask.sum().item()})")
            else:
                print(f"Step {m+1}/{M}: No masked tokens remaining")

        # Final result: input_latents contains the decoded sequence (last timestep)
        next_video_latents = input_latents

        # decode next video tokens to frames
        next_frames = video_tokenizer.detokenize(next_video_latents)  # [1, T, C, H, W]

        generated_frames = torch.cat([generated_frames, next_frames[:, -args.prediction_horizon:, :, :]], dim=1)

        if args.use_actions:
            print(f"Step {i+1}: Generated frame with action {sampled_action_index.item()}, sequence length: {context_frames.shape[1]}")

    # Determine how many frames were actually generated
    pred_len = min(effective_steps, generated_frames.shape[1] - args.context_window)

    predicted_frames = generated_frames
    ground_truth_frames = ground_truth_sequence

    visualize_inference(predicted_frames, ground_truth_frames, inferred_actions, args.fps, use_actions=args.use_actions)

# TODO: replace with yaml
def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with the trained video generation pipeline")
    parser.add_argument("--video_tokenizer_path", type=str, default="/workspace/nano-genie/video_tokenizer/results/vqvae_Sat_Sep_13_11_30_58_2025/checkpoints/videotokenizer_step_47500_Sat_Sep_13_15_24_15_2025.pth")
    parser.add_argument("--latent_actions_path", type=str, default="/workspace/nano-genie/latent_actions/results/latent_actions_Sat_Sep_13_15_36_38_2025/checkpoints/latent_actions_step_1900_Sat_Sep_13_15_48_55_2025.pth")
    parser.add_argument("--dynamics_path", type=str, default="/workspace/nano-genie/dynamics/results/dynamics_Sat_Sep_13_15_49_33_2025/checkpoints/dynamics_step_42500_Sun_Sep_14_02_43_30_2025.pth")
    parser.add_argument("--device", type=str, default=str(device), help="Device to use (cuda/cpu)")
    parser.add_argument("--generation_steps", type=int, default=4, help="Number of frames to generate")
    parser.add_argument("--context_window", type=int, default=3, help="Maximum sequence length for context window")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second for the MP4 video")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for sampling (lower = more conservative)")
    parser.add_argument("--use_actions", action="store_true", default=False, help="Whether to use action latents in the dynamics model (default: False)")
    parser.add_argument("--teacher_forced", action="store_true", default=False,
                        help="Run teacher-forced inference (always use ground-truth context).")
    parser.add_argument("--use_latest_checkpoints", action="store_true", default=False, help="If set, automatically find and use the latest video tokenizer, Latent Actions, and dynamics checkpoints.")
    parser.add_argument("--prediction_horizon", type=int, default=1, help="Number of frames to predict")
    parser.add_argument("--dataset", type=str, default="ZELDA", help="Dataset to use")
    parser.add_argument("--use_gt_actions", action="store_true", default=False, help="Whether to use ground-truth (latent_action_model inferred) action latents")
    return parser.parse_args()

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
    args = parse_args()

    # Ensure device is set correctly
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    if args.use_latest_checkpoints:
        base_dir = os.path.abspath(os.path.dirname(__file__))
        vt_ckpt = find_latest_checkpoint(base_dir, "video_tokenizer")
        lam_ckpt = find_latest_checkpoint(base_dir, "latent_actions")
        dyn_ckpt = find_latest_checkpoint(base_dir, "dynamics")

        if vt_ckpt:
            print(f"[INFO] Using latest video tokenizer checkpoint: {vt_ckpt}")
            args.video_tokenizer_path = vt_ckpt
        else:
            print("[WARN] No video tokenizer checkpoint found, using default.")

        if lam_ckpt:
            print(f"[INFO] Using latest Latent Actions checkpoint: {latent_actions_ckpt}")
            args.latent_actions_path = latent_actions_ckpt
        else:
            print("[WARN] No Latent Actions checkpoint found, using default.")

        if dyn_ckpt:
            print(f"[INFO] Using latest dynamics checkpoint: {dyn_ckpt}")
            args.dynamics_path = dyn_ckpt
        else:
            print("[WARN] No dynamics checkpoint found, using default.")
    main(args)
 