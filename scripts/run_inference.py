# python run_inference.py --config configs/inference.yaml

import torch
from datasets.data_utils import load_data_and_data_loaders
import matplotlib.pyplot as plt
import time
import os
import random
import glob
import re
from utils.utils import load_videotokenizer_from_checkpoint, load_latent_actions_from_checkpoint, load_dynamics_from_checkpoint, find_latest_checkpoint
from utils.config import InferenceConfig, load_config
from utils.inference_utils import load_models, visualize_inference, sample_random_action, get_action_latent
from einops import repeat


def main():
    args: InferenceConfig = load_config(InferenceConfig, default_config_path=os.path.join(os.getcwd(), 'configs', 'inference.yaml'))

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    use_latent_actions = (args.use_actions or args.use_gt_actions or args.use_interactive_mode)

    # Auto-resolve latest checkpoints if requested or any path missing
    def missing(path: str | None) -> bool:
        return (path is None) or (not os.path.isfile(path))

    base_dir = os.getcwd()
    if args.use_latest_checkpoints or missing(args.video_tokenizer_path):
        vt_ckpt = find_latest_checkpoint(base_dir, "video_tokenizer")
        args.video_tokenizer_path = vt_ckpt

    if (args.use_latest_checkpoints or missing(args.latent_actions_path)) and use_latent_actions:
        lam_ckpt = find_latest_checkpoint(base_dir, "latent_actions")
        args.latent_actions_path = lam_ckpt

    if args.use_latest_checkpoints or missing(args.dynamics_path):
        dyn_ckpt = find_latest_checkpoint(base_dir, "dynamics")
        args.dynamics_path = dyn_ckpt
    
    print(f"Using video_tokenizer checkpoint: {args.video_tokenizer_path}")
    if use_latent_actions:
        print(f"Using latent_actions checkpoint: {args.latent_actions_path}")
    print(f"Using dynamics checkpoint: {args.dynamics_path}")

    # Validate required paths
    if missing(args.video_tokenizer_path):
        raise FileNotFoundError("video_tokenizer_path is not set or not a file. Set it in configs/inference.yaml or enable use_latest_checkpoints with available runs.")
    if use_latent_actions and missing(args.latent_actions_path):
        raise FileNotFoundError("latent_actions_path is not set or not a file while actions are requested. Set it in configs/inference.yaml or enable use_latest_checkpoints.")
    if missing(args.dynamics_path):
        raise FileNotFoundError("dynamics_path is not set or not a file. Set it in configs/inference.yaml or enable use_latest_checkpoints.")
 
    # Move models to the specified device
    video_tokenizer, latent_action_model, dynamics_model = load_models(args.video_tokenizer_path, args.latent_actions_path, args.dynamics_path, args.device, use_actions=use_latent_actions)

    # Optionally compile the dynamics model (most compute heavy)
    if args.compile:
        video_tokenizer = torch.compile(video_tokenizer, mode="reduce-overhead", fullgraph=False, dynamic=True)
        if use_latent_actions:
            latent_action_model = torch.compile(latent_action_model, mode="reduce-overhead", fullgraph=False, dynamic=True)
        dynamics_model = torch.compile(dynamics_model, mode="reduce-overhead", fullgraph=False, dynamic=True)
        print("Compiled all models for inference.")

    # Determine how many ground-truth frames we need: context + generation steps + prediction horizon
    frames_to_load = args.context_window + args.generation_steps * args.prediction_horizon

    if hasattr(args, 'preload_ratio') and args.preload_ratio is not None:
        data_overrides = {'preload_ratio': args.preload_ratio}
    else:
        data_overrides = {}

    # Load data and get ground truth sequence
    _, _, data_loader, _, _ = load_data_and_data_loaders(
        dataset=args.dataset, batch_size=1, num_frames=frames_to_load, **data_overrides)

    random_idx = random.randint(0, len(data_loader.dataset) - 1)
    og_ground_truth_frames = data_loader.dataset[random_idx][0]  # full sequence
    og_ground_truth_frames = og_ground_truth_frames.unsqueeze(0).to(args.device)  # [1, frames_to_load, C, H, W]

    ground_truth_frames = og_ground_truth_frames[:, :frames_to_load, :, :, :]  # [1, frames_to_load, C, H, W]

    # Start with initial context (first context_window GT frames)
    context_frames = ground_truth_frames[:, :args.context_window, :, :, :]
    generated_frames = context_frames.clone()

    # # DEBUG: visualize what a pure mask-latent decode looks like
    # with torch.no_grad():
    #     video_indices_dbg = video_tokenizer.tokenize(context_frames[:, -1:])
    #     video_latents_dbg = video_tokenizer.quantizer.get_latents_from_indices(video_indices_dbg)
    #     B_dbg, T_dbg, P_dbg, L_dbg = video_latents_dbg.shape
    #     mask_latents_dbg = dynamics_model.mask_token.to(video_latents_dbg.device, video_latents_dbg.dtype).expand(B_dbg, 1, P_dbg, -1)
    #     mask_frames = video_tokenizer.detokenize(mask_latents_dbg)  # [B,1,C,H,W]
    #     img = (mask_frames[0, 0].cpu() + 1) / 2
    #     img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
    #     os.makedirs('inference_results', exist_ok=True)
    #     plt.figure(figsize=(4, 4))
    #     plt.imshow(img)
    #     plt.axis('off')
    #     save_path = os.path.join('inference_results', 'mask_token_decode.png')
    #     plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #     plt.close()
    #     print(f"Saved mask-only decode to: {save_path}")
    # return

    # Initialize action tracking
    n_actions = None
    inferred_actions = []
    if use_latent_actions:
        n_actions = latent_action_model.quantizer.codebook_size        

    # Ensure we don’t exceed available GT frames in teacher-forced mode
    max_possible_steps = ground_truth_frames.shape[1] - args.context_window
    if args.teacher_forced and args.generation_steps > max_possible_steps:
        print(f"[WARN] Requested {args.generation_steps} generation steps but only {max_possible_steps} are possible with teacher-forced context. Clamping.")
    effective_steps = args.generation_steps if not args.teacher_forced else min(args.generation_steps, max_possible_steps)

    for i in range(effective_steps):
        print(f"Inferring frame {i+1}/{effective_steps}")
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

        sampled_action_index, action_latent = get_action_latent(args, inferred_actions, n_actions, context_frames, latent_action_model, i)

        # Use model's iterative inference helper
        def idx_to_latents(idx):
            return video_tokenizer.quantizer.get_latents_from_indices(idx, dim=-1)

        # Autocast for inference if amp enabled (bfloat16 on CUDA by default)
        autocast_dtype = torch.bfloat16 if args.amp else None
        with torch.amp.autocast('cuda', enabled=args.amp, dtype=autocast_dtype):
            # Verification override: ensure full replacement of masked tokens on a single step
            if os.environ.get('NG_VERIFY_MASK') == '1':
                verification_horizon = 1
                verification_steps = max(16, 10)
                verification_temp = 0.0
            else:
                verification_horizon = args.prediction_horizon
                verification_steps = 10
                verification_temp = float(args.temperature) if hasattr(args, 'temperature') else 0.0
            next_video_latents = dynamics_model.forward_inference(
                context_latents=video_latents,
                prediction_horizon=verification_horizon,
                num_steps=verification_steps,
                index_to_latents_fn=idx_to_latents,
                conditioning=action_latent,
                temperature=verification_temp,
            )

        # decode next video tokens to frames
        next_frames = video_tokenizer.detokenize(next_video_latents)  # [1, T, C, H, W]

        generated_frames = torch.cat([generated_frames, next_frames[:, -verification_horizon:, :, :]], dim=1)
        # TODO: if using interactive mode, visualize next_frames[:, -1] (recently inferred frame) every time, probably with matplotlib is easiest
        # point is for user to be able to interact with it in real time

    pred_len = min(effective_steps, generated_frames.shape[1] - args.context_window)

    visualize_inference(generated_frames, ground_truth_frames, inferred_actions, args.fps, use_actions=use_latent_actions)


if __name__ == "__main__":
    main()
