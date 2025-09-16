import torch
import torch.nn as nn
import math
from models.positional_encoding import build_spatial_only_pe
from models.st_transformer import STTransformer

# TODO: make create mask function here
class DynamicsModel(nn.Module):
    """ST-Transformer decoder that reconstructs frames from latents"""
    def __init__(self, frame_size=(128, 128), patch_size=4, embed_dim=128, num_heads=8,
                 hidden_dim=128, num_blocks=4, num_bins=4, n_actions=8, conditioning_dim=3, latent_dim=5):
        super().__init__()
        H, W = frame_size

        codebook_size = num_bins**latent_dim
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True, conditioning_dim=conditioning_dim)

        # Latent embedding goes from latent_dim to embed_dim
        self.latent_embed = nn.Linear(latent_dim, embed_dim)

        # Shared spatial-only PE (zeros in temporal tail)
        pe_spatial = build_spatial_only_pe((H, W), patch_size, embed_dim, device='cpu', dtype=torch.float32)  # [1,P,E]
        self.register_buffer("pos_spatial_dec", pe_spatial, persistent=False)

        self.output_mlp = nn.Linear(embed_dim, codebook_size)

        # Learned mask token embedding
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, latent_dim) * 0.02)  # Small initialization

    def forward(self, discrete_latents, training=True, conditioning=None, targets=None):
        # discrete_latents: [B, T, P, L]
        # targets: [B, T, P] indices
        # conditioning: [B, T, A]
        B, T, P, L = discrete_latents.shape

        # Convert latents to float for embedding
        discrete_latents = discrete_latents.to(dtype=torch.float32)

        # Apply random masking during training (MaskGit-style)
        if training and self.training:
            # per-batch mask ratio in [0.5, 1.0)
            mask_ratio = 0.5 + torch.rand((), device=discrete_latents.device) * 0.5 
            mask_positions = (torch.rand(B, T, P, device=discrete_latents.device) < mask_ratio) # [B, T, P]

            # Guarantee at least one unmasked temporal anchor per (B, P)
            # Pick a random timestep for each (B,P) and force it to unmask
            anchor_idx = torch.randint(0, T, (B, P), device=discrete_latents.device)  # [B, P]
            mask_positions[torch.arange(B)[:, None], anchor_idx, torch.arange(P)[None, :]] = False # [B, T, P]

            # TODO: replace with repeat einops
            mask_token = self.mask_token.to(discrete_latents.device, discrete_latents.dtype).expand(B, T, P, -1) # [B, T, P, L]
            discrete_latents = torch.where(mask_positions.unsqueeze(-1), mask_token, discrete_latents) # [B, T, P, L]
        else:
            mask_positions = None

        embeddings = self.latent_embed(discrete_latents)  # [B, T, P, E]

        # Add spatial PE (affects only first 2/3 of dimensions)
        # STTransformer adds temporal PE to last 1/3 of dimensions
        embeddings = embeddings + self.pos_spatial_dec.to(embeddings.device, embeddings.dtype)
        transformed = self.transformer(embeddings, conditioning=conditioning)  # [B, T, P, E]

        # transform to logits for each token in codebook
        predicted_logits = self.output_mlp(transformed)  # [B, T, P, L^D]

        # compute masked cross-entropy loss with static shapes
        loss = None
        if training and self.training:
            assert targets is not None, "target indices are needed for training"
            K = predicted_logits.shape[-1]
            logits_flat = predicted_logits.reshape(-1, K)              # [(B*T*P), K]
            targets_flat = targets.reshape(-1)                          # [(B*T*P)]
            mask_flat = mask_positions.reshape(-1).to(torch.float32)    # [(B*T*P)]
            loss_per = nn.functional.cross_entropy(logits_flat, targets_flat, reduction='none')  # [(B*T*P)]
            denom = mask_flat.sum().clamp_min(1.0)
            loss = (loss_per * mask_flat).sum() / denom

        return predicted_logits, mask_positions, loss  # logits, mask, optional loss

    # TODO: make a util
    @torch.no_grad()
    def forward_inference(self, context_latents, prediction_horizon, num_steps, index_to_latents_fn, conditioning=None, schedule_k=5.0, temperature: float = 0.0):
        # TODO: review and clean
        # MaskGit-style iterative decoding
        # for timestep m in range M: 
        # 1. run inference with all tokens masked
        # 2. get argmax tokens and their corresponding probabilities
        # 3. choose top n tokens with highest probabilities and unmask them
        # 4. repeat until all tokens are unmasked or num_steps is reached
        # context_latents: [B, T_ctx, P, L]

        device = context_latents.device
        dtype = context_latents.dtype
        B, T_ctx, P, L = context_latents.shape

        # Append mask latents for horizon
        mask_latents = self.mask_token.to(device, dtype).expand(B, prediction_horizon, P, -1)
        input_latents = torch.cat([context_latents, mask_latents], dim=1)  # [B, T_ctx+H, P, L]

        # Boolean mask for horizon positions
        mask = torch.ones(B, prediction_horizon, P, 1, dtype=torch.bool, device=device)

        def exp_schedule_torch(t, T, P_total, k=schedule_k):
            x = t / max(T, 1)
            k_tensor = torch.tensor(k, device=device)
            result = P_total * torch.expm1(k_tensor * x) / torch.expm1(k_tensor)
            if t == T - 1:
                return torch.tensor(P_total, dtype=result.dtype, device=device)
            return result

        for m in range(num_steps):
            prev_unmask = (mask[:, -1, :, 0] == False).sum().item()
            n_tokens_raw = exp_schedule_torch(m, num_steps, P)
            n_tokens = int(n_tokens_raw)
            tokens_to_unmask = max(0, n_tokens - prev_unmask)

            # Predict logits for current input
            logits, _, _ = self.forward(input_latents, training=False, conditioning=conditioning, targets=None)
            # Temperature scaling
            if temperature and temperature > 0:
                scaled_logits = logits / float(temperature)
            else:
                scaled_logits = logits
            probs = torch.softmax(scaled_logits, dim=-1)  # [B, T, P, K]
            # Confidence for unmask selection always from max probability
            max_probs, _ = torch.max(probs, dim=-1)  # [B, T, P]
            # Choose indices either via argmax (temperature==0) or sampling
            if temperature and temperature > 0:
                # Sample per position from categorical distribution
                Bc, Tc, Pc, K = probs.shape
                sampled = torch.distributions.Categorical(probs=probs.reshape(-1, K)).sample()
                predicted_indices = sampled.view(Bc, Tc, Pc)  # [B, T, P]
            else:
                _, predicted_indices = torch.max(probs, dim=-1)  # [B, T, P]

            # Only operate on last timestep
            masked_probs = max_probs[:, -1, :]  # [B, P]
            masked_mask = mask[:, -1, :, 0]     # [B, P]

            # Ensure at least 1 token unmasked if any remain
            if masked_mask.any() and tokens_to_unmask == 0:
                tokens_to_unmask = 1

            # For each batch element, select top tokens_to_unmask among masked
            for b in range(B):
                masked_indices = torch.where(masked_mask[b])[0]  # [num_masked]
                if masked_indices.numel() == 0:
                    continue
                pos_probs = masked_probs[b, masked_indices]
                if pos_probs.numel() > tokens_to_unmask:
                    top_idx = torch.topk(pos_probs, tokens_to_unmask, largest=True).indices
                    tokens_to_unmask_indices = masked_indices[top_idx]
                else:
                    tokens_to_unmask_indices = masked_indices

                # Unmask and write predicted latents for those positions
                # Build indices tensor shape [1, 1, P_sel]
                sel = tokens_to_unmask_indices
                idx_sel = predicted_indices[b:b+1, -1:, sel]
                pred_latents_sel = index_to_latents_fn(idx_sel)  # [1,1,P_sel,L]
                input_latents[b:b+1, -1:, sel] = pred_latents_sel
                mask[b, -1, sel, 0] = False

            # Early exit if all unmasked
            if not mask[:, -1, :, 0].any():
                break

        return input_latents
