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
        # conditioning: [B, T, P, A]
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
            mask_token = self.mask_token.to(discrete_latents.device, discrete_latents.dtype).expand(B, T, P, -1) # [B, T, P, 1]
            discrete_latents = torch.where(mask_positions.unsqueeze(-1), mask_token, discrete_latents) # [B, T, P, 1]
        else:
            mask_positions = None

        embeddings = self.latent_embed(discrete_latents)  # [B, T, P, E]

        # Add spatial PE (affects only first 2/3 of dimensions)
        # STTransformer adds temporal PE to last 1/3 of dimensions
        embeddings = embeddings + self.pos_spatial_dec.to(embeddings.device, embeddings.dtype)
        transformed = self.transformer(embeddings, conditioning=conditioning)  # [B, T, P, E]

        # transform to logits for each token in codebook
        predicted_logits = self.output_mlp(transformed)  # [B, T, P, L^D]

        # compute masked cross-entropy loss
        loss = None
        if targets is not None:
            logits_flat = predicted_logits.reshape(-1, predicted_logits.shape[-1])
            targets_flat = targets.reshape(-1)
            mask_flat = mask_positions.reshape(-1)
            masked_logits = logits_flat[mask_flat]
            masked_targets = targets_flat[mask_flat]
            loss = nn.functional.cross_entropy(masked_logits, masked_targets)

        return predicted_logits, mask_positions, loss  # logits, mask, optional loss

    # TODO: make a util
    @torch.no_grad()
    def compute_action_diversity(self, actions_pre_vq, quantized_actions, quantizer):
        a = actions_pre_vq.reshape(-1, actions_pre_vq.size(-1))
        var = a.var(0, unbiased=False).mean()
        idx = quantizer.get_indices_from_latents(quantized_actions, dim=-1).reshape(-1)
        K = int(getattr(quantizer, 'codebook_size', quantizer.num_bins ** quantized_actions.size(-1)))
        p = torch.bincount(idx, minlength=K).float()
        p = p / p.sum().clamp_min(1)
        usage = (p > 0).float().mean()
        ent = -(p * (p + 1e-8).log()).sum() / math.log(max(K, 2))
        return {'pre_quant_var': var, 'action_usage': usage, 'action_entropy': ent}

    @torch.no_grad()
    def forward_inference(self, context_latents, prediction_horizon, num_steps, index_to_latents_fn, conditioning=None, schedule_k=5.0):
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
            logits, _mask_pos, _ = self.forward(input_latents, training=False, conditioning=conditioning, targets=None)
            probs = torch.softmax(logits, dim=-1)  # [B, T, P, K]
            max_probs, predicted_indices = torch.max(probs, dim=-1)  # [B, T, P]

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
