import torch
import torch.nn as nn
import math
from models.positional_encoding import build_spatial_only_pe
from models.st_transformer import STTransformer

# TODO: make create mask function here
# TODO: make run inference
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

    def forward(self, discrete_latents, training=True, conditioning=None):
        # discrete_latents: [B, T, P, L]
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

        return predicted_logits, mask_positions  # [B, T, P, L^D], [B, T, P] or None

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
