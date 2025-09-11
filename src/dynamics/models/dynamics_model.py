from src.vqvae.models.video_tokenizer import STTransformer, sincos_2d, sincos_1d, sincos_time
import torch
import torch.nn as nn
from einops import rearrange, repeat
import math

class DynamicsModel(nn.Module):
    """ST-Transformer decoder that reconstructs frames from latents"""
    def __init__(self, frame_size=(128, 128), patch_size=4, embed_dim=128, num_heads=8,
                 hidden_dim=128, num_blocks=4, num_bins=4, n_actions=8, conditioning_dim=3, latent_dim=5):
        super().__init__()
        H, W = frame_size
        Hp, Wp = H // patch_size, W // patch_size
        P = Hp * Wp
        
        # TODO: don't duplicate pos embeds between patch embed and here (need both to have, just make separate class for pos embeds, ideally including temporal too)
        # Split embedding dimensions, ensuring each split is even
        base_split = (embed_dim // 3) & ~1  # Round down to even number
        remaining_dim = embed_dim - base_split  # Remaining after temporal
        
        # Split remaining dimensions between x and y, ensuring even numbers
        self.spatial_x_dim = (remaining_dim // 2) & ~1  # Round down to even
        self.spatial_y_dim = remaining_dim - self.spatial_x_dim  # Give remainder to y
        self.temporal_dim = base_split  # Keep temporal dimension as base even split
        
        assert (self.spatial_x_dim + self.spatial_y_dim + self.temporal_dim) == embed_dim, \
            f"Dimension mismatch: {self.spatial_x_dim} + {self.spatial_y_dim} + {self.temporal_dim} != {embed_dim}"
        assert self.spatial_x_dim % 2 == 0 and self.spatial_y_dim % 2 == 0 and self.temporal_dim % 2 == 0, \
            f"All dimensions must be even: x={self.spatial_x_dim}, y={self.spatial_y_dim}, t={self.temporal_dim}"
        
        codebook_size = num_bins**latent_dim
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True, conditioning_dim=conditioning_dim)

        # Latent embedding goes from latent_dim to embed_dim
        self.latent_embed = nn.Linear(latent_dim, embed_dim)

        # Generate separate spatial x and y position encodings
        pe_x = sincos_1d(Wp, self.spatial_x_dim, device='cpu', dtype=torch.float32)  # [Wp, D/3+]
        pe_y = sincos_1d(Hp, self.spatial_y_dim, device='cpu', dtype=torch.float32)  # [Hp, D/3+]

        # Expand to 2D grid using rearrange
        pe_x = rearrange(pe_x, 'w d -> 1 w d')  # [1, Wp, D/3+]
        pe_x = repeat(pe_x, '1 w d -> h w d', h=Hp)  # [Hp, Wp, D/3+]
        
        pe_y = rearrange(pe_y, 'h d -> h 1 d')  # [Hp, 1, D/3+]
        pe_y = repeat(pe_y, 'h 1 d -> h w d', w=Wp)  # [Hp, Wp, D/3+]
        
        # Combine and pad with zeros for temporal part
        pe_spatial = torch.cat([
            pe_x,  # First third+: x position
            pe_y,  # Second third+: y position
            torch.zeros(Hp, Wp, self.temporal_dim, device='cpu', dtype=torch.float32)  # Last third: temporal (already in st transformer so just pad with 0s)
        ], dim=-1)  # [Hp, Wp, D]
        
        # Flatten spatial dimensions using rearrange
        pe_spatial = rearrange(pe_spatial, 'h w d -> (h w) d')  # [P, D]
        pe_spatial = rearrange(pe_spatial, 'p d -> 1 p d')  # [1, P, D]
        self.register_buffer("pos_spatial_dec", pe_spatial, persistent=False)
        
        self.output_mlp = nn.Linear(embed_dim, codebook_size)
        
        # Learned mask token embedding
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, latent_dim) * 0.02)  # Small initialization
        
    def forward(self, discrete_latents, training=True, conditioning=None):
        """
        Args:
            discrete_latents: [batch_size, seq_len, num_patches, latent_dim]
            training: Whether in training mode (for masking)
        Returns:
            next_token_logits: [batch_size, seq_len, num_patches, codebook_size]
            mask_positions: [batch_size, seq_len, num_patches] or None - positions that were masked
        """
        B, S, N, D = discrete_latents.shape
        
        # Convert latents to float for embedding
        discrete_latents = discrete_latents.to(dtype=torch.float32)
        
        # Apply random masking during training (MaskGit-style)
        if training and self.training:
            # TODO: change to thresholded binary mask
            # per-batch mask ratio in [0.5, 1.0)
            mask_ratio = 0.5 + torch.rand((), device=discrete_latents.device) * 0.5 
            mask_positions = (torch.rand(B, S, N, device=discrete_latents.device) < mask_ratio) # [B, S, N]

            # Guarantee at least one unmasked temporal anchor per (B,N)
            # Pick a random timestep for each (B,N) and force it to unmask
            anchor_idx = torch.randint(0, S, (B, N), device=discrete_latents.device)  # [B, N]
            mask_positions[torch.arange(B)[:, None], anchor_idx, torch.arange(N)[None, :]] = False # [B, S, N]

            # TODO: replace with repeat einops
            mask_token = self.mask_token.to(discrete_latents.device, discrete_latents.dtype).expand(B, S, N, -1) # [B, S, N, D]
            discrete_latents = torch.where(mask_positions.unsqueeze(-1), mask_token, discrete_latents) # [B, S, N, D]
        else:
            mask_positions = None

        embeddings = self.latent_embed(discrete_latents)  # [B, S, N, E]

        # Add spatial PE (affects only first 2/3 of dimensions)
        # STTransformer adds temporal PE to last 1/3 of dimensions
        embeddings = embeddings + self.pos_spatial_dec.to(embeddings.device, embeddings.dtype)
        transformed = self.transformer(embeddings, conditioning=conditioning)  # [B, S, N, E]

        # transform to logits for each token in codebook
        next_token_logits = self.output_mlp(transformed)  # [B, S, N, L^D]

        return next_token_logits, mask_positions  # [B, S, N, L^D], [B, S, N] or None

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
