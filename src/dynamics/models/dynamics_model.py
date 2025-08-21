from src.vqvae.models.video_tokenizer import STTransformer, sincos_2d, sincos_1d, sincos_time
import torch
import torch.nn as nn
from einops import rearrange

class Decoder(nn.Module):
    """ST-Transformer decoder that reconstructs frames from latents"""
    def __init__(self, frame_size=(64, 64), patch_size=4, embed_dim=512, num_heads=8,
                 hidden_dim=2048, num_blocks=6, latent_dim=6, num_bins=4):
        super().__init__()
        H, W = frame_size
        Hp, Wp = H // patch_size, W // patch_size
        P = Hp * Wp
        
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
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True)
        
        # Latent embedding goes from latent_dim to embed_dim
        self.latent_embed = nn.Linear(latent_dim, embed_dim)
        
        # Generate separate spatial x and y position encodings
        pe_x = sincos_1d(Wp, self.spatial_x_dim, device='cpu', dtype=torch.float32)  # [Wp, D/3+]
        pe_y = sincos_1d(Hp, self.spatial_y_dim, device='cpu', dtype=torch.float32)  # [Hp, D/3+]
        
        # Expand to 2D grid
        pe_x = pe_x.unsqueeze(0).expand(Hp, Wp, -1)  # [Hp, Wp, D/3+]
        pe_y = pe_y.unsqueeze(1).expand(Hp, Wp, -1)  # [Hp, Wp, D/3+]
        
        # Combine and pad with zeros for temporal part
        pe_spatial = torch.cat([
            pe_x,  # First third+: x position
            pe_y,  # Second third+: y position
            torch.zeros(Hp, Wp, self.temporal_dim, device='cpu', dtype=torch.float32)  # Last third: temporal
        ], dim=-1)  # [Hp, Wp, D]
        
        # Flatten spatial dimensions
        pe_spatial = pe_spatial.reshape(P, embed_dim)  # [P, D]
        self.register_buffer("pos_spatial_dec", pe_spatial[None, :, :], persistent=False)  # [1,P,D]
        
        self.mlp = nn.Linear(embed_dim, codebook_size)
        
        # Learned mask token embedding
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, latent_dim) * 0.02)  # Small initialization
        
    def forward(self, discrete_latents, training=True):
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
            # Sample masking rate uniformly from 0.5 to 1
            masking_rate = torch.rand(1).item() * 0.999 + 0.001   # [0.001, 1.0]
            
            # Create mask for each position (patch-level masking)
            mask_positions = torch.bernoulli(torch.full((B, S, N), masking_rate, device=discrete_latents.device))
            mask_positions = mask_positions.bool()
            
            # Use learned mask token
            # Expand mask token to match batch and sequence dimensions
            mask_token = self.mask_token.expand(B, S, N, -1)
            
            # Replace masked positions with learned mask tokens
            discrete_latents = torch.where(mask_positions.unsqueeze(-1), mask_token, discrete_latents)
        else:
            mask_positions = None

        embeddings = self.latent_embed(discrete_latents)  # [B, S, N, E]
        
        # Add spatial position encoding (affects only first 2/3 of dimensions)
        embeddings = embeddings + self.pos_spatial_dec.to(embeddings.device, embeddings.dtype)
        
        # The causal mask ensures each position can only attend to previous positions
        # STTransformer will add temporal position encoding to last 1/3 of dimensions
        transformed = self.transformer(embeddings)  # [B, S, N, E]
        
        # convert back to latent space
        next_token_logits = self.mlp(transformed)  # [B, S, N, L^D]
        
        return next_token_logits, mask_positions  # [B, S, N, L^D], [B, S, N] or None

class DynamicsModel(nn.Module):
    def __init__(self, frame_size=(64, 64), patch_size=16, embed_dim=512, num_heads=8,
                 hidden_dim=2048, num_blocks=6, latent_dim=32, num_bins=4):
        super().__init__()
        self.decoder = Decoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, latent_dim, num_bins)

    def forward(self, discrete_latents, training=True):
        """
        Args:
            discrete_latents: [batch_size, seq_len, num_patches, latent_dim] video latents with action latents added
            training: Whether in training mode (for masking)
        Returns:
            next_latents: [batch_size, seq_len, num_patches, latent_dim, num_bins]
            mask_positions: [batch_size, seq_len, num_patches] or None - positions that were masked
        """
        next_latents, mask_positions = self.decoder(discrete_latents, training)
        return next_latents, mask_positions
