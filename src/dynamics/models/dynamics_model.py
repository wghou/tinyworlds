from src.vqvae.models.video_tokenizer import STTransformer, sincos_2d, sincos_time
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
        
        codebook_size = num_bins**latent_dim
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True)

        # Latent embedding goes from latent_dim to embed_dim
        self.latent_embed = nn.Linear(latent_dim, embed_dim)
        
        # 2D spatial sincos PE buffer for decoder path
        pe_spatial = sincos_2d(Hp, Wp, embed_dim, device='cpu', dtype=torch.float32)
        self.register_buffer("pos_spatial_dec", pe_spatial[None, :, :], persistent=False)  # [1,P,E]
        
        # Output projection
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

        embeddings = self.latent_embed(discrete_latents) # [B, S, N, E]

        # Add spatial position encoding
        embeddings = embeddings + self.pos_spatial_dec.to(embeddings.device, embeddings.dtype)  # broadcast [1,P,E] over [B,S,P,E]
        
        # Add temporal position encoding
        tpe = sincos_time(S, embeddings.shape[-1], embeddings.device, embeddings.dtype)  # [S, E]
        embeddings = embeddings + tpe[None, :, None, :]  # broadcast [1,S,1,E] over [B,S,N,E]
    
        # The causal mask ensures each position can only attend to previous positions
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
