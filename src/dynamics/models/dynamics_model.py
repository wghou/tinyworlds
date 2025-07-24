from src.vqvae.models.video_tokenizer import STTransformer, PatchEmbedding
import torch
import torch.nn as nn
from einops import rearrange

class Decoder(nn.Module):
    """ST-Transformer decoder that reconstructs frames from latents"""
    def __init__(self, frame_size=(64, 64), patch_size=4, embed_dim=512, num_heads=8,
                 hidden_dim=2048, num_blocks=6, latent_dim=6, dropout=0.1, L=4):
        super().__init__()
        codebook_size = L**latent_dim
        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, dropout, causal=True)
        
        # Latent embedding goes from latent_dim to embed_dim
        self.latent_embed = nn.Linear(latent_dim, embed_dim)
        # Latent head goes from embed_dim to latent_dim
        self.latent_head = nn.Linear(embed_dim, latent_dim)

        self.mlp = nn.Linear(embed_dim, codebook_size)
        
    def forward(self, discrete_latents, training=True):
        """
        Args:
            discrete_latents: [batch_size, seq_len, num_patches, latent_dim]
            training: Whether in training mode (for masking)
        Returns:
            next_token_logits: [batch_size, seq_len, num_patches, codebook_size]
        """
        B, S, N, D = discrete_latents.shape
        
        # Apply random masking during training
        if training and self.training:
            # Sample masking rate uniformly from 0.5 to 1
            masking_rate = torch.rand(1).item() * 0.5 + 0.5  # [0.5, 1.0]
            
            # Create mask with Bernoulli distribution
            mask = torch.bernoulli(torch.full((B, S), masking_rate, device=discrete_latents.device))
            mask = mask.unsqueeze(-1).unsqueeze(-1)  # [B, S, 1, 1]
            
            # Apply mask (zero out masked latents)
            discrete_latents = discrete_latents * mask

        embeddings = self.latent_embed(discrete_latents) # [B, S, N, E]
    
        # The causal mask ensures each position can only attend to previous positions
        transformed = self.transformer(embeddings)  # [B, S, N, E]

        # convert back to latent space
        next_token_logits = self.mlp(transformed)  # [B, S, N, L^D]

        return next_token_logits  # [B, S, N, L^D]

class DynamicsModel(nn.Module):
    def __init__(self, frame_size=(64, 64), patch_size=16, embed_dim=512, num_heads=8,
                 hidden_dim=2048, num_blocks=6, latent_dim=32, dropout=0.1, codebook_size=4):
        super().__init__()
        self.decoder = Decoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, latent_dim, dropout,codebook_size)

    def forward(self, discrete_latents, training=True):
        """
        Args:
            discrete_latents: [batch_size, seq_len, num_patches, latent_dim] video latents with action latents added
            training: Whether in training mode (for masking)
        Returns:
            next_latents: [batch_size, seq_len, num_patches, codebook_size]
        """
        next_latents = self.decoder(discrete_latents, training)
        return next_latents
