from src.vqvae.models.video_tokenizer import STTransformer, PatchEmbedding
import torch
import torch.nn as nn
from einops import rearrange

class Decoder(nn.Module):
    """ST-Transformer decoder that reconstructs frames from latents"""
    def __init__(self, frame_size=(64, 64), patch_size=16, embed_dim=512, num_heads=8,
                 hidden_dim=2048, num_blocks=6, latent_dim=64, dropout=0.1, height=64, width=64, channels=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, dropout, causal=True)
        
        # Latent embedding goes from latent_dim to embed_dim
        self.latent_embed = nn.Linear(latent_dim, embed_dim)
        # Latent head goes from embed_dim to latent_dim
        self.latent_head = nn.Linear(embed_dim, latent_dim)
        
    def forward(self, latents, training=True):
        """
        Args:
            latents: [batch_size, seq_len, num_patches, latent_dim]
            training: Whether in training mode (for masking)
        Returns:
            next_latents: [batch_size, seq_len, num_patches, latent_dim]
        """
        batch_size, seq_len, num_patches, latent_dim = latents.shape
        
        # Apply random masking during training
        if training and self.training:
            # Sample masking rate uniformly from 0.5 to 1
            masking_rate = torch.rand(1).item() * 0.5 + 0.5  # [0.5, 1.0]
            
            # Create mask with Bernoulli distribution
            mask = torch.bernoulli(torch.full((batch_size, seq_len), masking_rate, device=latents.device))
            mask = mask.unsqueeze(-1).unsqueeze(-1)  # [batch_size, seq_len, 1, 1]
            
            # Apply mask (zero out masked latents)
            latents = latents * mask

        embeddings = self.latent_embed(latents) # [batch_size, seq_len, num_patches, embed_dim]
    
        # The causal mask ensures each position can only attend to previous positions
        transformed = self.transformer(embeddings)  # [batch_size, seq_len, num_patches, embed_dim]

        # convert back to latent space
        next_latents = self.latent_head(transformed)  # [batch_size, seq_len, num_patches, latent_dim]
        
        return next_latents  # [batch_size, seq_len, num_patches, latent_dim]

class DynamicsModel(nn.Module):
    def __init__(self, frame_size=(64, 64), patch_size=16, embed_dim=512, num_heads=8,
                 hidden_dim=2048, num_blocks=6, latent_dim=64, dropout=0.1, height=64, width=64, channels=3):
        super().__init__()
        self.decoder = Decoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, latent_dim, dropout, height, width, channels)

    def forward(self, latents, training=True):
        """
        Args:
            latents: [batch_size, seq_len, num_patches, latent_dim] video latents with action latents added
            training: Whether in training mode (for masking)
        Returns:
            next_latents: [batch_size, seq_len, num_patches, latent_dim]
        """
        next_latents = self.decoder(latents, training)
        return next_latents
