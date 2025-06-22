import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

class PatchEmbedding(nn.Module):
    """Convert frames to patch embeddings for ST-Transformer"""
    def __init__(self, frame_size=(64, 64), patch_size=4, embed_dim=512):
        super().__init__()
        self.frame_size = frame_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (frame_size[0] // patch_size) * (frame_size[1] // patch_size)
        
        # Linear projection for patches
        self.proj = nn.Linear(3 * patch_size * patch_size, embed_dim)
        
        # Position embeddings for patches within each frame
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
    def forward(self, frames):
        """
        Args:
            frames: [batch_size, seq_len, channels, height, width]
        Returns:
            embeddings: [batch_size, seq_len, num_patches, embed_dim]
        """
        batch_size, seq_len, C, H, W = frames.shape
        
        # Reshape to patches using einops: [batch_size * seq_len, num_patches, patch_size * patch_size * channels]
        # h and w here are actually number of patches in h and w dims, p1 and p2 are identical (patch size)
        patches = rearrange(frames, 'b s c (h p1) (w p2) -> (b s) (h w) (c p1 p2)', 
                          p1=self.patch_size, p2=self.patch_size)
        
        # Project to embeddings
        embeddings = self.proj(patches) # [batch_size * seq_len, num_patches, embed_dim]
        
        # Add position embeddings
        embeddings = embeddings + self.pos_embed # [batch_size * seq_len, num_patches, embed_dim]
        
        # Reshape back to sequence
        embeddings = rearrange(embeddings, '(b s) p e -> b s p e', b=batch_size, s=seq_len) # [batch_size, seq_len, num_patches, embed_dim]
        
        return embeddings

class SpatialAttention(nn.Module):
    """Spatial attention layer - attends over patches within each frame"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, num_patches, embed_dim]
        """
        # Always expects x of shape [batch, seq, num_patches, embed_dim]
        batch_size, seq_len, num_patches, embed_dim = x.shape
        
        # Project to Q, K, V and reshape: [batch, seq, num_patches, embed_dim] -> [batch, seq, num_heads, num_patches, head_dim]
        q = rearrange(self.q_proj(x), 'b s p (n d) -> b s n p d', n=self.num_heads)
        k = rearrange(self.k_proj(x), 'b s p (n d) -> b s n p d', n=self.num_heads)
        v = rearrange(self.v_proj(x), 'b s p (n d) -> b s n p d', n=self.num_heads)

        k_t = k.transpose(-2, -1) # [batch, seq, num_heads, head_dim, num_patches]
        
        # Compute attention: [batch, seq, num_heads, num_patches, num_patches]
        scores = torch.matmul(q, k_t) / math.sqrt(self.head_dim)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights) # TODO: do we need dropout here?
        
        # Apply attention to values: [batch, seq, num_heads, num_patches, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        attn_output = rearrange(attn_output, 'b s n p d -> b s p (n d)')
        
        # Final projection TODO: do we need this?
        attn_out = self.out_proj(attn_output)  # [batch_size, seq_len, num_patches, embed_dim]
        
        # Add residual and normalize
        out = self.norm(x + attn_out)
        
        return out

class TemporalAttention(nn.Module):
    """Temporal attention layer - attends over time steps for each patch"""
    def __init__(self, embed_dim, num_heads, dropout=0.1, causal=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.causal = causal
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, num_patches, embed_dim]
        """
        # Always expects x of shape [batch, seq, num_patches, embed_dim]
        batch_size, seq_len, num_patches, embed_dim = x.shape
        
        # Project to Q, K, V and reshape: [batch, seq, num_patches, embed_dim] -> [batch, num_patches, num_heads, seq, head_dim]
        q = rearrange(self.q_proj(x), 'b s p (n d) -> b p n s d', n=self.num_heads)
        k = rearrange(self.k_proj(x), 'b s p (n d) -> b p n s d', n=self.num_heads)
        v = rearrange(self.v_proj(x), 'b s p (n d) -> b p n s d', n=self.num_heads)

        k_t = k.transpose(-2, -1) # [batch, num_patches, num_heads, head_dim, seq]
        
        # Compute attention: [batch, num_patches, num_heads, seq, seq]
        scores = torch.matmul(q, k_t) / math.sqrt(self.head_dim)
        
        # Apply causal mask if needed (for each token t in seq, mask out all tokens to the right of t (after t))
        if self.causal:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            mask = mask.to(x.device)
            scores = scores.masked_fill(mask, float('-inf')) # [batch, num_patches, num_heads, seq, seq]
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights) # TODO: do we need dropout here?
        
        # Apply attention to values: [batch, num_patches, num_heads, seq, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        attn_output = rearrange(attn_output, 'b p n s d -> b s p (n d)') # [batch_size, seq_len, num_patches, embed_dim]
        
        # Final projection TODO: do we need this?
        attn_out = self.out_proj(attn_output)  # [batch_size, seq_len, num_patches, embed_dim]
        
        # Add residual and normalize
        out = self.norm(x + attn_out)
        
        return out

class FeedForward(nn.Module):
    """Feed-forward network"""
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, num_patches, embed_dim]
        """
        batch_size, seq_len, num_patches, embed_dim = x.shape
        
        # Reshape to process all elements at once: [batch_size * seq_len * num_patches, embed_dim]
        x_reshaped = rearrange(x, 'b s p e -> (b s p) e')
        
        # Apply feed-forward
        out = self.linear1(x_reshaped)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        
        # Add residual and normalize
        out = self.norm(x_reshaped + out)
        
        # Reshape back
        out = rearrange(out, '(b s p) e -> b s p e', b=batch_size, s=seq_len, p=num_patches)
        
        return out

class STTransformerBlock(nn.Module):
    """ST-Transformer block with spatial attention, temporal attention, and feed-forward"""
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1, causal=True):
        super().__init__()
        self.spatial_attn = SpatialAttention(embed_dim, num_heads, dropout)
        self.temporal_attn = TemporalAttention(embed_dim, num_heads, dropout, causal)
        self.ffn = FeedForward(embed_dim, hidden_dim, dropout)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, num_patches, embed_dim]
        """
        # Spatial attention - attends over patches within each frame
        x = self.spatial_attn(x)
        
        # Temporal attention - attends over time steps for each patch
        x = self.temporal_attn(x)
        
        # Feed-forward
        x = self.ffn(x)
        
        return x

class STTransformer(nn.Module):
    """ST-Transformer with multiple blocks"""
    def __init__(self, embed_dim, num_heads, hidden_dim, num_blocks, dropout=0.1, causal=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            STTransformerBlock(embed_dim, num_heads, hidden_dim, dropout, causal)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, num_patches, embed_dim]
        """
        for block in self.blocks:
            x = block(x)
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, latent_dim, beta=1.0):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.beta = beta
        
        self.embedding = nn.Embedding(self.codebook_size, self.latent_dim)
        
        # Use simple, stable initialization instead of Gram-Schmidt
        with torch.no_grad():
            # Initialize with small random values
            self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z):
        """
        Args:
            z: [batch_size, seq_len, num_patches, latent_dim]
        Returns:
            loss: scalar
            z_q: [batch_size, seq_len, num_patches, latent_dim]
            min_encoding_indices: [batch_size, seq_len, num_patches]
        """
        batch_size, seq_len, num_patches, latent_dim = z.shape
        
        # Debug input
        if torch.isnan(z).any():
            print("VQ WARNING: Input z contains NaN!")
        if torch.isinf(z).any():
            print("VQ WARNING: Input z contains inf!")
        
        # Reshape z to [batch_size * seq_len * num_patches, latent_dim] for distance calculation
        z_flat = z.view(-1, latent_dim)  # [batch_size * seq_len * num_patches, latent_dim]
        
        # Debug z_flat
        if torch.isnan(z_flat).any():
            print("VQ WARNING: z_flat contains NaN!")
        if torch.isinf(z_flat).any():
            print("VQ WARNING: z_flat contains inf!")
        
        # Debug embedding weights
        if torch.isnan(self.embedding.weight).any():
            print("VQ WARNING: embedding weights contain NaN!")
        if torch.isinf(self.embedding.weight).any():
            print("VQ WARNING: embedding weights contain inf!")
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        z_squared = torch.sum(z_flat ** 2, dim=1, keepdim=True)
        e_squared = torch.sum(self.embedding.weight**2, dim=1)
        z_e_product = torch.matmul(z_flat, self.embedding.weight.t())
        
        # Debug individual terms
        if torch.isnan(z_squared).any():
            print("VQ WARNING: z_squared contains NaN!")
        if torch.isinf(z_squared).any():
            print("VQ WARNING: z_squared contains inf!")
        if torch.isnan(e_squared).any():
            print("VQ WARNING: e_squared contains NaN!")
        if torch.isinf(e_squared).any():
            print("VQ WARNING: e_squared contains inf!")
        if torch.isnan(z_e_product).any():
            print("VQ WARNING: z_e_product contains NaN!")
        if torch.isinf(z_e_product).any():
            print("VQ WARNING: z_e_product contains inf!")
        
        d = z_squared + e_squared - 2 * z_e_product  # [batch_size * seq_len * num_patches, codebook_size]
        
        # Debug final distance
        if torch.isnan(d).any():
            print("VQ WARNING: distance matrix contains NaN!")
        if torch.isinf(d).any():
            print("VQ WARNING: distance matrix contains inf!")
            
        # Find nearest embedding
        min_encoding_indices = torch.argmin(d, dim=1)  # [batch_size * seq_len * num_patches]
        
        # Reshape back to original shape
        min_encoding_indices = min_encoding_indices.view(batch_size, seq_len, num_patches)
        
        z_q = self.embedding(min_encoding_indices)  # [batch_size, seq_len, num_patches, latent_dim]
        
        # Debug z_q
        if torch.isnan(z_q).any():
            print("VQ WARNING: z_q contains NaN!")
        if torch.isinf(z_q).any():
            print("VQ WARNING: z_q contains inf!")
        
        # Compute loss
        commitment_loss = torch.mean((z_q.detach()-z)**2)
        codebook_loss = torch.mean((z_q - z.detach()) ** 2)
        
        # Debug individual loss terms
        if torch.isnan(commitment_loss):
            print("VQ WARNING: commitment_loss is NaN!")
        if torch.isinf(commitment_loss):
            print("VQ WARNING: commitment_loss is inf!")
        if torch.isnan(codebook_loss):
            print("VQ WARNING: codebook_loss is NaN!")
        if torch.isinf(codebook_loss):
            print("VQ WARNING: codebook_loss is inf!")
        
        loss = commitment_loss + self.beta * codebook_loss
        
        # Debug final loss
        if torch.isnan(loss):
            print("VQ WARNING: final loss is NaN!")
        if torch.isinf(loss):
            print("VQ WARNING: final loss is inf!")
               
        # Preserve gradients
        z_q = z + (z_q - z).detach()
        
        return loss, z_q, min_encoding_indices

    def reset_codebook(self):
        """Reset codebook to stable initialization if it gets stuck"""
        with torch.no_grad():
            # Use simple, stable initialization
            self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

class Encoder(nn.Module):
    """ST-Transformer encoder that takes frames and outputs latent actions"""
    def __init__(self, frame_size=(64, 64), patch_size=16, embed_dim=512, num_heads=8, 
                 hidden_dim=1024, num_blocks=6, latent_dim=32, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, dropout, causal=True)
        
        # Latent prediction head
        self.latent_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, latent_dim)
        )
        
    def forward(self, frames):
        """
        Args:
            frames: [batch_size, seq_len, channels, height, width]
        Returns:
            tokens: [batch_size, seq_len, num_patches, latent_dim]
        """
        # Convert frames to patch embeddings
        embeddings = self.patch_embed(frames)  # [batch_size, seq_len, num_patches, embed_dim]
        
        # Apply ST-Transformer
        transformed = self.transformer(embeddings) # [batch_size, seq_len, num_patches, embed_dim]

        predicted_latents = self.latent_head(transformed) # [batch_size, seq_len, num_patches, latent_dim]
        
        return predicted_latents

class Decoder(nn.Module):
    """ST-Transformer decoder that takes frames and latents to predict next frame"""
    def __init__(self, frame_size=(64, 64), patch_size=16, embed_dim=512, num_heads=8,
                 hidden_dim=2048, num_blocks=6, latent_dim=64, dropout=0.1, height=64, width=64, channels=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, dropout, causal=True)
        
        # Latent embedding
        self.latent_embed = nn.Linear(latent_dim, embed_dim)
        
        num_patches = (height // patch_size) * (width // patch_size)
        # Frame prediction head goes from num_patches * embed_dim to height * width * channels
        self.frame_head = nn.Linear(num_patches * embed_dim, height * width * channels)
        self.height = height
        self.width = width
        self.channels = channels
        
        self.frame_size = frame_size
        self.patch_size = patch_size
        self.num_patches = (frame_size[0] // patch_size) * (frame_size[1] // patch_size)
        
    def forward(self, latents, training=True):
        """
        Args:
            latents: [batch_size, seq_len, num_patches, latent_dim]
            training: Whether in training mode (for masking)
        Returns:
            pred_frames: [batch_size, seq_len, channels, height, width]
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
        
        # convert back to original frames [batch_size, seq_len, channels, height, width]
        # convert to batch_size, seq_len, num_patches * embed_dim
        flattened = rearrange(transformed, 'b s p e -> b s (p e)')
        frames_out = self.frame_head(flattened)
        frames_out = rearrange(frames_out, 'b s (h w c) -> b s c h w', h=self.height, w=self.width)

        return frames_out  # [batch_size, seq_len, channels, height, width]

class Video_Tokenizer(nn.Module):
    def __init__(self, frame_size=(64, 64), patch_size=4, embed_dim=512, num_heads=8,
                 hidden_dim=2048, num_blocks=6, latent_dim=32, dropout=0.1, codebook_size=512, beta=1.0):
        super().__init__()
        self.encoder = Encoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, latent_dim, dropout)
        self.decoder = Decoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, latent_dim, dropout)
        self.vq = VectorQuantizer(codebook_size, latent_dim, beta)
        
    def forward(self, frames):
        # Debug input
        if torch.isnan(frames).any():
            print("WARNING: Input frames contain NaN!")
        if torch.isinf(frames).any():
            print("WARNING: Input frames contain inf!")
            
        # Encode frames to latent representations
        embeddings = self.encoder(frames)  # [batch_size, seq_len, num_patches, latent_dim]
        
        # Debug encoder output
        if torch.isnan(embeddings).any():
            print("WARNING: Encoder output contains NaN!")
        if torch.isinf(embeddings).any():
            print("WARNING: Encoder output contains inf!")
        
        # Apply vector quantization
        vq_loss, z_q, min_encoding_indices = self.vq(embeddings)
        
        # Debug VQ output
        if torch.isnan(z_q).any():
            print("WARNING: VQ output contains NaN!")
        if torch.isinf(z_q).any():
            print("WARNING: VQ output contains inf!")
        if torch.isnan(vq_loss):
            print("WARNING: VQ loss is NaN!")
        if torch.isinf(vq_loss):
            print("WARNING: VQ loss is inf!")
        
        # Decode quantized latents back to frames
        x_hat = self.decoder(z_q)  # [batch_size, seq_len, channels, height, width]
        
        # Debug decoder output
        if torch.isnan(x_hat).any():
            print("WARNING: Decoder output contains NaN!")
        if torch.isinf(x_hat).any():
            print("WARNING: Decoder output contains inf!")
        
        return x_hat, vq_loss
