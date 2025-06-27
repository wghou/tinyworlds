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
        Returns:
            out: [batch_size, seq_len, num_patches, embed_dim]
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
        Returns:
            out: [batch_size, seq_len, num_patches, embed_dim]
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
        Returns:
            out: [batch_size, seq_len, num_patches, embed_dim]
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
        Returns:
            out: [batch_size, seq_len, num_patches, embed_dim]
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
        Returns:
            out: [batch_size, seq_len, num_patches, embed_dim]
        """
        for block in self.blocks:
            x = block(x)
        return x

class FiniteScalarQuantizer(nn.Module):
    """
    Finite Scalar Quantizer - quantizes each dimension independently
    This helps prevent token collapse by allowing more flexible quantization
    """
    def __init__(self, latent_dim, num_bins=256, beta=1.0, ema_decay=0.99):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_bins = num_bins
        self.beta = beta
        self.ema_decay = ema_decay
        
        # Each dimension has its own set of bins
        self.bins = nn.Parameter(torch.randn(latent_dim, num_bins) * 0.1)
        
        # EMA tracking for bin updates
        self.register_buffer('ema_bin_usage', torch.zeros(latent_dim, num_bins))
        self.register_buffer('ema_bin_means', torch.zeros(latent_dim, num_bins))
        
        # Initialize bins to cover a reasonable range
        with torch.no_grad():
            # Initialize bins to be evenly spaced in [-1, 1] for each dimension
            for dim in range(latent_dim):
                self.bins.data[dim] = torch.linspace(-1.0, 1.0, num_bins)

    def forward(self, z):
        """
        Args:
            z: [batch_size, seq_len, num_patches, latent_dim]
        Returns:
            loss: scalar
            z_q: [batch_size, seq_len, num_patches, latent_dim]
            bin_indices: [batch_size, seq_len, num_patches, latent_dim]
        """
        batch_size, seq_len, num_patches, latent_dim = z.shape
        
        # Reshape to [batch_size * seq_len * num_patches, latent_dim] for quantization
        z_flat = z.view(-1, latent_dim)  # [batch_size * seq_len * num_patches, latent_dim]
        
        # Quantize each dimension independently
        z_q_flat = torch.zeros_like(z_flat)
        bin_indices_flat = torch.zeros(z_flat.shape[0], latent_dim, dtype=torch.long, device=z.device)
        
        commitment_loss = 0.0
        codebook_loss = 0.0
        
        for dim in range(latent_dim):
            # Get values for this dimension
            z_dim = z_flat[:, dim]  # [batch_size * seq_len * num_patches]
            bins_dim = self.bins[dim]  # [num_bins]
            
            # Find nearest bin for each value
            distances = torch.abs(z_dim.unsqueeze(1) - bins_dim.unsqueeze(0))  # [N, num_bins]
            bin_indices = torch.argmin(distances, dim=1)  # [N]
            
            # Get quantized values
            z_q_dim = bins_dim[bin_indices]  # [N]
            
            # Store results
            z_q_flat[:, dim] = z_q_dim
            bin_indices_flat[:, dim] = bin_indices
            
            # Compute losses for this dimension
            commitment_loss += torch.mean((z_dim - z_q_dim.detach())**2)
            codebook_loss += torch.mean((z_dim.detach() - z_q_dim)**2)
        
        # Average losses across dimensions
        commitment_loss /= latent_dim
        codebook_loss /= latent_dim
        
        # Total FSQ loss
        fsq_loss = commitment_loss + self.beta * codebook_loss
        
        # Reshape back to original shape
        z_q = z_q_flat.view(batch_size, seq_len, num_patches, latent_dim)
        bin_indices = bin_indices_flat.view(batch_size, seq_len, num_patches, latent_dim)
        
        # Straight-through estimator for gradients
        z_q = z + (z_q - z).detach()
        
        return fsq_loss, z_q, bin_indices

    def reset_bins(self):
        """Reset bins to stable initialization if they get stuck"""
        with torch.no_grad():
            # Reinitialize bins to be evenly spaced
            for dim in range(self.latent_dim):
                self.bins.data[dim] = torch.linspace(-1.0, 1.0, self.num_bins)
            
            # Add some noise to break symmetry
            noise = torch.randn_like(self.bins.data) * 0.1
            self.bins.data += noise

    def smart_reinitialize(self, z_flat, bin_indices):
        """
        Smart reinitialization using actual encoder outputs
        
        Args:
            z_flat: [batch_size * seq_len * num_patches, latent_dim] - encoder outputs
            bin_indices: [batch_size * seq_len * num_patches, latent_dim] - current bin assignments
        """
        with torch.no_grad():
            # For each dimension, check bin usage and reinitialize unused bins
            for dim in range(self.latent_dim):
                z_dim = z_flat[:, dim]
                bin_indices_dim = bin_indices[:, dim]
                
                # Count usage of each bin
                usage_counts = torch.bincount(bin_indices_dim, minlength=self.num_bins)
                
                # Find dead bins (unused or very rarely used)
                dead_threshold = max(1, len(z_dim) // (self.num_bins * 5))  # Allow 20% of bins to be "rare"
                dead_bins = usage_counts < dead_threshold
                
                if dead_bins.sum() == 0:
                    continue  # No dead bins in this dimension
                
                print(f"  🔄 Reinitializing {dead_bins.sum().item()} dead bins in dimension {dim}...")
                
                # For each dead bin, find a good replacement
                for dead_bin_idx in torch.where(dead_bins)[0]:
                    # Find encoder outputs that are far from existing bins
                    distances_to_bins = torch.abs(z_dim.unsqueeze(1) - self.bins[dim].unsqueeze(0))
                    min_distances = distances_to_bins.min(dim=1)[0]
                    
                    # Find encoder outputs that are far from all existing bins
                    far_threshold = min_distances.mean() + min_distances.std()
                    far_indices = torch.where(min_distances > far_threshold)[0]
                    
                    if len(far_indices) > 0:
                        # Pick a random far encoder output
                        replacement_idx = far_indices[torch.randint(0, len(far_indices), (1,))]
                        self.bins.data[dim, dead_bin_idx] = z_dim[replacement_idx]
                    else:
                        # Fallback: use a random encoder output
                        replacement_idx = torch.randint(0, len(z_dim), (1,))
                        self.bins.data[dim, dead_bin_idx] = z_dim[replacement_idx]
            
            # Add small noise to break any remaining symmetries
            noise = torch.randn_like(self.bins.data) * 0.01
            self.bins.data += noise

    def update_ema(self, z_flat, bin_indices):
        """
        Update EMA statistics for bin maintenance
        """
        with torch.no_grad():
            # Update usage counts and means for each dimension
            for dim in range(self.latent_dim):
                z_dim = z_flat[:, dim]
                bin_indices_dim = bin_indices[:, dim]
                
                # Update usage counts
                usage_counts = torch.bincount(bin_indices_dim, minlength=self.num_bins)
                self.ema_bin_usage.data[dim].mul_(self.ema_decay).add_((1 - self.ema_decay) * usage_counts)
                
                # Update bin means
                for bin_idx in range(self.num_bins):
                    mask = (bin_indices_dim == bin_idx)
                    if mask.sum() > 0:
                        bin_mean = z_dim[mask].mean()
                        self.ema_bin_means.data[dim, bin_idx].mul_(self.ema_decay).add_((1 - self.ema_decay) * bin_mean)

    def get_codebook_usage(self, bin_indices):
        """
        Calculate codebook usage across all dimensions
        """
        batch_size, seq_len, num_patches, latent_dim = bin_indices.shape
        bin_indices_flat = bin_indices.view(-1, latent_dim)
        
        total_used_bins = 0
        total_bins = latent_dim * self.num_bins
        
        for dim in range(latent_dim):
            unique_bins = torch.unique(bin_indices_flat[:, dim])
            total_used_bins += len(unique_bins)
        
        return total_used_bins / total_bins

class Encoder(nn.Module):
    """ST-Transformer encoder that takes frames and outputs latent representations"""
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
    """ST-Transformer decoder that reconstructs frames from latents"""
    def __init__(self, frame_size=(64, 64), patch_size=16, embed_dim=512, num_heads=8,
                 hidden_dim=2048, num_blocks=6, latent_dim=64, dropout=0.1, height=64, width=64, channels=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, dropout, causal=True)
        
        # Latent embedding
        self.latent_embed = nn.Linear(latent_dim, embed_dim)
        
        num_patches = (height // patch_size) * (width // patch_size)
        # Frame reconstruction head goes from num_patches * embed_dim to height * width * channels
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
                 hidden_dim=2048, num_blocks=6, latent_dim=32, dropout=0.1, codebook_size=512, beta=1.0, ema_decay=0.99):
        super().__init__()
        self.encoder = Encoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, latent_dim, dropout)
        self.decoder = Decoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, latent_dim, dropout)
        self.vq = FiniteScalarQuantizer(latent_dim, codebook_size, beta, ema_decay)
        
    def forward(self, frames):
            
        # Encode frames to latent representations
        embeddings = self.encoder(frames)  # [batch_size, seq_len, num_patches, latent_dim]
        
        # Apply vector quantization
        fsq_loss, z_q, bin_indices = self.vq(embeddings)
        
        # Decode quantized latents back to frames
        x_hat = self.decoder(z_q)  # [batch_size, seq_len, channels, height, width]
        
        return x_hat, fsq_loss, bin_indices
