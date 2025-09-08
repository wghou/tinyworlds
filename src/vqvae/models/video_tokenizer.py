import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

def sincos_1d(L, D, device, dtype):
    """
    Compute 1D sinusoidal position encodings.
    
    Each position i is encoded as:
    PE[i, 2j]   = sin(i / 10000^(2j/D))  # even indices
    PE[i, 2j+1] = cos(i / 10000^(2j/D))  # odd indices
    
    This creates a unique pattern for each position that:
    - Varies smoothly with position
    - Has different frequencies across dimensions
    - Allows relative position to be attended to via dot products
    
    Args:
        L: Length of sequence (number of positions)
        D: Dimension of encoding (must be even)
        device: torch device for tensors
        dtype: data type for tensors
    
    Returns:
        pe: Position encodings [L, D] where each row is a position's encoding
    """
    assert D % 2 == 0, "Encoding dimension must be even"
    
    # Generate position indices [L, 1] and dimension indices [1, D/2]
    pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)     # [L,1]
    i   = torch.arange(D // 2, device=device, dtype=dtype).unsqueeze(0)# [1,D/2]
    
    # Compute angular frequencies: 1/10000^(2i/D) for each dimension
    div = torch.pow(torch.tensor(10000.0, device=device, dtype=dtype), (2*i)/D)
    
    # Compute angles: pos * freq for each position-dimension pair
    angles = pos / div  # [L, D/2] = [L,1] * [1,D/2] (broadcasting)
    
    # Fill alternating sin/cos into output
    pe = torch.zeros(L, D, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(angles)  # even indices
    pe[:, 1::2] = torch.cos(angles)  # odd indices
    return pe

def sincos_2d(Hp, Wp, D, device, dtype):
    """
    Compute 2D sinusoidal position encodings for a grid of patches.
    
    The encoding combines:
    - Height position encoding using D/2 dimensions
    - Width position encoding using D/2 dimensions
    This gives each grid position (h,w) a unique encoding that:
    - Captures both vertical and horizontal position
    - Preserves spatial relationships in both dimensions
    
    Args:
        Hp: Height in patches
        Wp: Width in patches
        D: Total encoding dimension (must be even, split between H and W)
        device: torch device for tensors
        dtype: data type for tensors
    
    Returns:
        pe: Position encodings [P, D] where P = Hp*Wp is total patches,
            each row is the encoding for patch (h,w) flattened to 1D
    """
    assert D % 2 == 0, "Encoding dimension must be even to split between H and W"
    
    # Split dimensions between height and width encodings
    Dh = Dw = D // 2
    
    # Get 1D encodings for height and width positions
    pe_h = sincos_1d(Hp, Dh, device, dtype)        # [Hp, Dh]
    pe_w = sincos_1d(Wp, Dw, device, dtype)        # [Wp, Dw]
    
    # Combine into 2D grid:
    # 1. Expand height encodings across width: [Hp,1,Dh] → [Hp,Wp,Dh]
    # 2. Expand width encodings across height: [1,Wp,Dw] → [Hp,Wp,Dw]
    # 3. Concatenate along feature dimension: [Hp,Wp,D]
    pe = torch.cat([
        pe_h[:, None, :].expand(Hp, Wp, Dh),
        pe_w[None, :, :].expand(Hp, Wp, Dw)
    ], dim=-1)                                     # [Hp, Wp, D]
    
    # Flatten spatial dimensions: [Hp,Wp,D] → [P,D]
    return pe.reshape(Hp * Wp, D)                  # [P, D]

def sincos_time(S, D, device, dtype):
    """
    Compute temporal position encodings for sequence steps.
    
    This is identical to 1D position encoding but used specifically
    for encoding time steps in sequences. Each time step gets a unique
    encoding that captures temporal order and relative distances.
    
    Args:
        S: Sequence length (number of time steps)
        D: Encoding dimension (must be even)
        device: torch device for tensors
        dtype: data type for tensors
    
    Returns:
        pe: Position encodings [S, D] for each time step
    """
    return sincos_1d(S, D, device, dtype)  # reuse the same 1D builder

class PatchEmbedding(nn.Module):
    """Convert frames to patch embeddings for ST-Transformer"""
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128):
        super().__init__()
        H, W = frame_size
        self.frame_size = frame_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.Hp, self.Wp = H // patch_size, W // patch_size
        self.num_patches = self.Hp * self.Wp
        
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
        
        # Linear projection for patches
        self.proj = nn.Linear(3 * patch_size * patch_size, embed_dim)
        
        # Generate separate spatial x and y position encodings
        pe_x = sincos_1d(self.Wp, self.spatial_x_dim, device='cpu', dtype=torch.float32)  # [Wp, D/3+]
        pe_y = sincos_1d(self.Hp, self.spatial_y_dim, device='cpu', dtype=torch.float32)  # [Hp, D/3+]
        
        # Expand to 2D grid
        pe_x = pe_x.unsqueeze(0).expand(self.Hp, self.Wp, -1)  # [Hp, Wp, D/3+]
        pe_y = pe_y.unsqueeze(1).expand(self.Hp, self.Wp, -1)  # [Hp, Wp, D/3+]
        
        # Combine and pad with zeros for temporal part
        pe_spatial = torch.cat([
            pe_x,  # First third+: x position
            pe_y,  # Second third+: y position
            torch.zeros(self.Hp, self.Wp, self.temporal_dim, device='cpu', dtype=torch.float32)  # Last third: temporal
        ], dim=-1)  # [Hp, Wp, D]
        
        # Flatten spatial dimensions
        pe_spatial = pe_spatial.reshape(self.num_patches, embed_dim)  # [P, D]
        self.register_buffer("pos_spatial", pe_spatial[None, :, :], persistent=False)  # [1,P,D]
        
    def forward(self, frames):
        """
        Args:
            frames: [batch_size, seq_len, channels, height, width]
        Returns:
            embeddings: [batch_size, seq_len, num_patches, embed_dim]
        """
        batch_size, seq_len, C, H, W = frames.shape
        
        # Reshape to patches using einops
        x = rearrange(frames, 'b s c (h p1) (w p2) -> (b s) (h w) (c p1 p2)', 
                      p1=self.patch_size, p2=self.patch_size)
        
        # Project to embeddings
        x = self.proj(x)  # [(b*s), P, D]
        
        # Add position embeddings (spatial x,y only - temporal added in transformer)
        x = x + self.pos_spatial.to(dtype=x.dtype, device=x.device)
        
        # Reshape back to sequence
        x = rearrange(x, '(b s) p e -> b s p e', b=batch_size, s=seq_len)
        
        return x

class SpatialAttention(nn.Module):
    """Spatial attention layer - attends over patches within each frame"""
    def __init__(self, embed_dim, num_heads, conditioning_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.norm = AdaLN(embed_dim, conditioning_dim)
        
    def forward(self, x, conditioning=None):
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
        
        # Apply attention to values: [batch, seq, num_heads, num_patches, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        attn_output = rearrange(attn_output, 'b s n p d -> b s p (n d)')
        
        # Final projection TODO: do we need this?
        attn_out = self.out_proj(attn_output)  # [batch_size, seq_len, num_patches, embed_dim]
        
        # Add residual and normalize
        out = self.norm(x + attn_out, conditioning)
        
        return out

class TemporalAttention(nn.Module):
    """Temporal attention layer - attends over time steps for each patch"""
    def __init__(self, embed_dim, num_heads, causal=True, conditioning_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.norm = AdaLN(embed_dim, conditioning_dim)
        self.causal = causal
        
    def forward(self, x, conditioning=None):
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
        
        # Apply attention to values: [batch, num_patches, num_heads, seq, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        attn_output = rearrange(attn_output, 'b p n s d -> b s p (n d)') # [batch_size, seq_len, num_patches, embed_dim]
        
        # Final projection TODO: do we need this?
        attn_out = self.out_proj(attn_output)  # [batch_size, seq_len, num_patches, embed_dim]
        
        # Add residual and normalize
        out = self.norm(x + attn_out, conditioning)
        
        return out

class FeedForward(nn.Module):
    """Feed-forward network"""
    def __init__(self, embed_dim, hidden_dim, conditioning_dim=None):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.norm = AdaLN(embed_dim, conditioning_dim)

    def forward(self, x, conditioning=None):
        """
        Args:
            x: [batch_size, seq_len, num_patches, embed_dim]
        Returns:
            out: [batch_size, seq_len, num_patches, embed_dim]
        """
        batch_size, seq_len, num_patches, embed_dim = x.shape
        
        # Apply feed-forward
        out = self.linear1(x)
        out = F.relu(out) # TODO: try SiLU or SwiGLU
        out = self.linear2(out)
        
        # Add residual and normalize
        out = self.norm(x + out, conditioning)
        
        return out

class AdaLN(nn.Module):
    # Adaptive Layer Normalization, optionally unconditioned simple LayerNorm
    def __init__(self, embed_dim, conditioning_dim=None):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.to_gamma_beta = None
        if conditioning_dim is not None:
            self.to_gamma_beta = nn.Sequential(
                nn.SiLU(),
                    nn.Linear(conditioning_dim, 2 * embed_dim)
                )
            nn.init.zeros_(self.to_gamma_beta[-1].weight)
            nn.init.zeros_(self.to_gamma_beta[-1].bias)

    def forward(self, x, conditioning=None):
        # x: [B, S, P, E]
        # conditioning: [B, S or S-1, C]
        x = self.ln(x) # [B, S, P, E]

        # regular unconditioned layernorm
        # TODO: maybe use RMSNorm instead
        if self.to_gamma_beta is None or conditioning is None:
            return x

        B, S, P, E = x.shape
        out = self.to_gamma_beta(conditioning) # [B, S or S-1, 2 * E]

        out = repeat(out, 'b s twoe -> b s p twoe', p=P) # [B, S or S-1, P, 2 * E]

        gamma, beta = out.chunk(2, dim=-1) # each [B, S or S-1, P, E]

        # if x is len seq and gamma/beta are len seq-1, pad gamma/beta at the beginning with 0
        # so we do action conditioning on the "next frames"
        # TODO: confirm this is working properly
        if gamma.shape[1] == S - 1 and beta.shape[1] == S - 1:
            gamma = torch.cat([torch.zeros_like(gamma[:, :1]), gamma], dim=1)
            beta = torch.cat([torch.zeros_like(beta[:, :1]), beta], dim=1)
        
        x = x * (1 + gamma) + beta # [B, S, P, E]

        return x

class STTransformerBlock(nn.Module):
    """ST-Transformer block with spatial attention, temporal attention, and feed-forward"""
    def __init__(self, embed_dim, num_heads, hidden_dim, causal=True, conditioning_dim=None):
        super().__init__()
        # TODO: implement conditioning
        self.spatial_attn = SpatialAttention(embed_dim, num_heads, conditioning_dim)
        self.temporal_attn = TemporalAttention(embed_dim, num_heads, causal, conditioning_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim, conditioning_dim)

    def forward(self, x, conditioning=None):
        """
        Args:
            x: [batch_size, seq_len, num_patches, embed_dim]
        Returns:
            out: [batch_size, seq_len, num_patches, embed_dim]
        """
        # Spatial attention - attends over patches within each frame
        x = self.spatial_attn(x, conditioning)
        
        # Temporal attention - attends over time steps for each patch
        x = self.temporal_attn(x, conditioning)
        
        # Feed-forward
        x = self.ffn(x, conditioning)
        
        return x

class STTransformer(nn.Module):
    """ST-Transformer with multiple blocks"""
    def __init__(self, embed_dim, num_heads, hidden_dim, num_blocks, causal=True, conditioning_dim=None):
        super().__init__()
        # TODO: add conditioning based on actions (or technically any tensor)
        # Split dimensions, ensuring temporal is even
        self.temporal_dim = (embed_dim // 3) & ~1  # Round down to even number
        self.spatial_dims = embed_dim - self.temporal_dim  # Rest goes to spatial
        
        self.blocks = nn.ModuleList([
            STTransformerBlock(embed_dim, num_heads, hidden_dim, causal, conditioning_dim)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x, conditioning=None):
        """
        Args:
            x: [batch_size, seq_len, num_patches, embed_dim]
        Returns:
            out: [batch_size, seq_len, num_patches, embed_dim]
        """
        # Add temporal position encoding to last third of embedding
        B, S, P, E = x.shape
        tpe = sincos_time(S, self.temporal_dim, x.device, x.dtype)  # [S, D/3]
        
        # Create padded temporal encoding
        tpe_padded = torch.cat([
            torch.zeros(S, self.spatial_dims, device=x.device, dtype=x.dtype),  # Zeros for spatial
            tpe  # Temporal encoding in last third
        ], dim=-1)  # [S, D]
        
        # Add temporal encoding
        x = x + tpe_padded[None, :, None, :]  # broadcast [1,S,1,D] over [B,S,P,D]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, conditioning)
        return x

class FiniteScalarQuantizer(nn.Module):
    """
    Finite Scalar Quantization from https://arxiv.org/pdf/2309.15505
    Quantizes each dimension independently by bounding to 0, num_bins then rounding to nearest integer
    Prevents token collapse and no auxiliary losses necessary 
    """
    def __init__(self, latent_dim=6, num_bins=4):
        super().__init__()
        self.num_bins = num_bins
        self.levels_np = torch.tensor(latent_dim * [num_bins])
        self.codebook_size = num_bins**latent_dim
        # Register basis as a buffer so it gets moved to the correct device
        self.register_buffer('basis', num_bins**torch.arange(latent_dim))

    def scale_and_shift(self, z):
        # Scale and shift z from [-1, 1] to [0, num_bins - 1]
        return 0.5 * (z + 1) * (self.num_bins - 1)
    
    def unscale_and_unshift(self, z):
        # Unscale and unshift z from [0, num_bins - 1] to [-1, 1]
        return 2 * z / (self.num_bins - 1) - 1
        
    def forward(self, z):
        # z: [batch_size, seq_len, num_patches, latent_dim]
        tanh_z = torch.tanh(z)

        # apply 0.5 * (tanh(z) + 1) to go from z dimension to [0, num_bins - 1]
        bounded_z = self.scale_and_shift(tanh_z)

        # round to nearest integer
        rounded_z = torch.round(bounded_z)

        # apply stopgrad
        quantized_z = bounded_z + (rounded_z - bounded_z).detach()

        # normalize back to [-1, 1]
        quantized_z = self.unscale_and_unshift(quantized_z)

        return quantized_z

    def get_codebook_usage(self, quantized_z):
        unique_bins = torch.unique(quantized_z).shape[0]
        return unique_bins / self.num_bins
    
    def get_indices_from_latents(self, latents, dim=-1):
        # to get fsq indices, for each dimension, get the index and add it (so multiply by L then sum)
        # for each dimension of each latent, get sum of (value * L^current_dim) along latent dim which is the index of that latent in the codebook
        # codebook size = L^latent_dim
        # latents: [batch_size, seq_len, num_patches, latent_dim]

        # go from [-1, 1] to [0, num_bins - 1] in each dimension
        digits = torch.round(self.scale_and_shift(latents)).clamp(0, self.num_bins-1)
        
        # get indices for each latent by summing (value * L^current_dim_idx) along latent dim
        indices = torch.sum(digits * self.basis.to(latents.device), dim=dim).long() # [batch_size, seq_len, num_patches]
        return indices

    def get_latents_from_indices(self, indices, dim=-1):
        # indices: [batch_size, seq_len, num_patches]
        # recover each entry of latent in range [0, num_bins - 1] by repeatedly dividing by L^current_dim and taking mod
        digits = (indices.unsqueeze(-1) // self.basis) % self.num_bins   # [batch_size, seq_len, num_patches, latent_dim]

        # go from [0, num_bins - 1] to [-1, 1] in each dimension
        latents = self.unscale_and_unshift(digits) # [batch_size, seq_len, num_patches, latent_dim]
        return latents

class Encoder(nn.Module):
    """ST-Transformer encoder that takes frames and outputs latent representations"""
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, num_heads=8, 
                 hidden_dim=256, num_blocks=4, latent_dim=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True)
        
        # Latent prediction head
        self.latent_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, latent_dim)
        )
        
    def forward(self, frames):
        # frames: [batch_size, seq_len, channels, height, width]
        # Convert frames to patch embeddings
        embeddings = self.patch_embed(frames)  # [batch_size, seq_len, num_patches, embed_dim]
        
        # Apply ST-Transformer
        transformed = self.transformer(embeddings) # [batch_size, seq_len, num_patches, embed_dim]

        predicted_latents = self.latent_head(transformed) # [batch_size, seq_len, num_patches, latent_dim]
        
        return predicted_latents

class PatchWiseFrameHead(nn.Module):
    """Efficient patch-wise frame reconstruction head"""
    def __init__(self, embed_dim, patch_size=4, out_ch=3):
        super().__init__()
        self.p = patch_size
        self.out_ch = out_ch
        self.proj = nn.Linear(embed_dim, out_ch * patch_size * patch_size)

    def forward(self, x, H, W):
        # x: [B, S, P, E], P = (H/p)*(W/p)
        B, S, P, E = x.shape
        p = self.p
        # per-patch pixels
        y = self.proj(x)                            # [B,S,P,C*p*p]
        # fold patches back
        Hp, Wp = H // p, W // p
        y = y.view(B, S, Hp, Wp, self.out_ch, p, p) # [B,S,Hp,Wp,C,p,p]
        y = y.permute(0,1,4,2,5,3,6).contiguous()   # [B,S,C,Hp,p,Wp,p]
        y = y.view(B, S, self.out_ch, H, W)         # [B,S,C,H,W]
        return y

class Decoder(nn.Module):
    """ST-Transformer decoder that reconstructs frames from latents"""
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, num_heads=8,
                 hidden_dim=256, num_blocks=4, latent_dim=3):
        super().__init__()
        H, W = frame_size
        self.height = H
        self.width = W
        self.patch_size = patch_size
        self.Hp, self.Wp = H // patch_size, W // patch_size
        self.num_patches = self.Hp * self.Wp
        
        # Transformer and embeddings
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True)
        self.latent_embed = nn.Linear(latent_dim, embed_dim)
        
        # 2D spatial PE for the decoder path
        pe_spatial = sincos_2d(self.Hp, self.Wp, embed_dim, device='cpu', dtype=torch.float32)
        self.register_buffer("pos_spatial_dec", pe_spatial[None, :, :], persistent=False)
        
        # Efficient patch-wise frame reconstruction head
        self.frame_head = PatchWiseFrameHead(embed_dim, patch_size=patch_size, out_ch=3)
        
    def forward(self, latents):
        """
        Args:
            latents: [batch_size, seq_len, num_patches, latent_dim]
            training: Whether in training mode (for masking)
        Returns:
            pred_frames: [batch_size, seq_len, channels, height, width]
        """
        # Embed latents and add spatial PE
        embedding = self.latent_embed(latents)  # [batch_size, seq_len, num_patches, embed_dim]
        embedding = embedding + self.pos_spatial_dec.to(embedding.device, embedding.dtype)
    
        # Apply transformer (temporal PE added inside)
        embedding = self.transformer(embedding)  # [batch_size, seq_len, num_patches, embed_dim]
        
        # Reconstruct frames using patch-wise head
        frames_out = self.frame_head(embedding, self.height, self.width)  # [batch_size, seq_len, channels, height, width]

        return frames_out

class Video_Tokenizer(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, num_heads=8,
                 hidden_dim=256, num_blocks=4, latent_dim=3, num_bins=4):
        super().__init__()
        self.encoder = Encoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, latent_dim)
        self.decoder = Decoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, latent_dim)
        self.vq = FiniteScalarQuantizer(latent_dim, num_bins)
        self.codebook_size = num_bins**latent_dim

    def forward(self, frames):
        # Encode frames to latent representations
        embeddings = self.encoder(frames)  # [batch_size, seq_len, num_patches, latent_dim]

        # Apply vector quantization
        quantized_z = self.vq(embeddings)

        # Decode quantized latents back to frames
        x_hat = self.decoder(quantized_z)  # [batch_size, seq_len, channels, height, width]

        return x_hat
    
    def tokenize(self, frames):
        # Encode frames to latent representations
        embeddings = self.encoder(frames)  # [batch_size, seq_len, num_patches, latent_dim]

        # Apply vector quantization
        quantized_z = self.vq(embeddings)

        indices = self.vq.get_indices_from_latents(quantized_z, dim=-1)

        return indices

    def detokenize(self, quantized_z):
        # Decode quantized latents back to frames
        x_hat = self.decoder(quantized_z)  # [batch_size, seq_len, channels, height, width]

        return x_hat
