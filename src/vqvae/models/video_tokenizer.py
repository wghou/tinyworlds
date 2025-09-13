import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from src.vqvae.models.positional_encoding import build_spatial_only_pe

def sincos_1d(L, D, device, dtype):
    # 1d sinusoidal position encoding where position i is encoded as:
    # PE[i, 2j]   = sin(i / 10000^(2j/D))  # even indices
    # PE[i, 2j+1] = cos(i / 10000^(2j/D))  # odd indices
    
    assert D % 2 == 0, "Encoding dimension must be even"
    
    # generate position indices [L, 1] and dimension indices [1, D/2]
    pos = rearrange(torch.arange(L, device=device, dtype=dtype), 'l -> l 1')     # [L,1]
    i   = rearrange(torch.arange(D // 2, device=device, dtype=dtype), 'd -> 1 d')# [1,D/2]

    # compute angular frequencies: 1/10000^(2i/D) for each dimension
    div = torch.pow(torch.tensor(10000.0, device=device, dtype=dtype), (2*i)/D)
    
    # compute angles: pos * freq for each position-dimension pair
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
    
    # Combine into 2D grid using repeat to expand across axes
    pe = torch.cat([
        repeat(pe_h, 'hp dh -> hp wp dh', wp=Wp),
        repeat(pe_w, 'wp dw -> hp wp dw', hp=Hp)
    ], dim=-1)                                     # [Hp, Wp, D]
    
    # Flatten spatial dimensions: [Hp,Wp,D] → [P,D]
    return rearrange(pe, 'hp wp d -> (hp wp) d') 

def sincos_time(S, D, device, dtype):
    # temporal PE using 1d sinusoidal PE across time
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

        # TODO: clean and functionalize this
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

        # conv projection for patches
        self.proj = nn.Conv2d(3 * self.patch_size * self.patch_size, self.embed_dim, 1)

        # Shared spatial-only PE (zeros in temporal tail)
        pe_spatial = build_spatial_only_pe(self.frame_size, self.patch_size, self.embed_dim, device='cpu', dtype=torch.float32)  # [1,P,E]
        self.register_buffer("pos_spatial", pe_spatial, persistent=False)

    def forward(self, frames):
        B, S, C, H, W = frames.shape
        # space-to-depth
        x = rearrange(frames, 'b s c (h p1) (w p2) -> (b s) (c p1 p2) h w', p1=self.patch_size, p2=self.patch_size) # [(B*S), 3*p*p, Hp, Wp]
        x = self.proj(x) # [(B*S), E, Hp, Wp]
        x = rearrange(x, '(b s) e hp wp -> b s (hp wp) e', b=B, s=S) # [B, S, P, E]
        x = x + self.pos_spatial.to(dtype=x.dtype, device=x.device) # [B, S, P, E]
        return x

class SpatialAttention(nn.Module):
    """Spatial attention over patches within each frame"""
    def __init__(self, embed_dim, num_heads, conditioning_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, f"embed dim must be divisible by num heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm = AdaptiveNormalizer(embed_dim, conditioning_dim)

    def forward(self, x, conditioning=None):
        B, S, P, E = x.shape

        # Project to Q, K, V and reshape: [B, S, P, E] -> [(B*S), H, P, E/H] to work with torch compile attention
        q = rearrange(self.q_proj(x), 'B S P (H D) -> (B S) H P D', H=self.num_heads)
        k = rearrange(self.k_proj(x), 'B S P (H D) -> (B S) H P D', H=self.num_heads)
        v = rearrange(self.v_proj(x), 'B S P (H D) -> (B S) H P D', H=self.num_heads)

        k_t = k.transpose(-2, -1) # [(B*S), H, P, D, P]

        # attention(q, k, v) = softmax(qk^T / sqrt(d)) v
        scores = torch.matmul(q, k_t) / math.sqrt(self.head_dim) # [(B*S), H, P, P]
        attn_weights = F.softmax(scores, dim=-1) # [(B*S), H, P, P]
        attn_output = torch.matmul(attn_weights, v) # [(B*S), H, P, D]
        attn_output = rearrange(attn_output, '(B S) H P D -> B S P (H D)', B=B, S=S) # [B, S, P, E]

        # out proj to mix head information
        attn_out = self.out_proj(attn_output)  # [B, S, P, E]

        # residual and optionally conditioned norm
        out = self.norm(x + attn_out, conditioning) # [B, S, P, E]

        return out # [B, S, P, E]

class TemporalAttention(nn.Module):
    """Temporal attention over time steps (optionally causally) for the same patch)"""
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
        
        self.norm = AdaptiveNormalizer(embed_dim, conditioning_dim)
        self.causal = causal
        
    def forward(self, x, conditioning=None):
        B, S, P, E = x.shape
        
        # Project to Q, K, V and reshape: [B, S, P, E] -> [(B*P), H, S, D] to work with torch compile attention
        q = rearrange(self.q_proj(x), 'b s p (h d) -> (b p) h s d', h=self.num_heads)
        k = rearrange(self.k_proj(x), 'b s p (h d) -> (b p) h s d', h=self.num_heads)
        v = rearrange(self.v_proj(x), 'b s p (h d) -> (b p) h s d', h=self.num_heads) # [B, P, H, S, D]

        k_t = k.transpose(-2, -1) # [(B*P), H, S, D, S]

        # attention(q, k, v) = softmax(qk^T / sqrt(d)) v: [(B*P), H, S, D] -> [(B*P), H, S, S]
        scores = torch.matmul(q, k_t) / math.sqrt(self.head_dim)

        # causal mask for each token t in seq, mask out all tokens to the right of t (after t)
        if self.causal:
            mask = torch.triu(torch.ones(S, S), diagonal=1).bool().to(x.device)
            scores = scores.masked_fill(mask, -torch.inf) # [(B*P), H, S, S]

        attn_weights = F.softmax(scores, dim=-1) # [(B*P), H, S, S]
        attn_output = torch.matmul(attn_weights, v) # [(B*P), H, S, D]
        attn_output = rearrange(attn_output, '(b p) h s d -> b s p (h d)', b=B, p=P) # [B, S, P, E]

        # out proj to mix head information
        attn_out = self.out_proj(attn_output)  # [B, S, P, E]

        # residual and optionally conditioned norm
        out = self.norm(x + attn_out, conditioning) # [B, S, P, E]

        return out # [B, S, P, E]

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, conditioning_dim=None):
        super().__init__()
        # TODO: add MoE
        h = math.floor(2 * hidden_dim / 3)
        self.w_v = nn.Linear(embed_dim, h)
        self.w_g = nn.Linear(embed_dim, h)
        self.w_o = nn.Linear(h, embed_dim)
        self.norm = AdaptiveNormalizer(embed_dim, conditioning_dim)

    def forward(self, x, conditioning=None):
        v = F.silu(self.w_v(x)) # [B, S, P, h]
        g = self.w_g(x) # [B, S, P, h]
        out = self.w_o(v * g) # [B, S, P, E]
        return self.norm(x + out, conditioning) # [B, S, P, E]

class RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))  # learned scale for rmsnorm (γ)

    def forward(self, x):
        # RMSNorm(x) = x / sqrt(mean(x^2) + eps)
        mean_squared = torch.mean(x**2, dim=-1, keepdim=True)
        # rsqrt is reciprocal sqrt (faster and numerically stable vs dividing by sqrt)
        rms_normed = x * torch.rsqrt(mean_squared + self.eps)
        return rms_normed * self.weight

class SimpleLayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # Only center at 0 and make stddev 1 (for film)
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, unbiased=False, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps)

# TODO: move basic building blocks into a models.py or something better named
class AdaptiveNormalizer(nn.Module):
    # Conditioned Feature-wise Linear Modulation, optionally unconditioned simple RMSNorm
    def __init__(self, embed_dim, conditioning_dim=None):
        super().__init__()
        self.ln = None
        self.rms = None
        self.to_gamma_beta = None
        if conditioning_dim is None:
            # Prefer RMSNorm when unconditioned
            self.rms = RMSNorm(embed_dim)
        else:
            self.ln = SimpleLayerNorm(embed_dim)
            self.to_gamma_beta = nn.Sequential(
                nn.SiLU(),
                nn.Linear(conditioning_dim, 2 * embed_dim)
            )
            # for lam and dynamics we initialize with small non-zero weights 
            # so conditioning is active from step 1 (this helps prevent ignoring conditioning)
            nn.init.normal_(self.to_gamma_beta[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.to_gamma_beta[-1].bias)

    def forward(self, x, conditioning=None):
        # x: [B, S, P, E]
        # conditioning: [B, S, C] or [B, S - 1, C]
        if self.to_gamma_beta is None or conditioning is None:
            normed = self.rms(x) if self.rms is not None else self.ln(x)
            return normed

        x = self.ln(x)
        B, S, P, E = x.shape
        out = self.to_gamma_beta(conditioning) # [B, S, 2 * E]
        out = repeat(out, 'b s twoe -> b s p twoe', p=P) # [B, S, P, 2 * E]
        gamma, beta = out.chunk(2, dim=-1) # each [B, S, P, E]

        # preppend with zeros since a_t-1 should impact z_t (and a_0 is used for z_1 etc)
        if gamma.shape[1] == x.shape[1] - 1 and beta.shape[1] == x.shape[1] - 1:
            gamma = torch.cat([torch.zeros_like(gamma[:, :1]), gamma], dim=1)
            beta = torch.cat([torch.zeros_like(beta[:, :1]), beta], dim=1)

        assert gamma.shape[1] == x.shape[1], f"gamma shape: {gamma.shape} != x shape: {x.shape}"
        assert beta.shape[1] == x.shape[1], f"beta shape: {beta.shape} != x shape: {x.shape}"
        x = x * (1 + gamma) + beta # [B, S, P, E]
        return x

class STTransformerBlock(nn.Module):
    """ST-Transformer block with spatial attention, temporal attention, and feed-forward"""
    def __init__(self, embed_dim, num_heads, hidden_dim, causal=True, conditioning_dim=None):
        super().__init__()
        self.spatial_attn = SpatialAttention(embed_dim, num_heads, conditioning_dim)
        self.temporal_attn = TemporalAttention(embed_dim, num_heads, causal, conditioning_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim, conditioning_dim)

    def forward(self, x, conditioning=None):
        # x: [batch_size, seq_len, num_patches, embed_dim]
        # out: [batch_size, seq_len, num_patches, embed_dim]
        x = self.spatial_attn(x, conditioning)
        x = self.temporal_attn(x, conditioning)
        x = self.ffn(x, conditioning)
        return x

class STTransformer(nn.Module):
    """ST-Transformer with multiple blocks and temporal position encoding"""
    def __init__(self, embed_dim, num_heads, hidden_dim, num_blocks, causal=True, conditioning_dim=None):
        super().__init__()
        # Split dimensions, ensuring temporal is even
        self.temporal_dim = (embed_dim // 3) & ~1  # Round down to even number
        self.spatial_dims = embed_dim - self.temporal_dim  # Rest goes to spatial
        
        self.blocks = nn.ModuleList([
            STTransformerBlock(embed_dim, num_heads, hidden_dim, causal, conditioning_dim)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x, conditioning=None):
        # x: [batch_size, seq_len, num_patches, embed_dim]
        B, S, P, E = x.shape
        tpe = sincos_time(S, self.temporal_dim, x.device, x.dtype)  # [S, E/3]

        # temporal PE (pad with 0s for first 2/3s spatial PE, last 1/3 temporal PE)
        tpe_padded = torch.cat([
            torch.zeros(S, self.spatial_dims, device=x.device, dtype=x.dtype),
            tpe
        ], dim=-1)  # [S, E]
        x = x + tpe_padded[None, :, None, :]  # [B,S,P,E]

        # apply transformer blocks
        for block in self.blocks:
            x = block(x, conditioning)
        return x

class FiniteScalarQuantizer(nn.Module):
    # Finite Scalar Quantization from https://arxiv.org/pdf/2309.15505
    # quantizes each dimension independently by bounding to 0, num_bins then rounding to nearest integer
    # prevents token collapse and no auxiliary losses necessary 

    def __init__(self, latent_dim=5, num_bins=4):
        super().__init__()
        self.num_bins = num_bins
        self.levels_np = torch.tensor(latent_dim * [num_bins])
        self.codebook_size = num_bins**latent_dim
        # Register basis as a buffer so it gets moved to the correct device
        self.register_buffer('basis', (num_bins**torch.arange(latent_dim, dtype=torch.long)))

    def scale_and_shift(self, z):
        # Scale and shift z from [-1, 1] to [0, num_bins - 1]
        return 0.5 * (z + 1) * (self.num_bins - 1)
    
    def unscale_and_unshift(self, z):
        # Unscale and unshift z from [0, num_bins - 1] to [-1, 1]
        return 2 * z / (self.num_bins - 1) - 1
        
    def forward(self, z):
        # z: [batch_size, seq_len, num_patches, latent_dim]
    
        # apply 0.5 * (tanh(z) + 1) to go from z range to [0, num_bins - 1]
        tanh_z = torch.tanh(z)
        bounded_z = self.scale_and_shift(tanh_z)

        # round to nearest integer
        rounded_z = torch.round(bounded_z)

        # stopgrad for straight-through gradient bypassing quantization
        quantized_z = bounded_z + (rounded_z - bounded_z).detach()

        # normalize back to [-1, 1]
        quantized_z = self.unscale_and_unshift(quantized_z)

        return quantized_z

    def get_codebook_usage(self, quantized_z):
        unique_bins = torch.unique(quantized_z).shape[0]
        return unique_bins / self.num_bins
    
    def get_indices_from_latents(self, latents, dim=-1):
        # to get fsq indices, for each dimension, we get the index and add it (so multiply by L then sum)
        # for each dimension of each latent, get sum of (value * L^current_dim) along latent dim which is the index of that latent in the codebook
        # codebook size = L^latent_dim
        # latents: [B, S, P, L]

        # go from [-1, 1] to [0, num_bins - 1] in each dimension
        digits = torch.round(self.scale_and_shift(latents)).clamp(0, self.num_bins-1)
        
        # get indices for each latent by summing (value * L^current_dim_idx) along latent dim
        # basis is [L^0, L^1, ..., L^(L-1)]
        indices = torch.sum(digits * self.basis.to(latents.device), dim=dim).long() # [B, S, P]
        return indices
    
    def get_latents_from_indices(self, indices, dim=-1):
        # indices: [batch_size, seq_len, num_patches]
        # recover each entry of latent in range [0, num_bins - 1] by repeatedly dividing by L^current_dim and taking mod
        # basis is [L^0, L^1, ..., L^(L-1)]
        digits = (rearrange(indices, 'b s p -> b s p 1') // self.basis) % self.num_bins   # [B, S, P, L]

        # go from [0, num_bins - 1] to [-1, 1] in each dimension
        latents = self.unscale_and_unshift(digits) # [B, S, P, L]
        return latents

class Encoder(nn.Module):
    """ST-Transformer encoder that takes frames and outputs latent representations"""
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, num_heads=8, 
                 hidden_dim=256, num_blocks=4, latent_dim=5):
        super().__init__()
        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True)
        self.latent_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, latent_dim)
        )
        
    def forward(self, frames):
        # frames: [batch_size, seq_len, channels, height, width]
        # frames to patch embeddings, pass through transformer, project to latent dim
        embeddings = self.patch_embed(frames)  # [B, S, P, E]
        transformed = self.transformer(embeddings) # [B, S, P, E]
        predicted_latents = self.latent_head(transformed) # [B, S, P, L]
        return predicted_latents

class PixelShuffleFrameHead(nn.Module):
    def __init__(self, embed_dim, patch_size=8, channels=3, H=128, W=128):
        super().__init__()
        self.patch_size = patch_size
        self.Hp, self.Wp = H // patch_size, W // patch_size
        self.to_pixels = nn.Conv2d(embed_dim, channels * (patch_size ** 2), kernel_size=1)

    def forward(self, tokens):  # [B,S,P,E]
        B, S, P, E = tokens.shape
        x = rearrange(tokens, 'b s (hp wp) e -> (b s) e hp wp', hp=self.Hp, wp=self.Wp) # [(B*S), E, Hp, Wp]
        x = self.to_pixels(x)                  # [(B*S), C*p^2, Hp, Wp]
        x = rearrange(x, '(b s) (c p1 p2) hp wp -> b s c (hp p1) (wp p2)', p1=self.patch_size, p2=self.patch_size, b=B, s=S) # [B, S, C, H, W]
        return x

class Decoder(nn.Module):
    """ST-Transformer decoder that reconstructs frames from latents"""
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, num_heads=8,
                 hidden_dim=256, num_blocks=4, latent_dim=5):
        super().__init__()
        H, W = frame_size
        self.patch_size = patch_size
        self.Hp, self.Wp = H // patch_size, W // patch_size
        self.num_patches = self.Hp * self.Wp
        
        # Transformer and embeddings
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True)
        self.latent_embed = nn.Linear(latent_dim, embed_dim)
        
        # Shared spatial-only PE (zeros in temporal tail)
        pe_spatial_dec = build_spatial_only_pe((H, W), self.patch_size, embed_dim, device='cpu', dtype=torch.float32)  # [1,P,E]
        self.register_buffer("pos_spatial_dec", pe_spatial_dec, persistent=False)
        
        # Efficient patch-wise frame reconstruction head
        self.frame_head = PixelShuffleFrameHead(embed_dim, patch_size=patch_size, channels=3, H=H, W=W)
        
    def forward(self, latents):
        """
        Args:
            latents: [batch_size, seq_len, num_patches, latent_dim]
            training: Whether in training mode (for masking)
        Returns:
            pred_frames: [batch_size, seq_len, channels, height, width]
        """
        # Embed latents and add spatial PE
        embedding = self.latent_embed(latents)  # [B, S, P, E]
        embedding = embedding + self.pos_spatial_dec.to(dtype=embedding.dtype, device=embedding.device)
    
        # Apply transformer (temporal PE added inside)
        embedding = self.transformer(embedding)  # [B, S, P, E]
        
        # Reconstruct frames using patch-wise head
        frames_out = self.frame_head(embedding)  # [B, S, C, H, W]

        return frames_out

class Video_Tokenizer(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, num_heads=8,
                 hidden_dim=256, num_blocks=4, latent_dim=3, num_bins=4):
        super().__init__()
        self.encoder = Encoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, latent_dim)
        self.decoder = Decoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, latent_dim)
        self.quantizer = FiniteScalarQuantizer(latent_dim, num_bins)
        self.codebook_size = num_bins**latent_dim

    def forward(self, frames):
        # Encode frames to latent representations, quantize, and decode back to frames
        embeddings = self.encoder(frames)  # [B, S, P, L]
        quantized_z = self.quantizer(embeddings)
        x_hat = self.decoder(quantized_z)  # [B, S, C, H, W]
        return x_hat
    
    def tokenize(self, frames):
        # encode frames to latent representations, quantize, and return indices
        embeddings = self.encoder(frames)  # [B, S, P, L]
        quantized_z = self.quantizer(embeddings)
        indices = self.quantizer.get_indices_from_latents(quantized_z, dim=-1)
        return indices

    def detokenize(self, quantized_z):
        # decode quantized latents back to frames
        x_hat = self.decoder(quantized_z)  # [B, S, C, H, W]
        return x_hat
