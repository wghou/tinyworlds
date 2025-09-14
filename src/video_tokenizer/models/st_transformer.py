import torch
import torch.nn as nn
from einops import rearrange
from src.video_tokenizer.models.positional_encoding import build_spatial_only_pe, sincos_time
from src.video_tokenizer.models.norms import AdaptiveNormalizer
import math
import torch.nn.functional as F

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

        # split embed dim into thirds for spatial x, spatial y, and temporal
        base_split = (embed_dim // 3) & ~1
        remaining_dim = embed_dim - base_split
        self.spatial_x_dim = (remaining_dim // 2) & ~1
        self.spatial_y_dim = remaining_dim - self.spatial_x_dim
        self.temporal_dim = base_split

        # TODO: simpler assertions
        assert (self.spatial_x_dim + self.spatial_y_dim + self.temporal_dim) == embed_dim, \
            f"Dimension mismatch: {self.spatial_x_dim} + {self.spatial_y_dim} + {self.temporal_dim} != {embed_dim}"
        assert self.spatial_x_dim % 2 == 0 and self.spatial_y_dim % 2 == 0 and self.temporal_dim % 2 == 0, \
            f"Embed dim x={self.spatial_x_dim}, y={self.spatial_y_dim}, t={self.temporal_dim}"

        pe_spatial = build_spatial_only_pe(self.frame_size, self.patch_size, self.embed_dim, device='cpu', dtype=torch.float32)  # [1,P,E]
        self.register_buffer("pos_spatial", pe_spatial, persistent=False)

        # conv projection for patches
        self.proj = nn.Conv2d(3 * self.patch_size * self.patch_size, self.embed_dim, 1)


    def forward(self, frames):
        B, T, C, H, W = frames.shape
        # go from frames to patches
        x = rearrange(frames, 'b t c (h p1) (w p2) -> (b t) (c p1 p2) h w', p1=self.patch_size, p2=self.patch_size) # [(B*T), 3*p*p, Hp, Wp]
        x = self.proj(x) # [(B*T), E, Hp, Wp]
        x = rearrange(x, '(b t) e hp wp -> b t (hp wp) e', b=B, t=T) # [B, T, P, E]
        x = x + self.pos_spatial.to(dtype=x.dtype, device=x.device) # [B, T, P, E]
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
        B, T, P, E = x.shape

        # Project to Q, K, V and reshape: [B, T, P, E] -> [(B*T), H, P, E/H] to work with torch compile attention
        q = rearrange(self.q_proj(x), 'B T P (H D) -> (B T) H P D', H=self.num_heads)
        k = rearrange(self.k_proj(x), 'B T P (H D) -> (B T) H P D', H=self.num_heads)
        v = rearrange(self.v_proj(x), 'B T P (H D) -> (B T) H P D', H=self.num_heads)

        k_t = k.transpose(-2, -1) # [(B*T), H, P, D, P]

        # attention(q, k, v) = softmax(qk^T / sqrt(d)) v
        scores = torch.matmul(q, k_t) / math.sqrt(self.head_dim) # [(B*T), H, P, P]
        attn_weights = F.softmax(scores, dim=-1) # [(B*T), H, P, P]
        attn_output = torch.matmul(attn_weights, v) # [(B*T), H, P, D]
        attn_output = rearrange(attn_output, '(B T) H P D -> B T P (H D)', B=B, T=T) # [B, T, P, E]

        # out proj to mix head information
        attn_out = self.out_proj(attn_output)  # [B, T, P, E]

        # residual and optionally conditioned norm
        out = self.norm(x + attn_out, conditioning) # [B, T, P, E]

        return out # [B, T, P, E]

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
        B, T, P, E = x.shape
        
        # Project to Q, K, V and reshape: [B, T, P, E] -> [(B*P), H, T, D] to work with torch compile attention
        q = rearrange(self.q_proj(x), 'b t p (h d) -> (b p) h t d', h=self.num_heads)
        k = rearrange(self.k_proj(x), 'b t p (h d) -> (b p) h t d', h=self.num_heads)
        v = rearrange(self.v_proj(x), 'b t p (h d) -> (b p) h t d', h=self.num_heads) # [B, P, H, T, D]

        k_t = k.transpose(-2, -1) # [(B*P), H, T, D, T]

        # attention(q, k, v) = softmax(qk^T / sqrt(d)) v: [(B*P), H, T, D] -> [(B*P), H, T, T]
        scores = torch.matmul(q, k_t) / math.sqrt(self.head_dim)

        # causal mask for each token t in seq, mask out all tokens to the right of t (after t)
        if self.causal:
            mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
            scores = scores.masked_fill(mask, -torch.inf) # [(B*P), H, T, T]

        attn_weights = F.softmax(scores, dim=-1) # [(B*P), H, T, T]
        attn_output = torch.matmul(attn_weights, v) # [(B*P), H, T, D]
        attn_output = rearrange(attn_output, '(b p) h t d -> b t p (h d)', b=B, p=P) # [B, T, P, E]

        # out proj to mix head information
        attn_out = self.out_proj(attn_output)  # [B, T, P, E]

        # residual and optionally conditioned norm
        out = self.norm(x + attn_out, conditioning) # [B, T, P, E]

        return out # [B, T, P, E]

class SwiGLUFFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, conditioning_dim=None):
        super().__init__()
        # TODO: add MoE
        h = math.floor(2 * hidden_dim / 3)
        self.w_v = nn.Linear(embed_dim, h)
        self.w_g = nn.Linear(embed_dim, h)
        self.w_o = nn.Linear(h, embed_dim)
        self.norm = AdaptiveNormalizer(embed_dim, conditioning_dim)

    def forward(self, x, conditioning=None):
        v = F.silu(self.w_v(x)) # [B, T, P, h]
        g = self.w_g(x) # [B, T, P, h]
        out = self.w_o(v * g) # [B, T, P, E]
        return self.norm(x + out, conditioning) # [B, T, P, E]

class STTransformerBlock(nn.Module):
    """ST-Transformer block with spatial attention, temporal attention, and feed-forward"""
    def __init__(self, embed_dim, num_heads, hidden_dim, causal=True, conditioning_dim=None):
        super().__init__()
        self.spatial_attn = SpatialAttention(embed_dim, num_heads, conditioning_dim)
        self.temporal_attn = TemporalAttention(embed_dim, num_heads, causal, conditioning_dim)
        self.ffn = SwiGLUFFN(embed_dim, hidden_dim, conditioning_dim)

    def forward(self, x, conditioning=None):
        # x: [B, T, P, E]
        # out: [B, T, P, E]
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
        B, T, P, E = x.shape
        tpe = sincos_time(T, self.temporal_dim, x.device, x.dtype)  # [T, E/3]

        # temporal PE (pad with 0s for first 2/3s spatial PE, last 1/3 temporal PE)
        tpe_padded = torch.cat([
            torch.zeros(T, self.spatial_dims, device=x.device, dtype=x.dtype),
            tpe
        ], dim=-1)  # [T, E]
        x = x + tpe_padded[None, :, None, :]  # [B,T,P,E]

        # apply transformer blocks
        for block in self.blocks:
            x = block(x, conditioning)
        return x
