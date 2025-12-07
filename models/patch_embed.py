import torch
import torch.nn as nn
from einops import rearrange
from models.positional_encoding import build_spatial_only_pe

class PatchEmbedding(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128):
        super().__init__()
        H, W = frame_size
        # frame size of images, i.e. 128*128
        self.frame_size = frame_size
        # patch size, i.e. 8*8
        self.patch_size = patch_size
        # the dim of embedding vector
        self.embed_dim = embed_dim
        # number of patches in H/W dim
        self.Hp, self.Wp = H // patch_size, W // patch_size
        # number of patches in total
        self.num_patches = self.Hp * self.Wp

        # split embed dim into thirds for spatial x, spatial y, and temporal
        base_split = (embed_dim // 3) & ~1
        remaining_dim = embed_dim - base_split
        self.spatial_x_dim = (remaining_dim // 2) & ~1
        self.spatial_y_dim = remaining_dim - self.spatial_x_dim
        self.temporal_dim = base_split

        # ensure the embed dim is split wholy into thirds and each third is even
        assert (self.spatial_x_dim + self.spatial_y_dim + self.temporal_dim) == embed_dim, \
            f"Dimension mismatch: {self.spatial_x_dim} + {self.spatial_y_dim} + {self.temporal_dim} != {embed_dim}"
        assert self.spatial_x_dim % 2 == 0 and self.spatial_y_dim % 2 == 0 and self.temporal_dim % 2 == 0, \
            f"Embed dim x={self.spatial_x_dim}, y={self.spatial_y_dim}, t={self.temporal_dim}"

        pe_spatial = build_spatial_only_pe(self.frame_size, self.patch_size, self.embed_dim, device='cpu', dtype=torch.float32)  # [1,P,E]
        self.register_buffer("pos_spatial", pe_spatial, persistent=False)

        # pixel patches to embeddings
        self.proj = nn.Conv2d(3 * self.patch_size * self.patch_size, self.embed_dim, 1)


    def forward(self, frames):
        #* frames: [B, T, C, H, W]
        B, T, C, H, W = frames.shape
        # go from frames to patches
        #* [B*T, C*p*p, Hp, Wp], where p is patch size, Hp, Wp is number of patches in H/W dim
        x = rearrange(frames, 'b t c (hp p1) (wp p2) -> (b t) (c p1 p2) hp wp', p1=self.patch_size, p2=self.patch_size) # [(B*T), 3*p*p, Hp, Wp]
        # project each patch into embedding vector
        x = self.proj(x) #* [(B*T), E, Hp, Wp], where E is dim of embedding vector
        x = rearrange(x, '(b t) e hp wp -> b t (hp wp) e', b=B, t=T) #* [B, T, P, E], where P is number of patches, E is dim of embedding vector.
        # add 2d spatial pos encoding (first 2/3 of embed dim)
        x = x + self.pos_spatial.to(dtype=x.dtype, device=x.device) #* [B, T, P, E]
        # return image embedding vectors
        return x
