import torch
from einops import rearrange, repeat


def build_spatial_only_pe(frame_size, patch_size, embed_dim, device='cpu', dtype=torch.float32):
    """
    Build spatial-only positional encodings for a grid of patches.
    - First 2/3 (rounded) of embed_dim are spatial (split evenly into x/y with even dims)
    - Last 1/3 (rounded to even) is temporal and zero-filled here

    Returns:
        pos_spatial: [1, P, E] tensor suitable for broadcasting over [B, S, P, E]
    """
    H, W = frame_size
    Hp, Wp = H // patch_size, W // patch_size

    # Split dimensions, ensuring temporal is even
    temporal_dim = (embed_dim // 3) & ~1
    spatial_dims = embed_dim - temporal_dim

    # Split spatial dims between x and y (even)
    spatial_x_dim = (spatial_dims // 2) & ~1
    spatial_y_dim = spatial_dims - spatial_x_dim

    assert spatial_x_dim % 2 == 0 and spatial_y_dim % 2 == 0 and temporal_dim % 2 == 0

    # 1D sin/cos encodings for each axis
    def sincos_1d(L, D, device, dtype):
        assert D % 2 == 0
        pos = rearrange(torch.arange(L, device=device, dtype=dtype), 'l -> l 1')
        i = rearrange(torch.arange(D // 2, device=device, dtype=dtype), 'd -> 1 d')
        div = torch.pow(torch.tensor(10000.0, device=device, dtype=dtype), (2*i)/D)
        angles = pos / div
        pe = torch.zeros(L, D, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        return pe

    pe_x = sincos_1d(Wp, spatial_x_dim, device, dtype)  # [Wp, Dx]
    pe_y = sincos_1d(Hp, spatial_y_dim, device, dtype)  # [Hp, Dy]

    pe_x = repeat(pe_x, 'wp dx -> hp wp dx', hp=Hp)     # [Hp, Wp, Dx]
    pe_y = repeat(pe_y, 'hp dy -> hp wp dy', wp=Wp)     # [Hp, Wp, Dy]

    pe_spatial = torch.cat([
        pe_x,
        pe_y,
        torch.zeros(Hp, Wp, temporal_dim, device=device, dtype=dtype)  # zero temporal tail
    ], dim=-1)  # [Hp, Wp, E]

    pe_spatial = rearrange(pe_spatial, 'hp wp e -> 1 (hp wp) e')  # [1, P, E]
    return pe_spatial 