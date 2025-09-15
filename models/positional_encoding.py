import torch
from einops import rearrange, repeat


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

def sincos_2d(Hp, Wp, E, device, dtype):
    # 2d sinusoidal with half width and half height encodings
    assert E % 2 == 0, "Encoding dimension must be even to split between H and W"

    # Split dimensions between height and width encodings
    Dh = Dw = E // 2

    # Get 1D encodings for height and width positions
    pe_h = sincos_1d(Hp, Dh, device, dtype)        # [Hp, Dh]
    pe_w = sincos_1d(Wp, Dw, device, dtype)        # [Wp, Dw]

    # Combine into 2D grid using repeat to expand across axes
    pe = torch.cat([
        repeat(pe_h, 'hp dh -> hp wp dh', wp=Wp),
        repeat(pe_w, 'wp dw -> hp wp dw', hp=Hp)
    ], dim=-1)                                     # [Hp, Wp, E]

    return rearrange(pe, 'hp wp e -> (hp wp) e')  # [P, E]

def sincos_time(T, D, device, dtype):
    # temporal PE using 1d sinusoidal PE across time
    return sincos_1d(T, D, device, dtype)  # reuse the same 1D builder


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