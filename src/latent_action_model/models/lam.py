import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
from einops import rearrange, repeat, reduce
from src.vqvae.models.video_tokenizer import STTransformer, PatchEmbedding, FiniteScalarQuantizer

NUM_LAM_BINS = 2

class Encoder(nn.Module):
    """ST-Transformer encoder that takes frames and outputs latent actions"""
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, num_heads=8, 
                 hidden_dim=256, num_blocks=4, action_dim=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True)

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, 4 * action_dim),
            nn.GELU(),
            nn.Linear(4 * action_dim, action_dim)
        )

    def forward(self, frames):
        # frames: [batch_size, seq_len, channels, height, width]
        batch_size, seq_len, C, H, W = frames.shape

        # Convert frames to patch embeddings
        embeddings = self.patch_embed(frames)  # [batch_size, seq_len, num_patches, embed_dim]

        # Apply ST-Transformer
        transformed = self.transformer(embeddings)

        # TODO: find better method for outputting actions
        # Global average pooling over patches
        pooled = transformed.mean(dim=2)  # [batch_size, seq_len, embed_dim]

        # Predict actions between consecutive frames
        actions = []
        for t in range(seq_len - 1):
            # Concatenate current and next frame features
            combined = torch.cat([pooled[:, t], pooled[:, t+1]], dim=1)  # [batch_size, embed_dim*2]
            action = self.action_head(combined)  # [batch_size, action_dim]
            actions.append(action)

        actions = torch.stack(actions, dim=1)  # [batch_size, seq_len-1, action_dim]

        return actions

class Decoder(nn.Module):
    """ST-Transformer decoder that takes frames and actions to predict next frame"""
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, num_heads=8,
                 hidden_dim=256, num_blocks=4, conditioning_dim=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True, conditioning_dim=conditioning_dim)

        # Frame prediction head
        self.frame_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3 * patch_size * patch_size),
            nn.Tanh()
        )

        self.frame_size = frame_size
        self.patch_size = patch_size
        self.num_patches = (frame_size[0] // patch_size) * (frame_size[1] // patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

    def forward(self, frames, actions, training=True):
        # frames: [B, S, C, H, W]
        # actions: [B, S - 1, action_dim]
        B, S, C, H, W = frames.shape
        frames = frames[:, :-1] # [B, S-1, C, H, W]
        video_embeddings = self.patch_embed(frames)  # [B, S-1, P, E]
        _, _, P, E = video_embeddings.shape

        # Apply random masking during training
        if training and self.training:
            keep_rate = 0.0
            keep = (torch.rand(B, S-1, P, 1, device=frames.device) < keep_rate)
            keep[:, 0] = 1  # never mask first frame tokens (anchor) TODO: try rid of ablation
            video_embeddings = torch.where(
                keep, video_embeddings,
                self.mask_token.to(video_embeddings.dtype).expand_as(video_embeddings)
            )

        print(f"video_embeddings shape: {video_embeddings.shape}, action shape: {actions.shape}")
        transformed = self.transformer(video_embeddings, conditioning=actions)  # [B, S-1, P, E]
        patches = self.frame_head(transformed)  # [B, S-1, P, 3 * patch_size * patch_size]
        patches = rearrange(
            patches, 'b s p (c p1 p2) -> b s c p p1 p2', c=3, p1=self.patch_size, p2=self.patch_size
        ) # [B, S-1, C, P, patch_size, patch_size]
        pred_frames = rearrange(
            patches, 'b s c (h w) p1 p2 -> b s c (h p1) (w p2)', h=H//self.patch_size, w=W//self.patch_size
        ) # [B, S-1, C, H, W]
        return pred_frames  # [B, S-1, C, H, W]

class LAM(nn.Module):
    """ST-Transformer based Latent Action Model"""
    def __init__(self, frame_size=(128, 128), n_actions=8, patch_size=8, embed_dim=128, 
                 num_heads=8, hidden_dim=256, num_blocks=4):
        super().__init__()
        assert math.log(n_actions, NUM_LAM_BINS).is_integer(), f"n_actions must be a power of {NUM_LAM_BINS}"
        self.action_dim=int(math.log(n_actions, NUM_LAM_BINS))
        self.encoder = Encoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, action_dim=self.action_dim)
        self.quantizer = FiniteScalarQuantizer(latent_dim=self.action_dim, num_bins=NUM_LAM_BINS)
        self.decoder = Decoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, conditioning_dim=self.action_dim)
        self.var_target = 0.01
        self.var_lambda = 100.0

    def forward(self, frames):
        # frames: [B, S, C, H, W]
    
        # get quantized action latents
        action_latents = self.encoder(frames) # [B, S - 1, action_dim]
        action_latents_quantized = self.quantizer(action_latents) # [B, S - 1, action_dim]

        # decode to get predicted frames
        pred_frames = self.decoder(frames, action_latents_quantized, training=True)  # [B, S - 1, C, H, W]
        
        # Compute reconstruction loss
        target_frames = frames[:, 1:]  # All frames except first [B, S - 1, C, H, W]
        recon_loss = F.smooth_l1_loss(pred_frames, target_frames)
        
        # Encourage non-collapsed encoder variance per-dimension
        z_var = action_latents.var(dim=0, unbiased=False).mean()
        var_penalty = F.relu(self.var_target - z_var)
        total_loss = recon_loss + self.var_lambda * var_penalty

        return total_loss, pred_frames

    def encode(self, frames):
        action_latents = self.encoder(frames)  # [batch_size, seq_len, action_dim]
        action_latents_quantized = self.quantizer(action_latents) # [batch_size, seq_len, action_dim]
        return action_latents_quantized