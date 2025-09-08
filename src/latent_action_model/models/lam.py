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
        
    def forward(self, frames, actions, training=True):
        """
        Args:
            frames: [batch_size, seq_len, channels, height, width]
            actions: [batch_size, seq_len-1, action_dim] - actions between frames
            training: Whether in training mode (for masking)
        Returns:
            pred_frames: [batch_size, seq_len-1, channels, height, width] - predicted next frames from t=1 to t=seq_len-1
        """
        batch_size, seq_len, C, H, W = frames.shape

        frames = frames[:, :-1] # [batch_size, seq_len-1, channels, height, width]
        
        # # Apply random masking during training
        # if training and self.training:
        #     # Sample masking (keep) rate uniformly from 0.1 to 0.5
        #     masking_rate = torch.rand(1).item() * 0.3 + 0.1  # [0.1, 0.5]
            
        #     # Create mask with Bernoulli distribution
        #     mask = torch.bernoulli(torch.full((batch_size, seq_len, 1, 1, 1), masking_rate, device=frames.device))
        #     mask[:, 0] = 1  # never mask first frame (anchor) TODO: try rid of ablation
            
        #     # Apply mask (zero out masked frames)
        #     frames = frames * mask
        
        # Convert frames to patch embeddings (frame t context)
        video_embeddings = self.patch_embed(frames)  # [batch_size, seq_len-1, num_patches, embed_dim]

        transformed = self.transformer(video_embeddings, conditioning=actions)  # [batch_size, seq_len-1, num_patches, embed_dim]
         
        # Predict frame patches for all t (predict t+1)
        patches = self.frame_head(transformed)  # [batch_size, seq_len-1, num_patches, 3 * patch_size * patch_size]
         
        # Reshape to frames
        patches = rearrange(
            patches, 'b s p (c p1 p2) -> b s c p p1 p2', c=3, p1=self.patch_size, p2=self.patch_size
        ) # [batch_size, seq_len-1, channels, num_patches, patch_size, patch_size]
        frames_out = rearrange(
            patches, 'b s c (h w) p1 p2 -> b s c (h p1) (w p2)', h=H//self.patch_size, w=W//self.patch_size
        ) # [batch_size, seq_len-1, channels, pixel_height, pixel_width]
         
        # Return predictions with length S-1
        return frames_out  # [batch_size, seq_len-1, channels, pixel_height, pixel_width]

class LAM(nn.Module):
    """ST-Transformer based Latent Action Model"""
    def __init__(self, frame_size=(128, 128), n_actions=8, patch_size=8, embed_dim=128, 
                 num_heads=8, hidden_dim=256, num_blocks=4):
        super().__init__()
        assert math.log(n_actions, NUM_LAM_BINS).is_integer(), f"n_actions must be a power of {NUM_LAM_BINS}"
        self.action_dim=int(math.log(n_actions, NUM_LAM_BINS))
        print(f"action_dim: {self.action_dim}")
        self.encoder = Encoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, action_dim=self.action_dim)
        self.quantizer = FiniteScalarQuantizer(
            latent_dim=self.action_dim, num_bins=NUM_LAM_BINS
        )
        self.decoder = Decoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, conditioning_dim=self.action_dim)

    def forward(self, frames):
        # frames: Tensor of shape [batch_size, seq_len, channels, height, width]
 
        # Encode frames to get actions and features
        actions = self.encoder(frames)

        # Quantize actions with current step for warmup
        actions_quantized = self.quantizer(actions)
        
        # Decode to predict next frames
        pred_frames = self.decoder(frames, actions_quantized, training=True)  # [batch_size, seq_len-1, channels, height, width]
        
        # Compute reconstruction loss
        target_frames = frames[:, 1:]  # All frames except first [batch_size, seq_len-1, channels, height, width]
        recon_loss = F.mse_loss(pred_frames, target_frames)

        return recon_loss, pred_frames

    def encode(self, prev_frame, next_frame):
        # for inference when we only need to yield an action given sequence of previous frame
        # TODO: alter to take in a sequence of frames for more context
        # prev_frame: [batch_size, channels, height, width]
        # next_frame: [batch_size, channels, height, width]
        frames = torch.stack([prev_frame, next_frame], dim=1)  # [batch_size, 2, channels, height, width]
        
        action = self.encoder(frames)  # [batch_size, 1, action_dim]

        action_quantized = self.quantizer(action) # [batch_size, 1, action_dim]

        action_index = self.quantizer.get_indices_from_latents(action_quantized) # [batch_size, 1]

        print(f"action_index.shape: {action_index.shape}, action_quantized.shape: {action_quantized.shape}, action.shape: {action.shape}")

        return action_index