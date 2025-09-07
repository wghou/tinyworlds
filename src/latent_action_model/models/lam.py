import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
from einops import rearrange, repeat, reduce
from src.vqvae.models.video_tokenizer import STTransformer, PatchEmbedding

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, embedding_dim, beta=0.05):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.codebook_size, self.embedding_dim)

        # Better initialization: spread out the embeddings
        with torch.no_grad():
            weight = torch.randn(self.codebook_size, self.embedding_dim)
            weight = F.normalize(weight, p=2, dim=1)
            self.embedding.weight.data.copy_(weight)
        
    def forward(self, z, current_step=None):
        """
        Args:
            z: [..., D] encoder outputs; flattened to [M, D]
        Returns:
            loss: Scalar VQ objective (cosine-VQ via MSE on normalized reps)
            z_quantized: [M, D] quantized embeddings with straight-through estimator
            min_encoding_indices: [M] indices of nearest embeddings
        """
        z_flat = z.reshape(-1, z.shape[-1])                # [M, D]
        E = self.embedding.weight                          # [K, D]

        # cosine assignment
        z_n = F.normalize(z_flat, dim=1, eps=1e-8)         # unit
        E_n = F.normalize(E, dim=1, eps=1e-8)              # unit
        sims = z_n @ E_n.t()                               # [M, K]
        idx = sims.argmax(dim=1)                           # [M]
        zq = E[idx]                                        # unnormalized code for decoder

        # loss on directions (cosine-VQ via MSE on normalized reps)
        zq_n = F.normalize(zq, dim=1, eps=1e-8)
        loss = ((zq_n.detach() - z_n)**2).mean() + \
               self.beta * ((zq_n - z_n.detach())**2).mean()
        # straight-through
        zq_st = z_flat + (zq - z_flat).detach()
        return loss, zq_st, idx

    def get_codebook_usage(self, indices):
        """Calculate codebook usage"""
        unique_codes = torch.unique(indices)
        return len(unique_codes) / self.codebook_size

    @torch.no_grad()
    def smart_reinitialize(self, z_flat, indices):
        """
        z_flat: [M, D] encoder outputs (detached)
        indices: [M] current codebook assignments
        """
        K = self.codebook_size
        E = self.embedding.weight

        usage = torch.bincount(indices, minlength=K)
        dead_idx = (usage == 0).nonzero(as_tuple=False).squeeze(1)
        if dead_idx.numel() == 0:
            return

        # choose points least similar to any existing code (cosine)
        z_n = F.normalize(z_flat, dim=1, eps=1e-8)
        E_n = F.normalize(E, dim=1, eps=1e-8)
        sims = z_n @ E_n.t()                # [M, K]
        max_sim, _ = sims.max(dim=1)        # nearest-code similarity
        k = min(dead_idx.numel(), z_flat.size(0))
        _, far_idx = torch.topk(max_sim, k=k, largest=False)  # smallest similarity
        repl = F.normalize(z_flat[far_idx], dim=1, eps=1e-8)

        if k < dead_idx.numel():
            reps = (dead_idx.numel() + k - 1) // k
            repl = repl.repeat(reps, 1)[:dead_idx.numel()]

        E.data[dead_idx] = repl + 1e-3 * torch.randn_like(repl)

class Encoder(nn.Module):
    """ST-Transformer encoder that takes frames and outputs latent actions"""
    def __init__(self, frame_size=(64, 64), patch_size=16, embed_dim=512, num_heads=8, 
                 hidden_dim=2048, num_blocks=6, action_dim=64):
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
        """
        Args:
            frames: [batch_size, seq_len, channels, height, width]
        Returns:
            actions: [batch_size, seq_len-1, action_dim]
            embeddings: [batch_size, seq_len, num_patches, embed_dim]
        """
        batch_size, seq_len, C, H, W = frames.shape
        
        # Convert frames to patch embeddings
        embeddings = self.patch_embed(frames)  # [batch_size, seq_len, num_patches, embed_dim]
        
        # Apply ST-Transformer
        transformed = self.transformer(embeddings)
        
        # Global average pooling over patches
        pooled = transformed.mean(dim=2)  # [batch_size, seq_len, embed_dim]
        
        # Predict actions between consecutive frames
        actions = []
        for t in range(seq_len - 1):
            # Concatenate current and next frame features
            combined = torch.cat([pooled[:, t], pooled[:, t+1]], dim=1)  # [batch_size, embed_dim*2]
            action = self.action_head(combined)
            actions.append(action)
            
        actions = torch.stack(actions, dim=1)  # [batch_size, seq_len-1, action_dim]
        
        return actions, embeddings

class Decoder(nn.Module):
    """ST-Transformer decoder that takes frames and actions to predict next frame"""
    def __init__(self, frame_size=(64, 64), patch_size=16, embed_dim=512, num_heads=8,
                 hidden_dim=2048, num_blocks=6, action_dim=64):
        super().__init__()
        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True)
        
        # Action projection for action-dominant mixing
        self.act_proj = nn.Linear(action_dim, embed_dim)
        
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
        #     mask = torch.bernoulli(torch.full((batch_size, seq_len), masking_rate, device=frames.device))
        #     mask[:, 0] = 1  # never mask first frame (anchor) TODO: try rid of ablation
            
        #     # Apply mask (zero out masked frames)
        #     frames = frames * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        #     # frames = frames.clone()
        #     # frames[:, 1:] = 0.0
        
        # Convert frames to patch embeddings (frame t context)
        emb = self.patch_embed(frames)  # [batch_size, seq_len-1, num_patches, embed_dim]

        # Project actions to embedding and repeat per patch
        act = self.act_proj(actions)             # [batch_size, seq_len-1, embed_dim]
        act = repeat(act, 'b s e -> b s p e', p=self.num_patches)

        # Action-dominant mix; epsilon keeps slight visual context
        epsilon = 0.1
        combined = act + epsilon * emb
         
        # Apply ST-transformer with causal masking
        transformed = self.transformer(combined)  # [batch_size, seq_len-1, num_patches, embed_dim]
         
        # Predict frame patches for all t (predict t+1)
        patches = self.frame_head(transformed)  # [batch_size, seq_len-1, num_patches, 3 * patch_size * patch_size]
         
        # Reshape to frames using einops
        patches = rearrange(patches, 'b s p (c p1 p2) -> b s c p p1 p2', c=3, p1=self.patch_size, p2=self.patch_size)
        frames_out = rearrange(patches, 'b s c (h w) p1 p2 -> b s c (h p1) (w p2)', h=H//self.patch_size, w=W//self.patch_size)
         
        # Return predictions with length S-1
        return frames_out  # [batch_size, seq_len-1, channels, height, width]

class LAM(nn.Module):
    """ST-Transformer based Latent Action Model"""
    def __init__(self, frame_size=(128, 128), n_actions=8, patch_size=8, embed_dim=512, 
                 num_heads=8, hidden_dim=2048, num_blocks=6, action_dim=64, beta=1.0,
                 vq_warmup_steps=100, vq_tau_start=0.5, vq_tau_end=0.05, vq_cache_size=50000,
                 vq_reinit_check_interval=100, vq_dead_threshold_pct=0.2):
        super().__init__()
        self.encoder = Encoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, action_dim)
        self.quantizer = VectorQuantizer(
            n_actions, action_dim, beta=beta
        )
        self.decoder = Decoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, action_dim)
        self.register_buffer('step', torch.tensor(0))
        self.action_pre_norm = nn.LayerNorm(action_dim, elementwise_affine=False)

    def forward(self, frames):
        """
        Process a sequence of frames
        
        Args:
            frames: Tensor of shape [batch_size, seq_len, channels, height, width]
            
        Returns:
            loss: Total loss (reconstruction + VQ + diversity)
            pred_frames: Predicted next frames
            action_indices: Quantized action indices
            loss_dict: Dictionary containing individual loss components
        """
        # Encode frames to get actions and features
        actions, embeddings = self.encoder(frames)

        # Pre-normalize actions before quantization
        actions = self.action_pre_norm(actions)

        # Flatten actions for quantization
        batch_size, seq_len, action_dim = actions.shape
        actions_flat = rearrange(actions, 'b s a -> (b s) a')
        
        # Quantize actions with current step for warmup
        vq_loss, z_quantized_flat, action_indices_flat = self.quantizer(actions_flat)
        
        # Reshape quantized actions back to sequence
        z_quantized = rearrange(z_quantized_flat, '(b s) a -> b s a', b=batch_size, s=seq_len)
        action_indices = rearrange(action_indices_flat, '(b s) -> b s', b=batch_size, s=seq_len)
        
        # Decode to predict next frames
        pred_frames = self.decoder(frames, z_quantized, training=True)  # [batch_size, seq_len-1, channels, height, width]
        
        # Compute reconstruction loss
        target_frames = frames[:, 1:]  # All frames except first [batch_size, seq_len-1, channels, height, width]
        recon_loss = F.mse_loss(pred_frames, target_frames)

        # Total loss with better balancing
        total_loss = recon_loss + 0.5 * vq_loss

        # Diagnostics: pre-VQ variance and cosine top-1 vs top-2 gap (warmup lock-in)
        pre_vq_var = actions_flat.var(dim=0, unbiased=False).mean()

        # Create loss dictionary for monitoring
        loss_dict = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'diversity_loss': torch.tensor(0.0, device=vq_loss.device),
            'pre_vq_var': pre_vq_var,
        }

        return total_loss, pred_frames, action_indices, loss_dict

    def encode(self, prev_frame, next_frame):
        """
        Encode a single frame transition to get the latent action
        
        Args:
            prev_frame: Tensor of shape [batch_size, channels, height, width]
            next_frame: Tensor of shape [batch_size, channels, height, width]
            
        Returns:
            action_index: Quantized action index
        """
        # Add sequence dimension to make it compatible with encoder
        frames = torch.stack([prev_frame, next_frame], dim=1)  # [batch_size, 2, channels, height, width]
        
        # Encode to get actions
        actions, _ = self.encoder(frames)  # [batch_size, 1, action_dim]
        
        # Flatten for quantization
        actions_flat = actions.reshape(-1, actions.size(-1))
        
        # Quantize to get discrete action
        _, _, action_indices = self.quantizer(actions_flat)
        
        return action_indices[0]  # Return first (and only) action index
    
    def decode(self, prev_frame, action_index):
        # prev_frame: [batch_size, channels, height, width]

        batch_size = prev_frame.size(0)
        
        # Convert action index to embedding
        action_embedding = self.quantizer.embedding(action_index)  # [action_dim]
        action_embedding = action_embedding.unsqueeze(0).expand(batch_size, -1)  # [batch_size, action_dim]
        
        # Create sequence with just the previous frame
        frames = prev_frame.unsqueeze(1)  # [batch_size, 1, channels, height, width]
        actions = action_embedding.unsqueeze(1)  # [batch_size, 1, action_dim]
        
        # Decode to get next frame
        pred_frames = self.decoder(frames, actions, training=False)  # [batch_size, 1, channels, height, width]
        
        return pred_frames.squeeze(1)  # [batch_size, channels, height, width]
    
    # indices -> latent vectors
    def indices_to_latent(self, action_indices):
        return F.embedding(action_indices, self.quantizer.embedding)

    def latent_to_indices(self, latent_vectors):
        z_n = F.normalize(latent_vectors, dim=-1)
        E_n = F.normalize(self.quantizer.embedding.weight, dim=-1)
        return (z_n @ E_n.t()).argmax(dim=-1)
