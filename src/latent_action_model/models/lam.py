import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

class PatchEmbedding(nn.Module):
    """Convert frames to patch embeddings for ST-Transformer"""
    def __init__(self, frame_size=(64, 64), patch_size=16, embed_dim=512):
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
        """
        for block in self.blocks:
            x = block(x)
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, embedding_dim, beta=1.0):
        super().__init__()
        self.n_e = codebook_size
        self.e_dim = embedding_dim
        self.beta = beta
        self.register_buffer('step', torch.tensor(0))
        
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        
        # Better initialization: spread out the embeddings
        # Initialize with orthogonal vectors to ensure good separation
        with torch.no_grad():
            # Create orthogonal basis
            orthogonal_vectors = torch.randn(self.n_e, self.e_dim)
            # Apply Gram-Schmidt orthogonalization
            for i in range(self.n_e):
                for j in range(i):
                    proj = torch.dot(orthogonal_vectors[i], orthogonal_vectors[j]) / torch.dot(orthogonal_vectors[j], orthogonal_vectors[j])
                    orthogonal_vectors[i] = orthogonal_vectors[i] - proj * orthogonal_vectors[j]
                # Normalize
                orthogonal_vectors[i] = orthogonal_vectors[i] / torch.norm(orthogonal_vectors[i])
            
            # Scale to reasonable range - reduce scaling to bring vectors closer
            self.embedding.weight.data = orthogonal_vectors * 0.1  # Reduced from 0.5
    
    def get_warmup_beta(self, current_step, warmup_steps=1000):
        """Get warmup-adjusted beta value"""
        if current_step < warmup_steps:
            # Quadratic warmup: start at 0.01 and gradually increase to self.beta
            warmup_ratio = (current_step / warmup_steps) ** 2
            return 0.01 + (self.beta - 0.01) * warmup_ratio
        return self.beta
        
    def forward(self, z, current_step=None):
        # z shape: (batch, embedding_dim)
        
        # Update step counter
        if current_step is not None:
            self.step = torch.tensor(current_step, device=self.step.device)
        
        # Get warmup-adjusted beta
        warmup_beta = self.get_warmup_beta(self.step.item())
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z, self.embedding.weight.t())
            
        # Find nearest embedding
        min_encoding_indices = torch.argmin(d, dim=1)
        
        z_q = self.embedding(min_encoding_indices)
        
        # Compute loss with warmup-adjusted beta
        loss = torch.mean((z_q.detach()-z)**2) + warmup_beta * \
               torch.mean((z_q - z.detach()) ** 2)
               
        # Preserve gradients
        z_q = z + (z_q - z).detach()
        
        return loss, z_q, min_encoding_indices

    def reset_codebook(self):
        """Reset codebook to orthogonal initialization if it gets stuck"""
        with torch.no_grad():
            # Create orthogonal basis
            orthogonal_vectors = torch.randn(self.n_e, self.e_dim)
            # Apply Gram-Schmidt orthogonalization
            for i in range(self.n_e):
                for j in range(i):
                    proj = torch.dot(orthogonal_vectors[i], orthogonal_vectors[j]) / torch.dot(orthogonal_vectors[j], orthogonal_vectors[j])
                    orthogonal_vectors[i] = orthogonal_vectors[i] - proj * orthogonal_vectors[j]
                # Normalize
                orthogonal_vectors[i] = orthogonal_vectors[i] / torch.norm(orthogonal_vectors[i])
            
            # Scale to reasonable range - reduce scaling to bring vectors closer
            self.embedding.weight.data = orthogonal_vectors * 0.1  # Reduced from 0.5

    def smart_reinitialize(self, z_flat, indices):
        """
        Smart reinitialization using actual encoder outputs
        
        Args:
            z_flat: [batch_size, embedding_dim] - encoder outputs
            indices: [batch_size] - current codebook assignments
        """
        with torch.no_grad():
            # Count usage of each codebook entry
            usage_counts = torch.bincount(indices, minlength=self.n_e)
            
            # Find dead codes (unused or very rarely used)
            dead_threshold = max(1, len(z_flat) // (self.n_e * 5))  # Allow 20% of codes to be "rare"
            dead_codes = usage_counts < dead_threshold
            
            if dead_codes.sum() == 0:
                return  # No dead codes
            
            print(f"  🔄 Reinitializing {dead_codes.sum().item()} dead codes...")
            
            # For each dead code, find a good replacement
            for dead_code_idx in torch.where(dead_codes)[0]:
                # Find encoder outputs that are far from existing codes
                distances_to_codes = torch.sum((z_flat.unsqueeze(1) - self.embedding.weight.unsqueeze(0))**2, dim=2)
                min_distances = distances_to_codes.min(dim=1)[0]
                
                # Find encoder outputs that are far from all existing codes
                far_threshold = min_distances.mean() + min_distances.std()
                far_indices = torch.where(min_distances > far_threshold)[0]
                
                if len(far_indices) > 0:
                    # Pick a random far encoder output
                    replacement_idx = far_indices[torch.randint(0, len(far_indices), (1,))]
                    self.embedding.weight.data[dead_code_idx] = z_flat[replacement_idx]
                else:
                    # Fallback: use a random encoder output
                    replacement_idx = torch.randint(0, len(z_flat), (1,))
                    self.embedding.weight.data[dead_code_idx] = z_flat[replacement_idx]
            
            # Add small noise to break any remaining symmetries
            noise = torch.randn_like(self.embedding.weight.data) * 0.01
            self.embedding.weight.data += noise

    def get_codebook_usage(self, indices):
        """
        Calculate codebook usage
        """
        unique_codes = torch.unique(indices)
        return len(unique_codes) / self.n_e

class Encoder(nn.Module):
    """ST-Transformer encoder that takes frames and outputs latent actions"""
    def __init__(self, frame_size=(64, 64), patch_size=16, embed_dim=512, num_heads=8, 
                 hidden_dim=2048, num_blocks=6, action_dim=64, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, dropout, causal=True)
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, action_dim)
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
                 hidden_dim=2048, num_blocks=6, action_dim=64, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, dropout, causal=True)
        
        # Action embedding
        self.action_embed = nn.Linear(action_dim, embed_dim)
        
        # Frame prediction head
        self.frame_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3 * patch_size * patch_size)
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
        
        # Apply random masking during training
        if training and self.training:
            # Sample masking rate uniformly from 0.5 to 1
            masking_rate = torch.rand(1).item() * 0.5 + 0.5  # [0.5, 1.0]
            
            # Create mask with Bernoulli distribution
            mask = torch.bernoulli(torch.full((batch_size, seq_len), masking_rate, device=frames.device))
            mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [batch_size, seq_len, 1, 1, 1]
            
            # Apply mask (zero out masked frames)
            frames = frames * mask
        
        # Convert frames to patch embeddings
        embeddings = self.patch_embed(frames)  # [batch_size, seq_len, num_patches, embed_dim]
        
        # Convert actions to embeddings and pad to match sequence length
        action_embeds = self.action_embed(actions)  # [batch_size, seq_len-1, embed_dim]
        
        # Pad actions to match frame sequence length (add zero action for the last position)
        # This allows the transformer to predict the next frame for each position
        zero_action = torch.zeros(batch_size, 1, action_embeds.size(-1), device=action_embeds.device)
        action_embeds_padded = torch.cat([action_embeds, zero_action], dim=1)  # [batch_size, seq_len, embed_dim]
        
        # Add action embeddings to frame embeddings
        action_embeds_padded = repeat(action_embeds_padded, 'b s e -> b s p e', p=self.num_patches)
        combined = embeddings + action_embeds_padded  # [batch_size, seq_len, num_patches, embed_dim]
        
        # Apply ST-transformer with causal masking
        # The causal mask ensures each position can only attend to previous positions
        transformed = self.transformer(combined)  # [batch_size, seq_len, num_patches, embed_dim]
        
        # Predict frame patches for all positions
        patches = self.frame_head(transformed)  # [batch_size, seq_len, num_patches, patch_size^2 * 3]
        
        # Reshape to frames using einops
        patches = rearrange(patches, 'b s p (c p1 p2) -> b s c p p1 p2', c=3, p1=self.patch_size, p2=self.patch_size)
        frames_out = rearrange(patches, 'b s c (h w) p1 p2 -> b s c (h p1) (w p2)', h=H//self.patch_size, w=W//self.patch_size)
        
        # Return predictions for positions 1 to seq_len-1 (skip the first position)
        return frames_out[:, :-1]  # [batch_size, seq_len-1, channels, height, width]

class LAM(nn.Module):
    """ST-Transformer based Latent Action Model"""
    def __init__(self, frame_size=(64, 64), n_actions=8, patch_size=16, embed_dim=512, 
                 num_heads=8, hidden_dim=2048, num_blocks=6, action_dim=64, dropout=0.1, beta=1.0):
        super().__init__()
        self.encoder = Encoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, action_dim, dropout)
        self.quantizer = VectorQuantizer(n_actions, action_dim, beta)
        self.decoder = Decoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, action_dim, dropout)
        self.register_buffer('step', torch.tensor(0))
        
    def forward(self, frames, current_step=None):
        """
        Process a sequence of frames
        
        Args:
            frames: Tensor of shape [batch_size, seq_len, channels, height, width]
            current_step: Current training step for warmup scheduling
            
        Returns:
            loss: Total loss (reconstruction + VQ + diversity)
            pred_frames: Predicted next frames
            action_indices: Quantized action indices
            loss_dict: Dictionary containing individual loss components
        """
        # Update step counter
        if current_step is not None:
            self.step = torch.tensor(current_step, device=self.step.device)
        
        # Encode frames to get actions and features
        actions, embeddings = self.encoder(frames)
        
        # Add noise to actions during training to force diversity
        if self.training:
            # Add small amount of noise to break symmetry
            noise_std = 0.5  # Reduced from 1.0 for better stability
            actions = actions + torch.randn_like(actions) * noise_std
        
        # Flatten actions for quantization
        batch_size, seq_len, action_dim = actions.shape
        actions_flat = rearrange(actions, 'b s a -> (b s) a')
        
        # Quantize actions with current step for warmup
        vq_loss, z_q_flat, action_indices_flat = self.quantizer(actions_flat, current_step=self.step)
        
        # Reshape quantized actions back to sequence
        z_q = rearrange(z_q_flat, '(b s) a -> b s a', b=batch_size, s=seq_len)
        action_indices = rearrange(action_indices_flat, '(b s) -> b s', b=batch_size, s=seq_len)
        
        # Decode to predict next frames
        pred_frames = self.decoder(frames, z_q, training=True)  # [batch_size, seq_len-1, channels, height, width]
        
        # Compute reconstruction loss
        target_frames = frames[:, 1:]  # All frames except first [batch_size, seq_len-1, channels, height, width]
        recon_loss = F.mse_loss(pred_frames, target_frames)
        
        # Enhanced diversity loss
        diversity_loss = self.compute_enhanced_diversity_loss(actions, action_indices_flat)
        
        # Total loss with better balancing
        total_loss = recon_loss + 0.25 * vq_loss + 0.5 * diversity_loss
        
        # Create loss dictionary for monitoring
        loss_dict = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'diversity_loss': diversity_loss
        }
        
        return total_loss, pred_frames, action_indices, loss_dict
    
    def compute_enhanced_diversity_loss(self, actions, action_indices):
        """Compute enhanced diversity loss to encourage codebook usage"""
        # 1. Continuous diversity: encourage diverse encoder outputs
        actions_std = actions.std(dim=1)  # [batch_size, action_dim]
        continuous_diversity = -torch.mean(actions_std)  # Negative to maximize
        
        # 2. Sequence diversity: encourage different actions across time
        sequence_diversity = -torch.mean(torch.std(actions, dim=1))  # Negative to maximize
        
        # 3. Codebook usage penalty: directly penalize low codebook usage
        unique_codes = torch.unique(action_indices)
        usage_ratio = len(unique_codes) / self.quantizer.n_e
        usage_penalty = -torch.log(torch.tensor(usage_ratio + 1e-6))  # Negative log to encourage higher usage
        
        # 4. Entropy penalty: encourage uniform distribution over used codes
        if len(unique_codes) > 1:
            # Compute entropy of code usage
            code_counts = torch.bincount(action_indices, minlength=self.quantizer.n_e)
            used_codes = code_counts[unique_codes]
            probs = used_codes.float() / used_codes.sum()
            entropy = -torch.sum(probs * torch.log(probs + 1e-6))
            max_entropy = torch.log(torch.tensor(len(unique_codes), dtype=torch.float))
            entropy_penalty = -(entropy / max_entropy)  # Negative to maximize entropy
        else:
            entropy_penalty = torch.tensor(0.0, device=actions.device)
        
        # Combine all diversity components
        total_diversity = (
            0.5 * continuous_diversity + 
            0.3 * sequence_diversity + 
            0.15 * usage_penalty + 
            0.05 * entropy_penalty
        )
        
        return total_diversity
    
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
        """
        Decode a single frame transition using a discrete action
        
        Args:
            prev_frame: Tensor of shape [batch_size, channels, height, width]
            action_index: Integer action index
            
        Returns:
            next_frame: Predicted next frame
        """
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
    
