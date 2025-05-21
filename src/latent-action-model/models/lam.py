import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualLayer(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x

class ResidualStack(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super().__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList([ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, frame_size, h_dim, n_res_layers, res_h_dim, action_dim):
        super().__init__()
        kernel = 4
        stride = 2
        
        self.conv_stack = nn.Sequential(
            nn.Conv2d(6, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        )
        
        # Add batch norm to help with feature scaling
        self.pre_quantization = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(h_dim, action_dim),
            nn.BatchNorm1d(action_dim, track_running_stats=True, momentum=0.01),
            nn.Tanh(),
        )
        
    def forward(self, prev_frame, next_frame):
        x = torch.cat([prev_frame, next_frame], dim=1)
        x = self.conv_stack(x)
        
        # Handle single sample case during evaluation
        if x.size(0) == 1 and not self.training:
            self.pre_quantization.eval()  # Ensure using running statistics
            
        z = self.pre_quantization(x)
        z = z * 1.0  # Scale factor matches embedding init range
        return z.unsqueeze(2).unsqueeze(3)

class VectorQuantizer(nn.Module):
    def __init__(self, n_actions, action_dim, beta=1.0):
        super().__init__()
        self.n_e = n_actions
        self.e_dim = action_dim
        self.beta = beta
        
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0, 1.0)
        
    def forward(self, z):
        # z shape: (batch, action_dim, 1, 1)
        z_flattened = z.squeeze(-1).squeeze(-1)  # (batch, action_dim)
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
            
        # Find nearest embedding
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices)
        
        # Compute loss
        loss = torch.mean((z_q.detach()-z_flattened)**2) + self.beta * \
               torch.mean((z_q - z_flattened.detach()) ** 2)
               
        # Preserve gradients
        z_q = z_flattened + (z_q - z_flattened).detach()
        
        # Add spatial dimensions back
        z_q = z_q.unsqueeze(2).unsqueeze(3)  # (batch, action_dim, 1, 1)
        
        return loss, z_q, min_encoding_indices

class Decoder(nn.Module):
    def __init__(self, frame_size, h_dim, n_res_layers, res_h_dim, action_dim):
        super().__init__()
        kernel = 4
        stride = 2
        
        # Unpack frame size
        h, w = frame_size
        
        # Project action to initial spatial feature map
        self.initial_conv = nn.Sequential(
            nn.Conv2d(action_dim, h_dim, kernel_size=1),
            nn.Upsample(size=(h//4, w//4))  # Initial spatial size
        )
        
        self.inverse_conv_stack = nn.Sequential(
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim//2, 3, kernel_size=kernel, stride=stride, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z_q, prev_frame):
        x = self.initial_conv(z_q)
        x = self.inverse_conv_stack(x)
        return x

class LAM(nn.Module):
    def __init__(self, frame_size=(64, 64), n_actions=8, h_dim=128, res_h_dim=64, n_res_layers=2, action_dim=64):
        super().__init__()
        self.encoder = Encoder(frame_size, h_dim, n_res_layers, res_h_dim, action_dim)
        self.quantizer = VectorQuantizer(n_actions, action_dim)
        self.decoder = Decoder(frame_size, h_dim, n_res_layers, res_h_dim, action_dim)
        
    def forward(self, prev_frame, next_frame):
        z_e = self.encoder(prev_frame, next_frame)
        vq_loss, z_q, action_indices = self.quantizer(z_e)
        pred_next = self.decoder(z_q, prev_frame)
        
        recon_loss = F.mse_loss(pred_next, next_frame)
        
        return recon_loss + vq_loss, pred_next, action_indices
    
    def encode(self, prev_frame, next_frame):
        with torch.no_grad():
            self.eval()  # Set to evaluation mode
            z_e = self.encoder(prev_frame, next_frame)
            _, _, action_indices = self.quantizer(z_e)
            self.train()  # Set back to training mode
        return action_indices
    
    def decode(self, prev_frame, action_indices):
        with torch.no_grad():
            z_q = self.quantizer.embedding(action_indices)
            z_q = z_q.unsqueeze(2).unsqueeze(3)  # Add spatial dimensions
            return self.decoder(z_q, prev_frame)
