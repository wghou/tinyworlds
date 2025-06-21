# Latent Action Model (LAM)

This is a simplified implementation of the Latent Action Model component from the Genie World Model paper.

## Overview

The Latent Action Model learns to infer actions between video frames in an unsupervised manner. This allows conditioning dynamics on actions without requiring labeled action data.

### Architecture

The model consists of:

1. **Encoder**: Takes previous frames (x₁...xₜ) and next frame xₜ₊₁, outputs continuous latent actions (a₁...aₜ)
2. **VQ Layer**: Discretizes continuous actions into a small set of codes
3. **Decoder**: Takes previous frames and latent actions to predict next frame xₜ₊₁

### Key Features

- Unsupervised learning of discrete actions
- Small VQ codebook size (8 unique actions)
- Decoder only sees history and latent action
- Action should encode meaningful frame-to-frame changes

## Implementation Details

This implementation includes:
- Basic encoder-decoder architecture
- Vector quantization for action discretization
- Training with reconstruction loss
- Simple frame prediction

### Usage
Initialize model
model = LAM(
frame_size=(64, 64),
n_actions=8,
hidden_dim=128
)
Training
loss = model(prev_frames, next_frame)
Inference
action = model.encode(prev_frames, next_frame)
pred_next_frame = model.decode(prev_frames, action)


## Training

The model is trained to:
1. Encode frame transitions into discrete actions
2. Reconstruct future frames given history and actions
3. Learn a minimal set of meaningful actions

The decoder provides the training signal but is replaced with user actions at inference time.

## Verifying

Can get lam proper and isolated by training to infer between frames, then test whether similar frame to frame transitions are the same action, and different frame to frame transitions are different actions





