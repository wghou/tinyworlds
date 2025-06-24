# Dynamics Model Trainer

This module trains a dynamics model that predicts next frames in latent space given video tokenizer latents and action latents.

## Overview

The dynamics model takes as input:
1. **Video tokenizer latents**: Encoded representations of current frames
2. **Action latents**: Encoded representations of actions between frames

It predicts the next frame latents by combining these inputs and processing them through a ST-Transformer architecture.

## Prerequisites

Before training the dynamics model, you need:

1. **Pre-trained Video Tokenizer**: A trained video tokenizer checkpoint (from `src/vqvae/main.py`)
2. **Pre-trained Latent Action Model**: A trained LAM checkpoint (from `src/latent_action_model/main.py`)
3. **Dataset**: The Pong dataset in `data/pong_frames.h5`

## Usage

### Basic Usage

```bash
python src/dynamics/main.py \
    --video_tokenizer_path src/vqvae/results/videotokenizer_timestamp/checkpoints/videotokenizer_checkpoint_timestamp.pth \
    --lam_path src/latent_action_model/results/lam_timestamp/checkpoints/lam_checkpoint_timestamp.pth \
    --batch_size 16 \
    --n_updates 5000 \
    --learning_rate 1e-4
```

### Example Script

Use the provided example script:

```bash
python src/dynamics/example_run.py
```

Make sure to update the checkpoint paths in `example_run.py` to match your actual trained models.

## Architecture

The dynamics model consists of:

- **ST-Transformer Decoder**: Processes combined video and action latents
- **Latent Prediction**: Outputs predicted next frame latents
- **Causal Masking**: Ensures predictions only depend on previous frames

## Training Process

1. **Load Pre-trained Models**: Video tokenizer and LAM are loaded and set to evaluation mode
2. **Encode Frames**: Current frames are encoded to video latents using the video tokenizer
3. **Encode Actions**: Frame transitions are encoded to action latents using the LAM
4. **Combine Latents**: Video and action latents are added together
5. **Predict Next Latents**: The dynamics model predicts next frame latents
6. **Compute Loss**: MSE loss between predicted and target next frame latents

## Key Parameters

- `--video_tokenizer_path`: Path to pre-trained video tokenizer checkpoint
- `--lam_path`: Path to pre-trained LAM checkpoint
- `--batch_size`: Training batch size (default: 32)
- `--n_updates`: Number of training iterations (default: 10000)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--context_length`: Sequence length for training (default: 4)
- `--latent_dim`: Latent dimension (must match video tokenizer and LAM)

## Output

The trainer creates an organized directory structure:

```
src/dynamics/results/
└── dynamics_{timestamp}/
    ├── checkpoints/
    │   └── dynamics_checkpoint_{timestamp}.pth
    └── visualizations/
        └── dynamics_prediction_step_{iteration}_{timestamp}.png
```

- **Model Checkpoints**: `src/dynamics/results/dynamics_{timestamp}/checkpoints/dynamics_checkpoint_{timestamp}.pth`
- **Visualizations**: `src/dynamics/results/dynamics_{timestamp}/visualizations/`
- **Training Logs**: Console output with loss values and debug information

## Model Architecture Details

The dynamics model uses the same ST-Transformer architecture as the video tokenizer and LAM, ensuring compatibility:

- **Patch Size**: 8x8 patches (matches video tokenizer)
- **Embedding Dimension**: 128 (matches video tokenizer)
- **Attention Heads**: 4 (matches video tokenizer)
- **Hidden Dimension**: 512 (matches video tokenizer)
- **Number of Blocks**: 2 (matches video tokenizer)
- **Latent Dimension**: 16 (matches video tokenizer latent_dim)

**Important**: All three models (video tokenizer, LAM, and dynamics model) now use the same hyperparameters for consistency. The LAM's `action_dim` matches the video tokenizer's `latent_dim` (16), ensuring proper latent space compatibility.

## Troubleshooting

### Common Issues

1. **Checkpoint Not Found**: Ensure video tokenizer and LAM checkpoints exist and paths are correct
2. **Dimension Mismatch**: Make sure `latent_dim` matches between all three models (16), and all other parameters match the video tokenizer exactly
3. **Parameter Mismatch**: All three models must use the same parameters (patch_size=8, embed_dim=128, hidden_dim=512, num_blocks=2, latent_dim=16)
4. **Memory Issues**: Reduce batch size or sequence length
5. **NaN Losses**: Check if pre-trained models are stable and producing valid outputs

### Debug Information

The trainer prints debug information every 10 iterations:
- Dynamics loss values
- Variance of different latent representations
- Model output statistics

## Integration with Full Pipeline

The trained dynamics model can be used in the full video generation pipeline:

1. **Video Tokenizer**: Encodes frames to latents
2. **LAM**: Encodes actions to latents  
3. **Dynamics Model**: Predicts next frame latents
4. **Video Tokenizer Decoder**: Converts latents back to frames

This enables end-to-end video generation from initial frames and action sequences.
