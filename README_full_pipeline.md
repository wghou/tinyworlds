# Full Pipeline Training Script

This script automates the complete training pipeline for the video generation system on the Pong dataset.

## Overview

The `train_full_pipeline.py` script trains all three models in sequence:

1. **Video Tokenizer** - Learns to encode/decode video frames to/from latent representations
2. **LAM (Latent Action Model)** - Learns to infer discrete actions between frame transitions
3. **Dynamics Model** - Learns to predict next frame latents given current latents and actions

## Prerequisites

Before running the full pipeline, ensure you have:

1. **Pong Dataset**: The preprocessed Pong dataset at `data/pong_frames.h5`
2. **Dependencies**: All required Python packages installed
3. **Directory Structure**: Run from the nano-genie root directory

## Usage

```bash
# From the nano-genie root directory
python train_full_pipeline.py
```

## What the Script Does

### Step 1: Video Tokenizer Training
- **Dataset**: Pong
- **Duration**: 5000 iterations (reduced for faster training)
- **Output**: `src/vqvae/results/videotokenizer_{timestamp}/`

### Step 2: LAM Training  
- **Dataset**: Pong
- **Duration**: 2000 iterations (reduced for faster training)
- **Output**: `src/latent_action_model/results/lam_{timestamp}/`

### Step 3: Checkpoint Discovery
- Automatically finds the latest checkpoints from previous steps
- Validates that both checkpoints exist before proceeding

### Step 4: Dynamics Model Training
- **Uses**: Pre-trained video tokenizer and LAM checkpoints
- **Duration**: 3000 iterations (reduced for faster training)
- **Output**: `src/dynamics/results/dynamics_{timestamp}/`

## Hyperparameters

All models use consistent hyperparameters for compatibility:

- **Patch Size**: 8x8
- **Embedding Dimension**: 128
- **Attention Heads**: 4
- **Hidden Dimension**: 512
- **Number of Blocks**: 2
- **Latent/Action Dimension**: 16
- **Dropout**: 0.1

## Output Structure

After successful completion, you'll have:

```
nano-genie/
├── src/
│   ├── vqvae/results/videotokenizer_{timestamp}/
│   │   ├── checkpoints/videotokenizer_checkpoint_{timestamp}.pth
│   │   └── visualizations/reconstruction_step_*.png
│   ├── latent_action_model/results/lam_{timestamp}/
│   │   ├── checkpoints/lam_checkpoint_{timestamp}.pth
│   │   └── visualizations/reconstructions_lam_epoch_*.png
│   └── dynamics/results/dynamics_{timestamp}/
│       ├── checkpoints/dynamics_checkpoint_{timestamp}.pth
│       └── visualizations/dynamics_prediction_step_*.png
```

## Error Handling

The script includes robust error handling:

- ✅ **Directory validation**: Ensures you're in the correct directory
- ✅ **Dataset validation**: Checks that Pong dataset exists
- ✅ **Checkpoint discovery**: Automatically finds latest checkpoints
- ✅ **Training monitoring**: Shows progress and handles interruptions
- ✅ **Graceful failure**: Stops pipeline if any step fails

## Customization

To modify training parameters, edit the command lists in the script:

```python
# Example: Increase training iterations
video_tokenizer_cmd = [
    sys.executable, "src/vqvae/main.py",
    "--n_updates", "10000",  # Increased from 5000
    # ... other parameters
]
```

## Monitoring Progress

The script provides clear progress indicators:

- 🚀 **Start**: Shows timestamp and dataset validation
- 📊 **Step progress**: Clear section headers for each step
- ✅ **Success indicators**: Confirms each step completion
- 📁 **Results summary**: Shows all checkpoint paths at the end

## Next Steps

After successful training:

1. **Check visualizations**: Review the generated images in each model's `visualizations/` directory
2. **Inference**: Use the trained models for video generation
3. **Fine-tuning**: Optionally continue training with more iterations
4. **Evaluation**: Test the models on new data

## Troubleshooting

### Common Issues

1. **"Pong dataset not found"**: Ensure `data/pong_frames.h5` exists
2. **"Please run from nano-genie root"**: Make sure you're in the correct directory
3. **Training failures**: Check GPU memory and reduce batch size if needed
4. **Checkpoint not found**: Verify previous training steps completed successfully

### Performance Tips

- **GPU memory**: Reduce batch size if you encounter OOM errors
- **Training time**: Adjust `--n_updates` for faster/slower training
- **Logging**: Modify `--log_interval` for more/fewer checkpoints

## Example Output

```
🚀 Starting Full Pipeline Training on Pong Dataset
Timestamp: sun_jun_22_23_50_36_2025
✅ Found Pong dataset at data/pong_frames.h5

============================================================
Starting: Video Tokenizer Training
============================================================
✅ Video Tokenizer Training completed successfully!

============================================================
Starting: LAM Training  
============================================================
✅ LAM Training completed successfully!

============================================================
Starting: Dynamics Model Training
============================================================
✅ Dynamics Model Training completed successfully!

============================================================
🎉 FULL PIPELINE TRAINING COMPLETED SUCCESSFULLY!
============================================================

📁 Results Summary:
Video Tokenizer: src/vqvae/results/videotokenizer_sun_jun_22_23_50_36_2025/checkpoints/videotokenizer_checkpoint_sun_jun_22_23_50_36_2025.pth
LAM: src/latent_action_model/results/lam_sun_jun_22_23_50_36_2025/checkpoints/lam_checkpoint_sun_jun_22_23_50_36_2025.pth
Dynamics Model: src/dynamics/results/dynamics_sun_jun_22_23_50_36_2025/checkpoints/dynamics_checkpoint_sun_jun_22_23_50_36_2025.pth 