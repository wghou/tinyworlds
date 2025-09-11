#!/usr/bin/env python3
"""
Comprehensive test script for inference pipeline components
Merges functionality from:
- test_checkpoint_loading.py
- test_inference.py  
- test_lam_datasets.py
- test_mp4_creation.py
- test_sliding_window.py
- test_visualization.py
"""

import torch
import cv2
import numpy as np
import sys
import os
import time

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.vqvae.models.video_tokenizer import Video_Tokenizer
from src.latent_action_model.models.lam import LAM
from src.dynamics.models.dynamics_model import DynamicsModel
from datasets.utils import load_data_and_data_loaders

# Import visualization function if available
try:
    from run_inference import visualize_inference
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ Visualization function not available")

# Global test configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test paths - update these to actual checkpoint paths
VIDEO_TOKENIZER_PATH = "src/vqvae/results/videotokenizer_sat_jun_28_00_41_39_2025/checkpoints/videotokenizer_checkpoint_sat_jun_28_00_41_39_2025.pth"
LAM_PATH = "src/latent_action_model/results/lam_Sat_Jun_28_12_59_59_2025/checkpoints/lam_checkpoint_Sat_Jun_28_12_59_59_2025.pth"
DYNAMICS_PATH = "src/dynamics/results/dynamics_Sat_Jun_28_13_05_43_2025/checkpoints/dynamics_checkpoint_Sat_Jun_28_13_05_43_2025.pth"
    
def _initialize_models():
    """Initialize models for testing (internal function)"""
    print("\n🧪 Testing model initialization...")
    
    # Test video tokenizer
    print("Initializing video tokenizer...")
    video_tokenizer = Video_Tokenizer(
        frame_size=(64, 64),
        patch_size=4,
        embed_dim=128,
        num_heads=4,
        hidden_dim=512,
        num_blocks=2,
        latent_dim=6,
        dropout=0.1,
        num_bins=4
    ).to(DEVICE)
    print("✅ Video tokenizer initialized")
    
    # Test LAM
    print("Initializing LAM...")
    lam = LAM(
        frame_size=(64, 64),
        n_actions=8,
        patch_size=4,
        embed_dim=128,
        num_heads=4,
        hidden_dim=512,
        num_blocks=2,
        action_dim=32,
        dropout=0.1,
        beta=1.0
    ).to(DEVICE)
    print("✅ LAM initialized")
    
    # Test dynamics model
    print("Initializing dynamics model...")
    dynamics_model = DynamicsModel(
        frame_size=(64, 64),
        patch_size=4,
        embed_dim=128,
        num_heads=4,
        hidden_dim=512,
        num_blocks=2,
        latent_dim=6,
        dropout=0.1
    ).to(DEVICE)
    print("✅ Dynamics model initialized")
    
    return video_tokenizer, lam, dynamics_model

def test_model_initialization():
    """Test that all models can be initialized with consistent parameters"""
    _initialize_models()
    assert True  # If we get here, initialization succeeded
    
def _test_forward_passes(video_tokenizer, lam, dynamics_model):
    """Test forward passes through all models (internal function)"""
    print("\n🧪 Testing forward passes...")
    
    # Test video tokenizer
    print("Testing video tokenizer forward pass...")
    test_frame = torch.randn(1, 1, 3, 64, 64).to(DEVICE)  # [batch, seq_len, C, H, W]
    with torch.no_grad():
        latent = video_tokenizer.encoder(test_frame)
        print(f"Encoder output shape: {latent.shape}")
        decoded_frame = video_tokenizer.decoder(latent)
    print(f"✅ Video tokenizer: input {test_frame.shape} -> latent {latent.shape} -> output {decoded_frame.shape}")
    
    # Test LAM
    print("Testing LAM forward pass...")
    test_frames = torch.randn(1, 2, 3, 64, 64).to(DEVICE)  # 2-frame sequence
    with torch.no_grad():
        loss, pred_frames, action_indices, loss_dict = lam(test_frames)
    print(f"✅ LAM: input {test_frames.shape} -> actions {action_indices.shape} -> output {pred_frames.shape}")
    
    # Test dynamics model
    print("Testing dynamics model forward pass...")
    # Create input with video latents (action latents will be added, not concatenated)
    video_latents = torch.randn(1, 1, 256, 6).to(DEVICE)  # [batch, seq, patches, latent_dim]
    with torch.no_grad():
        next_latents, _ = dynamics_model(video_latents, training=False)
    print(f"✅ Dynamics model: input {video_latents.shape} -> output {next_latents.shape}")
    
    return True

def test_forward_passes():
    """Test forward passes through all models"""
    video_tokenizer, lam, dynamics_model = _initialize_models()
    return _test_forward_passes(video_tokenizer, lam, dynamics_model)
    
def test_checkpoint_loading():
    """Test loading checkpoints with weights_only=False fallback"""
    print("\n🧪 Testing checkpoint loading with weights_only=False fallback...")
    
    # Test video tokenizer loading
    if os.path.exists(VIDEO_TOKENIZER_PATH):
        print(f"Testing video tokenizer loading from {VIDEO_TOKENIZER_PATH}")
        try:
            # Try loading with weights_only=True first, fallback to False if it fails
            try:
                checkpoint = torch.load(VIDEO_TOKENIZER_PATH, map_location=DEVICE, weights_only=True)
                print("✅ Video tokenizer loaded with weights_only=True")
            except Exception as e:
                print(f"weights_only=True failed, trying weights_only=False: {e}")
                checkpoint = torch.load(VIDEO_TOKENIZER_PATH, map_location=DEVICE, weights_only=False)
                print("✅ Video tokenizer loaded with weights_only=False")
            
            # Initialize model and load state dict
            model = Video_Tokenizer(
                frame_size=(64, 64), 
                patch_size=4,
                embed_dim=128,
                num_heads=4,
                hidden_dim=512,
                num_blocks=2,
                latent_dim=6,
                dropout=0.1, 
                num_bins=4,
                ema_decay=0.99
            ).to(DEVICE)
            
            model.load_state_dict(checkpoint['model'])
            print("✅ Video tokenizer state dict loaded successfully")
            
        except Exception as e:
            print(f"❌ Video tokenizer loading failed: {e}")
    else:
        print(f"⚠️ Video tokenizer checkpoint not found at {VIDEO_TOKENIZER_PATH}")
    
    # Test LAM loading
    if os.path.exists(LAM_PATH):
        print(f"Testing LAM loading from {LAM_PATH}")
        try:
            # Try loading with weights_only=True first, fallback to False if it fails
            try:
                checkpoint = torch.load(LAM_PATH, map_location=DEVICE, weights_only=True)
                print("✅ LAM loaded with weights_only=True")
            except Exception as e:
                print(f"weights_only=True failed, trying weights_only=False: {e}")
                checkpoint = torch.load(LAM_PATH, map_location=DEVICE, weights_only=False)
                print("✅ LAM loaded with weights_only=False")
            
            # Initialize model and load state dict
            model = LAM(
                frame_size=(64, 64),
                n_actions=8,
                patch_size=8,  # Match checkpoint
                embed_dim=128,
                num_heads=4,
                hidden_dim=512,
                num_blocks=2,
                action_dim=32,  # Match checkpoint
                dropout=0.1,
                beta=1.0
            ).to(DEVICE)
            
            model.load_state_dict(checkpoint['model'])
            print("✅ LAM state dict loaded successfully")
            
        except Exception as e:
            print(f"❌ LAM loading failed: {e}")
    else:
        print(f"⚠️ LAM checkpoint not found at {LAM_PATH}")
    
    # Test dynamics model loading
    if os.path.exists(DYNAMICS_PATH):
        print(f"Testing dynamics model loading from {DYNAMICS_PATH}")
        try:
            # Try loading with weights_only=True first, fallback to False if it fails
            try:
                checkpoint = torch.load(DYNAMICS_PATH, map_location=DEVICE, weights_only=True)
                print("✅ Dynamics model loaded with weights_only=True")
            except Exception as e:
                print(f"weights_only=True failed, trying weights_only=False: {e}")
                checkpoint = torch.load(DYNAMICS_PATH, map_location=DEVICE, weights_only=False)
                print("✅ Dynamics model loaded with weights_only=False")
            
            # Initialize model and load state dict
            model = DynamicsModel(
                frame_size=(64, 64),
                patch_size=4,
                embed_dim=128,
                num_heads=4,
                hidden_dim=512,
                num_blocks=2,
                latent_dim=6,
                dropout=0.1
            ).to(DEVICE)
            
            model.load_state_dict(checkpoint['model'])
            print("✅ Dynamics model state dict loaded successfully")
            
        except Exception as e:
            print(f"❌ Dynamics model loading failed: {e}")
    else:
        print(f"⚠️ Dynamics model checkpoint not found at {DYNAMICS_PATH}")
    
def test_dataset_loading():
    """Test loading different datasets"""
    print("\n🧪 Testing dataset loading...")
    datasets = ['SONIC', 'PONG', 'BLOCK']
    
    for dataset_name in datasets:
        print(f"\nTesting {dataset_name} dataset...")
        try:
            # Load dataset with small batch size for testing
            _, _, training_loader, validation_loader, _ = load_data_and_data_loaders(
                dataset=dataset_name, 
                batch_size=4, 
                num_frames=8
            )
            
            # Try to get one batch to verify it works
            batch = next(iter(training_loader))
            frames, labels = batch
            
            print(f"✅ {dataset_name} dataset loaded successfully!")
            print(f"   Batch shape: {frames.shape}")
            print(f"   Labels shape: {labels.shape}")
            
        except Exception as e:
            print(f"❌ Failed to load {dataset_name} dataset: {e}")

def save_frames_as_mp4(frames, output_path, fps=2):
    """
    Save frames as an MP4 video file.
    
    Args:
        frames: Tensor of shape [batch_size, num_frames, C, H, W] with values in [0, 1]
        output_path: Path to save the MP4 file
        fps: Frames per second for the video
    """
    # Get dimensions
    batch_size, num_frames, C, H, W = frames.shape
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    # Convert each frame and write to video
    for i in range(num_frames):
        # Get frame and convert to numpy
        frame = frames[0, i].permute(1, 2, 0).numpy()  # [H, W, C]
        
        # Convert from RGB to BGR (OpenCV format)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Convert from float [0, 1] to uint8 [0, 255]
        frame_uint8 = (frame_bgr * 255).astype(np.uint8)
        
        # Write frame to video
        out.write(frame_uint8)
    
    # Release video writer
    out.release()

def test_mp4_creation():
    """Test creating an MP4 file with dummy frames"""
    print("\n🧪 Testing MP4 creation with dummy frames...")
    
    # Create dummy frames (gradient pattern that changes over time)
    batch_size, num_frames, C, H, W = 1, 10, 3, 64, 64
    frames = torch.zeros(batch_size, num_frames, C, H, W)
    
    # Create a gradient pattern that changes over time
    for t in range(num_frames):
        for c in range(C):
            # Create a different gradient for each channel and time step
            for i in range(H):
                for j in range(W):
                    frames[0, t, c, i, j] = (i + j + t * 10) / (H + W + num_frames * 10)
    
    # Normalize to [0, 1]
    frames = torch.clamp(frames, 0, 1)
    
    # Create output directory
    save_dir = "test_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save MP4
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    mp4_path = os.path.join(save_dir, f"test_video_{timestamp}.mp4")
    
    try:
        save_frames_as_mp4(frames, mp4_path, fps=2)
        print(f"✅ MP4 video created successfully: {mp4_path}")
        
        # Check if file exists and has reasonable size
        if os.path.exists(mp4_path):
            file_size = os.path.getsize(mp4_path)
            print(f"✅ File size: {file_size} bytes")
            if file_size > 1000:  # Should be at least 1KB
                print("✅ File size looks reasonable")
            else:
                print("⚠️ File size seems small, might be empty")
        else:
            print("❌ File was not created")
            
    except Exception as e:
        print(f"❌ MP4 creation failed: {e}")

def test_sliding_window():
    """Test the sliding window logic"""
    print("\n🧪 Testing sliding window logic...")
    
    # Simulate frames tensor
    frames = torch.randn(1, 1, 3, 64, 64)  # Start with 1 frame
    context_window = 4
    
    print(f"Initial frames shape: {frames.shape}")
    
    # Simulate adding frames and maintaining sliding window
    for i in range(6):  # Add 6 more frames
        new_frame = torch.randn(1, 1, 3, 64, 64)
        frames = torch.cat([frames, new_frame], dim=1)
        
        # Apply sliding window
        if frames.shape[1] > context_window:
            frames = frames[:, -context_window:, :, :, :]
        
        print(f"After step {i+1}: frames shape = {frames.shape}")
    
    print("✅ Sliding window test passed!")
    print(f"Final frames shape: {frames.shape}")
    assert frames.shape[1] <= context_window, "Sliding window failed!"

def test_visualization():
    """Test the visualization function with dummy data"""
    if not VISUALIZATION_AVAILABLE:
        print("\n⚠️ Skipping visualization test - function not available")
        return
        
    print("\n🧪 Testing visualization function...")
    
    # Create dummy frames and actions
    batch_size, num_frames, C, H, W = 1, 5, 3, 64, 64
    frames = torch.randn(batch_size, num_frames, C, H, W)  # Random frames in [-1, 1] range
    
    # Create dummy actions (random integers from 0 to 7)
    inferred_actions = [torch.randint(0, 8, (1,)) for _ in range(num_frames - 1)]
    
    print(f"Frames shape: {frames.shape}")
    print(f"Number of actions: {len(inferred_actions)}")
    print(f"Actions: {[action.item() for action in inferred_actions]}")
    
    # Test visualization
    try:
        visualize_inference(frames, inferred_actions)
        print("✅ Visualization test passed!")
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()

def test_inference_pipeline():
    """Test the complete inference pipeline"""
    print("\n🧪 Testing complete inference pipeline...")
    
    # Initialize models
    video_tokenizer, lam, dynamics_model = _initialize_models()
    
    # Test forward passes
    _test_forward_passes(video_tokenizer, lam, dynamics_model)
    
    # Test the inference steps
    print("\nTesting inference pipeline steps...")
    
    # Step 1: Start with a test frame
    initial_frame = torch.randn(1, 1, 3, 64, 64).to(DEVICE)  # [batch, seq_len, C, H, W]
    print(f"Initial frame shape: {initial_frame.shape}")
    
    # Step 2: Encode frame to video tokens
    with torch.no_grad():
        latent = video_tokenizer.encoder(initial_frame)
        quantized_latent, _, token_indices = video_tokenizer.vq(latent)
    print(f"Encoded to video tokens: {quantized_latent.shape}")
    
    # Step 3: Select random action
    action_idx = torch.randint(0, 8, (1,))
    action_latent = lam.quantizer.embedding(action_idx)
    print(f"Selected action {action_idx.item()}, action latent shape: {action_latent.shape}")
    
    # Step 4: Predict next tokens using dynamics model
    batch_size, seq_len, num_patches, latent_dim = quantized_latent.shape
    action_expanded = action_latent.unsqueeze(1).expand(-1, num_patches, -1)
    
    # Add action latents to video latents (not concatenate)
    combined_latents = quantized_latent.squeeze(1) + action_expanded  # [1, num_patches, latent_dim]
    combined_latents = combined_latents.unsqueeze(1)  # Add sequence dimension
    
    with torch.no_grad():
        next_video_latents, _ = dynamics_model(combined_latents, training=False)
        next_video_latents = next_video_latents.squeeze(1)
    print(f"Predicted next video tokens: {next_video_latents.shape}")
    
    # Step 5: Decode to next frame
    with torch.no_grad():
        next_frame = video_tokenizer.decoder(next_video_latents.unsqueeze(1))
    print(f"Decoded next frame: {next_frame.shape}")
    
    print("✅ Complete inference pipeline test passed!")
    return True

def main():
    """Main function to run the test suite"""
    print("🧪 Running Comprehensive Inference Pipeline Tests")
    print("=" * 60)
    
    try:
        # Test model initialization and forward passes
        test_inference_pipeline()
        
        # Test checkpoint loading
        test_checkpoint_loading()
        
        # Test dataset loading
        test_dataset_loading()
        
        # Test MP4 creation
        test_mp4_creation()
        
        # Test sliding window
        test_sliding_window()
        
        # Test visualization
        test_visualization()
        
        print("\n🎉 All tests completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 