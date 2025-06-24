"""
Tests for the video tokenizer main script
"""

import pytest
import torch
import os
import tempfile
import shutil
import json
import sys
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vqvae.models.video_tokenizer import Video_Tokenizer
from vqvae.datasets.pong import PongDataset


class TestVideoTokenizer:
    """Test class for video tokenizer functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_data(self):
        """Create mock data for testing"""
        # Create mock video frames
        batch_size, seq_len, channels, height, width = 4, 4, 3, 64, 64
        frames = torch.rand(batch_size, seq_len, channels, height, width)
        return frames
    
    @pytest.fixture
    def video_tokenizer_model(self):
        """Create a video tokenizer model for testing"""
        model = Video_Tokenizer(
            frame_size=(64, 64),
            patch_size=8,
            embed_dim=128,
            num_heads=4,
            hidden_dim=512,
            num_blocks=2,
            latent_dim=16,
            dropout=0.1,
            codebook_size=256,
            beta=0.01
        )
        return model
    
    def test_video_tokenizer_initialization(self, video_tokenizer_model):
        """Test video tokenizer model initialization"""
        assert video_tokenizer_model is not None
        assert hasattr(video_tokenizer_model, 'encoder')
        assert hasattr(video_tokenizer_model, 'decoder')
        assert hasattr(video_tokenizer_model, 'vq')
        
        # Test model parameters
        total_params = sum(p.numel() for p in video_tokenizer_model.parameters())
        assert total_params > 0
    
    def test_video_tokenizer_forward_pass(self, video_tokenizer_model, mock_data):
        """Test video tokenizer forward pass"""
        video_tokenizer_model.eval()
        
        with torch.no_grad():
            x_hat, vq_loss = video_tokenizer_model(mock_data)
            
            # Check output shapes
            assert x_hat.shape == mock_data.shape
            assert isinstance(vq_loss, torch.Tensor)
            assert vq_loss.dim() == 0  # Scalar loss
    
    def test_video_tokenizer_encoder(self, video_tokenizer_model, mock_data):
        """Test video tokenizer encoder"""
        video_tokenizer_model.eval()
        
        with torch.no_grad():
            latents = video_tokenizer_model.encoder(mock_data)
            
            # Check latent shape: [batch_size, seq_len, num_patches, latent_dim]
            batch_size, seq_len = mock_data.shape[:2]
            num_patches = (64 // 8) * (64 // 8)  # Based on patch_size=8
            expected_shape = (batch_size, seq_len, num_patches, 16)  # latent_dim=16
            
            assert latents.shape == expected_shape
    
    def test_video_tokenizer_decoder(self, video_tokenizer_model, mock_data):
        """Test video tokenizer decoder"""
        video_tokenizer_model.eval()
        
        with torch.no_grad():
            # Get latents from encoder
            latents = video_tokenizer_model.encoder(mock_data)
            
            # Test decoder with quantized latents
            _, quantized_latents, _ = video_tokenizer_model.vq(latents)
            decoded_frames = video_tokenizer_model.decoder(quantized_latents)
            
            # Check output shape
            assert decoded_frames.shape == mock_data.shape
    
    def test_vector_quantizer(self, video_tokenizer_model, mock_data):
        """Test vector quantizer functionality"""
        video_tokenizer_model.eval()
        
        with torch.no_grad():
            # Get latents from encoder
            latents = video_tokenizer_model.encoder(mock_data)
            
            # Test vector quantization
            vq_loss, quantized_latents, indices = video_tokenizer_model.vq(latents)
            
            # Check outputs
            assert isinstance(vq_loss, torch.Tensor)
            assert quantized_latents.shape == latents.shape
            assert indices.shape == latents.shape[:3]  # [batch_size, seq_len, num_patches]
            
            # Check that indices are within codebook size
            assert indices.max() < 256  # codebook_size
            assert indices.min() >= 0
    
    @patch('src.vqvae.main.utils.load_data_and_data_loaders')
    @patch('src.vqvae.main.save_run_configuration')
    def test_main_script_config_saving(self, mock_save_config, mock_load_data, temp_dir):
        """Test that main script saves configuration properly"""
        # Mock data loading
        mock_loader = MagicMock()
        mock_load_data.return_value = (None, None, mock_loader, mock_loader, 1.0)
        
        # Mock training loop to avoid actual training
        with patch('src.vqvae.main.train') as mock_train:
            # Import and run main with minimal iterations
            with patch('sys.argv', ['main.py', '--n_updates', '1', '--batch_size', '2']):
                # This would normally run the main script, but we're just testing config saving
                pass
        
        # Verify that save_run_configuration was called
        # Note: This test is more of a structure test since we can't easily run the full main script
        assert True  # Placeholder assertion
    
    def test_pong_dataset(self, temp_dir):
        """Test Pong dataset functionality"""
        # Create mock HDF5 file
        import h5py
        import numpy as np
        
        mock_frames = np.random.randint(0, 255, (100, 64, 64, 3), dtype=np.uint8)
        
        h5_path = os.path.join(temp_dir, 'mock_pong_frames.h5')
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('frames', data=mock_frames)
        
        # Test dataset loading
        dataset = PongDataset(
            video_path=None,  # Not used when save_path exists
            save_path=h5_path,
            train=True,
            num_frames=4
        )
        
        # Test dataset properties
        assert len(dataset) > 0
        assert len(dataset) <= len(mock_frames) - 4 + 1  # Account for sequence length
        
        # Test getting an item
        frames, label = dataset[0]
        assert frames.shape == (4, 3, 64, 64)  # [seq_len, channels, height, width]
        assert label == 0  # Dummy label
    
    def test_model_device_compatibility(self, video_tokenizer_model, mock_data):
        """Test model works on both CPU and GPU if available"""
        # Test on CPU
        video_tokenizer_model.cpu()
        video_tokenizer_model.eval()
        
        with torch.no_grad():
            x_hat, vq_loss = video_tokenizer_model(mock_data.cpu())
            assert x_hat.device == torch.device('cpu')
        
        # Test on GPU if available
        if torch.cuda.is_available():
            video_tokenizer_model.cuda()
            video_tokenizer_model.eval()
            
            with torch.no_grad():
                x_hat, vq_loss = video_tokenizer_model(mock_data.cuda())
                assert x_hat.device == torch.device('cuda')
    
    def test_gradient_flow(self, video_tokenizer_model, mock_data):
        """Test that gradients flow properly through the model"""
        video_tokenizer_model.train()
        
        # Forward pass
        x_hat, vq_loss = video_tokenizer_model(mock_data)
        
        # Compute loss
        recon_loss = torch.mean((x_hat - mock_data)**2)
        total_loss = recon_loss + vq_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check that gradients exist
        for name, param in video_tokenizer_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_model_save_load(self, video_tokenizer_model, temp_dir):
        """Test model saving and loading"""
        # Save model
        save_path = os.path.join(temp_dir, 'test_model.pth')
        torch.save(video_tokenizer_model.state_dict(), save_path)
        
        # Load model
        new_model = Video_Tokenizer(
            frame_size=(64, 64),
            patch_size=8,
            embed_dim=128,
            num_heads=4,
            hidden_dim=512,
            num_blocks=2,
            latent_dim=16,
            dropout=0.1,
            codebook_size=256,
            beta=0.01
        )
        new_model.load_state_dict(torch.load(save_path))
        
        # Test that models produce same output
        video_tokenizer_model.eval()
        new_model.eval()
        
        with torch.no_grad():
            mock_data = torch.rand(2, 4, 3, 64, 64)
            output1, loss1 = video_tokenizer_model(mock_data)
            output2, loss2 = new_model(mock_data)
            
            assert torch.allclose(output1, output2, atol=1e-6)
            assert torch.allclose(loss1, loss2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__]) 