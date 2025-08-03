"""
Tests for the dynamics model main script
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

from dynamics.models.dynamics_model import DynamicsModel
from vqvae.models.video_tokenizer import Video_Tokenizer
from latent_action_model.models.lam import LAM


class TestDynamicsModel:
    """Test class for dynamics model functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_sequence_data(self):
        """Create mock sequence data for testing"""
        # Create mock frame sequences
        batch_size, seq_len, channels, height, width = 4, 4, 3, 64, 64
        frames = torch.rand(batch_size, seq_len, channels, height, width)
        return frames
    
    @pytest.fixture
    def dynamics_model(self):
        """Create a dynamics model for testing"""
        model = DynamicsModel(
            frame_size=(64, 64),
            patch_size=8,
            embed_dim=128,
            num_heads=4,
            hidden_dim=512,
            num_blocks=2,
            latent_dim=6,
            dropout=0.1,
            height=64,
            width=64,
            channels=3
        )
        return model
    
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
            latent_dim=6,
            dropout=0.1,
            num_bins=4,
            beta=0.01
        )
        return model
    
    @pytest.fixture
    def lam_model(self):
        """Create a LAM model for testing"""
        model = LAM(
            frame_size=(64, 64),
            n_actions=4,
            patch_size=8,
            embed_dim=128,
            num_heads=4,
            hidden_dim=512,
            num_blocks=2,
            action_dim=6,
            dropout=0.1
        )
        return model
    
    def test_dynamics_model_initialization(self, dynamics_model):
        """Test dynamics model initialization"""
        assert dynamics_model is not None
        # Check for decoder which contains the actual components
        assert hasattr(dynamics_model, 'decoder')
        assert hasattr(dynamics_model.decoder, 'patch_embed')
        assert hasattr(dynamics_model.decoder, 'transformer')
        assert hasattr(dynamics_model.decoder, 'latent_embed')
        assert hasattr(dynamics_model.decoder, 'latent_head')
        
        # Test model parameters
        total_params = sum(p.numel() for p in dynamics_model.parameters())
        assert total_params > 0
    
    def test_dynamics_model_forward_pass(self, dynamics_model, mock_sequence_data):
        """Test dynamics model forward pass"""
        dynamics_model.eval()
        
        # Create mock latents: [batch_size, seq_len, num_patches, latent_dim]
        batch_size, seq_len = mock_sequence_data.shape[:2]
        num_patches = (64 // 8) * (64 // 8)  # Based on patch_size=8
        latents = torch.rand(batch_size, seq_len, num_patches, 6)  # latent_dim=6
        
        with torch.no_grad():
            predicted_latents, _ = dynamics_model(latents, training=False)
            
            # Check output shape: should be same as input latents
            assert predicted_latents.shape == latents.shape
    
    def test_dynamics_model_training_mode(self, dynamics_model, mock_sequence_data):
        """Test dynamics model in training mode with masking"""
        dynamics_model.train()
        
        # Create mock latents
        batch_size, seq_len = mock_sequence_data.shape[:2]
        num_patches = (64 // 8) * (64 // 8)
        latents = torch.rand(batch_size, seq_len, num_patches, 6)
        
        with torch.no_grad():
            predicted_latents, _ = dynamics_model(latents, training=True)
            
            # Check output shape
            assert predicted_latents.shape == latents.shape
    
    def test_full_pipeline_integration(self, dynamics_model, video_tokenizer_model, lam_model, mock_sequence_data):
        """Test the full pipeline integration"""
        # Set all models to eval mode
        dynamics_model.eval()
        video_tokenizer_model.eval()
        lam_model.eval()
        
        with torch.no_grad():
            # Step 1: Get video tokenizer latents
            video_latents = video_tokenizer_model.encoder(mock_sequence_data)
            _, quantized_video_latents, _ = video_tokenizer_model.vq(video_latents)
            
            # Step 2: Get action latents
            actions, _ = lam_model.encoder(mock_sequence_data)
            actions_flat = actions.reshape(-1, actions.size(-1))
            _, quantized_actions_flat, _ = lam_model.quantizer(actions_flat)
            quantized_actions = quantized_actions_flat.reshape(actions.shape)
            
            # Step 3: Pad actions to match video latents sequence length
            batch_size, seq_len, num_patches, latent_dim = quantized_video_latents.shape
            zero_action = torch.zeros(batch_size, 1, latent_dim)
            quantized_actions_padded = torch.cat([quantized_actions, zero_action], dim=1)
            quantized_actions_padded = quantized_actions_padded.unsqueeze(2).expand(-1, -1, num_patches, -1)
            
            # Step 4: Combine latents
            combined_latents = quantized_video_latents + quantized_actions_padded
            
            # Step 5: Predict next latents using dynamics model
            predicted_next_latents, _ = dynamics_model(combined_latents, training=False)
            
            # Check shapes
            assert predicted_next_latents.shape == combined_latents.shape
            assert predicted_next_latents.shape == quantized_video_latents.shape
    
    def test_model_device_compatibility(self, dynamics_model, mock_sequence_data):
        """Test model works on both CPU and GPU if available"""
        # Create mock latents
        batch_size, seq_len = mock_sequence_data.shape[:2]
        num_patches = (64 // 8) * (64 // 8)
        latents = torch.rand(batch_size, seq_len, num_patches, 6)
        
        # Test on CPU
        dynamics_model.cpu()
        dynamics_model.eval()
        
        with torch.no_grad():
            predicted_latents, _ = dynamics_model(latents.cpu())
            assert predicted_latents.device == torch.device('cpu')
        
        # Test on GPU if available
        if torch.cuda.is_available():
            dynamics_model.cuda()
            dynamics_model.eval()
            
            with torch.no_grad():
                predicted_latents, _ = dynamics_model(latents.cuda())
                assert predicted_latents.device == torch.device('cuda')
    
    def test_gradient_flow(self, dynamics_model, mock_sequence_data):
        """Test that gradients flow properly through the model"""
        dynamics_model.train()
        
        # Create mock latents
        batch_size, seq_len = mock_sequence_data.shape[:2]
        num_patches = (64 // 8) * (64 // 8)
        latents = torch.rand(batch_size, seq_len, num_patches, 6)
        
        # Forward pass
        predicted_latents, _ = dynamics_model(latents, training=True)
        
        # Compute loss
        loss = torch.mean((predicted_latents - latents)**2)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist for key parameters
        # Focus on the main components that should definitely have gradients
        key_params = [
            'decoder.latent_embed.weight',
            'decoder.latent_embed.bias',
            'decoder.latent_head.weight',
            'decoder.latent_head.bias'
        ]
        
        for param_name in key_params:
            param = dict(dynamics_model.named_parameters())[param_name]
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {param_name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {param_name}"
        
        # Check that at least some parameters have gradients
        has_gradients = False
        for name, param in dynamics_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                break
        assert has_gradients, "No gradients found in any parameters"
    
    def test_model_save_load(self, dynamics_model, temp_dir):
        """Test model saving and loading"""
        # Save model
        save_path = os.path.join(temp_dir, 'test_dynamics_model.pth')
        torch.save(dynamics_model.state_dict(), save_path)
        
        # Load model
        new_model = DynamicsModel(
            frame_size=(64, 64),
            patch_size=8,
            embed_dim=128,
            num_heads=4,
            hidden_dim=512,
            num_blocks=2,
            latent_dim=6,
            dropout=0.1,
            height=64,
            width=64,
            channels=3
        )
        new_model.load_state_dict(torch.load(save_path))
        
        # Test that models produce same output
        dynamics_model.eval()
        new_model.eval()
        
        with torch.no_grad():
            latents = torch.rand(2, 4, 64, 6)  # [batch_size, seq_len, num_patches, latent_dim]
            output1, _ = dynamics_model(latents)
            output2, _ = new_model(latents)
            
            assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_latent_consistency(self, dynamics_model, mock_sequence_data):
        """Test that the model maintains latent space consistency"""
        dynamics_model.eval()
        
        # Create mock latents
        batch_size, seq_len = mock_sequence_data.shape[:2]
        num_patches = (64 // 8) * (64 // 8)
        latents = torch.rand(batch_size, seq_len, num_patches, 6)
        
        with torch.no_grad():
            predicted_latents, _ = dynamics_model(latents, training=False)
            
            # Check that predicted latents have reasonable values
            assert not torch.isnan(predicted_latents).any()
            assert not torch.isinf(predicted_latents).any()
            
            # Check that predictions are not identical to inputs (model should learn something)
            # This is a weak test, but better than nothing
            assert not torch.allclose(predicted_latents, latents, atol=1e-3)
    
    def test_masking_effect(self, dynamics_model, mock_sequence_data):
        """Test that masking works in training mode"""
        dynamics_model.train()
        
        # Create mock latents
        batch_size, seq_len = mock_sequence_data.shape[:2]
        num_patches = (64 // 8) * (64 // 8)
        latents = torch.rand(batch_size, seq_len, num_patches, 6)
        
        with torch.no_grad():
            # Test with training=True (should apply masking)
            predicted_training, mask_training = dynamics_model(latents, training=True)
            
            # Test with training=False (should not apply masking)
            predicted_eval, mask_eval = dynamics_model(latents, training=False)
            
            # Both should have same shape
            assert predicted_training.shape == predicted_eval.shape
            
            # During training, mask should be returned; during eval, it should be None
            assert mask_training is not None or not dynamics_model.training
            assert mask_eval is None
    
    def test_checkpoint_loading(self, dynamics_model, video_tokenizer_model, lam_model, temp_dir):
        """Test checkpoint loading functionality"""
        # Create mock checkpoints
        vt_checkpoint = {'model': video_tokenizer_model.state_dict()}
        lam_checkpoint = {'model': lam_model.state_dict()}
        dynamics_checkpoint = {'model': dynamics_model.state_dict()}
        
        # Save checkpoints
        vt_path = os.path.join(temp_dir, 'vt_checkpoint.pth')
        lam_path = os.path.join(temp_dir, 'lam_checkpoint.pth')
        dynamics_path = os.path.join(temp_dir, 'dynamics_checkpoint.pth')
        
        torch.save(vt_checkpoint, vt_path)
        torch.save(lam_checkpoint, lam_path)
        torch.save(dynamics_checkpoint, dynamics_path)
        
        # Test loading
        loaded_vt_checkpoint = torch.load(vt_path)
        loaded_lam_checkpoint = torch.load(lam_path)
        loaded_dynamics_checkpoint = torch.load(dynamics_path)
        
        assert 'model' in loaded_vt_checkpoint
        assert 'model' in loaded_lam_checkpoint
        assert 'model' in loaded_dynamics_checkpoint


if __name__ == "__main__":
    pytest.main([__file__]) 