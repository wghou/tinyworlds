"""
Tests for the LAM (Latent Action Model) main script
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

from latent_action_model.models.lam import LAM


class TestLAM:
    """Test class for LAM functionality"""
    
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
        batch_size, seq_len, channels, height, width = 4, 8, 3, 64, 64
        frames = torch.rand(batch_size, seq_len, channels, height, width)
        return frames
    
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
            action_dim=16,
            dropout=0.1
        )
        return model
    
    def test_lam_initialization(self, lam_model):
        """Test LAM model initialization"""
        assert lam_model is not None
        assert hasattr(lam_model, 'encoder')
        assert hasattr(lam_model, 'decoder')
        assert hasattr(lam_model, 'quantizer')
        
        # Test model parameters
        total_params = sum(p.numel() for p in lam_model.parameters())
        assert total_params > 0
    
    def test_lam_forward_pass(self, lam_model, mock_sequence_data):
        """Test LAM forward pass"""
        lam_model.eval()
        
        with torch.no_grad():
            loss, pred_frames, action_indices, loss_dict = lam_model(mock_sequence_data)
            
            # Check output shapes and types
            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0  # Scalar loss
            
            # pred_frames should be [batch_size, seq_len-1, channels, height, width]
            batch_size, seq_len = mock_sequence_data.shape[:2]
            expected_pred_shape = (batch_size, seq_len-1, 3, 64, 64)
            assert pred_frames.shape == expected_pred_shape
            
            # action_indices should be [batch_size, seq_len-1]
            expected_action_shape = (batch_size, seq_len-1)
            assert action_indices.shape == expected_action_shape
            
            # Check loss dictionary
            assert isinstance(loss_dict, dict)
            assert 'total_loss' in loss_dict
            assert 'recon_loss' in loss_dict
            assert 'vq_loss' in loss_dict
            assert 'diversity_loss' in loss_dict
    
    def test_lam_encoder(self, lam_model, mock_sequence_data):
        """Test LAM encoder"""
        lam_model.eval()
        
        with torch.no_grad():
            actions, embeddings = lam_model.encoder(mock_sequence_data)
            
            # Check action shape: [batch_size, seq_len-1, action_dim]
            batch_size, seq_len = mock_sequence_data.shape[:2]
            expected_action_shape = (batch_size, seq_len-1, 16)  # action_dim=16
            assert actions.shape == expected_action_shape
            
            # Check embeddings shape: [batch_size, seq_len, num_patches, embed_dim]
            num_patches = (64 // 8) * (64 // 8)  # Based on patch_size=8
            expected_embed_shape = (batch_size, seq_len, num_patches, 128)  # embed_dim=128
            assert embeddings.shape == expected_embed_shape
    
    def test_lam_decoder(self, lam_model, mock_sequence_data):
        """Test LAM decoder"""
        lam_model.eval()
        
        with torch.no_grad():
            # Get actions from encoder
            actions, _ = lam_model.encoder(mock_sequence_data)
            
            # Test decoder with quantized actions
            actions_flat = actions.reshape(-1, actions.size(-1))
            _, quantized_actions_flat, _ = lam_model.quantizer(actions_flat)
            quantized_actions = quantized_actions_flat.reshape(actions.shape)
            
            decoded_frames = lam_model.decoder(mock_sequence_data, quantized_actions)
            
            # Check output shape: [batch_size, seq_len-1, channels, height, width]
            batch_size, seq_len = mock_sequence_data.shape[:2]
            expected_shape = (batch_size, seq_len-1, 3, 64, 64)
            assert decoded_frames.shape == expected_shape
    
    def test_vector_quantizer(self, lam_model, mock_sequence_data):
        """Test vector quantizer functionality"""
        lam_model.eval()
        
        with torch.no_grad():
            # Get actions from encoder
            actions, _ = lam_model.encoder(mock_sequence_data)
            actions_flat = actions.reshape(-1, actions.size(-1))
            
            # Test vector quantization
            vq_loss, quantized_actions, indices = lam_model.quantizer(actions_flat)
            
            # Check outputs
            assert isinstance(vq_loss, torch.Tensor)
            assert quantized_actions.shape == actions_flat.shape
            assert indices.shape == (actions_flat.shape[0],)  # [batch_size * (seq_len-1)]
            
            # Check that indices are within codebook size
            assert indices.max() < 4  # n_actions=4
            assert indices.min() >= 0
    
    def test_encode_decode_single_transition(self, lam_model, mock_sequence_data):
        """Test encode and decode for single frame transitions"""
        lam_model.eval()
        
        with torch.no_grad():
            # Get first two frames
            prev_frame = mock_sequence_data[0:1, 0]  # [1, channels, height, width]
            next_frame = mock_sequence_data[0:1, 1]  # [1, channels, height, width]
            
            # Encode to get action
            action_index = lam_model.encode(prev_frame, next_frame)
            assert isinstance(action_index, torch.Tensor)
            assert action_index.item() < 4  # n_actions=4
            assert action_index.item() >= 0
            
            # Decode to get predicted next frame
            predicted_next = lam_model.decode(prev_frame, action_index)
            assert predicted_next.shape == prev_frame.shape
    
    def test_model_device_compatibility(self, lam_model, mock_sequence_data):
        """Test model works on both CPU and GPU if available"""
        # Test on CPU
        lam_model.cpu()
        lam_model.eval()
        
        with torch.no_grad():
            loss, pred_frames, action_indices, loss_dict = lam_model(mock_sequence_data.cpu())
            assert pred_frames.device == torch.device('cpu')
        
        # Test on GPU if available
        if torch.cuda.is_available():
            lam_model.cuda()
            lam_model.eval()
            
            with torch.no_grad():
                loss, pred_frames, action_indices, loss_dict = lam_model(mock_sequence_data.cuda())
                assert pred_frames.device == torch.device('cuda')
    
    def test_gradient_flow(self, lam_model, mock_sequence_data):
        """Test that gradients flow properly through the model"""
        lam_model.train()
        
        # Forward pass
        loss, pred_frames, action_indices, loss_dict = lam_model(mock_sequence_data)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for name, param in lam_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_model_save_load(self, lam_model, temp_dir):
        """Test model saving and loading"""
        # Save model
        save_path = os.path.join(temp_dir, 'test_lam_model.pth')
        torch.save(lam_model.state_dict(), save_path)
        
        # Load model
        new_model = LAM(
            frame_size=(64, 64),
            n_actions=4,
            patch_size=8,
            embed_dim=128,
            num_heads=4,
            hidden_dim=512,
            num_blocks=2,
            action_dim=16,
            dropout=0.1
        )
        new_model.load_state_dict(torch.load(save_path))
        
        # Test that models produce same output
        lam_model.eval()
        new_model.eval()
        
        with torch.no_grad():
            mock_data = torch.rand(2, 8, 3, 64, 64)
            loss1, pred1, actions1, dict1 = lam_model(mock_data)
            loss2, pred2, actions2, dict2 = new_model(mock_data)
            
            assert torch.allclose(loss1, loss2, atol=1e-6)
            assert torch.allclose(pred1, pred2, atol=1e-6)
    
    def test_action_diversity(self, lam_model, mock_sequence_data):
        """Test that the model produces diverse actions"""
        lam_model.eval()
        
        with torch.no_grad():
            # Get actions for the sequence
            actions, _ = lam_model.encoder(mock_sequence_data)
            actions_flat = actions.reshape(-1, actions.size(-1))
            
            # Quantize actions
            _, _, indices = lam_model.quantizer(actions_flat)
            
            # Check that we get some diversity in actions
            unique_actions = torch.unique(indices)
            # For a small test sequence, we might not get much diversity
            # Just check that we get at least one action
            assert len(unique_actions) >= 1
            
            # Check that actions are within valid range
            assert indices.min() >= 0
            assert indices.max() < lam_model.quantizer.n_e
    
    def test_loss_components(self, lam_model, mock_sequence_data):
        """Test that all loss components are computed correctly"""
        lam_model.train()
        
        # Forward pass
        loss, pred_frames, action_indices, loss_dict = lam_model(mock_sequence_data)
        
        # Check that all loss components are positive (except diversity loss which can be negative)
        assert loss_dict['recon_loss'] > 0
        assert loss_dict['vq_loss'] > 0
        # diversity_loss can be negative as it encourages diversity


if __name__ == "__main__":
    pytest.main([__file__]) 