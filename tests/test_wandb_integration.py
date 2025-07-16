"""
Tests for Weights & Biases integration
"""

import pytest
import os
import sys
import tempfile
import shutil
import subprocess
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestWandbIntegration:
    """Test class for W&B integration functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_wandb_utils_import(self):
        """Test that wandb utilities can be imported"""
        try:
            from src.utils.wandb_utils import (
                init_wandb, log_training_metrics, log_model_gradients,
                log_model_parameters, log_learning_rate, log_reconstruction_comparison,
                log_video_sequence, log_codebook_usage, log_action_distribution,
                log_system_metrics, finish_wandb, create_wandb_config
            )
            assert True  # Import successful
        except ImportError as e:
            pytest.fail(f"Failed to import wandb utilities: {e}")
    
    def test_wandb_config_creation(self):
        """Test W&B config creation"""
        from src.utils.wandb_utils import create_wandb_config
        
        # Mock args object
        class MockArgs:
            def __init__(self):
                self.batch_size = 16
                self.n_updates = 1000
                self.learning_rate = 1e-4
                self.log_interval = 100
                self.dataset = 'SONIC'
                self.context_length = 4
                self.seq_length = 8
        
        args = MockArgs()
        model_config = {
            'patch_size': 4,
            'embed_dim': 128,
            'num_heads': 4,
            'hidden_dim': 512,
            'num_blocks': 2,
            'latent_dim': 32,
            'dropout': 0.1
        }
        
        config = create_wandb_config(args, model_config)
        
        # Check that config contains expected keys
        assert 'batch_size' in config
        assert 'n_updates' in config
        assert 'learning_rate' in config
        assert 'model_architecture' in config
        assert config['batch_size'] == 16
        assert config['model_architecture']['patch_size'] == 4
    
    @patch('wandb.init')
    def test_wandb_initialization(self, mock_wandb_init):
        """Test W&B initialization"""
        from src.utils.wandb_utils import init_wandb
        
        # Mock wandb.run
        mock_run = MagicMock()
        mock_run.name = "test_run"
        mock_run.get_url.return_value = "https://wandb.ai/test/project/runs/test_run"
        mock_wandb_init.return_value = mock_run
        
        config = {'batch_size': 16, 'learning_rate': 1e-4}
        
        run = init_wandb("test-project", config, "test_run")
        
        # Verify wandb.init was called with correct arguments
        mock_wandb_init.assert_called_once()
        call_args = mock_wandb_init.call_args
        assert call_args[1]['project'] == "test-project"
        assert call_args[1]['config'] == config
        assert call_args[1]['name'] == "test_run"
    
    def test_video_tokenizer_wandb_args(self, temp_dir):
        """Test that video tokenizer accepts W&B arguments"""
        # This test verifies that the argument parser includes W&B options
        from src.vqvae.main import parser
        
        # Test parsing W&B arguments
        test_args = [
            '--use_wandb',
            '--wandb_project', 'test-project',
            '--wandb_run_name', 'test-run'
        ]
        
        parsed_args = parser.parse_args(test_args)
        
        assert parsed_args.use_wandb is True
        assert parsed_args.wandb_project == 'test-project'
        assert parsed_args.wandb_run_name == 'test-run'
    
    def test_lam_wandb_args(self, temp_dir):
        """Test that LAM accepts W&B arguments"""
        from src.latent_action_model.main import parser
        
        # Test parsing W&B arguments
        test_args = [
            '--use_wandb',
            '--wandb_project', 'test-project',
            '--wandb_run_name', 'test-run'
        ]
        
        parsed_args = parser.parse_args(test_args)
        
        assert parsed_args.use_wandb is True
        assert parsed_args.wandb_project == 'test-project'
        assert parsed_args.wandb_run_name == 'test-run'
    
    def test_dynamics_wandb_args(self, temp_dir):
        """Test that dynamics model accepts W&B arguments"""
        from src.dynamics.main import parser
        
        # Test parsing W&B arguments
        test_args = [
            '--use_wandb',
            '--wandb_project', 'test-project',
            '--wandb_run_name', 'test-run',
            '--video_tokenizer_path', '/path/to/vt.pth',
            '--lam_path', '/path/to/lam.pth'
        ]
        
        parsed_args = parser.parse_args(test_args)
        
        assert parsed_args.use_wandb is True
        assert parsed_args.wandb_project == 'test-project'
        assert parsed_args.wandb_run_name == 'test-run'
    
    def test_full_pipeline_wandb_args(self, temp_dir):
        """Test that full pipeline accepts W&B arguments"""
        from train_full_pipeline import main
        
        # Mock argument parsing
        with patch('argparse.ArgumentParser.parse_args') as mock_parse:
            mock_args = MagicMock()
            mock_args.use_wandb = True
            mock_args.wandb_project = "test-pipeline"
            mock_parse.return_value = mock_args
            
            # Mock directory and file existence
            with patch('os.path.exists', return_value=True), \
                 patch('builtins.print') as mock_print:
                
                # Mock subprocess calls
                with patch('subprocess.run') as mock_run:
                    mock_run.return_value = MagicMock(returncode=0)
                    
                    # Mock checkpoint finding
                    with patch('train_full_pipeline.find_latest_checkpoint') as mock_find:
                        mock_find.return_value = 'mock_checkpoint.pth'
                        
                        # Run the pipeline
                        main()
                        
                        # Verify W&B arguments were passed to subprocess
                        subprocess_calls = mock_run.call_args_list
                        assert len(subprocess_calls) > 0
                        
                        # Check that W&B arguments are included in dynamics command
                        dynamics_call = subprocess_calls[-1]  # Last call should be dynamics
                        cmd = dynamics_call[0][0]
                        assert '--use_wandb' in cmd
                        assert '--wandb_project' in cmd
    
    def test_wandb_metrics_logging(self):
        """Test W&B metrics logging functions"""
        from src.utils.wandb_utils import log_training_metrics
        import torch
        
        # Mock wandb.log
        with patch('wandb.log') as mock_log:
            metrics = {
                'loss': 0.5,
                'accuracy': 0.8
            }
            
            log_training_metrics(100, metrics, "train")
            
            # Verify wandb.log was called with correct arguments
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            
            assert 'train/loss' in call_args
            assert 'train/accuracy' in call_args
            assert call_args['step'] == 100
            assert call_args['train/loss'] == 0.5
            assert call_args['train/accuracy'] == 0.8
    
    def test_wandb_reconstruction_logging(self):
        """Test W&B reconstruction comparison logging"""
        from src.utils.wandb_utils import log_reconstruction_comparison
        import torch
        
        # Create mock tensors
        original = torch.randn(4, 3, 64, 64)  # [B, C, H, W]
        reconstructed = torch.randn(4, 3, 64, 64)
        
        # Mock wandb.log
        with patch('wandb.log') as mock_log:
            log_reconstruction_comparison(original, reconstructed, 100)
            
            # Verify wandb.log was called
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            
            assert 'reconstruction_comparison' in call_args
            assert call_args['step'] == 100
    
    def test_wandb_video_logging(self):
        """Test W&B video sequence logging"""
        from src.utils.wandb_utils import log_video_sequence
        import torch
        
        # Create mock video tensor
        video = torch.randn(1, 8, 3, 64, 64)  # [B, T, C, H, W]
        
        # Mock wandb.log
        with patch('wandb.log') as mock_log:
            log_video_sequence(video, 100, "Test Video")
            
            # Verify wandb.log was called
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            
            assert 'video_sequence' in call_args
            assert call_args['step'] == 100


if __name__ == "__main__":
    pytest.main([__file__])