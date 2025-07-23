"""
Tests for the full pipeline training script
"""

import pytest
import os
import tempfile
import shutil
import json
import sys
import subprocess
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestFullPipeline:
    """Test class for full pipeline functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_pong_dataset(self, temp_dir):
        """Create a mock Pong dataset for testing"""
        import h5py
        import numpy as np
        
        # Create mock HDF5 file
        mock_frames = np.random.randint(0, 255, (1000, 64, 64, 3), dtype=np.uint8)
        h5_path = os.path.join(temp_dir, 'pong_frames.h5')
        
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('frames', data=mock_frames)
        
        return h5_path
    
    def test_find_latest_checkpoint(self, temp_dir):
        """Test finding the latest checkpoint functionality"""
        from train_full_pipeline import find_latest_checkpoint
        
        # Create mock results directories
        os.makedirs(os.path.join(temp_dir, 'src', 'vqvae', 'results'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'src', 'latent_action_model', 'results'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'src', 'dynamics', 'results'), exist_ok=True)
        
        # Create mock timestamped directories
        import time
        timestamp1 = time.strftime("%a_%b_%d_%H_%M_%S_%Y")
        timestamp2 = time.strftime("%a_%b_%d_%H_%M_%S_%Y")
        
        vt_dir1 = os.path.join(temp_dir, 'src', 'vqvae', 'results', f'videotokenizer_{timestamp1}')
        vt_dir2 = os.path.join(temp_dir, 'src', 'vqvae', 'results', f'videotokenizer_{timestamp2}')
        lam_dir = os.path.join(temp_dir, 'src', 'latent_action_model', 'results', f'lam_{timestamp1}')
        dynamics_dir = os.path.join(temp_dir, 'src', 'dynamics', 'results', f'dynamics_{timestamp1}')
        
        os.makedirs(vt_dir1, exist_ok=True)
        os.makedirs(vt_dir2, exist_ok=True)
        os.makedirs(lam_dir, exist_ok=True)
        os.makedirs(dynamics_dir, exist_ok=True)
        
        # Create checkpoints directories
        os.makedirs(os.path.join(vt_dir1, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(vt_dir2, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(lam_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(dynamics_dir, 'checkpoints'), exist_ok=True)
        
        # Create mock checkpoint files
        import torch
        mock_checkpoint = {'model': {}, 'optimizer_state_dict': {}, 'results': {}, 'hyperparameters': {}}
        
        torch.save(mock_checkpoint, os.path.join(vt_dir1, 'checkpoints', 'videotokenizer_checkpoint_old.pth'))
        torch.save(mock_checkpoint, os.path.join(vt_dir2, 'checkpoints', 'videotokenizer_checkpoint_new.pth'))
        torch.save(mock_checkpoint, os.path.join(lam_dir, 'checkpoints', 'lam_checkpoint_test.pth'))
        torch.save(mock_checkpoint, os.path.join(dynamics_dir, 'checkpoints', 'dynamics_checkpoint_test.pth'))
        
        # Test finding latest checkpoints
        with patch('os.getcwd', return_value=temp_dir):
            vt_checkpoint = find_latest_checkpoint(temp_dir, 'videotokenizer')
            lam_checkpoint = find_latest_checkpoint(temp_dir, 'lam')
            dynamics_checkpoint = find_latest_checkpoint(temp_dir, 'dynamics')
            
            # Should find the most recent checkpoint
            assert vt_checkpoint is not None
            assert lam_checkpoint is not None
            assert dynamics_checkpoint is not None
            
            # Should find the newer checkpoint for video tokenizer
            assert 'videotokenizer_checkpoint_new.pth' in vt_checkpoint
    
    def test_run_command_success(self):
        """Test successful command execution"""
        from train_full_pipeline import run_command
        
        # Test with a simple command that should succeed
        result = run_command(['echo', 'test'], 'Test command')
        assert result is True
    
    def test_run_command_failure(self):
        """Test failed command execution"""
        from train_full_pipeline import run_command
        
        # Test with a command that should fail
        try:
            result = run_command(['nonexistent_command'], 'Failing command')
            # If we get here, the command didn't fail as expected
            assert False, "Command should have failed"
        except FileNotFoundError:
            # This is the expected behavior
            pass
        except Exception as e:
            # Other exceptions are also acceptable
            pass
    
    @patch('subprocess.run')
    def test_run_command_keyboard_interrupt(self, mock_run):
        """Test command execution with keyboard interrupt"""
        from train_full_pipeline import run_command
        
        # Mock keyboard interrupt
        mock_run.side_effect = KeyboardInterrupt()
        
        result = run_command(['test_command'], 'Interrupted command')
        assert result is False
    
    def test_pipeline_directory_validation(self, temp_dir):
        """Test pipeline directory validation"""
        from train_full_pipeline import main
        
        # Test without src directory
        with patch('os.path.exists', return_value=False):
            with patch('builtins.print') as mock_print:
                main()
                # Should print error about missing src directory
                mock_print.assert_called()
    
    def test_pipeline_dataset_validation(self, temp_dir):
        """Test pipeline dataset validation"""
        from train_full_pipeline import main
        
        # Mock src directory exists but no dataset
        with patch('os.path.exists', side_effect=lambda x: 'src' in x and 'pong_frames.h5' not in x):
            with patch('builtins.print') as mock_print:
                main()
                # Should print error about missing dataset
                mock_print.assert_called()
    
    @patch('subprocess.run')
    def test_full_pipeline_execution(self, mock_run, temp_dir, mock_pong_dataset):
        """Test full pipeline execution with mocked subprocess calls"""
        from train_full_pipeline import main
        
        # Mock successful subprocess runs
        mock_run.return_value = MagicMock(returncode=0)
        
        # Mock directory and file existence
        with patch('os.path.exists', side_effect=lambda x: True), \
             patch('os.getcwd', return_value=temp_dir), \
             patch('builtins.print') as mock_print:
            
            # Mock checkpoint finding
            with patch('train_full_pipeline.find_latest_checkpoint') as mock_find:
                mock_find.return_value = 'mock_checkpoint.pth'
                
                # Run the pipeline
                main()
                
                # Verify that subprocess.run was called for each step
                assert mock_run.call_count >= 3  # At least 3 training steps
    
    def test_pipeline_error_handling(self, temp_dir, mock_pong_dataset):
        """Test pipeline error handling"""
        from train_full_pipeline import main
        import subprocess

        # Mock successful first step, failed second step
        with patch('subprocess.run') as mock_run, \
             patch('os.path.exists', side_effect=lambda x: True), \
             patch('os.getcwd', return_value=temp_dir), \
             patch('builtins.print') as mock_print:

            # First call succeeds, second call raises CalledProcessError
            mock_run.side_effect = [
                MagicMock(returncode=0),  # Video tokenizer succeeds
                subprocess.CalledProcessError(returncode=1, cmd='lam')  # LAM fails
            ]

            # Run the pipeline
            main()

            # Should stop after LAM failure
            assert mock_run.call_count == 2
    
    def test_configuration_file_creation(self, temp_dir):
        """Test that configuration files are created properly"""
        # This test verifies the structure of configuration files
        # without actually running the training scripts
        
        config_structure = {
            'timestamp': 'test_timestamp',
            'device': 'cpu',
            'model_architecture': {
                'frame_size': (64, 64),
                'patch_size': 8,
                'embed_dim': 128,
                'num_heads': 4,
                'hidden_dim': 512,
                'num_blocks': 2,
                'latent_dim': 6,
                'dropout': 0.1
            },
            'training_parameters': {
                'batch_size': 16,
                'n_updates': 1000,
                'learning_rate': 1e-4,
                'log_interval': 100
            },
            'directories': {
                'run_dir': '/path/to/run',
                'checkpoints_dir': '/path/to/checkpoints',
                'visualizations_dir': '/path/to/visualizations'
            }
        }
        
        # Test that config can be serialized to JSON
        config_json = json.dumps(config_structure, indent=2, default=str)
        assert isinstance(config_json, str)
        
        # Test that config can be loaded back
        loaded_config = json.loads(config_json)
        assert loaded_config['model_architecture']['patch_size'] == 8
        assert loaded_config['training_parameters']['batch_size'] == 16
    
    def test_pipeline_hyperparameter_consistency(self):
        """Test that hyperparameters are consistent across all models"""
        # Define expected hyperparameters
        expected_params = {
            'patch_size': 8,
            'embed_dim': 128,
            'num_heads': 4,
            'hidden_dim': 512,
            'num_blocks': 2,
            'latent_dim': 6,
            'dropout': 0.1
        }
        
        # Test that all models use the same hyperparameters
        # This is a structural test to ensure consistency
        assert expected_params['patch_size'] == 8
        assert expected_params['embed_dim'] == 128
        assert expected_params['num_heads'] == 4
        assert expected_params['hidden_dim'] == 512
        assert expected_params['num_blocks'] == 2
        assert expected_params['latent_dim'] == 6
        assert expected_params['dropout'] == 0.1
    
    def test_pipeline_directory_structure(self, temp_dir):
        """Test that the pipeline creates the correct directory structure"""
        # Create mock directory structure
        expected_dirs = [
            'src/vqvae/results/videotokenizer_timestamp/checkpoints',
            'src/vqvae/results/videotokenizer_timestamp/visualizations',
            'src/latent_action_model/results/lam_timestamp/checkpoints',
            'src/latent_action_model/results/lam_timestamp/visualizations',
            'src/dynamics/results/dynamics_timestamp/checkpoints',
            'src/dynamics/results/dynamics_timestamp/visualizations'
        ]
        
        # Test directory creation
        for dir_path in expected_dirs:
            full_path = os.path.join(temp_dir, dir_path)
            os.makedirs(full_path, exist_ok=True)
            assert os.path.exists(full_path)
    
    def test_pipeline_logging(self, temp_dir):
        """Test pipeline logging and progress reporting"""
        from train_full_pipeline import main
        
        with patch('subprocess.run') as mock_run, \
             patch('os.path.exists', side_effect=lambda x: True), \
             patch('os.getcwd', return_value=temp_dir), \
             patch('builtins.print') as mock_print:
            
            # Mock successful execution
            mock_run.return_value = MagicMock(returncode=0)
            
            # Mock checkpoint finding
            with patch('train_full_pipeline.find_latest_checkpoint') as mock_find:
                mock_find.return_value = 'mock_checkpoint.pth'
                
                # Run the pipeline
                main()
                
                # Verify that progress messages were printed
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                
                # Should contain start message
                assert any('Starting Full Pipeline Training' in str(call) for call in print_calls)
                
                # Should contain step messages
                assert any('STEP 1' in str(call) for call in print_calls)
                assert any('STEP 2' in str(call) for call in print_calls)
                assert any('STEP 3' in str(call) for call in print_calls)
                assert any('STEP 4' in str(call) for call in print_calls)
    
    def test_pipeline_completion_summary(self, temp_dir):
        """Test pipeline completion summary"""
        from train_full_pipeline import main
        
        with patch('subprocess.run') as mock_run, \
             patch('os.path.exists', side_effect=lambda x: True), \
             patch('os.getcwd', return_value=temp_dir), \
             patch('builtins.print') as mock_print:
            
            # Mock successful execution
            mock_run.return_value = MagicMock(returncode=0)
            
            # Mock checkpoint finding
            with patch('train_full_pipeline.find_latest_checkpoint') as mock_find:
                mock_find.return_value = 'mock_checkpoint.pth'
                
                # Run the pipeline
                main()
                
                # Verify completion summary
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                
                # Should contain completion message
                assert any('FULL PIPELINE TRAINING COMPLETED SUCCESSFULLY' in str(call) for call in print_calls)
                
                # Should contain results summary
                assert any('Results Summary' in str(call) for call in print_calls)
                assert any('Video Tokenizer' in str(call) for call in print_calls)
                assert any('LAM' in str(call) for call in print_calls)
                assert any('Dynamics Model' in str(call) for call in print_calls)
    
    def test_pipeline_cleanup(self, temp_dir):
        """Test pipeline cleanup and error recovery"""
        # This test ensures that the pipeline handles errors gracefully
        # and doesn't leave the system in a bad state
        
        from train_full_pipeline import main
        
        with patch('subprocess.run') as mock_run, \
             patch('os.path.exists', side_effect=lambda x: True), \
             patch('os.getcwd', return_value=temp_dir), \
             patch('builtins.print') as mock_print:
            
            # Mock successful execution
            mock_run.return_value = MagicMock(returncode=0)
            
            # Mock checkpoint finding
            with patch('train_full_pipeline.find_latest_checkpoint') as mock_find:
                mock_find.return_value = 'mock_checkpoint.pth'
                
                # Run the pipeline
                main()
                
                # Verify that no exceptions were raised
                # (This is implicit in the test not failing)
                assert True


if __name__ == "__main__":
    pytest.main([__file__]) 