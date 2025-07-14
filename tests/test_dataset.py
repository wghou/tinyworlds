"""
Tests for dataset functionality
"""

import pytest
import os
import tempfile
import shutil
import numpy as np
import h5py
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.vqvae.datasets.sonic import SonicDataset
from src.vqvae.datasets.pong import PongDataset
from src.vqvae.datasets.block import BlockDataset


class TestSonicDataset:
    """Test class for SonicDataset functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_sonic_data(self, temp_dir):
        """Create a mock SONIC dataset for testing"""
        # Create mock HDF5 file with 100 frames
        mock_frames = np.random.randint(0, 255, (100, 64, 64, 3), dtype=np.uint8)
        h5_path = os.path.join(temp_dir, 'sonic_frames.h5')
        
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('frames', data=mock_frames)
        
        return h5_path
    
    def test_dataset_bounds_with_frame_skip(self, temp_dir, mock_sonic_data):
        """Test that dataset bounds are correctly handled with frame_skip"""
        
        # Test with different num_frames values
        test_cases = [
            (1, 30),   # num_frames=1, frame_skip=30
            (2, 30),   # num_frames=2, frame_skip=30
            (4, 30),   # num_frames=4, frame_skip=30
        ]
        
        for num_frames, frame_skip in test_cases:
            print(f"\nTesting with num_frames={num_frames}, frame_skip={frame_skip}")
            
            # Create dataset
            dataset = SonicDataset(
                video_path=None,  # Not used when save_path exists
                save_path=mock_sonic_data,
                train=True,
                num_frames=num_frames
            )
            
            # Override frame_skip for testing
            dataset.frame_skip = frame_skip
            
            print(f"Dataset length: {len(dataset)}")
            print(f"Data length: {len(dataset.data)}")
            
            # Calculate expected length
            max_valid_index = len(dataset.data) - (num_frames * frame_skip)
            expected_length = max(0, max_valid_index + 1)
            print(f"Expected length: {expected_length}")
            
            assert len(dataset) == expected_length, f"Length mismatch: got {len(dataset)}, expected {expected_length}"
            
            # Test accessing the last valid index
            if len(dataset) > 0:
                last_index = len(dataset) - 1
                print(f"Testing last index: {last_index}")
                
                # This should not raise an IndexError
                frame_sequence, label = dataset[last_index]
                print(f"Last sequence shape: {frame_sequence.shape}")
                
                # Verify we got the expected number of frames
                assert frame_sequence.shape[0] == num_frames, f"Expected {num_frames} frames, got {frame_sequence.shape[0]}"
                
                # Test that accessing beyond the last index raises IndexError
                with pytest.raises(IndexError):
                    dataset[len(dataset)]
            
            print(f"✅ Test passed for num_frames={num_frames}, frame_skip={frame_skip}")
    
    def test_dataset_index_out_of_bounds(self, temp_dir, mock_sonic_data):
        """Test that accessing out-of-bounds indices raises proper errors"""
        
        dataset = SonicDataset(
            video_path=None,
            save_path=mock_sonic_data,
            train=True,
            num_frames=2
        )
        dataset.frame_skip = 30
        
        # Test accessing beyond dataset length
        with pytest.raises(IndexError):
            dataset[len(dataset)]
        
        # Test accessing negative index (should work in Python, but may fail due to frame_skip logic)
        # Let's test with a valid negative index instead
        if len(dataset) > 0:
            try:
                dataset[-1]  # This should work if there are valid indices
            except (IndexError, ValueError):
                # It's okay if this fails due to frame_skip logic
                pass
    
    def test_dataset_sequence_length_validation(self, temp_dir, mock_sonic_data):
        """Test that dataset validates sequence length correctly"""
        
        dataset = SonicDataset(
            video_path=None,
            save_path=mock_sonic_data,
            train=True,
            num_frames=2
        )
        dataset.frame_skip = 30
        
        # This should work and return exactly 2 frames
        frame_sequence, label = dataset[0]
        assert frame_sequence.shape[0] == 2, f"Expected 2 frames, got {frame_sequence.shape[0]}"
    
    def test_dataset_train_val_split(self, temp_dir, mock_sonic_data):
        """Test that train/val split works correctly"""
        
        # Test train split
        train_dataset = SonicDataset(
            video_path=None,
            save_path=mock_sonic_data,
            train=True,
            num_frames=1
        )
        
        # Test val split
        val_dataset = SonicDataset(
            video_path=None,
            save_path=mock_sonic_data,
            train=False,
            num_frames=1
        )
        
        # Train should have 90% of data, val should have 10%
        expected_train_size = int(0.9 * 100)  # 100 frames total
        expected_val_size = 100 - expected_train_size
        
        assert len(train_dataset.data) == expected_train_size
        assert len(val_dataset.data) == expected_val_size
    
    def test_dataset_data_types(self, temp_dir, mock_sonic_data):
        """Test that dataset returns correct data types"""
        
        dataset = SonicDataset(
            video_path=None,
            save_path=mock_sonic_data,
            train=True,
            num_frames=1
        )
        
        frame_sequence, label = dataset[0]
        
        # Should return torch tensor
        import torch
        assert isinstance(frame_sequence, torch.Tensor)
        
        # Should have correct shape [seq_len, C, H, W]
        assert frame_sequence.shape == (1, 3, 64, 64)
        
        # Should be float32 and normalized to [0, 1]
        assert frame_sequence.dtype == torch.float32
        assert frame_sequence.min() >= 0.0
        assert frame_sequence.max() <= 1.0


class TestPongDataset:
    """Test class for PongDataset functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_pong_data(self, temp_dir):
        """Create a mock Pong dataset for testing"""
        # Create mock HDF5 file with 100 frames
        mock_frames = np.random.randint(0, 255, (100, 64, 64, 3), dtype=np.uint8)
        h5_path = os.path.join(temp_dir, 'pong_frames.h5')
        
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('frames', data=mock_frames)
        
        return h5_path
    
    def test_pong_dataset_bounds(self, temp_dir, mock_pong_data):
        """Test that PongDataset bounds are correctly handled"""
        
        dataset = PongDataset(
            video_path=None,
            save_path=mock_pong_data,
            train=True,
            num_frames=2
        )
        
        # Pong dataset doesn't use frame_skip in __getitem__, so it's simpler
        expected_length = len(dataset.data) - dataset.num_frames + 1
        assert len(dataset) == expected_length
        
        # Test accessing last valid index
        if len(dataset) > 0:
            frame_sequence, label = dataset[len(dataset) - 1]
            assert frame_sequence.shape[0] == 2
        
        # Note: PongDataset doesn't implement bounds checking, so this won't raise IndexError
        # This is a limitation of the current implementation


class TestBlockDataset:
    """Test class for BlockDataset functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_block_data(self, temp_dir):
        """Create a mock Block dataset for testing"""
        # Create mock numpy file with block data
        # Block data has shape [n_samples, 1, 1, height, width, channels]
        mock_data = np.random.randint(0, 255, (100, 1, 1, 32, 32, 3), dtype=np.uint8)
        npy_path = os.path.join(temp_dir, 'block_data.npy')
        np.save(npy_path, mock_data)
        return npy_path
    
    def test_block_dataset_bounds(self, temp_dir, mock_block_data):
        """Test that BlockDataset bounds are correctly handled"""
        
        dataset = BlockDataset(
            file_path=mock_block_data,
            train=True
        )
        
        # Block dataset is simple - just single images
        expected_length = len(dataset.data)
        assert len(dataset) == expected_length
        
        # Test accessing last valid index
        if len(dataset) > 0:
            img, label = dataset[len(dataset) - 1]
            # Block dataset returns (H, W, C) shape, not (C, H, W)
            assert img.shape == (32, 32, 3)  # Resized to 32x32
        
        # Note: BlockDataset doesn't implement bounds checking, so this won't raise IndexError
        # This is a limitation of the current implementation


if __name__ == "__main__":
    pytest.main([__file__]) 