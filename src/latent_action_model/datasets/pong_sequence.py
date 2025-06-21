import torch
import h5py
from torch.utils.data import Dataset
import numpy as np

class PongSequenceDataset(Dataset):
    """
    Dataset for loading sequences of Pong frames from preprocessed HDF5 file
    """
    def __init__(self, data_path, seq_length=8, transform=None):
        """
        Args:
            data_path: Path to HDF5 file containing preprocessed frames
            seq_length: Number of consecutive frames to return in each sequence
            transform: Optional transform to apply to the frames
        """
        self.data_path = data_path
        self.seq_length = seq_length
        self.transform = transform
        
        # Open HDF5 file to get metadata
        with h5py.File(data_path, 'r') as f:
            self.num_frames = f['frames'].shape[0]
            self.frame_shape = f['frames'].shape[1:]
        
        # Calculate number of valid sequences
        self.num_sequences = self.num_frames - self.seq_length + 1
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        """
        Returns a sequence of frames starting at index idx
        """
        # Open HDF5 file (we open it here to avoid keeping it open across processes)
        with h5py.File(self.data_path, 'r') as f:
            # Get sequence of frames
            frames = f['frames'][idx:idx+self.seq_length]
        
        # Convert to torch tensor and normalize to [0, 1]
        frames = frames.astype(np.float32)  / 255.0
        
        # Apply transform if provided
        if self.transform:
            # Check if frames are already in the right format
            if frames.shape[-3:] == (3, 64, 64):  # Already in CxHxW format with 3 channels
                transformed_frames = frames
            else:
                # Apply transform to each frame in the sequence
                transformed_frames = []
                for frame in frames:
                    transformed_frames.append(self.transform(frame))
                transformed_frames = torch.stack(transformed_frames)
        else:
            transformed_frames = frames
        
        # Return sequence of frames and a dummy label
        return transformed_frames, torch.zeros(1)  # Dummy label for compatibility 