"""Dataset for loading PicoDoom video frames using HDF5 storage"""

from torch.utils.data import Dataset
import cv2
import h5py
import os
from tqdm import tqdm
import numpy as np
import torch

class PicoDoomDataset(Dataset):
    def __init__(self, video_path, transform=None, save_path=None, train=True, num_frames=4):
        self.transform = transform
        self.train = train
        self.num_frames = num_frames
        self.fps = 15
        self.frame_skip = 60 // self.fps
        self.fraction_of_frames = 1.0
        
        if save_path and os.path.exists(save_path):
            print(f"Loading preprocessed frames from {save_path}")
            with h5py.File(save_path, 'r') as h5_file:  # Use context manager
                frames = h5_file['frames']
                n_frames = int(len(frames))
                print(f"Loading {n_frames} frames from {save_path}")
                
                # Load frames into memory in chunks
                chunk_size = 1000  # Adjust based on available RAM
                self.data = []
                for i in tqdm(range(100, n_frames, chunk_size), desc="Loading frames"):
                    chunk = frames[i:min(i+chunk_size, n_frames)][:]  # [:] forces load into memory
                    self.data.extend(chunk)
                self.data = np.array(self.data)
                print(f"Loaded {len(self.data)} frames")
                
                # Split into train/val (90/10 split)
                split_idx = int(0.9 * len(self.data))
                self.data = self.data[:split_idx] if train else self.data[split_idx:]
        else:
            frames = self.preprocess_video(video_path)
            if save_path:
                print(f"Saving preprocessed frames to {save_path}")
                with h5py.File(save_path, 'w') as f:
                    # Use lighter compression for better read speed
                    f.create_dataset('frames', data=frames, 
                                   compression='lzf')  # LZF is faster than gzip
                
                # Reload the saved data
                with h5py.File(save_path, 'r') as h5_file:  # Use context manager
                    frames = h5_file['frames'][:]  # Load all into memory
                
                # Split into train/val
                split_idx = int(0.9 * len(frames))
                self.data = frames[:split_idx] if train else frames[split_idx:]
            else:
                split_idx = int(0.9 * len(frames))
                self.data = frames[:split_idx] if train else frames[split_idx:]

    def preprocess_video(self, video_path):
        """Preprocess video to get frames"""
        print(f"Preprocessing video {video_path}")
        video = cv2.VideoCapture(video_path)
        
        # Get total frame count for progress bar
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        
        for i in tqdm(range(0, total_frames), desc="Processing video frames"):
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
            frames.append(frame)
            
        video.release()
        return np.array(frames)

    def __len__(self):
        # how many starting positions can give us a full sequence?
        max_valid_index = int((len(self.data) - (self.num_frames * self.frame_skip)) * self.fraction_of_frames)
        return max(0, max_valid_index)          # training and val alike
    
    def __getitem__(self, index):
        # Ensure index is within bounds
        if index >= len(self):
            raise IndexError(f"Index {index} out of bounds for dataset of length {len(self)}")
        
        # Get sequence of frames starting from index
        frame_sequence = self.data[index:index + (self.num_frames * self.frame_skip):self.frame_skip]
        
        # Verify we got the expected number of frames
        if len(frame_sequence) != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {len(frame_sequence)} frames")
        
        # Convert to float and normalize to [0, 1]
        frame_sequence = frame_sequence.astype(np.float32) / 255.0
        
        if self.transform:
            transformed_frames = []
            for frame in frame_sequence:
                # Pass numpy array directly to transform (ToTensor will handle conversion)
                transformed_frame = self.transform(frame)
                transformed_frames.append(transformed_frame)
            frame_sequence = torch.stack(transformed_frames, dim=0)  # [seq_len, C, H, W]
        else:
            frame_sequence = torch.from_numpy(frame_sequence).permute(0, 3, 1, 2)  # [seq_len, C, H, W]
        
        return frame_sequence, 0  # Return 0 as dummy label
    
    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()
