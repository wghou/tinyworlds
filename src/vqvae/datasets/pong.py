"""Dataset for loading Pong video frames using HDF5 storage"""

import torch
from torch.utils.data import Dataset
import cv2
import h5py
import os
from tqdm import tqdm
import numpy as np

class PongDataset(Dataset):
    def __init__(self, video_path, transform=None, save_path=None, train=True):
        self.transform = transform
        self.train = train
        
        if save_path and os.path.exists(save_path):
            print(f"Loading preprocessed frames from {save_path}")
            self.h5_file = h5py.File(save_path, 'r')
            frames = self.h5_file['frames']
            n_frames = len(frames)
            
            # Load frames into memory in chunks
            chunk_size = 1000  # Adjust based on available RAM
            self.data = []
            for i in tqdm(range(0, n_frames, chunk_size), desc="Loading frames"):
                chunk = frames[i:min(i+chunk_size, n_frames)][:]  # [:] forces load into memory
                self.data.extend(chunk)
            self.data = np.array(self.data)
            
            # Split into train/val (90/10 split)
            split_idx = int(0.9 * len(self.data))
            self.data = self.data[:split_idx] if train else self.data[split_idx:]
            
            # Close HDF5 file since we loaded everything
            self.h5_file.close()
        else:
            frames = self.preprocess_video(video_path)
            if save_path:
                print(f"Saving preprocessed frames to {save_path}")
                with h5py.File(save_path, 'w') as f:
                    # Use lighter compression for better read speed
                    f.create_dataset('frames', data=frames, 
                                   compression='lzf')  # LZF is faster than gzip
                
                # Reload the saved data
                self.h5_file = h5py.File(save_path, 'r')
                frames = self.h5_file['frames'][:]  # Load all into memory
                self.h5_file.close()
                
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
        
        # Maybe skip frames to reduce dataset size
        frame_skip = 2  # Only keep every 2nd frame
        
        for i in tqdm(range(0, total_frames, frame_skip), desc="Processing video frames"):
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
        return len(self.data)
    
    def __getitem__(self, index):
        frame = self.data[index]
        
        # Convert to float and normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        if self.transform:
            frame = self.transform(frame)
        return frame, 0  # Return 0 as dummy label
    
    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()
