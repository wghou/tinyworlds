import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from datasets.datasets import PongDataset, SonicDataset, PolePositionDataset, PicoDoomDataset, ZeldaDataset


def _default_video_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def _load_video_dataset_pair(dataset_cls, video_rel_path, h5_rel_path, num_frames, transform=None, **kwargs):
    current_folder_path = os.getcwd()
    video_path = current_folder_path + video_rel_path
    preprocessed_path = current_folder_path + h5_rel_path
    transform = _default_video_transform() if transform is None else transform

    train = dataset_cls(
        video_path,
        transform=transform,
        save_path=preprocessed_path,
        train=True,
        num_frames=num_frames,
        **kwargs
    )
    val = dataset_cls(
        video_path,
        transform=transform,
        save_path=preprocessed_path,
        train=False,
        num_frames=num_frames,
        **kwargs
    )
    return train, val


def load_pong(num_frames=1):
    return _load_video_dataset_pair(
        PongDataset,
        '/data/pong.mp4',
        '/data/pong_frames.h5',
        num_frames=num_frames
    )

def load_sonic(num_frames=4):
    return _load_video_dataset_pair(
        SonicDataset,
        '/data/Sonic Cleaned.mp4',
        '/data/Sonic Cleaned_frames.h5',
        num_frames=num_frames
    )

def load_pole_position(num_frames=4):
    return _load_video_dataset_pair(
        PolePositionDataset,
        '/data/pole_position.mp4',
        '/data/pole_position_frames.h5',
        num_frames=num_frames
    )

def load_picodoom(num_frames=4):
    return _load_video_dataset_pair(
        PicoDoomDataset,
        '/data/picodoom cleaned.mp4',
        '/data/picodoom_frames.h5',
        num_frames=num_frames
    )

def load_zelda(num_frames=4):
    return _load_video_dataset_pair(
        ZeldaDataset,
        '/data/Zelda oot2d 1 Cut.mp4',
        '/data/zelda_frames.h5',
        num_frames=num_frames
    )

def data_loaders(train_data, val_data, batch_size):
    # Use most of available CPU cores (leave 2 for system), minimum 2
    env_workers = os.environ.get("NG_NUM_WORKERS")
    if env_workers is not None and env_workers.isdigit():
        num_workers = max(2, int(env_workers))
    else:
        num_workers = max(2, (os.cpu_count() or 4) - 2)
    print(f"Using {num_workers} workers for data loading")

    # Prefetch factor override
    env_prefetch = os.environ.get("NG_PREFETCH_FACTOR")
    if env_prefetch is not None and env_prefetch.isdigit():
        prefetch_factor = max(2, int(env_prefetch))
    else:
        prefetch_factor = 4

    pin_memory = os.environ.get("NG_PIN_MEMORY", "1") != "0"
    persistent_workers = os.environ.get("NG_PERSISTENT_WORKERS", "1") != "0"

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True
    )
    return train_loader, val_loader


def load_data_and_data_loaders(dataset, batch_size, num_frames=1):
    if dataset == 'PONG':
        training_data, validation_data = load_pong(num_frames=num_frames)
    elif dataset == 'SONIC':
        training_data, validation_data = load_sonic(num_frames=num_frames)
    elif dataset == 'POLE_POSITION':
        training_data, validation_data = load_pole_position(num_frames=num_frames)
    elif dataset == 'PICODOOM':
        training_data, validation_data = load_picodoom(num_frames=num_frames)
    elif dataset == 'ZELDA':
        training_data, validation_data = load_zelda(num_frames=num_frames)
    else:
        raise ValueError('Invalid dataset')

    training_loader, validation_loader = data_loaders(training_data, validation_data, batch_size)
    x_train_var = np.var(training_data.data)

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(model, optimizer, results, hyperparameters, timestamp):
    SAVE_MODEL_PATH = os.getcwd() + '/results'

    results_to_save = {
        'model': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + '.pth')

def visualize_reconstruction(original, reconstruction, save_path=None):
    # original: (B, C, H, W) or (B, seq_len, C, H, W)
    # reconstruction: (B, C, H, W) or (B, seq_len, C, H, W) 

    # Move tensors to CPU and convert to float32 for matplotlib compatibility
    original = original.detach().to('cpu', dtype=torch.float32)
    reconstruction = reconstruction.detach().to('cpu', dtype=torch.float32)

    # Handle single frames by expanding to sequences
    if original.dim() == 4:  # (B, C, H, W)
        original = original.unsqueeze(1)  # Add sequence dimension
    if reconstruction.dim() == 4:  # (B, C, H, W)
        reconstruction = reconstruction.unsqueeze(1)  # Add sequence dimension

    # Take first 4 sequences, each of length 4 (or available length)
    num_sequences = min(4, original.shape[0])
    seq_length = min(4, original.shape[1])

    original = original[:num_sequences, :seq_length]  # (4, seq_len, C, H, W)
    reconstruction = reconstruction[:num_sequences, :seq_length]  # (4, seq_len, C, H, W)

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # For original sequences
    # Reshape to (num_sequences * seq_length, C, H, W) for make_grid
    orig_flat = original.reshape(-1, *original.shape[2:])  # (4*seq_len, C, H, W)
    grid_orig = make_grid(orig_flat, nrow=seq_length, normalize=True, padding=2).clamp(0, 1)
    ax1.imshow(grid_orig.permute(1, 2, 0).contiguous().numpy())
    ax1.axis('off')
    ax1.set_title(f'Original Sequences (4 sequences × {seq_length} frames)')

    # For reconstructed sequences
    recon_flat = reconstruction.reshape(-1, *reconstruction.shape[2:])  # (4*seq_len, C, H, W)
    grid_recon = make_grid(recon_flat, nrow=seq_length, normalize=True, padding=2).clamp(0, 1)
    ax2.imshow(grid_recon.permute(1, 2, 0).contiguous().numpy())
    ax2.axis('off')
    ax2.set_title(f'Reconstructed Sequences (4 sequences × {seq_length} frames)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()
