import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from src.vqvae.datasets.block import BlockDataset, LatentBlockDataset
from src.vqvae.datasets.pong import PongDataset
from src.vqvae.datasets.sonic import SonicDataset
from src.vqvae.datasets.pole_position import PolePositionDataset
from src.vqvae.datasets.picodoom import PicoDoomDataset


def load_cifar():
    train = datasets.CIFAR10(root="data", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    val = datasets.CIFAR10(root="data", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    return train, val


def load_block():
    current_folder_path = os.getcwd()
    data_file_path = current_folder_path + \
        '/data/randact_traj_length_100_n_trials_1000_n_contexts_1.npy'

    train = BlockDataset(data_file_path, train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))

    val = BlockDataset(data_file_path, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    return train, val

def load_latent_block():
    current_folder_path = os.getcwd()
    data_file_path = current_folder_path + \
        '/data/latent_e_indices.npy'

    train = LatentBlockDataset(data_file_path, train=True,
                         transform=None)

    val = LatentBlockDataset(data_file_path, train=False,
                       transform=None)
    return train, val

def load_pong(num_frames=1):
    """loads pong dataset"""
    current_folder_path = os.getcwd()
    video_path = current_folder_path + '/data/pong.mp4'
    preprocessed_path = current_folder_path + '/data/pong_frames.h5'
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train = PongDataset(video_path, 
                       transform=transform,
                       save_path=preprocessed_path,
                       train=True,
                       num_frames=num_frames)
    val = PongDataset(video_path,
                      transform=transform,
                      save_path=preprocessed_path,
                      train=False,
                      num_frames=num_frames)
    
    return train, val

def load_sonic(num_frames=4):
    """loads sonic dataset"""
    current_folder_path = os.getcwd()
    video_path = current_folder_path + '/data/Sonic Cleaned.mp4'
    preprocessed_path = current_folder_path + '/data/Sonic Cleaned_frames.h5'
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train = SonicDataset(video_path, 
                       transform=transform,
                       save_path=preprocessed_path,
                       train=True,
                       num_frames=num_frames)
    val = SonicDataset(video_path, 
                       transform=transform,
                       save_path=preprocessed_path,
                       train=False,
                       num_frames=num_frames)
    return train, val

def load_pole_position(num_frames=4):
    current_folder_path = os.getcwd()
    video_path = current_folder_path + '/data/pole_position.mp4'
    preprocessed_path = current_folder_path + '/data/pole_position_frames.h5'
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train = PolePositionDataset(video_path, 
                       transform=transform,
                       save_path=preprocessed_path,
                       train=True,
                       num_frames=num_frames)
    val = PolePositionDataset(video_path, 
                       transform=transform,
                       save_path=preprocessed_path,
                       train=False,
                       num_frames=num_frames)
    return train, val

def load_picodoom(num_frames=4):
    current_folder_path = os.getcwd()
    video_path = current_folder_path + '/data/picodoom cleaned.mp4'
    preprocessed_path = current_folder_path + '/data/picodoom_frames.h5'
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train = PicoDoomDataset(video_path, 
                       transform=transform,
                       save_path=preprocessed_path,
                       train=True,
                       num_frames=num_frames)
    val = PicoDoomDataset(video_path, 
                       transform=transform,
                       save_path=preprocessed_path,
                       train=False,
                       num_frames=num_frames)
    return train, val



def data_loaders(train_data, val_data, batch_size):
    # Use most of available CPU cores (leave 2 for system), minimum 2
    env_workers = os.environ.get("NG_NUM_WORKERS")
    if env_workers is not None and env_workers.isdigit():
        num_workers = max(2, int(env_workers))
    else:
        num_workers = max(2, (os.cpu_count() or 4) - 2)
    print(f"os.cpu_count(): {os.cpu_count()}")
    print(f"num_workers: {num_workers}")

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
    if dataset == 'CIFAR10':
        training_data, validation_data = load_cifar()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data / 255.0)

    elif dataset == 'BLOCK':
        training_data, validation_data = load_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data / 255.0)
    elif dataset == 'LATENT_BLOCK':
        training_data, validation_data = load_latent_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data)
    # TODO: add pong dataset
    elif dataset == 'PONG':
        training_data, validation_data = load_pong(num_frames=num_frames)
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data)
    elif dataset == 'SONIC':
        training_data, validation_data = load_sonic(num_frames=num_frames)
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data)
    elif dataset == 'POLE_POSITION':
        training_data, validation_data = load_pole_position(num_frames=num_frames)
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data)
    elif dataset == 'PICODOOM':
        training_data, validation_data = load_picodoom(num_frames=num_frames)
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data)
    else:
        raise ValueError(
            'Invalid dataset: only CIFAR10 and BLOCK datasets are supported.')

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(model, optimizer, results, hyperparameters, timestamp):
    """
    Save model checkpoint including model state, optimizer state, results and hyperparameters
    
    Args:
        model: The PyTorch model
        optimizer: The optimizer
        results: Dictionary containing training results
        hyperparameters: Dictionary of hyperparameters
        timestamp: String timestamp for filename
    """
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
    """
    Visualizes original sequences and their reconstructions side by side
    
    Args:
        original: Tensor of original images (B, C, H, W) or sequences (B, seq_len, C, H, W)
        reconstruction: Tensor of reconstructed images (B, C, H, W) or sequences (B, seq_len, C, H, W) 
        save_path: Optional path to save the visualization
    """
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