import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
from datasets.block import BlockDataset, LatentBlockDataset
from datasets.pong import PongDataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


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

def load_pong():
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
                       train=True)
    val = PongDataset(video_path,
                      transform=transform,
                      save_path=preprocessed_path,
                      train=False)
    
    return train, val

def data_loaders(train_data, val_data, batch_size):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return train_loader, val_loader


def load_data_and_data_loaders(dataset, batch_size):
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
        training_data, validation_data = load_pong()
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
    Visualizes original images and their reconstructions side by side
    
    Args:
        original: Tensor of original images (B, C, H, W)
        reconstruction: Tensor of reconstructed images (B, C, H, W) 
        save_path: Optional path to save the visualization
    """
    # Move tensors to CPU and convert to numpy arrays
    original = original.detach().cpu()
    reconstruction = reconstruction.detach().cpu()
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Show original images
    grid_orig = make_grid(original, nrow=4, normalize=True, padding=2)
    ax1.imshow(grid_orig.permute(1, 2, 0))
    ax1.axis('off')
    ax1.set_title('Original Images')
    
    # Show reconstructed images
    grid_recon = make_grid(reconstruction, nrow=4, normalize=True, padding=2)
    ax2.imshow(grid_recon.permute(1, 2, 0))
    ax2.axis('off')
    ax2.set_title('Reconstructed Images')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()