import numpy as np
import torch
import torch.optim as optim
import argparse
import utils
from cnn_models.vqvae import VQVAE
import os
from utils import visualize_reconstruction
from tqdm import tqdm

parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
timestamp = utils.readable_timestamp()

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=10000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=500)
parser.add_argument("--dataset",  type=str, default='CIFAR10')

# whether or not to save model
parser.add_argument("-save", action="store_true", default=True)
parser.add_argument("--filename",  type=str, default=timestamp)

# Add checkpoint arguments
parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file to resume from")
parser.add_argument("--start_iteration", type=int, default=0, help="Iteration to start from")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.save:
    print('Results will be saved in ./results/vqvae_' + args.filename + '.pth')

"""
Load data and define batch data loaders
"""

training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
    args.dataset, args.batch_size)
"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(device)

"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

"""
Load checkpoint if specified
"""
results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}

if args.checkpoint:
    if os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        results = checkpoint['results']
        # Load hyperparameters but don't override current args
        saved_hyperparameters = checkpoint['hyperparameters']
        print(f"Resuming from update {results['n_updates']}")
        
        # Restore optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print(f"No checkpoint found at {args.checkpoint}")

model.train()

def train():
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.getcwd(), 'results')
    reconstructions_dir = os.path.join(results_dir, 'reconstructions')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(reconstructions_dir):
        os.makedirs(reconstructions_dir)
        
    start_iter = max(args.start_iteration, results['n_updates'])
    
    for i in tqdm(range(start_iter, args.n_updates)):
        (x, _) = next(iter(training_loader))
        x = x.to(device)
        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity = model(x)
        recon_loss = torch.mean((x_hat - x)**2) / x_train_var
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        if i % args.log_interval == 0:
            """
            save model and print values
            """
            if args.save:
                hyperparameters = args.__dict__
                utils.save_model_and_results(
                    model, optimizer, results, hyperparameters, args.filename)
                
                # Add visualization
                save_path = os.path.join(reconstructions_dir, f'reconstruction_step_{i}_{args.filename}.png')
                visualize_reconstruction(x[:16], x_hat[:16], save_path)  # Visualize first 16 images

            print('Update #', i, 'Recon Error:',
                  np.mean(results["recon_errors"][-args.log_interval:]),
                  'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
                  'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]))


if __name__ == "__main__":
    train()