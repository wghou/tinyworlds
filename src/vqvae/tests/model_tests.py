import unittest
import torch
import numpy as np
from models.encoder import Encoder
from models.decoder import Decoder
from models.quantizer import VectorQuantizer
from models.residual import ResidualLayer, ResidualStack
from models.vqvae import VQVAE

class TestModels(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.channels = 3
        self.height = 64
        self.width = 64
        self.h_dim = 128
        self.res_h_dim = 32
        self.n_res_layers = 2
        self.embedding_dim = 64
        self.n_embeddings = 512
        self.beta = 0.25
        
        # Create sample input
        self.x = torch.randn(self.batch_size, self.channels, self.height, self.width)

    def test_residual_layer(self):
        layer = ResidualLayer(self.h_dim, self.h_dim, self.res_h_dim)
        x = torch.randn(self.batch_size, self.h_dim, self.height, self.width)
        out = layer(x)
        
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.is_tensor(out))

    def test_residual_stack(self):
        stack = ResidualStack(self.h_dim, self.h_dim, self.res_h_dim, self.n_res_layers)
        x = torch.randn(self.batch_size, self.h_dim, self.height, self.width)
        out = stack(x)
        
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.is_tensor(out))

    def test_encoder(self):
        encoder = Encoder(self.channels, self.h_dim, self.n_res_layers, self.res_h_dim)
        out = encoder(self.x)
        
        # Check output shape (should be downsampled by factor of 4)
        expected_shape = (self.batch_size, self.h_dim, self.height//4, self.width//4)
        self.assertEqual(out.shape, expected_shape)
        self.assertTrue(torch.is_tensor(out))

    def test_decoder(self):
        decoder = Decoder(self.embedding_dim, self.h_dim, self.n_res_layers, self.res_h_dim)
        x = torch.randn(self.batch_size, self.embedding_dim, self.height//4, self.width//4)
        out = decoder(x)
        
        expected_shape = (self.batch_size, self.channels, self.height, self.width)
        self.assertEqual(out.shape, expected_shape)
        self.assertTrue(torch.is_tensor(out))

    def test_vector_quantizer(self):
        quantizer = VectorQuantizer(self.n_embeddings, self.embedding_dim, self.beta)
        x = torch.randn(self.batch_size, self.embedding_dim, self.height//4, self.width//4)
        loss, quantized, perplexity, _, _ = quantizer(x)
        
        self.assertEqual(quantized.shape, x.shape)
        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(torch.is_tensor(perplexity))
        self.assertTrue(perplexity > 0)

    def test_vqvae(self):
        vqvae = VQVAE(self.h_dim, self.res_h_dim, self.n_res_layers,
                      self.n_embeddings, self.embedding_dim, self.beta)
        
        # Test forward pass
        embedding_loss, x_hat, perplexity = vqvae(self.x)
        
        # Check shapes
        self.assertEqual(x_hat.shape, self.x.shape)
        
        # Check types
        self.assertTrue(torch.is_tensor(embedding_loss))
        self.assertTrue(torch.is_tensor(x_hat))
        self.assertTrue(torch.is_tensor(perplexity))
        
        # Check values
        self.assertTrue(embedding_loss >= 0)
        self.assertTrue(perplexity > 0)

    def test_vqvae_training(self):
        vqvae = VQVAE(self.h_dim, self.res_h_dim, self.n_res_layers,
                      self.n_embeddings, self.embedding_dim, self.beta)
        optimizer = torch.optim.Adam(vqvae.parameters(), lr=1e-3)
        
        # Simple training loop
        initial_loss = None
        for _ in range(5):
            optimizer.zero_grad()
            embedding_loss, x_hat, _ = vqvae(self.x)
            recon_loss = torch.mean((x_hat - self.x)**2)
            loss = recon_loss + embedding_loss
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
        
        final_loss = loss.item()
        self.assertLess(final_loss, initial_loss)

if __name__ == '__main__':
    unittest.main()
