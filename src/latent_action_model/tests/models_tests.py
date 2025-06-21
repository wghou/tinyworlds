import unittest
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.lam import LAM, Encoder, Decoder, VectorQuantizer, PatchEmbedding

class TestPatchEmbedding(unittest.TestCase):
    def test_patch_embedding_shape(self):
        patch_embed = PatchEmbedding(frame_size=(64, 64), patch_size=16, embed_dim=512)
        frames = torch.randn(2, 8, 3, 64, 64)
        out = patch_embed(frames)
        self.assertEqual(out.shape, (2, 8, 16, 512))  # 16 patches for 64x64 with 16x16

class TestEncoder(unittest.TestCase):
    def test_encoder_shapes(self):
        encoder = Encoder(frame_size=(64, 64), patch_size=16, embed_dim=512, num_heads=8, hidden_dim=2048, num_blocks=2, action_dim=64)
        frames = torch.randn(2, 8, 3, 64, 64)
        actions, embeddings = encoder(frames)
        self.assertEqual(actions.shape, (2, 7, 64))  # seq_len-1 actions
        self.assertEqual(embeddings.shape, (2, 8, 16, 512))

class TestDecoder(unittest.TestCase):
    def test_decoder_shapes(self):
        decoder = Decoder(frame_size=(64, 64), patch_size=16, embed_dim=512, num_heads=8, hidden_dim=2048, num_blocks=2, action_dim=64)
        frames = torch.randn(2, 8, 3, 64, 64)
        actions = torch.randn(2, 7, 64)
        pred_frames = decoder(frames, actions)
        self.assertEqual(pred_frames.shape, (2, 7, 3, 64, 64))

class TestVectorQuantizer(unittest.TestCase):
    def test_quantizer_shapes(self):
        quantizer = VectorQuantizer(codebook_size=8, embedding_dim=64)
        z = torch.randn(4, 64)
        loss, z_q, indices = quantizer(z)
        self.assertEqual(z_q.shape, (4, 64))
        self.assertEqual(indices.shape, (4,))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue((indices >= 0).all() and (indices < 8).all())

class TestLAM(unittest.TestCase):
    def setUp(self):
        self.lam = LAM(frame_size=(64, 64), n_actions=8, patch_size=16, embed_dim=512, num_heads=8, hidden_dim=2048, num_blocks=2, action_dim=64)
    def test_lam_forward(self):
        frames = torch.randn(2, 8, 3, 64, 64)
        loss, pred_frames, action_indices, loss_dict = self.lam(frames)
        self.assertEqual(pred_frames.shape, (2, 7, 3, 64, 64))
        self.assertEqual(action_indices.shape, (2, 7))
        self.assertTrue(torch.isfinite(loss))
        self.assertIn('total_loss', loss_dict)
        self.assertIn('recon_loss', loss_dict)
        self.assertIn('vq_loss', loss_dict)
        self.assertIn('diversity_loss', loss_dict)

if __name__ == '__main__':
    unittest.main()
