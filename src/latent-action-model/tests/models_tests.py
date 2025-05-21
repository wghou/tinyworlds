import unittest
import torch
from models.lam import LAM, Encoder, Decoder, VectorQuantizer

class TestLAM(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.frame_size = (64, 64)
        self.n_actions = 8
        self.hidden_dim = 128
        self.action_dim = 32
        
        # Create sample inputs
        self.prev_frame = torch.randn(self.batch_size, 3, *self.frame_size)
        self.next_frame = torch.randn(self.batch_size, 3, *self.frame_size)
        
    def test_encoder(self):
        encoder = Encoder(self.frame_size, self.hidden_dim, self.action_dim)
        out = encoder(self.prev_frame, self.next_frame)
        
        self.assertEqual(out.shape, (self.batch_size, self.action_dim))
        self.assertTrue(torch.is_tensor(out))
        
    def test_vector_quantizer(self):
        vq = VectorQuantizer(self.n_actions, self.action_dim)
        z = torch.randn(self.batch_size, self.action_dim)
        loss, z_q, indices = vq(z)
        
        self.assertEqual(z_q.shape, z.shape)
        self.assertEqual(indices.shape, (self.batch_size,))
        self.assertTrue(torch.all(indices < self.n_actions))
        self.assertTrue(loss >= 0)
        
    def test_decoder(self):
        decoder = Decoder(self.frame_size, self.hidden_dim, self.action_dim)
        action = torch.randn(self.batch_size, self.action_dim)
        out = decoder(self.prev_frame, action)
        
        self.assertEqual(out.shape, self.prev_frame.shape)
        self.assertTrue(torch.all(out >= -1) and torch.all(out <= 1))
        
    def test_lam_forward(self):
        model = LAM(self.frame_size, self.n_actions, self.hidden_dim, self.action_dim)
        loss, pred_next, actions = model(self.prev_frame, self.next_frame)
        
        self.assertTrue(loss >= 0)
        self.assertEqual(pred_next.shape, self.next_frame.shape)
        self.assertEqual(actions.shape, (self.batch_size,))
        self.assertTrue(torch.all(actions < self.n_actions))
        
    def test_lam_training(self):
        model = LAM(self.frame_size, self.n_actions, self.hidden_dim, self.action_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        initial_loss = None
        for _ in range(5):
            optimizer.zero_grad()
            loss, _, _ = model(self.prev_frame, self.next_frame)
            
            if initial_loss is None:
                initial_loss = loss.item()
                
            loss.backward()
            optimizer.step()
            
        final_loss = loss.item()
        self.assertLess(final_loss, initial_loss)

if __name__ == '__main__':
    unittest.main()
