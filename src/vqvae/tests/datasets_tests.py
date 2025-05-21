import unittest
import torch
import numpy as np
import os
import h5py
import cv2
from torchvision import transforms
from datasets.pong import PongDataset
from datasets.block import BlockDataset, LatentBlockDataset

class TestDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create test data directory
        os.makedirs("data/test", exist_ok=True)
        
        # Create a small test video
        cls.test_video_path = "data/test/test_pong.mp4"
        cls.create_test_video()
        
        # Create test block data
        cls.test_block_path = "data/test/test_block.npy"
        cls.create_test_block_data()
        
        # Create test latent block data
        cls.test_latent_path = "data/test/test_latent.npy"
        cls.create_test_latent_data()
        
        cls.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    @classmethod
    def create_test_video(cls):
        # Create a small test video file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(cls.test_video_path, fourcc, 30.0, (64, 64))
        
        for _ in range(100):  # 100 frames
            frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

    @classmethod
    def create_test_block_data(cls):
        # Create test block dataset
        test_data = []
        for _ in range(100):
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            test_data.append(((frame,),))
        np.save(cls.test_block_path, test_data)

    @classmethod
    def create_test_latent_data(cls):
        # Create test latent dataset
        test_data = np.random.rand(1000, 10)  # 1000 samples, 10 features
        np.save(cls.test_latent_path, test_data)

    def test_pong_dataset(self):
        # Test PongDataset
        dataset = PongDataset(
            self.test_video_path,
            transform=self.transform,
            save_path="data/test/test_pong.h5"
        )
        
        # Test length
        self.assertGreater(len(dataset), 0)
        
        # Test getting an item
        item, label = dataset[0]
        self.assertEqual(item.shape[0], 3)  # 3 channels
        self.assertEqual(item.shape[1], 64)  # Height
        self.assertEqual(item.shape[2], 64)  # Width
        self.assertEqual(label, 0)
        
        # Test data type and range
        self.assertTrue(torch.is_tensor(item))
        self.assertTrue(-1.0 <= item.min() <= item.max() <= 1.0)

    def test_block_dataset(self):
        # Test BlockDataset
        dataset = BlockDataset(
            self.test_block_path,
            transform=self.transform
        )
        
        # Test length
        self.assertGreater(len(dataset), 0)
        
        # Test getting an item
        item, label = dataset[0]
        self.assertEqual(item.shape[0], 3)  # 3 channels
        self.assertEqual(item.shape[1], 32)  # Height
        self.assertEqual(item.shape[2], 32)  # Width
        self.assertEqual(label, 0)

    def test_latent_block_dataset(self):
        # Test LatentBlockDataset
        dataset = LatentBlockDataset(
            self.test_latent_path
        )
        
        # Test length
        self.assertGreater(len(dataset), 0)
        
        # Test getting an item
        item, label = dataset[0]
        self.assertEqual(len(item.shape), 1)  # Should be 1D
        self.assertEqual(item.shape[0], 10)  # 10 features
        self.assertEqual(label, 0)

    @classmethod
    def tearDownClass(cls):
        # Clean up test files
        if os.path.exists("data/test/test_pong.mp4"):
            os.remove("data/test/test_pong.mp4")
        if os.path.exists("data/test/test_pong.h5"):
            os.remove("data/test/test_pong.h5")
        if os.path.exists("data/test/test_block.npy"):
            os.remove("data/test/test_block.npy")
        if os.path.exists("data/test/test_latent.npy"):
            os.remove("data/test/test_latent.npy")
        if os.path.exists("data/test"):
            os.rmdir("data/test")

if __name__ == '__main__':
    unittest.main()
