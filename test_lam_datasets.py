#!/usr/bin/env python3
"""
Test script to verify that LAM can load different datasets using the updated dataloader.
"""

import sys
import os
sys.path.insert(0, '.')

from src.latent_action_model.utils import load_sequence_dataset

def test_dataset_loading():
    """Test loading different datasets"""
    datasets = ['SONIC', 'PONG', 'BLOCK']
    
    for dataset_name in datasets:
        print(f"\nTesting {dataset_name} dataset...")
        try:
            # Load dataset with small batch size for testing
            loader = load_sequence_dataset(batch_size=4, seq_length=8, dataset_name=dataset_name)
            
            # Try to get one batch to verify it works
            batch = next(iter(loader))
            frames, labels = batch
            
            print(f"✅ {dataset_name} dataset loaded successfully!")
            print(f"   Batch shape: {frames.shape}")
            print(f"   Labels shape: {labels.shape}")
            
        except Exception as e:
            print(f"❌ Failed to load {dataset_name} dataset: {e}")

if __name__ == "__main__":
    test_dataset_loading() 