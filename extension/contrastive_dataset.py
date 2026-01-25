"""
Task 5: SimSaC Contrastive Fine-tuning
Dataset and DataLoader implementation

This module provides PyTorch Dataset for loading contrastive pairs.
"""

import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class ContrastivePairsDataset(Dataset):
    """
    PyTorch Dataset for contrastive learning pairs.
    
    Loads pairs from pickle files created by load_and_create_pairs_v2.py
    """
    
    def __init__(self, pairs_path, transform=None, target_size=(256, 256)):
        """
        Initialize dataset.
        
        Args:
            pairs_path: Path to .pkl file with pairs
            transform: Optional torchvision transforms
            target_size: Resize images to this size
        """
        # Load pairs
        with open(pairs_path, 'rb') as f:
            self.pairs = pickle.load(f)
        
        print(f"Loaded {len(self.pairs)} pairs from {pairs_path}")
        
        # Count positive/negative
        self.num_positive = sum(1 for p in self.pairs if p['label'] == 1)
        self.num_negative = len(self.pairs) - self.num_positive
        
        print(f"  Positive: {self.num_positive}")
        print(f"  Negative: {self.num_negative}")
        
        self.target_size = target_size
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet mean
                    std=[0.229, 0.224, 0.225]     # ImageNet std
                )
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Get a pair.

        Returns:
            img1: First image tensor [3, H, W]
            img2: Second image tensor [3, H, W]
            label: 1 for positive, 0 for negative
        """
        pair = self.pairs[idx]

        # Get images - support both 'image1'/'image2' and 'surface1'/'surface2' keys
        # for backward compatibility with both full UV map pairs and surface-level pairs
        img1 = pair.get('image1', pair.get('surface1'))
        img2 = pair.get('image2', pair.get('surface2'))

        if img1 is None or img2 is None:
            raise KeyError(f"Pair must contain either 'image1'/'image2' or 'surface1'/'surface2' keys. Found keys: {list(pair.keys())}")

        # Convert to PIL if they're numpy arrays
        if isinstance(img1, np.ndarray):
            img1 = Image.fromarray(img1)
        if isinstance(img2, np.ndarray):
            img2 = Image.fromarray(img2)

        # Ensure RGB
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')

        # Apply transforms
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        label = torch.tensor(pair['label'], dtype=torch.float32)

        return img1, img2, label
    
    def get_statistics(self):
        """Get dataset statistics."""
        return {
            'total_pairs': len(self.pairs),
            'positive_pairs': self.num_positive,
            'negative_pairs': self.num_negative,
            'positive_ratio': self.num_positive / len(self.pairs),
        }


def create_dataloaders(train_pairs_path, val_pairs_path, batch_size=16, 
                       num_workers=2, target_size=(256, 256)):
    """
    Create train and validation dataloaders.
    
    Args:
        train_pairs_path: Path to train_pairs.pkl
        val_pairs_path: Path to val_pairs.pkl
        batch_size: Batch size
        num_workers: Number of workers for data loading
        target_size: Image size
    
    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    print(f"\n{'='*70}")
    print("Creating DataLoaders")
    print(f"{'='*70}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Target size: {target_size}")
    
    # Create datasets
    train_dataset = ContrastivePairsDataset(
        train_pairs_path,
        target_size=target_size
    )
    
    val_dataset = ContrastivePairsDataset(
        val_pairs_path,
        target_size=target_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✓ DataLoaders created")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    return train_loader, val_loader, train_dataset, val_dataset


if __name__ == "__main__":
    # Test dataset loading
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python contrastive_dataset.py <path_to_train_pairs.pkl>")
        sys.exit(1)
    
    pairs_path = sys.argv[1]
    
    print(f"\nTesting dataset loading from: {pairs_path}")
    
    dataset = ContrastivePairsDataset(pairs_path)
    
    # Test loading a batch
    print(f"\nTesting data loading...")
    img1, img2, label = dataset[0]
    
    print(f"Image 1 shape: {img1.shape}")
    print(f"Image 2 shape: {img2.shape}")
    print(f"Label: {label}")
    print(f"Label dtype: {label.dtype}")
    
    # Print statistics
    print(f"\nDataset statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n✓ Dataset test passed!")
