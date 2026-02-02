#!/usr/bin/env python3
"""
SimSAC Adversarial Robustness Fine-Tuning

Fine-tune SimSAC with adversarial positive/negative pairs to make it robust
to adversarial perturbations while maintaining clean accuracy.

This satisfies the capstone objective: "Fine-tune SimSAC with positive/negative pairs"

Usage:
    # Step 1: Create adversarial pair dataset
    python train_simsac_adversarial_robust.py \
        --mode create_pairs \
        --clean_dir data/validation/carpet \
        --adversarial_dir data/adversarial_strategy1_carpet \
        --output_dir data/adversarial_pairs_carpet

    # Step 2: Fine-tune SimSAC
    python train_simsac_adversarial_robust.py \
        --mode train \
        --data_dir data/adversarial_pairs_carpet \
        --output_dir checkpoints/simsac_robust \
        --epochs 20 \
        --batch_size 4

    # Step 3: Evaluate robustness improvement
    python train_simsac_adversarial_robust.py \
        --mode evaluate \
        --checkpoint checkpoints/simsac_robust/best_model.pth \
        --clean_dir data/validation/carpet \
        --adversarial_dir data/adversarial_strategy1_carpet
"""

import argparse
from pathlib import Path
import pickle
import sys
from tqdm import tqdm
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd

# Add src to path - handle both local and Colab environments
import os
parent_dir = Path(__file__).parent.parent

# IMPORTANT: Add parent_dir FIRST so that 'from src.simsac...' works
# The simsac/inference.py file imports as 'from src.simsac...'
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Set PYTHONPATH for subprocesses
os.environ['PYTHONPATH'] = f"{parent_dir}:{os.environ.get('PYTHONPATH', '')}"

from src.simsac.inference import SimSaC
from src.tampering.utils import get_side_surface_patches
from src.tampering.parcel import PATCH_ORDER


class AdversarialPairDataset(Dataset):
    """
    Dataset for adversarial robustness training.

    Each sample contains:
    - reference_patch: Reference surface patch
    - field_patch: Field surface patch (clean or adversarial)
    - label: 0 = similar (untampered), 1 = dissimilar (tampered)
    - is_adversarial: Whether field is adversarial
    """

    def __init__(self, pairs_file):
        """
        Load pairs from pickle file.

        Args:
            pairs_file: Path to pickle file containing list of pair dicts
        """
        with open(pairs_file, 'rb') as f:
            self.pairs = pickle.load(f)

        print(f"Loaded {len(self.pairs)} pairs from {pairs_file}")

        # Count by type
        positive = sum(1 for p in self.pairs if p['label'] == 0)
        negative = sum(1 for p in self.pairs if p['label'] == 1)
        adversarial = sum(1 for p in self.pairs if p['is_adversarial'])

        print(f"  Positive (untampered): {positive}")
        print(f"  Negative (tampered): {negative}")
        print(f"  Adversarial samples: {adversarial}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Load and preprocess images
        ref_patch = cv2.imread(str(pair['reference_patch']))
        ref_patch = cv2.cvtColor(ref_patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        field_patch = cv2.imread(str(pair['field_patch']))
        field_patch = cv2.cvtColor(field_patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Convert to torch tensors (C, H, W)
        ref_patch = torch.from_numpy(ref_patch).permute(2, 0, 1)
        field_patch = torch.from_numpy(field_patch).permute(2, 0, 1)

        return {
            'reference': ref_patch,
            'field': field_patch,
            'label': torch.tensor(pair['label'], dtype=torch.long),
            'is_adversarial': torch.tensor(pair['is_adversarial'], dtype=torch.bool)
        }


class SimSaCRobust(nn.Module):
    """
    SimSAC wrapper for adversarial robustness fine-tuning.

    Adds a contrastive head on top of SimSAC's change detection features.
    """

    def __init__(self, freeze_backbone=True):
        """
        Initialize robust SimSAC model.

        Args:
            freeze_backbone: If True, freeze SimSAC backbone and only train head
        """
        super().__init__()

        # Load pretrained SimSAC
        self.simsac = SimSaC.get_instance()

        # SimSAC is a wrapper, get the actual model
        self.simsac_model = self.simsac.model

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.simsac_model.parameters():
                param.requires_grad = False
            print("Froze SimSAC backbone - only training contrastive head")
        else:
            print("Fine-tuning full SimSAC model")

        # Contrastive projection head
        # SimSAC outputs change maps of shape (H, W, 3)
        # We'll use global average pooling + MLP to get embeddings
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pool
            nn.Flatten(),
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # 32-dim embedding
        )

    def forward(self, reference, field):
        """
        Forward pass.

        Args:
            reference: Reference image (B, C, H, W)
            field: Field image (B, C, H, W)

        Returns:
            embedding: 32-dim embedding vector (B, 32)
            change_map: Raw change map from SimSAC (B, H, W, 3)
        """
        # Convert to numpy for SimSAC
        batch_size = reference.shape[0]
        embeddings = []
        change_maps = []

        for i in range(batch_size):
            ref_np = (reference[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            field_np = (field[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            # Run SimSAC
            imgs = self.simsac.inference(ref_np, field_np)
            change1 = imgs[0]  # First change map

            # Convert to tensor (H, W, 3)
            change1_tensor = torch.from_numpy(change1).float().to(reference.device)
            change1_tensor = change1_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

            # Get embedding
            embedding = self.projection_head(change1_tensor)

            embeddings.append(embedding)
            change_maps.append(change1_tensor)

        embeddings = torch.cat(embeddings, dim=0)  # (B, 32)
        change_maps = torch.cat(change_maps, dim=0)  # (B, 3, H, W)

        return embeddings, change_maps

    def get_trainable_parameters(self):
        """Get parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for adversarial robustness.

    Pulls similar pairs (untampered) together, pushes dissimilar pairs (tampered) apart.
    """

    def __init__(self, margin=1.0):
        """
        Args:
            margin: Margin for dissimilar pairs
        """
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        Compute contrastive loss.

        Args:
            embeddings: Embeddings (B, D)
            labels: 0 = similar (untampered), 1 = dissimilar (tampered)

        Returns:
            loss: Contrastive loss
        """
        # Pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Create label matrix (1 if same label, 0 if different)
        label_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # Similar pairs: minimize distance
        similar_loss = label_matrix * distances.pow(2)

        # Dissimilar pairs: maximize distance up to margin
        dissimilar_loss = (1 - label_matrix) * torch.clamp(self.margin - distances, min=0).pow(2)

        # Average over all pairs
        loss = (similar_loss + dissimilar_loss).mean()

        return loss


def create_adversarial_pairs(clean_dir, adversarial_dir, output_dir, backgrounds=['carpet_adv_fgsm', 'carpet_adv_pgd'], max_parcels=None):
    """
    Create adversarial training pairs from clean and adversarial UV maps.

    Creates positive pairs (untampered) and negative pairs (tampered) from both
    clean and adversarial samples.

    Args:
        clean_dir: Directory with clean UV maps
        adversarial_dir: Directory with adversarial UV maps
        output_dir: Output directory for pairs
        backgrounds: List of adversarial background names
        max_parcels: Maximum number of parcels to process (None = all). Use 2-3 for quick testing.
    """
    clean_dir = Path(clean_dir)
    adversarial_dir = Path(adversarial_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tampering labels from tampering_mapping.csv
    # Try multiple possible locations
    possible_paths = [
        parent_dir / 'src' / 'tampering' / 'tampering_mapping.csv',  # Standard location
        Path('/content/tampar/src/tampering/tampering_mapping.csv'),  # Colab location
    ]

    tampering_mapping_path = None
    for path in possible_paths:
        if path.exists():
            tampering_mapping_path = path
            break

    if tampering_mapping_path is None:
        print(f"Error: tampering_mapping.csv not found. Tried:")
        for path in possible_paths:
            print(f"  - {path}")
        return

    print(f"Loading tampering labels from: {tampering_mapping_path}")
    df_tampering = pd.read_csv(tampering_mapping_path)
    df_tampering.fillna('', inplace=True)

    # Create dict: (parcel_id, sideface) -> tampering_code
    tampering_dict = {}
    for _, row in df_tampering.iterrows():
        parcel_id = row['id']
        for sideface in ['top', 'bottom', 'left', 'right', 'center']:
            tampering_code = row.get(sideface, '')
            if tampering_code != '':
                tampering_dict[(parcel_id, sideface)] = tampering_code

    print(f"Loaded {len(tampering_dict)} tampering labels")

    pairs = []

    # Get unique parcel IDs and limit if requested
    clean_files_all = sorted(clean_dir.glob("*_uvmap_gt.png"))

    if max_parcels is not None:
        # Get unique parcel IDs
        parcel_ids = set()
        for f in clean_files_all:
            parcel_id = int(f.stem.split('_')[1])
            parcel_ids.add(parcel_id)

        selected_parcels = sorted(list(parcel_ids))[:max_parcels]
        clean_files = [f for f in clean_files_all if int(f.stem.split('_')[1]) in selected_parcels]
        print(f"\n⚠️  Limited to {max_parcels} parcels: {selected_parcels}")
    else:
        clean_files = clean_files_all
        print(f"\nProcessing all parcels")

    # Process clean samples
    print("\nProcessing clean samples...")

    for clean_file in tqdm(clean_files):
        # Parse filename
        parts = clean_file.stem.split('_')
        parcel_id = int(parts[1])

        # Load UV map
        uvmap = cv2.imread(str(clean_file))
        if uvmap is None:
            continue

        uvmap = cv2.cvtColor(uvmap, cv2.COLOR_BGR2RGB)

        # Extract patches
        patches = list(get_side_surface_patches(uvmap))

        # Create pairs for each surface
        for surf_idx, patch in enumerate(patches):
            surface_name = PATCH_ORDER[surf_idx]

            # Skip if mostly white
            if np.mean(patch) >= 250:
                continue

            # Get tampering label
            tampering_code = tampering_dict.get((parcel_id, surface_name), '')
            is_tampered = bool(tampering_code)

            # Save reference and field patches
            ref_patch_path = output_dir / f"ref_id{parcel_id:02d}_{surface_name}_clean.png"
            field_patch_path = output_dir / f"field_id{parcel_id:02d}_{surface_name}_clean.png"

            # Save patches
            success_ref = cv2.imwrite(str(ref_patch_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            success_field = cv2.imwrite(str(field_patch_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

            if not success_ref or not success_field:
                print(f"Warning: Failed to save patches for parcel {parcel_id}, surface {surface_name}")
                continue

            pairs.append({
                'reference_patch': ref_patch_path,
                'field_patch': field_patch_path,
                'label': 1 if is_tampered else 0,  # 0=untampered, 1=tampered
                'is_adversarial': False,
                'parcel_id': parcel_id,
                'surface': surface_name,
                'tampering': tampering_code
            })

    # Process adversarial samples
    print("\nProcessing adversarial samples...")
    for background in backgrounds:
        adv_dir = adversarial_dir / background
        if not adv_dir.exists():
            print(f"Warning: {adv_dir} not found, skipping")
            continue

        adv_files_all = sorted(adv_dir.glob("*_uvmap_gt.png"))

        # Apply same parcel limit
        if max_parcels is not None:
            adv_files = [f for f in adv_files_all if int(f.stem.split('_')[1]) in selected_parcels]
        else:
            adv_files = adv_files_all

        for adv_file in tqdm(adv_files, desc=background):
            # Parse filename
            parts = adv_file.stem.split('_')
            parcel_id = int(parts[1])

            # Load UV map
            uvmap = cv2.imread(str(adv_file))
            if uvmap is None:
                continue

            uvmap = cv2.cvtColor(uvmap, cv2.COLOR_BGR2RGB)

            # Extract patches
            patches = list(get_side_surface_patches(uvmap))

            # Create pairs for each surface
            for surf_idx, patch in enumerate(patches):
                surface_name = PATCH_ORDER[surf_idx]

                # Skip if mostly white
                if np.mean(patch) >= 250:
                    continue

                # Get tampering label
                tampering_code = tampering_dict.get((parcel_id, surface_name), '')
                is_tampered = bool(tampering_code)

                # Save reference (use clean) and adversarial field patches
                ref_patch_path = output_dir / f"ref_id{parcel_id:02d}_{surface_name}_clean.png"
                field_patch_path = output_dir / f"field_id{parcel_id:02d}_{surface_name}_{background}.png"

                # Reference already saved from clean processing
                cv2.imwrite(str(field_patch_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

                pairs.append({
                    'reference_patch': ref_patch_path,
                    'field_patch': field_patch_path,
                    'label': 1 if is_tampered else 0,
                    'is_adversarial': True,
                    'parcel_id': parcel_id,
                    'surface': surface_name,
                    'tampering': tampering_code,
                    'attack': background
                })

    # Save pairs
    pairs_file = output_dir / 'adversarial_pairs.pkl'
    with open(pairs_file, 'wb') as f:
        pickle.dump(pairs, f)

    print(f"\n✓ Created {len(pairs)} pairs")
    print(f"  Saved to: {pairs_file}")

    # Print statistics
    positive = sum(1 for p in pairs if p['label'] == 0)
    negative = sum(1 for p in pairs if p['label'] == 1)
    adversarial = sum(1 for p in pairs if p['is_adversarial'])
    clean = len(pairs) - adversarial

    print(f"\nStatistics:")
    print(f"  Positive (untampered): {positive}")
    print(f"  Negative (tampered): {negative}")
    print(f"  Clean samples: {clean}")
    print(f"  Adversarial samples: {adversarial}")

    return pairs_file


def train_robust_simsac(data_dir, output_dir, epochs=20, batch_size=4, learning_rate=1e-4, freeze_backbone=True):
    """
    Train SimSAC for adversarial robustness.

    Args:
        data_dir: Directory with adversarial pairs
        output_dir: Output directory for checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        freeze_backbone: If True, freeze SimSAC backbone
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    pairs_file = data_dir / 'adversarial_pairs.pkl'
    dataset = AdversarialPairDataset(pairs_file)

    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\nDataset split:")
    print(f"  Training: {train_size} pairs")
    print(f"  Validation: {val_size} pairs")

    # Create model
    model = SimSaCRobust(freeze_backbone=freeze_backbone).to(device)

    # Loss and optimizer
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.get_trainable_parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }

    best_val_loss = float('inf')

    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            reference = batch['reference'].to(device)
            field = batch['field'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            embeddings, _ = model(reference, field)
            loss = criterion(embeddings, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                reference = batch['reference'].to(device)
                field = batch['field'].to(device)
                labels = batch['label'].to(device)

                embeddings, _ = model(reference, field)
                loss = criterion(embeddings, labels)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")

    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, output_dir / 'final_model.pth')

    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='SimSAC Adversarial Robustness Fine-Tuning')
    parser.add_argument('--mode', type=str, required=True, choices=['create_pairs', 'train', 'evaluate'],
                       help='Mode: create_pairs, train, or evaluate')

    # Pair creation args
    parser.add_argument('--clean_dir', type=str, help='Directory with clean UV maps')
    parser.add_argument('--adversarial_dir', type=str, help='Directory with adversarial UV maps')
    parser.add_argument('--max_parcels', type=int, default=None,
                       help='Maximum number of parcels to use (for quick testing). Default: None (all)')

    # Training args
    parser.add_argument('--data_dir', type=str, help='Directory with adversarial pairs')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                       help='Freeze SimSAC backbone (only train head)')
    parser.add_argument('--full_finetune', action='store_true',
                       help='Fine-tune full model (override freeze_backbone)')

    args = parser.parse_args()

    if args.mode == 'create_pairs':
        create_adversarial_pairs(args.clean_dir, args.adversarial_dir, args.output_dir, max_parcels=args.max_parcels)

    elif args.mode == 'train':
        freeze = args.freeze_backbone and not args.full_finetune
        train_robust_simsac(
            args.data_dir,
            args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            freeze_backbone=freeze
        )

    elif args.mode == 'evaluate':
        print("Evaluation mode not yet implemented")
        print("Use src/tools/predict_tampering_adversarial_eval.py with the fine-tuned checkpoint")


if __name__ == '__main__':
    main()
