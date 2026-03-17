
import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from PIL import Image
from torchvision import transforms

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())

from extension.simsac_multitask_model import create_multitask_model


class TripletDataset(Dataset):

    def __init__(self, triplet_csv, transform=None, img_size=512):
        self.df = pd.read_csv(triplet_csv)
        self.transform = transform
        self.img_size = img_size

        # Resize transform for 256x256 version
        self.resize_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        # Full resolution transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load images
        anchor_img = Image.open(row['anchor']).convert('RGB')
        positive_img = Image.open(row['positive']).convert('RGB')
        negative_img = Image.open(row['negative']).convert('RGB')

        # Apply transforms
        anchor = self.transform(anchor_img)
        anchor_256 = self.resize_transform(anchor_img)

        positive = self.transform(positive_img)
        positive_256 = self.resize_transform(positive_img)

        negative = self.transform(negative_img)
        negative_256 = self.resize_transform(negative_img)

        # Labels
        anchor_label = torch.tensor(row['anchor_label'], dtype=torch.long)
        positive_label = torch.tensor(row['positive_label'], dtype=torch.long)
        negative_label = torch.tensor(row['negative_label'], dtype=torch.long)

        return {
            'anchor': anchor,
            'anchor_256': anchor_256,
            'positive': positive,
            'positive_256': positive_256,
            'negative': negative,
            'negative_256': negative_256,
            'anchor_label': anchor_label,
            'positive_label': positive_label,
            'negative_label': negative_label,
        }


class QuadrupletDataset(Dataset):

    def __init__(self, quadruplet_csv, transform=None, img_size=512):
        self.df = pd.read_csv(quadruplet_csv)
        self.transform = transform
        self.img_size = img_size

        # Resize transform for 256x256 version
        self.resize_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        # Full resolution transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load images
        anchor_img = Image.open(row['anchor']).convert('RGB')
        positive_img = Image.open(row['positive']).convert('RGB')
        negative1_img = Image.open(row['negative1']).convert('RGB')
        negative2_img = Image.open(row['negative2']).convert('RGB')

        # Apply transforms
        anchor = self.transform(anchor_img)
        anchor_256 = self.resize_transform(anchor_img)

        positive = self.transform(positive_img)
        positive_256 = self.resize_transform(positive_img)

        negative1 = self.transform(negative1_img)
        negative1_256 = self.resize_transform(negative1_img)

        negative2 = self.transform(negative2_img)
        negative2_256 = self.resize_transform(negative2_img)

        # Labels
        anchor_label = torch.tensor(row['anchor_label'], dtype=torch.long)
        positive_label = torch.tensor(row['positive_label'], dtype=torch.long)
        negative1_label = torch.tensor(row['negative1_label'], dtype=torch.long)
        negative2_label = torch.tensor(row['negative2_label'], dtype=torch.long)

        return {
            'anchor': anchor,
            'anchor_256': anchor_256,
            'positive': positive,
            'positive_256': positive_256,
            'negative1': negative1,
            'negative1_256': negative1_256,
            'negative2': negative2,
            'negative2_256': negative2_256,
            'anchor_label': anchor_label,
            'positive_label': positive_label,
            'negative1_label': negative1_label,
            'negative2_label': negative2_label,
        }


class TripletLoss(nn.Module):

    def __init__(self, margin=0.5, mining='hard'):
        super().__init__()
        self.margin = margin
        self.mining = mining

    def forward(self, anchor_emb, positive_emb, negative_emb):
        # Compute distances
        pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2)
        neg_dist = F.pairwise_distance(anchor_emb, negative_emb, p=2)

        # Triplet loss: max(0, margin + pos_dist - neg_dist)
        losses = F.relu(self.margin + pos_dist - neg_dist)

        if self.mining == 'hard':
            hard_losses = losses[losses > 0]
            if len(hard_losses) > 0:
                return hard_losses.mean()
            else:
                return losses.mean()
        else:
            return losses.mean()


class QuadrupletLoss(nn.Module):

    def __init__(self, margin1=0.5, margin2=0.3):
        super().__init__()
        self.margin1 = margin1
        self.margin2 = margin2

    def forward(self, anchor_emb, positive_emb, negative1_emb, negative2_emb):
        # Compute distances
        pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2)
        neg1_dist = F.pairwise_distance(anchor_emb, negative1_emb, p=2)
        neg2_dist = F.pairwise_distance(anchor_emb, negative2_emb, p=2)

        # Quadruplet loss with two margins
        loss1 = F.relu(self.margin1 + pos_dist - neg1_dist)
        loss2 = F.relu(self.margin2 + pos_dist - neg2_dist)

        return (loss1 + loss2).mean()


class MultiTaskTrainer:

    def __init__(
        self,
        model,
        triplet_loader,
        quadruplet_loader,
        output_dir,
        device='cuda',
        triplet_weight=1.0,
        classification_weight=0.5,
        quadruplet_weight=0.3,
        triplet_margin=0.5,
        learning_rate=1e-4,
        weight_decay=1e-5
    ):
        self.model = model
        self.triplet_loader = triplet_loader
        self.quadruplet_loader = quadruplet_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Loss weights
        self.triplet_weight = triplet_weight
        self.classification_weight = classification_weight
        self.quadruplet_weight = quadruplet_weight

        # Losses
        self.triplet_loss_fn = TripletLoss(margin=triplet_margin, mining='hard')
        self.quadruplet_loss_fn = QuadrupletLoss(margin1=0.5, margin2=0.3)
        self.classification_loss_fn = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Tracking
        self.history = {
            'train_loss': [],
            'train_triplet_loss': [],
            'train_classification_loss': [],
            'train_quadruplet_loss': [],
            'train_accuracy': [],
            'learning_rate': []
        }

        self.best_loss = float('inf')
        self.best_epoch = 0

    def train_epoch_triplets(self, epoch):
        self.model.train()

        total_loss = 0
        total_triplet_loss = 0
        total_classification_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.triplet_loader, desc=f"Epoch {epoch+1} (Triplets)")

        for batch in pbar:
            # Move to device
            anchor = batch['anchor'].to(self.device)
            anchor_256 = batch['anchor_256'].to(self.device)
            positive = batch['positive'].to(self.device)
            positive_256 = batch['positive_256'].to(self.device)
            negative = batch['negative'].to(self.device)
            negative_256 = batch['negative_256'].to(self.device)

            anchor_label = batch['anchor_label'].to(self.device)
            negative_label = batch['negative_label'].to(self.device)


            # Anchor-Positive embeddings
            anchor_pos_emb, anchor_logits = self.model(
                anchor, positive, anchor_256, positive_256
            )

            # Anchor-Negative embeddings
            anchor_neg_emb, negative_logits = self.model(
                anchor, negative, anchor_256, negative_256
            )


            triplet_loss = self.triplet_loss_fn(
                anchor_pos_emb,
                anchor_pos_emb,
                anchor_neg_emb
            )

            # Classification loss (on negative samples - tampered)
            classification_loss = self.classification_loss_fn(negative_logits, negative_label)

            # Combined loss
            loss = (
                self.triplet_weight * triplet_loss +
                self.classification_weight * classification_loss
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_triplet_loss += triplet_loss.item()
            total_classification_loss += classification_loss.item()

            # Accuracy
            _, predicted = negative_logits.max(1)
            total += negative_label.size(0)
            correct += predicted.eq(negative_label).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'triplet': f'{triplet_loss.item():.4f}',
                'cls': f'{classification_loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        avg_loss = total_loss / len(self.triplet_loader)
        avg_triplet_loss = total_triplet_loss / len(self.triplet_loader)
        avg_classification_loss = total_classification_loss / len(self.triplet_loader)
        accuracy = 100. * correct / total

        return avg_loss, avg_triplet_loss, avg_classification_loss, 0, accuracy

    def train_epoch_quadruplets(self, epoch):
        if self.quadruplet_loader is None:
            return 0, 0, 0, 0, 0

        self.model.train()

        total_loss = 0
        total_quadruplet_loss = 0
        total_classification_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.quadruplet_loader, desc=f"Epoch {epoch+1} (Quadruplets)")

        for batch in pbar:
            # Move to device
            anchor = batch['anchor'].to(self.device)
            anchor_256 = batch['anchor_256'].to(self.device)
            positive = batch['positive'].to(self.device)
            positive_256 = batch['positive_256'].to(self.device)
            negative1 = batch['negative1'].to(self.device)
            negative1_256 = batch['negative1_256'].to(self.device)
            negative2 = batch['negative2'].to(self.device)
            negative2_256 = batch['negative2_256'].to(self.device)

            negative1_label = batch['negative1_label'].to(self.device)
            negative2_label = batch['negative2_label'].to(self.device)

            # Forward pass for all pairs
            anchor_pos_emb, _ = self.model(anchor, positive, anchor_256, positive_256)
            anchor_neg1_emb, neg1_logits = self.model(anchor, negative1, anchor_256, negative1_256)
            anchor_neg2_emb, neg2_logits = self.model(anchor, negative2, anchor_256, negative2_256)

            # Compute quadruplet loss
            quadruplet_loss = self.quadruplet_loss_fn(
                anchor_pos_emb,
                anchor_pos_emb,
                anchor_neg1_emb,
                anchor_neg2_emb
            )

            # Classification loss
            classification_loss = (
                self.classification_loss_fn(neg1_logits, negative1_label) +
                self.classification_loss_fn(neg2_logits, negative2_label)
            ) / 2

            # Combined loss
            loss = (
                self.quadruplet_weight * quadruplet_loss +
                self.classification_weight * classification_loss
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_quadruplet_loss += quadruplet_loss.item()
            total_classification_loss += classification_loss.item()

            # Accuracy
            _, pred1 = neg1_logits.max(1)
            _, pred2 = neg2_logits.max(1)
            total += negative1_label.size(0) + negative2_label.size(0)
            correct += pred1.eq(negative1_label).sum().item() + pred2.eq(negative2_label).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'quad': f'{quadruplet_loss.item():.4f}',
                'cls': f'{classification_loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        avg_loss = total_loss / len(self.quadruplet_loader)
        avg_quadruplet_loss = total_quadruplet_loss / len(self.quadruplet_loader)
        avg_classification_loss = total_classification_loss / len(self.quadruplet_loader)
        accuracy = 100. * correct / total

        return avg_loss, 0, avg_classification_loss, avg_quadruplet_loss, accuracy

    def train(self, num_epochs):
        print("Starting Multi-Task Training")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
        print(f"Triplet weight: {self.triplet_weight}")
        print(f"Classification weight: {self.classification_weight}")
        print(f"Quadruplet weight: {self.quadruplet_weight}")

        for epoch in range(num_epochs):
            # Train on triplets
            trip_loss, trip_triplet, trip_cls, _, trip_acc = self.train_epoch_triplets(epoch)

            # Train on quadruplets (if available)
            quad_loss, _, quad_cls, quad_quad, quad_acc = self.train_epoch_quadruplets(epoch)

            # Combined metrics
            total_loss = (trip_loss + quad_loss) / 2 if quad_loss > 0 else trip_loss
            total_accuracy = (trip_acc + quad_acc) / 2 if quad_acc > 0 else trip_acc

            # Update scheduler
            self.scheduler.step(total_loss)

            # Track history
            self.history['train_loss'].append(total_loss)
            self.history['train_triplet_loss'].append(trip_triplet)
            self.history['train_classification_loss'].append((trip_cls + quad_cls) / 2 if quad_cls > 0 else trip_cls)
            self.history['train_quadruplet_loss'].append(quad_quad)
            self.history['train_accuracy'].append(total_accuracy)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Total Loss: {total_loss:.4f}")
            print(f"  Triplet Loss: {trip_triplet:.4f}")
            print(f"  Classification Loss: {(trip_cls + quad_cls) / 2 if quad_cls > 0 else trip_cls:.4f}")
            if quad_quad > 0:
                print(f"  Quadruplet Loss: {quad_quad:.4f}")
            print(f"  Accuracy: {total_accuracy:.2f}%")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save checkpoint
            self.save_checkpoint(epoch, is_best=(total_loss < self.best_loss))

            if total_loss < self.best_loss:
                self.best_loss = total_loss
                self.best_epoch = epoch

            # Save training history
            self.save_history()

        print("Training Complete!")
        print(f"Best loss: {self.best_loss:.4f} at epoch {self.best_epoch+1}")

    def save_checkpoint(self, epoch, is_best=False):
        # Extract only the base SimSaC model weights
        full_state_dict = self.model.state_dict()
        simsac_state_dict = {}

        for key, value in full_state_dict.items():
            if key.startswith('simsac.'):
                # Remove "simsac." prefix to match TAMPAR's format
                new_key = key[7:]
                simsac_state_dict[new_key] = value
            # Skip projection_head, classification_head, tampering_encoder, feature_projector

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': simsac_state_dict,
            'full_model_state_dict': full_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history
        }

        # Save latest checkpoint
        latest_path = self.output_dir / 'latest_model.pth'
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"   Saved best model (loss: {self.best_loss:.4f})")

    def save_history(self):
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Multi-task training for tampering detection")

    # Data
    parser.add_argument('--triplet_pairs', type=str, required=True,
                       help='Path to triplet pairs CSV')
    parser.add_argument('--quadruplet_pairs', type=str, default=None,
                       help='Path to quadruplet pairs CSV (optional)')

    # Model
    parser.add_argument('--simsac_checkpoint', type=str, default=None,
                       help='Path to pre-trained SimSaC checkpoint')
    parser.add_argument('--freeze_simsac', action='store_true',
                       help='Freeze SimSaC weights')
    parser.add_argument('--projection_dim', type=int, default=512,
                       help='Projection dimension for contrastive learning')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of tampering classes')

    # Training
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')

    # Loss weights
    parser.add_argument('--triplet_weight', type=float, default=1.0,
                       help='Weight for triplet loss')
    parser.add_argument('--classification_weight', type=float, default=0.5,
                       help='Weight for classification loss')
    parser.add_argument('--quadruplet_weight', type=float, default=0.3,
                       help='Weight for quadruplet loss')
    parser.add_argument('--triplet_margin', type=float, default=0.5,
                       help='Margin for triplet loss')

    # Other
    parser.add_argument('--img_size', type=int, default=512,
                       help='Image size')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loading workers')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    print("\nCreating multi-task model")
    model = create_multitask_model(
        simsac_checkpoint=args.simsac_checkpoint,
        freeze_simsac=args.freeze_simsac,
        projection_dim=args.projection_dim,
        num_tampering_classes=args.num_classes,
        device=device
    )

    # Create datasets
    print("\nLoading triplet data")
    triplet_dataset = TripletDataset(
        args.triplet_pairs,
        img_size=args.img_size
    )
    triplet_loader = DataLoader(
        triplet_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    quadruplet_loader = None
    if args.quadruplet_pairs:
        print("Loading quadruplet data")
        quadruplet_dataset = QuadrupletDataset(
            args.quadruplet_pairs,
            img_size=args.img_size
        )
        quadruplet_loader = DataLoader(
            quadruplet_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

    # Create trainer
    trainer = MultiTaskTrainer(
        model=model,
        triplet_loader=triplet_loader,
        quadruplet_loader=quadruplet_loader,
        output_dir=args.output_dir,
        device=device,
        triplet_weight=args.triplet_weight,
        classification_weight=args.classification_weight,
        quadruplet_weight=args.quadruplet_weight,
        triplet_margin=args.triplet_margin,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Train
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()
