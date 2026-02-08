"""
Task 5: SimSaC Contrastive Training Script

Main training loop for fine-tuning SimSaC with contrastive learning.

Supports both full UV map pairs and surface-level pairs.

Usage:
    # Phase 1: Frozen backbone (auto-detects surface-level pairs)
    python train_simsac_contrastive.py --phase 1 --data_dir /content/tampar/data/tampar_sample/contrastive_pairs_surface

    # Phase 2: Full fine-tuning
    python train_simsac_contrastive.py --phase 2 --data_dir /content/tampar/data/tampar_sample/contrastive_pairs_surface --checkpoint phase1_final.pth

    # Explicitly specify pair files
    python train_simsac_contrastive.py --phase 1 --data_dir /path/to/dir --train_pairs train_pairs_surface_level.pkl --val_pairs val_pairs_surface_level.pkl
"""

import os
import argparse
from pathlib import Path
import json
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from contrastive_dataset import create_dataloaders
from contrastive_losses import CombinedLoss
from simsac_contrastive_model import create_simsac_contrastive


class Trainer:
    """Trainer for SimSaC contrastive learning."""
    
    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        """
        Initialize trainer.
        
        Args:
            model: SimSaCContrastive model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dict
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = CombinedLoss(
            lambda_contrastive=config['lambda_contrastive'],
            lambda_flow=config.get('lambda_flow', 0.0),
            lambda_change=config.get('lambda_change', 0.0),
            temperature=config['temperature'],
            use_simplified=True  # Use simplified loss with explicit labels
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.get_trainable_parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.start_epoch = 0
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        loss_components = {
            'contrastive': 0,
            'total': 0
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Train]")
        
        for batch_idx, (img1, img2, labels) in enumerate(pbar):
            # Move to device
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            z1, z2 = self.model(img1, img2)
            
            # Compute loss
            loss, loss_dict = self.criterion(z1, z2, labels=labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (optional)
            if self.config.get('gradient_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            for key in loss_dict:
                if key in loss_components:
                    loss_components[key] += loss_dict[key]
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/(batch_idx+1):.4f}"
            })
        
        # Average losses
        avg_loss = total_loss / len(self.train_loader)
        for key in loss_components:
            loss_components[key] /= len(self.train_loader)
        
        return avg_loss, loss_components
    
    def validate(self, epoch):
        """Validate on validation set."""
        self.model.eval()
        
        total_loss = 0
        loss_components = {
            'contrastive': 0,
            'total': 0
        }
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Val]  ")
        
        with torch.no_grad():
            for img1, img2, labels in pbar:
                # Move to device
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                z1, z2 = self.model(img1, img2)
                
                # Compute loss
                loss, loss_dict = self.criterion(z1, z2, labels=labels)
                
                # Update statistics
                total_loss += loss.item()
                for key in loss_dict:
                    if key in loss_components:
                        loss_components[key] += loss_dict[key]
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss/(pbar.n+1):.4f}"
                })
        
        # Average losses
        avg_loss = total_loss / len(self.val_loader)
        for key in loss_components:
            loss_components[key] /= len(self.val_loader)
        
        return avg_loss, loss_components
    
    def train(self, output_dir):
        """Main training loop."""
        print(f"\n{'='*70}")
        print(f"Starting Training - Phase {self.config['phase']}")
        print(f"{'='*70}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Freeze backbone: {self.config['freeze_backbone']}")
        print(f"Output directory: {output_dir}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            # Train
            train_loss, train_components = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_components = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if (epoch + 1) % self.config.get('save_every', 5) == 0 or is_best:
                self.save_checkpoint(
                    epoch,
                    output_dir,
                    is_best=is_best
                )
            
            # Plot progress
            if (epoch + 1) % 5 == 0:
                self.plot_progress(output_dir)
        
        # Save final model
        self.save_checkpoint(
            self.config['epochs'] - 1,
            output_dir,
            filename=f"phase{self.config['phase']}_final.pth"
        )
        
        # Final plots
        self.plot_progress(output_dir)
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"{'='*70}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final checkpoint: {output_dir}/phase{self.config['phase']}_final.pth")
    
    def save_checkpoint(self, epoch, output_dir, is_best=False, filename=None):
        """Save model checkpoint in TAMPAR-compatible format."""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch+1}.pth"

        # Extract only the base SimSaC model weights (remove "simsac." prefix and projection head)
        # This ensures compatibility with TAMPAR's inference.py
        full_state_dict = self.model.state_dict()
        simsac_state_dict = {}

        for key, value in full_state_dict.items():
            if key.startswith('simsac.'):
                # Remove "simsac." prefix to match TAMPAR's format
                new_key = key[7:]  # Remove "simsac." (7 characters)
                simsac_state_dict[new_key] = value
            # Skip projection_head.* keys - they're not needed for TAMPAR inference

        # Save checkpoint with TAMPAR-compatible format
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': simsac_state_dict,  # Only base SimSaC weights, no prefix
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }

        checkpoint_path = output_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")

        if is_best:
            # Save with phase-specific name for clarity
            phase = self.config.get('phase', 1)
            best_path = output_dir / f'phase{phase}_best.pth'
            torch.save(checkpoint, best_path)
            print(f"  ✓ New best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint to resume training."""
        print(f"\nLoading checkpoint from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # The checkpoint contains only base SimSaC weights (no "simsac." prefix)
        # We need to add the prefix back to load into our wrapped model
        simsac_state_dict = checkpoint['state_dict']

        # Add "simsac." prefix to all keys
        wrapped_state_dict = {}
        for key, value in simsac_state_dict.items():
            wrapped_state_dict[f'simsac.{key}'] = value

        # Load into the wrapped model (this only loads the simsac part, projection_head keeps its random init)
        self.model.load_state_dict(wrapped_state_dict, strict=False)

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        self.start_epoch = checkpoint['epoch']

        print(f"✓ Checkpoint loaded")
        print(f"  Resuming from epoch: {self.start_epoch}")
        print(f"  Best val loss so far: {self.best_val_loss:.4f}")
    
    def plot_progress(self, output_dir):
        """Plot training progress."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.history['train_loss'], label='Train Loss', marker='o')
        ax1.plot(epochs, self.history['val_loss'], label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax2.plot(epochs, self.history['learning_rate'], label='Learning Rate', marker='o', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / 'training_progress.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"  Progress plot saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Train SimSaC with contrastive learning")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing pair files (supports both surface-level and full UV map pairs)')
    parser.add_argument('--train_pairs', type=str, default=None,
                       help='Path to train pairs file (auto-detected if not specified)')
    parser.add_argument('--val_pairs', type=str, default=None,
                       help='Path to val pairs file (auto-detected if not specified)')
    parser.add_argument('--weights_path', type=str,
                       default='/content/tampar/src/simsac/weight/synth_then_joint_synth_changesim.pth',
                       help='Path to pre-trained SimSaC weights')
    
    # Training arguments
    parser.add_argument('--phase', type=int, choices=[1, 2], required=True,
                       help='Training phase: 1=frozen backbone, 2=full fine-tuning')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (default: 10 for phase1, 20 for phase2)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (default: 1e-3 for phase1, 1e-4 for phase2)')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature for contrastive loss')
    
    # Resume training
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint to resume from')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='/content/outputs/training',
                       help='Output directory for checkpoints')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Set defaults based on phase
    if args.epochs is None:
        args.epochs = 10 if args.phase == 1 else 20
    
    if args.lr is None:
        args.lr = 1e-3 if args.phase == 1 else 1e-4
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Create config
    config = {
        'phase': args.phase,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'temperature': args.temperature,
        'freeze_backbone': (args.phase == 1),
        'lambda_contrastive': 1.0,
        'lambda_flow': 0.0,  # Not using flow loss for now
        'lambda_change': 0.0,  # Not using change loss for now
        'weight_decay': 1e-4,
        'min_lr': 1e-6,
        'gradient_clip': 1.0,
        'save_every': 5
    }
    
    print(f"\n{'='*70}")
    print("SimSaC Contrastive Training")
    print(f"{'='*70}")
    print(f"Phase: {args.phase}")
    print(f"Device: {device}")

    # Auto-detect pair files if not specified
    data_dir = Path(args.data_dir)

    if args.train_pairs is None:
        # Try surface-level pairs first, then fall back to regular pairs
        candidates = [
            data_dir / 'train_pairs_surface_level.pkl',
            data_dir / 'train_pairs.pkl'
        ]
        train_path = None
        for candidate in candidates:
            if candidate.exists():
                train_path = candidate
                break
        if train_path is None:
            raise FileNotFoundError(f"Could not find train pairs file in {data_dir}. Tried: {[str(c) for c in candidates]}")
    else:
        train_path = Path(args.train_pairs)

    if args.val_pairs is None:
        # Try surface-level pairs first, then fall back to regular pairs
        candidates = [
            data_dir / 'val_pairs_surface_level.pkl',
            data_dir / 'val_pairs.pkl'
        ]
        val_path = None
        for candidate in candidates:
            if candidate.exists():
                val_path = candidate
                break
        if val_path is None:
            raise FileNotFoundError(f"Could not find val pairs file in {data_dir}. Tried: {[str(c) for c in candidates]}")
    else:
        val_path = Path(args.val_pairs)

    print(f"\nUsing pair files:")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")

    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
        str(train_path),
        str(val_path),
        batch_size=args.batch_size,
        num_workers=2
    )
    
    # Create model
    model = create_simsac_contrastive(
        weights_path=args.weights_path,
        projection_dim=128,
        freeze_backbone=config['freeze_backbone'],
        device=device
    )
    
    # If phase 2, may need to load phase 1 checkpoint
    if args.phase == 2 and args.checkpoint is None:
        phase1_checkpoint = Path(args.output_dir) / 'phase1_final.pth'
        if phase1_checkpoint.exists():
            args.checkpoint = str(phase1_checkpoint)
            print(f"\nPhase 2 detected, loading phase 1 checkpoint: {args.checkpoint}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Load checkpoint if resuming
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        
        # If phase 2, unfreeze backbone
        if args.phase == 2:
            model.unfreeze_all()
            # Update optimizer with new parameters
            trainer.optimizer = optim.Adam(
                model.get_trainable_parameters(),
                lr=config['learning_rate'],
                weight_decay=config.get('weight_decay', 1e-4)
            )
    
    # Train
    trainer.train(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
