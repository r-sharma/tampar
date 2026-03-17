
import os
import sys
import argparse
import types
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add extension dir to path (for local imports)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'extension'))

from contrastive_dataset import create_dataloaders
from contrastive_losses import CombinedLoss
from simsac_contrastive_model import create_simsac_contrastive
from src.simsac.models.our_models.SimSaC import SimSaC_Model


# SimSAC Change-Map Attack (same approach as generate_adversarial_simsac_targeted.py)

def enable_simsac_gradients(model):
    def forward_with_grad(self, im_target, im_source, im_target_256, im_source_256, disable_flow=None):
        im1_pyr = self.pyramid(im_target, eigth_resolution=True)
        im2_pyr = self.pyramid(im_source, eigth_resolution=True)

        im1_pyr_256 = self.pyramid(im_target_256)
        im2_pyr_256 = self.pyramid(im_source_256)
        c14 = im1_pyr_256[-3]
        c24 = im2_pyr_256[-3]

        flow4, corr4 = self.coarsest_resolution_flow(c14, c24, h_256=256, w_256=256, return_corr=True)

        return {
            "flow": ([flow4], [flow4]),
            "change": ([corr4], [corr4])
        }

    model.forward_sigle_ref = types.MethodType(forward_with_grad, model)
    return model


def simsac_change_loss(simsac_model, field_img, reference_img):
    field_512 = F.interpolate(field_img, size=(512, 512), mode='bilinear', align_corners=False)
    ref_512 = F.interpolate(reference_img, size=(512, 512), mode='bilinear', align_corners=False)
    field_256 = F.interpolate(field_img, size=(256, 256), mode='bilinear', align_corners=False)
    ref_256 = F.interpolate(reference_img, size=(256, 256), mode='bilinear', align_corners=False)

    output = simsac_model(field_512, ref_512, field_256, ref_256)

    # Extract change map
    change = None
    if isinstance(output, dict):
        change_tuple = output.get('change', None)
        if change_tuple and isinstance(change_tuple, tuple) and len(change_tuple) == 2:
            change = change_tuple[1][-1] if len(change_tuple[1]) > 0 else change_tuple[0][-1]
    elif isinstance(output, (list, tuple)) and len(output) >= 2:
        change = output[1]

    if change is None:
        raise ValueError(f"SimSAC did not return change output. Got: {type(output)}")

    change_magnitude = torch.sqrt(change[:, 0] ** 2 + change[:, 1] ** 2 + 1e-8)
    return change_magnitude.mean()


def compute_change_map_loss(simsac_model, img2_adv, img1, labels):
    loss_terms = []
    for i in range(img2_adv.shape[0]):
        change_mag = simsac_change_loss(simsac_model, img2_adv[i:i+1], img1[i:i+1])
        label_val = labels[i].item()
        # Tampered  maximize (negate), Clean  minimize (keep positive)
        signed_loss = -(2.0 * label_val - 1.0) * change_mag
        loss_terms.append(signed_loss)

    return torch.stack(loss_terms).mean()


class OnlineAttackGenerator:

    def __init__(self, simsac_model, attack_type='fgsm', epsilon=0.05,
                 pgd_steps=5, pgd_step_size=None, device='cuda'):
        self.simsac = simsac_model
        self.simsac = enable_simsac_gradients(self.simsac)
        self.simsac.train()

        self.attack_type = attack_type
        self.epsilon = epsilon
        self.pgd_steps = pgd_steps
        self.pgd_step_size = pgd_step_size if pgd_step_size else epsilon / 4
        self.device = device

        if attack_type != 'none':
            print(f"  Online attack: {attack_type.upper()}, ε={epsilon}"
                  + (f", steps={pgd_steps}" if attack_type == 'pgd' else ""))

    def fgsm(self, field_batch, ref_batch):
        adv_results = []
        for i in range(field_batch.shape[0]):
            field = field_batch[i:i+1].clone().detach().requires_grad_(True)
            ref = ref_batch[i:i+1].clone().detach()

            loss = simsac_change_loss(self.simsac, field, ref)
            loss.backward()

            with torch.no_grad():
                adv = field + self.epsilon * field.grad.sign()
                adv = torch.clamp(adv, 0, 1)

            adv_results.append(adv.detach())

        return torch.cat(adv_results, dim=0)

    def pgd(self, field_batch, ref_batch):
        adv_results = []
        for i in range(field_batch.shape[0]):
            original = field_batch[i:i+1].clone().detach()
            ref = ref_batch[i:i+1].clone().detach()
            adv = original.clone().detach()

            for _ in range(self.pgd_steps):
                adv.requires_grad_(True)
                loss = simsac_change_loss(self.simsac, adv, ref)
                loss.backward()

                with torch.no_grad():
                    adv = adv + self.pgd_step_size * adv.grad.sign()
                    perturbation = torch.clamp(adv - original, -self.epsilon, self.epsilon)
                    adv = torch.clamp(original + perturbation, 0, 1)

                adv = adv.detach()

            adv_results.append(adv)

        return torch.cat(adv_results, dim=0)

    def generate(self, field_batch, ref_batch):
        if self.attack_type == 'none':
            return None

        was_training = self.simsac.training
        self.simsac.train()

        saved_grad_states = {n: p.requires_grad
                             for n, p in self.simsac.named_parameters()}
        for p in self.simsac.parameters():
            p.requires_grad_(False)

        try:
            if self.attack_type == 'fgsm':
                adv = self.fgsm(field_batch, ref_batch)
            elif self.attack_type == 'pgd':
                adv = self.pgd(field_batch, ref_batch)
            else:
                adv = None
        finally:
            # Restore backbone grad state and training mode
            for name, p in self.simsac.named_parameters():
                p.requires_grad_(saved_grad_states[name])
            if not was_training:
                self.simsac.eval()

        return adv


# Trainer with online adversarial generation + change map loss

class OnlineAdvTrainer:

    def __init__(self, model, attack_generator, train_loader, val_loader,
                 config, device='cuda'):
        self.model = model
        self.attacker = attack_generator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.criterion = CombinedLoss(
            lambda_contrastive=config['lambda_contrastive'],
            lambda_flow=0.0,
            lambda_change=0.0,
            temperature=config['temperature'],
            use_simplified=True,
            use_weighted=config.get('use_weighted', False),
            adversarial_weight=config.get('adversarial_weight', 3.0)
        )

        self.optimizer = optim.Adam(
            model.get_trainable_parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            threshold=0.01,
            threshold_mode='rel'
        )

        self.history = {
            'train_loss': [], 'train_loss_clean': [], 'train_loss_adv': [],
            'train_loss_cm': [], 'val_loss': [], 'learning_rate': []
        }
        self.best_val_loss = float('inf')
        self.start_epoch = 0

    def _forward_loss(self, img1, img2, labels, is_adversarial=None):
        z1, z2 = self.model(img1, img2)
        loss, loss_dict = self.criterion(z1, z2, labels=labels,
                                         is_adversarial=is_adversarial)
        return loss, loss_dict

    def train_epoch(self, epoch):
        self.model.train()
        # Also keep SimSAC backbone in train mode for gradient flow during attack
        if self.attacker and self.attacker.attack_type != 'none':
            self.attacker.simsac.train()

        total_loss = 0.0
        total_clean = 0.0
        total_adv = 0.0
        total_cm = 0.0

        adv_weight = self.config.get('adversarial_weight', 3.0)
        lambda_cm = self.config.get('lambda_cm', 0.0)
        use_online_attack = (self.attacker is not None
                             and self.attacker.attack_type != 'none')
        use_cm_loss = (use_online_attack and lambda_cm > 0.0
                       and not self.config.get('freeze_backbone', False))

        pbar = tqdm(self.train_loader,
                    desc=f"Epoch {epoch+1}/{self.config['epochs']} [Train]")

        for batch_idx, batch_data in enumerate(pbar):
            # Unpack
            if len(batch_data) == 4:
                img1, img2, labels, is_adversarial = batch_data
            else:
                img1, img2, labels = batch_data
                is_adversarial = None

            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.to(self.device)
            if is_adversarial is not None:
                is_adversarial = is_adversarial.to(self.device)

            self.optimizer.zero_grad()

            # --- Clean contrastive loss ---
            loss_clean, _ = self._forward_loss(img1, img2, labels, is_adversarial)
            total_loss_batch = loss_clean

            # --- Online adversarial loss ---
            img2_adv = None
            if use_online_attack:
                # Generate adversarial field images (detached — separate graph)
                img2_adv = self.attacker.generate(img2.detach(), img1.detach())

                if img2_adv is not None:
                    img2_adv = img2_adv.to(self.device)
                    # All adversarial samples are treated as adversarial
                    is_adv_flag = torch.ones(img1.shape[0],
                                             dtype=torch.bool,
                                             device=self.device)
                    loss_adv, _ = self._forward_loss(img1, img2_adv, labels,
                                                     is_adversarial=is_adv_flag)
                    total_loss_batch = total_loss_batch + adv_weight * loss_adv
                    total_adv += loss_adv.item()

            if use_cm_loss and img2_adv is not None:
                loss_cm = compute_change_map_loss(
                    self.attacker.simsac,
                    img2_adv.detach(),
                    img1.detach(),
                    labels.float()
                )
                total_loss_batch = total_loss_batch + lambda_cm * loss_cm
                total_cm += loss_cm.item()

            total_clean += loss_clean.item()

            # Backward
            total_loss_batch.backward()

            if self.config.get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config['gradient_clip']
                )

            self.optimizer.step()
            total_loss += total_loss_batch.item()

            pbar.set_postfix({
                'loss': f"{total_loss_batch.item():.4f}",
                'avg': f"{total_loss / (batch_idx + 1):.4f}"
            })

        n = len(self.train_loader)
        return total_loss / n, total_clean / n, total_adv / n, total_cm / n

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0

        pbar = tqdm(self.val_loader,
                    desc=f"Epoch {epoch+1}/{self.config['epochs']} [Val]  ")

        with torch.no_grad():
            for batch_data in pbar:
                if len(batch_data) == 4:
                    img1, img2, labels, is_adversarial = batch_data
                else:
                    img1, img2, labels = batch_data
                    is_adversarial = None

                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.to(self.device)
                if is_adversarial is not None:
                    is_adversarial = is_adversarial.to(self.device)

                loss, _ = self._forward_loss(img1, img2, labels, is_adversarial)
                total_loss += loss.item()

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        return total_loss / len(self.val_loader)

    def train(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        attack_name = self.attacker.attack_type if self.attacker else 'none'
        lambda_cm = self.config.get('lambda_cm', 0.0)
        freeze_backbone = self.config.get('freeze_backbone', False)
        cm_active = (lambda_cm > 0.0 and not freeze_backbone
                     and attack_name != 'none')

        print(f"Online Adversarial Training - Phase {self.config['phase']}")
        print(f"Epochs:          {self.config['epochs']}")
        print(f"Learning rate:   {self.config['learning_rate']}")
        print(f"Batch size:      {self.config['batch_size']}")
        print(f"Freeze backbone: {freeze_backbone}")
        print(f"Online attack:   {attack_name.upper()}")
        print(f"Epsilon:         {self.config.get('epsilon', 'N/A')}")
        print(f"Adv weight:      {self.config.get('adversarial_weight', 3.0)}")
        print(f"Lambda CM:       {lambda_cm}"
              + (" (ACTIVE)" if cm_active else " (inactive — backbone frozen or no attack)"))
        print(f"Output dir:      {output_dir}")

        for epoch in range(self.start_epoch, self.config['epochs']):
            train_loss, clean_loss, adv_loss, cm_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['train_loss_clean'].append(clean_loss)
            self.history['train_loss_adv'].append(adv_loss)
            self.history['train_loss_cm'].append(cm_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)

            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print(f"  Train Loss (total):     {train_loss:.4f}")
            print(f"  Train Loss (clean):     {clean_loss:.4f}")
            print(f"  Train Loss (adv):       {adv_loss:.4f}")
            if cm_active:
                print(f"  Train Loss (change map):{cm_loss:.4f}")
            print(f"  Val Loss:               {val_loss:.4f}")
            print(f"  LR:                     {current_lr:.6f}")

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            if (epoch + 1) % self.config.get('save_every', 5) == 0 or is_best:
                self._save_checkpoint(epoch, output_dir, is_best=is_best)

            if (epoch + 1) % 5 == 0:
                self._plot_progress(output_dir)

        # Final checkpoint
        self._save_checkpoint(
            self.config['epochs'] - 1,
            output_dir,
            filename=f"phase{self.config['phase']}_final.pth"
        )
        self._plot_progress(output_dir)

        print(f"Training Complete! Best val loss: {self.best_val_loss:.4f}")

    def _save_checkpoint(self, epoch, output_dir, is_best=False, filename=None):
        if filename is None:
            filename = f"checkpoint_epoch_{epoch+1}.pth"

        full_state = self.model.state_dict()
        simsac_state = {
            k[7:]: v for k, v in full_state.items()
            if k.startswith('simsac.')
        }

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': simsac_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }

        path = output_dir / filename
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")

        if is_best:
            best_path = output_dir / f"phase{self.config['phase']}_best.pth"
            torch.save(checkpoint, best_path)
            print(f"   New best model: {best_path}")

    def load_checkpoint(self, checkpoint_path):
        print(f"\nLoading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        wrapped = {f'simsac.{k}': v for k, v in ckpt['state_dict'].items()}
        self.model.load_state_dict(wrapped, strict=False)
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.best_val_loss = ckpt['best_val_loss']
        self.history = ckpt.get('history', self.history)
        self.start_epoch = ckpt['epoch']

        print(f" Loaded. Resuming from epoch {self.start_epoch}, "
              f"best val loss: {self.best_val_loss:.4f}")

    def _plot_progress(self, output_dir):
        epochs = range(1, len(self.history['train_loss']) + 1)
        has_cm = any(v > 0 for v in self.history['train_loss_cm'])
        fig, axes = plt.subplots(1, 4, figsize=(24, 4))

        # Total loss
        axes[0].plot(epochs, self.history['train_loss'], label='Train', marker='o')
        axes[0].plot(epochs, self.history['val_loss'], label='Val', marker='s')
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Clean vs Adv contrastive loss
        axes[1].plot(epochs, self.history['train_loss_clean'],
                     label='Clean (contrastive)', marker='o', color='green')
        axes[1].plot(epochs, self.history['train_loss_adv'],
                     label='Adversarial (contrastive)', marker='s', color='red')
        axes[1].set_title('Contrastive: Clean vs Adversarial')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Change map loss
        axes[2].plot(epochs, self.history['train_loss_cm'],
                     label='Change Map Loss', marker='^', color='darkorange')
        axes[2].set_title('Change Map Loss\n(direct detection supervision)')
        axes[2].set_xlabel('Epoch')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        if not has_cm:
            axes[2].text(0.5, 0.5, 'Inactive\n(Phase 1 / no attack)',
                         ha='center', va='center', transform=axes[2].transAxes,
                         fontsize=10, color='gray')

        # LR
        axes[3].plot(epochs, self.history['learning_rate'],
                     marker='o', color='purple')
        axes[3].set_title('Learning Rate')
        axes[3].set_xlabel('Epoch')
        axes[3].set_yscale('log')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'training_progress.png', dpi=150)
        plt.close()
        print(f"  Progress plot saved.")


# Main

def main():
    parser = argparse.ArgumentParser(
        description="SimSAC Online Adversarial Training with Change Map Loss",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory with train/val pairs .pkl files')
    parser.add_argument('--train_pairs', type=str, default=None,
                        help='Train pairs pkl (auto-detected if not set)')
    parser.add_argument('--val_pairs', type=str, default=None,
                        help='Val pairs pkl (auto-detected if not set)')
    parser.add_argument('--weights_path', type=str,
                        default='/content/tampar/src/simsac/weight/synthetic.pth',
                        help='Pre-trained SimSAC weights path (default: synthetic.pth)')

    # Training phase
    parser.add_argument('--phase', type=int, choices=[1, 2], required=True,
                        help='1=frozen backbone, 2=full fine-tune')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to resume from (Phase 2 loads Phase 1 automatically)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints')

    # Training hyperparams
    parser.add_argument('--epochs', type=int, default=None,
                        help='Epochs (default: 10 phase1, 20 phase2)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=None,
                        help='LR (default: 1e-3 phase1, 1e-4 phase2)')
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--use_weighted_loss', action='store_true',
                        help='Apply adversarial_weight to pre-labeled adversarial pairs in dataloader')
    parser.add_argument('--adversarial_weight', type=float, default=3.0,
                        help='Contrastive loss weight multiplier for adversarial pairs (default: 3.0)')

    # Online attack settings
    parser.add_argument('--attack', type=str, default='pgd',
                        choices=['fgsm', 'pgd', 'none'],
                        help='Online attack type (default: pgd)')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Perturbation magnitude (default: 0.05)')
    parser.add_argument('--pgd_steps', type=int, default=5,
                        help='PGD steps per batch (default: 5, keep small for speed)')
    parser.add_argument('--pgd_step_size', type=float, default=None,
                        help='PGD step size (default: epsilon/4)')

    # Change map loss
    parser.add_argument('--lambda_cm', type=float, default=0.0,
                        help=(
                            'Weight for change map loss (default: 0.0). '
                            'Set > 0 in Phase 2 to directly supervise the change map. '
                            'Recommended: 1.0 for Phase 2. Has no effect in Phase 1 '
                            '(backbone frozen → zero gradient from this loss).'
                        ))

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])

    args = parser.parse_args()

    # Phase defaults
    if args.epochs is None:
        args.epochs = 10 if args.phase == 1 else 20
    if args.lr is None:
        args.lr = 1e-3 if args.phase == 1 else 1e-4

    # Warn if lambda_cm > 0 in Phase 1
    if args.lambda_cm > 0.0 and args.phase == 1:
        print(f"\nWARNING: --lambda_cm {args.lambda_cm} has no effect in Phase 1 "
              f"(backbone is frozen, zero gradient). Set --lambda_cm 0.0 for Phase 1 "
              f"to avoid the extra forward pass overhead.")

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        device = 'cpu'

    config = {
        'phase': args.phase,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'temperature': args.temperature,
        'freeze_backbone': (args.phase == 1),
        'lambda_contrastive': 1.0,
        'lambda_cm': args.lambda_cm,
        'weight_decay': 1e-4,
        'gradient_clip': 1.0,
        'save_every': 5,
        'use_weighted': args.use_weighted_loss,
        'adversarial_weight': args.adversarial_weight,
        'attack': args.attack,
        'epsilon': args.epsilon,
        'pgd_steps': args.pgd_steps,
    }

    # Auto-detect pair files
    data_dir = Path(args.data_dir)

    def find_pairs(name_override, candidates):
        if name_override:
            return Path(name_override)
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(f"No pair file found. Tried: {[str(c) for c in candidates]}")

    train_path = find_pairs(args.train_pairs, [
        data_dir / 'train_pairs_surface_level.pkl',
        data_dir / 'train_pairs.pkl'
    ])
    val_path = find_pairs(args.val_pairs, [
        data_dir / 'val_pairs_surface_level.pkl',
        data_dir / 'val_pairs.pkl'
    ])

    print(f"\nTrain pairs: {train_path}")
    print(f"Val pairs:   {val_path}")

    # Dataloaders
    train_loader, val_loader, _, _ = create_dataloaders(
        str(train_path), str(val_path),
        batch_size=args.batch_size,
        num_workers=2
    )

    # Contrastive model (with projection head)
    model = create_simsac_contrastive(
        weights_path=args.weights_path,
        projection_dim=128,
        freeze_backbone=config['freeze_backbone'],
        device=device
    )

    # Online attack generator — uses the raw SimSAC backbone from the model
    if args.attack != 'none':
        attacker = OnlineAttackGenerator(
            simsac_model=model.simsac,
            attack_type=args.attack,
            epsilon=args.epsilon,
            pgd_steps=args.pgd_steps,
            pgd_step_size=args.pgd_step_size,
            device=device
        )
    else:
        attacker = None
        print("  No online attack (running as standard contrastive training)")

    # Phase 2: auto-load phase 1 checkpoint
    if args.phase == 2 and args.checkpoint is None:
        phase1_ckpt = Path(args.output_dir) / 'phase1_final.pth'
        if phase1_ckpt.exists():
            args.checkpoint = str(phase1_ckpt)
            print(f"\nPhase 2: auto-loading phase 1 checkpoint: {args.checkpoint}")

    # Trainer
    trainer = OnlineAdvTrainer(
        model=model,
        attack_generator=attacker,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

        if args.phase == 2:
            model.unfreeze_all()
            trainer.optimizer = optim.Adam(
                model.get_trainable_parameters(),
                lr=config['learning_rate'],
                weight_decay=config.get('weight_decay', 1e-4)
            )
            trainer.scheduler = ReduceLROnPlateau(
                trainer.optimizer, mode='min', factor=0.5,
                patience=3, min_lr=1e-6, threshold=0.01,
                threshold_mode='rel'
            )
            # Reset history so Phase 2 plots start from epoch 1
            trainer.history = {
                'train_loss': [], 'train_loss_clean': [], 'train_loss_adv': [],
                'train_loss_cm': [], 'val_loss': [], 'learning_rate': []
            }
            trainer.start_epoch = 0
            print(" Phase 2: backbone unfrozen, scheduler and history reset")

    trainer.train(output_dir=args.output_dir)


if __name__ == '__main__':
    main()
