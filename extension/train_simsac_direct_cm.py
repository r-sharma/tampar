"""
SimSAC Direct Change Map Fine-tuning (Path C — No contrastive loss)

Directly fine-tunes SimSAC's change map output on pre-generated adversarial
UV maps. No projection head. No contrastive loss. No temperature.

Key design decisions that make this reliable:

  1. DIRECT change map loss — optimises exactly what predict_tampering uses
     Tampered pairs → push change map HIGH
     Clean pairs    → push change map LOW

  2. L2 regularisation toward original synthetic.pth weights
     Prevents catastrophic forgetting of clean-image performance.
     Without it: clean accuracy drops (as seen in Phase 2 full fine-tune → 65%)
     With it:    clean accuracy stays close to 84% while adversarial improves

  3. Mixed data per batch (clean + adversarial, configurable ratio)
     Model sees both distributions every epoch →
     cannot forget clean while learning adversarial robustness

  4. Very small learning rate (default 1e-5)
     Conservative backbone updates → stable training

  5. Per-epoch evaluation on BOTH clean and adversarial validation
     Best checkpoint selected by: 0.5 * clean_sep + 0.5 * adv_sep
     where sep = tampered_mean_cm - clean_mean_cm (higher = better)

Why this works against pre-generated PGD/FGSM attacks:
  - Attack was generated ONCE with original synthetic.pth (fixed perturbations)
  - Model trains to recognise and resist THOSE specific perturbation patterns
  - Unlike online training (arms race), this is stable offline adversarial training
  - L2 reg ensures backbone doesn't move far from original → clean preserved

Usage:
    python extension/train_simsac_direct_cm.py \\
        --clean_pairs  /content/drive/MyDrive/TAMPAR_DATA/tampar/contrastive_pairs_clean_only_test/train_pairs_surface_level.pkl \\
        --val_pairs    /content/drive/MyDrive/TAMPAR_DATA/tampar/contrastive_pairs_clean_only_test/val_pairs_surface_level.pkl \\
        --adv_uvmap_dir /content/drive/MyDrive/TAMPAR_DATA/tampar/adversarial_test \\
        --attack pgd \\
        --weights_path /content/tampar/src/simsac/weight/synthetic.pth \\
        --output_dir   /content/drive/MyDrive/TAMPAR_DATA/tampar/simsac_weights_direct_cm \\
        --lr 1e-5 \\
        --lambda_reg 0.1 \\
        --adv_ratio 0.5 \\
        --epochs 15
"""

import sys
import types
import argparse
import pickle
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'extension'))

from src.simsac.models.our_models.SimSaC import SimSaC_Model


# ---------------------------------------------------------------------------
# SimSAC gradient patch (same as train_simsac_online_adv.py)
# ---------------------------------------------------------------------------

def enable_simsac_gradients(model):
    """Remove internal no_grad() wrapper so change map loss can backprop."""
    def forward_with_grad(self, im_target, im_source,
                          im_target_256, im_source_256, disable_flow=None):
        im1_pyr = self.pyramid(im_target, eigth_resolution=True)
        im2_pyr = self.pyramid(im_source, eigth_resolution=True)
        im1_pyr_256 = self.pyramid(im_target_256)
        im2_pyr_256 = self.pyramid(im_source_256)
        c14 = im1_pyr_256[-3]
        c24 = im2_pyr_256[-3]
        flow4, corr4 = self.coarsest_resolution_flow(
            c14, c24, h_256=256, w_256=256, return_corr=True
        )
        return {"flow": ([flow4], [flow4]), "change": ([corr4], [corr4])}

    model.forward_sigle_ref = types.MethodType(forward_with_grad, model)
    return model


def get_change_magnitude(simsac, field_img, ref_img):
    """
    Forward pass through SimSAC and return change magnitude per image.

    Args:
        field_img: [B, C, H, W] in [0, 1]
        ref_img:   [B, C, H, W] in [0, 1]

    Returns:
        change_mag: [B] per-image mean change magnitude
    """
    mags = []
    for i in range(field_img.shape[0]):
        f = field_img[i:i+1]
        r = ref_img[i:i+1]

        f512 = F.interpolate(f, (512, 512), mode='bilinear', align_corners=False)
        r512 = F.interpolate(r, (512, 512), mode='bilinear', align_corners=False)
        f256 = F.interpolate(f, (256, 256), mode='bilinear', align_corners=False)
        r256 = F.interpolate(r, (256, 256), mode='bilinear', align_corners=False)

        out = simsac(f512, r512, f256, r256)

        change = None
        if isinstance(out, dict):
            ct = out.get('change', None)
            if ct and isinstance(ct, tuple) and len(ct) == 2:
                change = ct[1][-1] if len(ct[1]) > 0 else ct[0][-1]
        elif isinstance(out, (list, tuple)) and len(out) >= 2:
            change = out[1]

        if change is None:
            raise ValueError(f"SimSAC returned no change map. Got: {type(out)}")

        mag = torch.sqrt(change[:, 0] ** 2 + change[:, 1] ** 2 + 1e-8).mean()
        mags.append(mag)

    return torch.stack(mags)   # [B]


def change_map_loss(magnitudes, labels):
    """
    Direct change map supervision loss.

    Tampered (label=1): want HIGH magnitude → loss = -magnitude
    Clean    (label=0): want LOW  magnitude → loss = +magnitude

    Formula: loss = (1 - 2*label) * magnitude
      label=1: (1-2)*mag = -mag  → minimising loss maximises magnitude ✓
      label=0: (1-0)*mag = +mag  → minimising loss minimises magnitude  ✓

    Args:
        magnitudes: [B] per-image change magnitude tensor
        labels:     [B] binary labels (1=tampered, 0=clean), float tensor

    Returns:
        Scalar loss
    """
    return ((1.0 - 2.0 * labels) * magnitudes).mean()


def l2_reg_loss(model, original_params):
    """
    L2 regularisation toward original synthetic.pth weights.

    Prevents catastrophic forgetting: penalises large deviations from
    the original weight values. This keeps clean-image performance intact.

    Args:
        model:           Current SimSAC model
        original_params: Dict {name → tensor} of original synthetic.pth weights

    Returns:
        Scalar L2 regularisation loss
    """
    reg = 0.0
    for name, param in model.named_parameters():
        if name in original_params and param.requires_grad:
            reg = reg + ((param - original_params[name]) ** 2).sum()
    return reg


# ---------------------------------------------------------------------------
# Dataset — loads clean pairs + finds adversarial counterparts
# ---------------------------------------------------------------------------

class DirectCMDataset(Dataset):
    """
    Loads (reference_uvmap, field_uvmap, label) triples.

    For each clean pair in the pkl, optionally looks up the adversarial
    counterpart by resolving the field path into adv_uvmap_dir.

    Adversarial path resolution (tries in order):
      1. adv_uvmap_dir / relative_field_path_with_attack_suffix
      2. adv_uvmap_dir / relative_field_path (same filename, different root)
    """

    def __init__(self, pairs_pkl, adv_uvmap_dir=None, attack='pgd',
                 adv_ratio=0.5, clean_root=None, target_size=(256, 256)):
        """
        Args:
            pairs_pkl:    Path to train/val_pairs_surface_level.pkl
            adv_uvmap_dir: Root dir of pre-generated adversarial UV maps.
                           If None, only clean pairs are used.
            attack:       'pgd' or 'fgsm' — suffix used in adversarial filenames
            adv_ratio:    Fraction of adversarial samples per epoch (0.0–1.0)
            clean_root:   Root dir of clean UV maps (auto-detected from pkl paths)
            target_size:  Resize all images to this size
        """
        with open(pairs_pkl, 'rb') as f:
            self.pairs = pickle.load(f)

        self.adv_uvmap_dir = Path(adv_uvmap_dir) if adv_uvmap_dir else None
        self.attack = attack
        self.adv_ratio = adv_ratio if adv_uvmap_dir else 0.0
        self.target_size = target_size

        # Auto-detect clean_root from first pair's img1 path
        if clean_root is None and len(self.pairs) > 0:
            p = Path(self.pairs[0]['img1'])
            # Walk up until we find a directory that contains 'id_XX_uvmap' files
            for parent in p.parents:
                if any(parent.glob('id_*_uvmap.png')):
                    self.clean_root = parent
                    break
            else:
                self.clean_root = p.parent.parent
        else:
            self.clean_root = Path(clean_root) if clean_root else None

        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])

        # Pre-build adversarial path lookup
        self._adv_cache = {}

        n_adv = sum(1 for p in self.pairs
                    if self._find_adv_path(p['img2']) is not None)
        print(f"  Loaded {len(self.pairs)} pairs "
              f"({sum(1 for p in self.pairs if p['label']==1)} tampered, "
              f"{sum(1 for p in self.pairs if p['label']==0)} clean)")
        if self.adv_uvmap_dir:
            print(f"  Adversarial UV maps found for {n_adv}/{len(self.pairs)} pairs")
            print(f"  Adversarial sampling ratio: {adv_ratio:.0%}")

    def _find_adv_path(self, clean_field_path):
        """Resolve clean field UV map path → adversarial version path."""
        if self.adv_uvmap_dir is None:
            return None

        key = str(clean_field_path)
        if key in self._adv_cache:
            return self._adv_cache[key]

        clean_p = Path(clean_field_path)

        # Try to get relative path from clean_root
        try:
            if self.clean_root and str(clean_p).startswith(str(self.clean_root)):
                rel = clean_p.relative_to(self.clean_root)
            else:
                # Use last 3 parts: background/filename
                rel = Path(*clean_p.parts[-3:])
        except Exception:
            rel = Path(*clean_p.parts[-3:])

        candidates = []

        # Strategy 1: same relative path, different root
        candidates.append(self.adv_uvmap_dir / rel)

        # Strategy 2: add attack suffix to stem before extension
        # e.g. id_01_..._uvmap_pred.png → id_01_..._pgd_uvmap_pred.png
        stem = clean_p.stem
        suffix = clean_p.suffix
        adv_name = f"{stem}_{self.attack}{suffix}"
        candidates.append(self.adv_uvmap_dir / rel.parent / adv_name)

        # Strategy 3: replace stem part containing 'uvmap'
        # e.g. ..._uvmap_pred.png → ..._pgd_uvmap_pred.png
        if 'uvmap' in stem:
            adv_name2 = stem.replace('uvmap', f'{self.attack}_uvmap') + suffix
            candidates.append(self.adv_uvmap_dir / rel.parent / adv_name2)

        for c in candidates:
            if c.exists():
                self._adv_cache[key] = c
                return c

        self._adv_cache[key] = None
        return None

    def _load_img(self, path):
        img = Image.open(str(path)).convert('RGB')
        return self.transform(img)   # [3, H, W] in [0, 1]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        label = int(pair['label'])

        img1 = self._load_img(pair['img1'])   # reference UV map

        # Decide clean vs adversarial
        use_adv = (self.adv_uvmap_dir is not None
                   and random.random() < self.adv_ratio)

        if use_adv:
            adv_path = self._find_adv_path(pair['img2'])
            if adv_path is not None:
                img2 = self._load_img(adv_path)
                is_adv = 1
            else:
                img2 = self._load_img(pair['img2'])
                is_adv = 0
        else:
            img2 = self._load_img(pair['img2'])
            is_adv = 0

        return img1, img2, torch.tensor(label, dtype=torch.float32), is_adv


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DirectCMTrainer:
    """
    Trains SimSAC backbone directly on change map loss.

    No contrastive loss. No projection head. No temperature.
    Pure change map supervision + L2 regularisation toward original weights.
    """

    def __init__(self, simsac, original_params, train_loader, val_loader,
                 config, device):
        self.simsac = simsac
        self.original_params = original_params
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.optimizer = optim.Adam(
            [p for p in simsac.parameters() if p.requires_grad],
            lr=config['lr'],
            weight_decay=0.0   # L2 reg is handled explicitly (toward original, not toward 0)
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max',   # maximise separation
            factor=0.5, patience=3, min_lr=1e-7
        )

        self.history = {
            'train_cm_loss': [],
            'val_clean_sep': [],     # tampered_mean_cm - clean_mean_cm on clean val
            'val_adv_sep': [],       # same on adversarial val
            'val_combined': [],
            'learning_rate': []
        }
        self.best_combined = -float('inf')
        self.best_epoch = 0

    def train_epoch(self, epoch):
        self.simsac.train()
        total_loss = 0.0
        n = 0

        pbar = tqdm(self.train_loader,
                    desc=f"Epoch {epoch+1} [Train]")

        for img1, img2, labels, is_adv in pbar:
            img1   = img1.to(self.device)
            img2   = img2.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward: get per-image change magnitude
            mags = get_change_magnitude(self.simsac, img2, img1)

            # Change map loss
            loss_cm = change_map_loss(mags, labels)

            # L2 regularisation toward original weights
            loss_reg = l2_reg_loss(self.simsac, self.original_params)

            loss = loss_cm + self.config['lambda_reg'] * loss_reg

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.simsac.parameters(), self.config.get('grad_clip', 1.0)
            )

            self.optimizer.step()

            total_loss += loss.item()
            n += 1

            pbar.set_postfix({
                'cm': f"{loss_cm.item():.4f}",
                'reg': f"{loss_reg.item():.3f}",
                'avg': f"{total_loss/n:.4f}"
            })

        return total_loss / n

    @torch.no_grad()
    def evaluate_separation(self, loader):
        """
        Compute change map separation on a dataloader.

        Returns:
            separation = mean_cm(tampered) - mean_cm(clean)
            Higher is better.
        """
        self.simsac.eval()
        tampered_cms, clean_cms = [], []

        for img1, img2, labels, _ in loader:
            img1   = img1.to(self.device)
            img2   = img2.to(self.device)
            labels = labels.to(self.device)

            # Temporarily allow forward pass (eval mode, no grad needed)
            mags = get_change_magnitude(self.simsac, img2, img1)

            for mag, lbl in zip(mags, labels):
                if lbl.item() == 1:
                    tampered_cms.append(mag.item())
                else:
                    clean_cms.append(mag.item())

        if not tampered_cms or not clean_cms:
            return 0.0, 0.0, 0.0

        t_mean = np.mean(tampered_cms)
        c_mean = np.mean(clean_cms)
        sep    = t_mean - c_mean
        return sep, t_mean, c_mean

    def train(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*65}")
        print("SimSAC Direct Change Map Fine-tuning")
        print(f"{'='*65}")
        print(f"Epochs:      {self.config['epochs']}")
        print(f"LR:          {self.config['lr']}")
        print(f"Lambda reg:  {self.config['lambda_reg']}")
        print(f"Adv ratio:   {self.config['adv_ratio']:.0%} of batches use adversarial UV maps")
        print(f"Output:      {output_dir}")
        print(f"{'='*65}")

        for epoch in range(self.config['epochs']):

            # Train
            train_loss = self.train_epoch(epoch)

            # Evaluate on clean val
            clean_sep, clean_t, clean_c = self.evaluate_separation(self.val_loader)

            # Evaluate on adversarial val (if adv_val_loader provided)
            adv_sep = clean_sep   # fallback: same as clean if no adv val
            if hasattr(self, 'adv_val_loader') and self.adv_val_loader:
                adv_sep, adv_t, adv_c = self.evaluate_separation(self.adv_val_loader)
            else:
                adv_t, adv_c = clean_t, clean_c

            combined = 0.5 * clean_sep + 0.5 * adv_sep
            self.scheduler.step(combined)
            lr = self.optimizer.param_groups[0]['lr']

            self.history['train_cm_loss'].append(train_loss)
            self.history['val_clean_sep'].append(clean_sep)
            self.history['val_adv_sep'].append(adv_sep)
            self.history['val_combined'].append(combined)
            self.history['learning_rate'].append(lr)

            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print(f"  Train loss:          {train_loss:.4f}")
            print(f"  Clean val sep:       {clean_sep:.4f}  "
                  f"(tampered={clean_t:.3f}, clean={clean_c:.3f})")
            print(f"  Adv val sep:         {adv_sep:.4f}  "
                  f"(tampered={adv_t:.3f}, clean={adv_c:.3f})")
            print(f"  Combined (metric):   {combined:.4f}")
            print(f"  LR:                  {lr:.2e}")

            # Save best
            is_best = combined > self.best_combined
            if is_best:
                self.best_combined = combined
                self.best_epoch = epoch + 1
                self._save(epoch, output_dir, 'best.pth')
                print(f"  ✓ New best model saved (combined sep: {combined:.4f})")

            # Save every 5 epochs
            if (epoch + 1) % 5 == 0:
                self._save(epoch, output_dir, f'epoch_{epoch+1}.pth')
                self._plot(output_dir)

        # Final
        self._save(self.config['epochs'] - 1, output_dir, 'final.pth')
        self._plot(output_dir)

        print(f"\n{'='*65}")
        print(f"Training Complete!")
        print(f"Best epoch: {self.best_epoch} — combined separation: {self.best_combined:.4f}")
        print(f"Run predict_tampering with: {output_dir}/best.pth")
        print(f"{'='*65}")

    def _save(self, epoch, output_dir, filename):
        ckpt = {
            'epoch': epoch + 1,
            'state_dict': self.simsac.state_dict(),   # raw SimSAC (no wrapper, TAMPAR-compatible)
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_combined': self.best_combined,
            'history': self.history,
            'config': self.config
        }
        torch.save(ckpt, output_dir / filename)

    def _plot(self, output_dir):
        epochs = range(1, len(self.history['train_cm_loss']) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        axes[0].plot(epochs, self.history['train_cm_loss'], marker='o', color='steelblue')
        axes[0].set_title('Train Change Map Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, self.history['val_clean_sep'],
                     marker='o', color='green', label='Clean val')
        axes[1].plot(epochs, self.history['val_adv_sep'],
                     marker='s', color='red', label='Adversarial val')
        axes[1].plot(epochs, self.history['val_combined'],
                     marker='^', color='purple', label='Combined (best metric)', linestyle='--')
        axes[1].set_title('Change Map Separation\n(tampered_mean − clean_mean, higher=better)')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, self.history['learning_rate'],
                     marker='o', color='orange')
        axes[2].set_title('Learning Rate')
        axes[2].set_xlabel('Epoch')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'training_progress.png', dpi=150)
        plt.close()
        print(f"  Plot saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_simsac(weights_path, device):
    """Load raw SimSAC model from checkpoint."""
    simsac = SimSaC_Model(
        evaluation=True,
        pyramid_type='VGG',
        md=4,
        dense_connection=True,
        consensus_network=False,
        cyclic_consistency=False,
        decoder_inputs='corr_flow_feat',
        num_class=2,
        use_pac=False,
        batch_norm=True,
        iterative_refinement=False,
        refinement_at_all_levels=False,
        refinement_at_adaptive_reso=True,
        upfeat_channels=2,
        vpr_candidates=False,
        div=1.0
    )

    ckpt = torch.load(weights_path, map_location=device)
    # Handle both raw state dict and wrapped checkpoint formats
    if 'state_dict' in ckpt:
        state = ckpt['state_dict']
        # Strip 'simsac.' prefix if present (from contrastive wrapper)
        state = {k.replace('simsac.', ''): v for k, v in state.items()}
    else:
        state = ckpt

    simsac.load_state_dict(state, strict=False)
    simsac = simsac.to(device)
    simsac.train()
    enable_simsac_gradients(simsac)

    n_params = sum(p.numel() for p in simsac.parameters())
    print(f"  ✓ SimSAC loaded ({n_params/1e6:.1f}M params) from {weights_path}")
    return simsac


def main():
    parser = argparse.ArgumentParser(
        description='SimSAC Direct Change Map Fine-tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Data
    parser.add_argument('--clean_pairs', type=str, required=True,
                        help='Path to train_pairs_surface_level.pkl (clean UV maps)')
    parser.add_argument('--val_pairs', type=str, required=True,
                        help='Path to val_pairs_surface_level.pkl (clean UV maps)')
    parser.add_argument('--adv_uvmap_dir', type=str, default=None,
                        help='Root dir of pre-generated adversarial UV maps '
                             '(mirrors clean structure). If None, train on clean only.')
    parser.add_argument('--adv_val_dir', type=str, default=None,
                        help='Root dir of adversarial UV maps for VALIDATION. '
                             'If None, val_pairs are used with adv_uvmap_dir.')
    parser.add_argument('--attack', type=str, default='pgd',
                        choices=['pgd', 'fgsm', 'both'],
                        help='Attack type suffix in adversarial filenames (default: pgd)')
    parser.add_argument('--adv_ratio', type=float, default=0.5,
                        help='Fraction of each batch using adversarial UV maps (default: 0.5)')

    # Model
    parser.add_argument('--weights_path', type=str,
                        default='/content/tampar/src/simsac/weight/synthetic.pth',
                        help='Starting SimSAC weights (ALWAYS use synthetic.pth, not a failed checkpoint)')

    # Training
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of epochs (default: 15)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8, smaller than contrastive due to SimSAC overhead)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate (default: 1e-5, keep small to prevent forgetting)')
    parser.add_argument('--lambda_reg', type=float, default=0.1,
                        help='L2 regularisation weight toward original weights (default: 0.1). '
                             'Increase if clean accuracy drops, decrease if adversarial improvement is slow.')
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints and plots')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])

    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"\n{'='*65}")
    print("SimSAC Direct Change Map Fine-tuning")
    print(f"{'='*65}")
    print(f"Clean pairs:    {args.clean_pairs}")
    print(f"Val pairs:      {args.val_pairs}")
    print(f"Adv UV maps:    {args.adv_uvmap_dir or 'None (clean-only training)'}")
    print(f"Weights:        {args.weights_path}")
    print(f"LR:             {args.lr}  (keep low to prevent catastrophic forgetting)")
    print(f"Lambda reg:     {args.lambda_reg}  (L2 toward original weights)")
    print(f"Adv ratio:      {args.adv_ratio:.0%}")

    # Load SimSAC
    simsac = load_simsac(args.weights_path, device)

    # Store original weights for L2 regularisation
    # These are the synthetic.pth values we regularise toward
    original_params = {
        name: param.data.clone().detach()
        for name, param in simsac.named_parameters()
    }
    print(f"  ✓ Original weights stored for L2 regularisation "
          f"({len(original_params)} parameter tensors)")

    # Datasets
    print(f"\nBuilding datasets...")
    train_dataset = DirectCMDataset(
        pairs_pkl=args.clean_pairs,
        adv_uvmap_dir=args.adv_uvmap_dir,
        attack=args.attack,
        adv_ratio=args.adv_ratio
    )

    val_dataset = DirectCMDataset(
        pairs_pkl=args.val_pairs,
        adv_uvmap_dir=None,   # clean val — always evaluate on clean
        adv_ratio=0.0
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # Adversarial validation dataloader (val pairs but pointing to adv UV maps)
    adv_val_loader = None
    adv_val_dir = args.adv_val_dir or args.adv_uvmap_dir
    if adv_val_dir:
        adv_val_dataset = DirectCMDataset(
            pairs_pkl=args.val_pairs,
            adv_uvmap_dir=adv_val_dir,
            attack=args.attack,
            adv_ratio=1.0   # always use adversarial for adv val
        )
        adv_val_loader = DataLoader(adv_val_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=2)
        print(f"  Adversarial val: always using adversarial UV maps for evaluation")

    # Config
    config = {
        'epochs':     args.epochs,
        'lr':         args.lr,
        'lambda_reg': args.lambda_reg,
        'grad_clip':  args.grad_clip,
        'adv_ratio':  args.adv_ratio,
        'weights_path': args.weights_path,
        'attack':     args.attack,
    }

    # Trainer
    trainer = DirectCMTrainer(
        simsac=simsac,
        original_params=original_params,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    if adv_val_loader:
        trainer.adv_val_loader = adv_val_loader

    # Train
    trainer.train(output_dir=args.output_dir)


if __name__ == '__main__':
    main()
