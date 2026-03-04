"""
SimSAC Direct Change Map Fine-tuning (Path C — No contrastive loss)

Directly fine-tunes SimSAC's change map output on pre-generated adversarial
UV maps from contrastive_pairs_surface_wb_test. No projection head. No
contrastive loss. No temperature.

Key design decisions that make this reliable:

  1. DIRECT change map loss — optimises exactly what predict_tampering uses
     Tampered pairs → push change map HIGH
     Clean pairs    → push change map LOW

  2. L2 regularisation toward original synthetic.pth weights
     Prevents catastrophic forgetting of clean-image performance.
     Without it: clean accuracy drops (as seen in Phase 2 full fine-tune → 65%)
     With it:    clean accuracy stays close to 84% while adversarial improves

  3. Mixed data per batch (clean + adversarial together in the pkl)
     The contrastive_pairs_surface_wb_test pkl already contains both clean and
     adversarial pairs (adversarial_simsac_test_wb_ep_10). Model sees both
     distributions every epoch → cannot forget clean while learning robustness.

  4. Very small learning rate (default 1e-5)
     Conservative backbone updates → stable training

  5. Per-epoch evaluation on BOTH clean and adversarial validation
     val_loader is the mixed val pkl; evaluate_separation splits by is_adv flag.
     Best checkpoint selected by: 0.5 * clean_sep + 0.5 * adv_sep
     where sep = tampered_mean_cm - clean_mean_cm (higher = better)

Why this works against pre-generated PGD/FGSM attacks:
  - Attack was generated ONCE with original synthetic.pth (fixed perturbations)
  - Model trains to recognise and resist THOSE specific perturbation patterns
  - Unlike online training (arms race), this is stable offline adversarial training
  - L2 reg ensures backbone doesn't move far from original → clean preserved

Usage:
    python extension/train_simsac_direct_cm.py \\
        --clean_pairs  /content/drive/MyDrive/TAMPAR_DATA/tampar/contrastive_pairs_surface_wb_test/train_pairs_surface_level.pkl \\
        --val_pairs    /content/drive/MyDrive/TAMPAR_DATA/tampar/contrastive_pairs_surface_wb_test/val_pairs_surface_level.pkl \\
        --weights_path /content/tampar/src/simsac/weight/synthetic.pth \\
        --output_dir   /content/drive/MyDrive/TAMPAR_DATA/tampar/simsac_weights_direct_cm \\
        --lr 1e-5 \\
        --lambda_reg 0.1 \\
        --epochs 15
"""

import sys
import types
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'extension'))

from src.simsac.models.our_models.SimSaC import SimSaC_Model
from contrastive_dataset import ContrastivePairsDataset


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
        field_img: [B, C, H, W] in ImageNet-normalised range
        ref_img:   [B, C, H, W] in ImageNet-normalised range

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

        # +1e-8 inside sqrt prevents NaN gradient when both channels are near 0
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
# Trainer
# ---------------------------------------------------------------------------

class DirectCMTrainer:
    """
    Trains SimSAC backbone directly on change map loss.

    No contrastive loss. No projection head. No temperature.
    Pure change map supervision + L2 regularisation toward original weights.

    val_loader should be the MIXED val pkl (contrastive_pairs_surface_wb_test)
    which contains both clean and adversarial pairs. evaluate_separation splits
    these by the is_adversarial flag so we track both clean_sep and adv_sep from
    a single loader.
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
            'val_clean_sep': [],     # tampered_mean_cm - clean_mean_cm on non-adv pairs
            'val_adv_sep': [],       # tampered_mean_cm - clean_mean_cm on adversarial pairs
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
        Compute change map separation on a mixed dataloader, split by is_adv.

        Returns:
            (clean_sep, clean_t_mean, clean_c_mean): stats on non-adversarial pairs
            (adv_sep, adv_t_mean, adv_c_mean):       stats on adversarial pairs

        separation = mean_cm(tampered) - mean_cm(clean), higher is better.
        If no adversarial pairs are found, adv stats fall back to clean stats.
        """
        self.simsac.eval()

        # (is_adv=False/True, label=0/1) → list of magnitudes
        cms = {(False, 0): [], (False, 1): [], (True, 0): [], (True, 1): []}

        for img1, img2, labels, is_advs in loader:
            img1   = img1.to(self.device)
            img2   = img2.to(self.device)
            labels = labels.to(self.device)

            mags = get_change_magnitude(self.simsac, img2, img1)

            for mag, lbl, is_adv_flag in zip(mags, labels, is_advs):
                key = (bool(is_adv_flag.item()), int(lbl.item()))
                cms[key].append(mag.item())

        def compute_sep(tampered_list, clean_list):
            if not tampered_list or not clean_list:
                return 0.0, 0.0, 0.0
            t = float(np.mean(tampered_list))
            c = float(np.mean(clean_list))
            return t - c, t, c

        clean_sep, clean_t, clean_c = compute_sep(cms[(False, 1)], cms[(False, 0)])
        adv_sep,   adv_t,   adv_c   = compute_sep(cms[(True, 1)],  cms[(True, 0)])

        # If no adversarial pairs in this split, report clean stats for both
        if not cms[(True, 0)] and not cms[(True, 1)]:
            adv_sep, adv_t, adv_c = clean_sep, clean_t, clean_c

        n_clean_pairs = len(cms[(False, 0)]) + len(cms[(False, 1)])
        n_adv_pairs   = len(cms[(True, 0)])  + len(cms[(True, 1)])
        print(f"    [eval] clean pairs: {n_clean_pairs}  |  adv pairs: {n_adv_pairs}")

        return (clean_sep, clean_t, clean_c), (adv_sep, adv_t, adv_c)

    def train(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*65}")
        print("SimSAC Direct Change Map Fine-tuning")
        print(f"{'='*65}")
        print(f"Epochs:      {self.config['epochs']}")
        print(f"LR:          {self.config['lr']}")
        print(f"Lambda reg:  {self.config['lambda_reg']}")
        print(f"Output:      {output_dir}")
        print(f"{'='*65}")

        for epoch in range(self.config['epochs']):

            # Train
            train_loss = self.train_epoch(epoch)

            # Evaluate on mixed val (splits by is_adv internally)
            (clean_sep, clean_t, clean_c), (adv_sep, adv_t, adv_c) = \
                self.evaluate_separation(self.val_loader)

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

    # Data — both pkls come from contrastive_pairs_surface_wb_test
    # which already contains mixed clean + adversarial (adversarial_simsac_test_wb_ep_10) pairs
    parser.add_argument('--clean_pairs', type=str, required=True,
                        help='Path to train_pairs_surface_level.pkl '
                             '(from contrastive_pairs_surface_wb_test — '
                             'already contains both clean and adversarial pairs)')
    parser.add_argument('--val_pairs', type=str, required=True,
                        help='Path to val_pairs_surface_level.pkl '
                             '(from contrastive_pairs_surface_wb_test — '
                             'already contains both clean and adversarial pairs)')

    # Model
    parser.add_argument('--weights_path', type=str,
                        default='/content/tampar/src/simsac/weight/synthetic.pth',
                        help='Starting SimSAC weights (ALWAYS use synthetic.pth, not a failed checkpoint)')

    # Training
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of epochs (default: 15)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate (default: 1e-5, keep small to prevent forgetting)')
    parser.add_argument('--lambda_reg', type=float, default=0.1,
                        help='L2 regularisation weight toward original weights (default: 0.1). '
                             'Increase if clean accuracy drops, decrease if adv improvement is slow.')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=2,
                        help='DataLoader workers (default: 2)')

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
    print(f"Train pairs:    {args.clean_pairs}")
    print(f"Val pairs:      {args.val_pairs}")
    print(f"Weights:        {args.weights_path}")
    print(f"LR:             {args.lr}  (keep low to prevent catastrophic forgetting)")
    print(f"Lambda reg:     {args.lambda_reg}  (L2 toward original weights)")
    print(f"Note: pkls from contrastive_pairs_surface_wb_test already contain")
    print(f"      both clean and adversarial (adversarial_simsac_test_wb_ep_10) pairs.")

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

    # Datasets — ContrastivePairsDataset handles:
    #   - keys 'image1'/'surface1' and 'image2'/'surface2' (numpy arrays in pkl)
    #   - numpy → PIL conversion
    #   - ImageNet normalisation
    #   - is_adversarial detection from image2_path / pair_type / metadata
    print(f"\nBuilding datasets...")
    train_dataset = ContrastivePairsDataset(args.clean_pairs)
    val_dataset   = ContrastivePairsDataset(args.val_pairs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda')
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    # Config
    config = {
        'epochs':       args.epochs,
        'lr':           args.lr,
        'lambda_reg':   args.lambda_reg,
        'grad_clip':    args.grad_clip,
        'weights_path': args.weights_path,
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

    # Train
    trainer.train(output_dir=args.output_dir)


if __name__ == '__main__':
    main()
