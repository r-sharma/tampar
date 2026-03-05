"""
SimSAC Laplacian Distillation Fine-tuning

Key insight from adversarial attack analysis (comparison_results_adv_test_wb_ep_10.csv):
  - PGD/FGSM attack targeted SimSAC's correlation features specifically
  - Result: simsac simple_threshold dropped 84% → 70.7% on adversarial test
  - But: laplacian XGBoost alone = 91.4%, canny XGBoost alone = 86.3%
  - Edge detectors are completely IMMUNE because the attack never touched them

Why Laplacian works as a teacher:
  The PGD attack minimised ||cm_simsac(tampered+δ, ref)||, making SimSAC's VGG
  pyramid features of (tampered+δ) look identical to features of (ref).
  But the actual pixel-level tampering (UV map changes) creates REAL geometric
  differences that Laplacian edge maps still detect clearly.

Training signal:
  For every pair (ref, field):
    teacher = Laplacian_spatial_diff(ref, field)   ← fixed, no grad, attack-immune
    student = SimSAC_pyramid_feature_dist(ref, field)  ← grad flows through VGG
    loss    = MSE(normalise(student), normalise(teacher))

  For adversarial tampered pairs:
    teacher is HIGH  (real tampering → Laplacian sees edge differences)
    student is LOW   (attack fooled SimSAC's correlation features)
    → loss pushes student HIGH → change map becomes discriminative again

  For clean pairs:
    teacher is LOW   (no tampering → Laplacian sees similar edges)
    → loss keeps student LOW → no false positives

  L2 regularisation toward synthetic.pth prevents catastrophic forgetting.

Key differences from train_simsac_direct_cm.py (which failed):
  - NO enable_simsac_gradients() proxy — we access simsac.pyramid directly
    (a standard nn.Module, gradient flow works natively)
  - NO change map proxy that mismatched predict_tampering's actual output
  - Laplacian provides an EXACT per-sample regression target, not just
    a push direction — stronger, more informative gradient signal

Evaluation during training:
  - Feature distances are thresholded to classify tampered/clean
  - Best threshold found per epoch on full val set
  - Accuracy reported separately for clean pairs and adversarial pairs
  - Final verification: run compute_similarity_scores.py with best.pth

Usage:
    python extension/train_simsac_laplacian_distill.py \\
        --train_pairs /content/drive/MyDrive/TAMPAR_DATA/tampar/contrastive_pairs_surface_wb_test/train_pairs_surface_level.pkl \\
        --val_pairs   /content/drive/MyDrive/TAMPAR_DATA/tampar/contrastive_pairs_surface_wb_test/val_pairs_surface_level.pkl \\
        --weights_path /content/tampar/src/simsac/weight/synthetic.pth \\
        --output_dir  /content/drive/MyDrive/TAMPAR_DATA/tampar/simsac_weights_lap_distill \\
        --lr 1e-5 --lambda_reg 0.1 --epochs 15
"""

import sys
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
# Laplacian teacher — fixed operation, immune to SimSAC-targeted attacks
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_laplacian_diff(img1, img2):
    """
    Compute spatial Laplacian edge difference between two images.

    This is the TEACHER signal. The adversarial perturbation δ was optimised
    to fool SimSAC's correlation features only. Real UV-map tampering creates
    genuine geometric changes that Laplacian edges still detect clearly —
    even through the adversarial perturbation.

    Args:
        img1, img2: [B, C, H, W]  ImageNet-normalised tensors (on device)

    Returns:
        diff: [B]  per-pair mean squared Laplacian spatial difference
              High → images differ in edge space (tampered)
              Low  → images look similar in edge space (clean)
    """
    C, device, dtype = img1.shape[1], img1.device, img1.dtype

    # Standard discrete Laplacian kernel — sensitive to step edges
    kernel = torch.tensor(
        [[0.,  1., 0.],
         [1., -4., 1.],
         [0.,  1., 0.]], dtype=dtype, device=device
    ).view(1, 1, 3, 3).expand(C, 1, 3, 3).contiguous()

    # Depthwise convolution: apply the same kernel to every channel
    lap1 = F.conv2d(img1, kernel, padding=1, groups=C)   # [B, C, H, W]
    lap2 = F.conv2d(img2, kernel, padding=1, groups=C)   # [B, C, H, W]

    # Spatial mean-squared difference of edge maps → per-image scalar
    diff = (lap1 - lap2).pow(2).mean(dim=[1, 2, 3])      # [B]
    return diff


# ---------------------------------------------------------------------------
# SimSAC pyramid feature distance — student, gradients flow here
# ---------------------------------------------------------------------------

def get_simsac_feature_dist(simsac, img1, img2):
    """
    Compute feature-space L2 distance using SimSAC's pyramid backbone.

    Directly accesses `simsac.pyramid` (VGG feature extractor).
    This is a standard nn.Module → gradient computation is native.
    No enable_simsac_gradients() proxy needed.

    Processes one image at a time (as SimSAC is designed for single-pair
    input), collects distances across all pyramid levels, returns the
    mean distance — a scalar per image in the batch.

    Args:
        simsac:     SimSaC_Model in .train() mode
        img1, img2: [B, C, H, W]  ImageNet-normalised, on device

    Returns:
        dist: [B]  per-pair mean feature distance across pyramid levels
    """
    B = img1.shape[0]
    dists = []

    for i in range(B):
        f1 = img1[i:i+1]   # [1, C, H, W]
        f2 = img2[i:i+1]   # [1, C, H, W]

        # eigth_resolution=True → compute all scales down to 1/8 resolution
        # Fallback: some builds may not accept the kwarg
        try:
            pyr1 = simsac.pyramid(f1, eigth_resolution=True)
            pyr2 = simsac.pyramid(f2, eigth_resolution=True)
        except TypeError:
            pyr1 = simsac.pyramid(f1)
            pyr2 = simsac.pyramid(f2)

        if not pyr1:
            raise RuntimeError("simsac.pyramid returned empty feature list")

        # Average L2 distance across all pyramid levels
        level_dists = []
        for p1, p2 in zip(pyr1, pyr2):
            d = (p1 - p2).pow(2).mean()   # scalar — collapses C, H, W
            level_dists.append(d)

        dists.append(torch.stack(level_dists).mean())

    return torch.stack(dists)   # [B]


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def distillation_loss(simsac_dist, lap_dist):
    """
    MSE between batch-normalised SimSAC feature distance and Laplacian distance.

    Batch-normalisation (min-max to [0, 1]) removes the scale mismatch
    between the two distance spaces while preserving their relative ordering
    and within-batch magnitudes.

    Gradient intuition:
      If simsac_dist(adv_tampered) < lap_dist(adv_tampered):
        → gradient pushes simsac_dist UP (toward Laplacian's high value)
        → pyramid features become more sensitive to the real tampering signal

    Args:
        simsac_dist: [B]  pyramid feature distances (requires grad)
        lap_dist:    [B]  Laplacian differences (detached, no grad)

    Returns:
        Scalar MSE loss
    """
    target = lap_dist.detach().float()
    pred   = simsac_dist.float()

    def batch_minmax(x):
        lo  = x.min()
        hi  = x.max()
        rng = (hi - lo).clamp(min=1e-8)
        return (x - lo) / rng

    # Degenerate batch: all values identical → zero loss (avoid div by zero)
    if pred.max() == pred.min() or target.max() == target.min():
        return pred.mean() * 0.0   # zero, keeps gradient graph alive

    return F.mse_loss(batch_minmax(pred), batch_minmax(target))


def l2_reg_loss(model, original_params):
    """
    L2 regularisation toward original synthetic.pth weights.

    Prevents catastrophic forgetting: penalises deviation from the original
    weight values. Keeps clean-image performance intact while the backbone
    adjusts to be Laplacian-guided.
    """
    reg = 0.0
    for name, param in model.named_parameters():
        if name in original_params and param.requires_grad:
            reg = reg + ((param - original_params[name]) ** 2).sum()
    return reg


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_val_stats(simsac, loader, device):
    """
    Run feature distance computation over the entire validation set.

    Returns:
        dists:    np.ndarray [N]   per-pair feature distances
        labels:   np.ndarray [N]   0=clean, 1=tampered
        is_adv:   np.ndarray [N]   bool, True if pair uses adversarial field
    """
    simsac.eval()
    all_dists, all_labels, all_is_adv = [], [], []

    for img1, img2, labels, is_advs in loader:
        img1 = img1.to(device)
        img2 = img2.to(device)

        dists = get_simsac_feature_dist(simsac, img1, img2)

        all_dists.extend(dists.cpu().float().numpy().tolist())
        all_labels.extend(labels.numpy().astype(int).tolist())
        all_is_adv.extend(is_advs.numpy().astype(bool).tolist())

    return (np.array(all_dists, dtype=np.float32),
            np.array(all_labels, dtype=np.int32),
            np.array(all_is_adv, dtype=bool))


def best_threshold_accuracy(dists, labels):
    """
    Sweep thresholds and return (best_accuracy, best_threshold).
    Prediction rule: tampered if dist > threshold.
    """
    thresholds = np.percentile(dists, np.linspace(0, 100, 200))
    best_acc, best_t = 0.0, thresholds[0]
    for t in thresholds:
        preds = (dists > t).astype(int)
        acc   = (preds == labels).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t
    return float(best_acc), float(best_t)


def feature_separation(dists, labels):
    """mean_dist(tampered) − mean_dist(clean). Higher = better separation."""
    t_dists = dists[labels == 1]
    c_dists = dists[labels == 0]
    if len(t_dists) == 0 or len(c_dists) == 0:
        return 0.0
    return float(t_dists.mean() - c_dists.mean())


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class LapDistillTrainer:
    """
    Fine-tunes SimSAC's pyramid backbone with Laplacian distillation.

    Loss per batch:
        distillation_loss(simsac_feat_dist, laplacian_diff.detach())
        + lambda_reg * l2_reg_loss(model, original_params)

    Validation metric:
        Threshold accuracy on feature distances, split by is_adv.
        Best checkpoint = highest (0.5 * clean_acc + 0.5 * adv_acc).

    NOTE: Val accuracy here uses feature distances as the classifier.
    FINAL verification must use compute_similarity_scores.py with best.pth
    (which uses the real SimSAC change map + all compare types).
    """

    def __init__(self, simsac, original_params, train_loader, val_loader,
                 config, device):
        self.simsac          = simsac
        self.original_params = original_params
        self.train_loader    = train_loader
        self.val_loader      = val_loader
        self.config          = config
        self.device          = device

        self.optimizer = optim.Adam(
            [p for p in simsac.parameters() if p.requires_grad],
            lr=config['lr'],
            weight_decay=0.0   # explicit L2 toward original, not toward zero
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max',
            factor=0.5, patience=3, min_lr=1e-7
        )

        self.history = {
            'train_distill_loss': [],
            'train_reg_loss':     [],
            'val_clean_acc':      [],
            'val_adv_acc':        [],
            'val_combined':       [],
            'val_separation':     [],
            'learning_rate':      [],
        }
        self.best_combined = -float('inf')
        self.best_epoch    = 0

    # ------------------------------------------------------------------
    def train_epoch(self, epoch):
        self.simsac.train()
        total_distill = 0.0
        total_reg     = 0.0
        n = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")

        for img1, img2, labels, is_adv in pbar:
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            # labels and is_adv not needed for loss — Laplacian carries
            # the discriminative signal implicitly

            self.optimizer.zero_grad()

            # Teacher: Laplacian spatial edge difference — no grad
            lap_diff  = compute_laplacian_diff(img1, img2)   # [B]

            # Student: SimSAC pyramid feature distance — grad flows here
            feat_dist = get_simsac_feature_dist(self.simsac, img1, img2)  # [B]

            # Distillation loss: student → teacher (batch-normalised MSE)
            loss_distill = distillation_loss(feat_dist, lap_diff)

            # L2 regularisation toward original synthetic.pth
            loss_reg = l2_reg_loss(self.simsac, self.original_params)

            loss = loss_distill + self.config['lambda_reg'] * loss_reg
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.simsac.parameters(), self.config.get('grad_clip', 1.0)
            )
            self.optimizer.step()

            total_distill += loss_distill.item()
            total_reg     += loss_reg.item()
            n += 1

            pbar.set_postfix({
                'distill': f"{loss_distill.item():.4f}",
                'reg':     f"{loss_reg.item():.3f}",
                'avg':     f"{total_distill/n:.4f}",
            })

        return total_distill / n, total_reg / n

    # ------------------------------------------------------------------
    def evaluate(self):
        """
        Evaluate on val set.
        Finds the best threshold on the full val set, then reports
        accuracy separately for clean and adversarial pairs using that
        same threshold.
        """
        dists, labels, is_adv = collect_val_stats(
            self.simsac, self.val_loader, self.device
        )

        overall_acc, best_t = best_threshold_accuracy(dists, labels)
        sep = feature_separation(dists, labels)

        # Apply same threshold to each split
        clean_mask = ~is_adv
        adv_mask   =  is_adv

        def split_acc(mask):
            if not mask.any():
                return overall_acc
            preds = (dists[mask] > best_t).astype(int)
            return float((preds == labels[mask]).mean())

        clean_acc = split_acc(clean_mask)
        adv_acc   = split_acc(adv_mask)

        n_clean = int(clean_mask.sum())
        n_adv   = int(adv_mask.sum())
        print(f"    [eval] clean pairs: {n_clean}  |  adv pairs: {n_adv}"
              f"  |  threshold: {best_t:.6f}")

        return clean_acc, adv_acc, overall_acc, sep

    # ------------------------------------------------------------------
    def train(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*65}")
        print("SimSAC Laplacian Distillation Fine-tuning")
        print(f"{'='*65}")
        print(f"Epochs:      {self.config['epochs']}")
        print(f"LR:          {self.config['lr']}")
        print(f"Lambda reg:  {self.config['lambda_reg']}")
        print(f"Baselines:   clean=84.0%  |  adversarial=70.7%")
        print(f"Output:      {output_dir}")
        print(f"{'='*65}\n")

        for epoch in range(self.config['epochs']):

            # Train
            train_distill, train_reg = self.train_epoch(epoch)

            # Evaluate
            clean_acc, adv_acc, overall_acc, sep = self.evaluate()

            combined = 0.5 * clean_acc + 0.5 * adv_acc
            self.scheduler.step(combined)
            lr = self.optimizer.param_groups[0]['lr']

            self.history['train_distill_loss'].append(train_distill)
            self.history['train_reg_loss'].append(train_reg)
            self.history['val_clean_acc'].append(clean_acc)
            self.history['val_adv_acc'].append(adv_acc)
            self.history['val_combined'].append(combined)
            self.history['val_separation'].append(sep)
            self.history['learning_rate'].append(lr)

            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print(f"  Train distill loss:  {train_distill:.4f}")
            print(f"  Train reg loss:      {train_reg:.4f}")
            print(f"  Val clean acc:       {clean_acc*100:.1f}%  "
                  f"(baseline 84.0%)")
            print(f"  Val adv acc:         {adv_acc*100:.1f}%  "
                  f"(baseline 70.7%)")
            print(f"  Val overall acc:     {overall_acc*100:.1f}%")
            print(f"  Val separation:      {sep:.4f}")
            print(f"  Combined (metric):   {combined:.4f}")
            print(f"  LR:                  {lr:.2e}")

            is_best = combined > self.best_combined
            if is_best:
                self.best_combined = combined
                self.best_epoch    = epoch + 1
                self._save(epoch, output_dir, 'best.pth')
                print(f"  ✓ New best saved  (combined: {combined:.4f})")

            if (epoch + 1) % 5 == 0:
                self._save(epoch, output_dir, f'epoch_{epoch+1}.pth')
                self._plot(output_dir)

        # Final checkpoint
        self._save(self.config['epochs'] - 1, output_dir, 'final.pth')
        self._plot(output_dir)

        print(f"\n{'='*65}")
        print(f"Training complete!")
        print(f"Best epoch:  {self.best_epoch}")
        print(f"Best combined acc: {self.best_combined:.4f}")
        print(f"\nNext step — verify with real change map:")
        print(f"  python src/tools/compute_similarity_scores.py \\")
        print(f"    --checkpoint {output_dir}/best.pth")
        print(f"  Then compare simsac simple_threshold to baseline 70.7%")
        print(f"{'='*65}")

    # ------------------------------------------------------------------
    def _save(self, epoch, output_dir, filename):
        torch.save({
            'epoch':                epoch + 1,
            'state_dict':           self.simsac.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_combined':        self.best_combined,
            'history':              self.history,
            'config':               self.config,
        }, output_dir / filename)

    def _plot(self, output_dir):
        epochs = range(1, len(self.history['train_distill_loss']) + 1)
        fig, axes = plt.subplots(1, 4, figsize=(22, 4))

        # --- Loss ---
        axes[0].plot(epochs, self.history['train_distill_loss'],
                     marker='o', color='steelblue', label='Distill loss')
        axes[0].plot(epochs, self.history['train_reg_loss'],
                     marker='s', color='orange', label='Reg loss', linestyle='--')
        axes[0].set_title('Train Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # --- Accuracy ---
        axes[1].plot(epochs, [v * 100 for v in self.history['val_clean_acc']],
                     marker='o', color='green', label='Clean val acc')
        axes[1].plot(epochs, [v * 100 for v in self.history['val_adv_acc']],
                     marker='s', color='red', label='Adversarial val acc')
        axes[1].plot(epochs, [v * 100 for v in self.history['val_combined']],
                     marker='^', color='purple', label='Combined', linestyle='--')
        axes[1].axhline(84.0,  color='green', linestyle=':', alpha=0.5,
                        label='Clean baseline 84%')
        axes[1].axhline(70.7,  color='red',   linestyle=':', alpha=0.5,
                        label='Adv baseline 70.7%')
        axes[1].set_title('Validation Accuracy\n(feature dist threshold)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('%')
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)

        # --- Separation ---
        axes[2].plot(epochs, self.history['val_separation'],
                     marker='o', color='darkorange')
        axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[2].set_title('Feature Distance Separation\n'
                          '(mean_dist(tampered) − mean_dist(clean))')
        axes[2].set_xlabel('Epoch')
        axes[2].grid(True, alpha=0.3)

        # --- LR ---
        axes[3].plot(epochs, self.history['learning_rate'],
                     marker='o', color='gray')
        axes[3].set_title('Learning Rate')
        axes[3].set_xlabel('Epoch')
        axes[3].set_yscale('log')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'training_progress.png', dpi=150)
        plt.close()
        print(f"  Plot saved → {output_dir}/training_progress.png")


# ---------------------------------------------------------------------------
# SimSAC loader
# ---------------------------------------------------------------------------

def load_simsac(weights_path, device):
    """
    Load SimSAC from checkpoint.

    NOTE: No enable_simsac_gradients() patch is applied here.
    We access simsac.pyramid directly, which is a standard nn.Module
    and supports gradient computation natively.
    """
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
    if 'state_dict' in ckpt:
        state = {k.replace('simsac.', ''): v
                 for k, v in ckpt['state_dict'].items()}
    else:
        state = ckpt

    simsac.load_state_dict(state, strict=False)
    simsac = simsac.to(device)
    simsac.train()

    n_params = sum(p.numel() for p in simsac.parameters())
    n_train  = sum(p.numel() for p in simsac.parameters() if p.requires_grad)
    print(f"  ✓ SimSAC loaded ({n_params/1e6:.1f}M total, "
          f"{n_train/1e6:.1f}M trainable) from {weights_path}")
    return simsac


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='SimSAC Laplacian Distillation Fine-tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Data — use contrastive_pairs_surface_wb_test (mixed clean + adversarial)
    parser.add_argument('--train_pairs', required=True,
                        help='train_pairs_surface_level.pkl from '
                             'contrastive_pairs_surface_wb_test')
    parser.add_argument('--val_pairs', required=True,
                        help='val_pairs_surface_level.pkl from '
                             'contrastive_pairs_surface_wb_test')

    # Model — always start from original synthetic.pth
    parser.add_argument('--weights_path',
                        default='/content/tampar/src/simsac/weight/synthetic.pth',
                        help='Starting SimSAC weights. Always use synthetic.pth '
                             '(not a failed checkpoint).')

    # Training
    parser.add_argument('--epochs',      type=int,   default=15)
    parser.add_argument('--batch_size',  type=int,   default=8)
    parser.add_argument('--lr',          type=float, default=1e-5,
                        help='Keep small (1e-5 or lower) to prevent forgetting')
    parser.add_argument('--lambda_reg',  type=float, default=0.1,
                        help='L2 reg weight toward synthetic.pth. '
                             'Increase if clean acc drops, decrease if '
                             'adv acc improvement is too slow.')
    parser.add_argument('--grad_clip',   type=float, default=1.0)
    parser.add_argument('--num_workers', type=int,   default=2)

    # Output
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])

    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"\n{'='*65}")
    print("SimSAC Laplacian Distillation Fine-tuning")
    print(f"{'='*65}")
    print(f"Train pairs:  {args.train_pairs}")
    print(f"Val pairs:    {args.val_pairs}")
    print(f"Weights:      {args.weights_path}")
    print(f"LR:           {args.lr}")
    print(f"Lambda reg:   {args.lambda_reg}")
    print(f"Device:       {device}")

    # Load SimSAC (no proxy patch)
    simsac = load_simsac(args.weights_path, device)

    # Store original weights for L2 regularisation
    original_params = {
        name: param.data.clone().detach()
        for name, param in simsac.named_parameters()
    }
    print(f"  ✓ Original weights stored ({len(original_params)} tensors)")

    # Datasets — ContrastivePairsDataset handles numpy arrays,
    # 'image1'/'surface1' keys, ImageNet normalisation, is_adversarial flag
    print(f"\nBuilding datasets...")
    train_dataset = ContrastivePairsDataset(args.train_pairs)
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

    config = {
        'epochs':       args.epochs,
        'lr':           args.lr,
        'lambda_reg':   args.lambda_reg,
        'grad_clip':    args.grad_clip,
        'weights_path': args.weights_path,
    }

    trainer = LapDistillTrainer(
        simsac=simsac,
        original_params=original_params,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    trainer.train(output_dir=args.output_dir)


if __name__ == '__main__':
    main()
