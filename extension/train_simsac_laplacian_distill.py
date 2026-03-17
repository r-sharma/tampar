
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


# SimSAC gradient proxy  (same as train_simsac_direct_cm.py)

def enable_simsac_gradients(model):
    def forward_with_grad(self, im_target, im_source,
                          im_target_256, im_source_256, disable_flow=None):
        im1_pyr     = self.pyramid(im_target,     eigth_resolution=True)
        im2_pyr     = self.pyramid(im_source,     eigth_resolution=True)
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


# Proxy change magnitude  (differentiable, used as student in distillation)

def get_proxy_change_magnitude(simsac, img1, img2):
    mags = []
    for i in range(img1.shape[0]):
        f = img1[i:i+1]
        r = img2[i:i+1]

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

    return torch.stack(mags)


# Laplacian teacher  (fixed, no grad, immune to SimSAC-targeted attacks)

@torch.no_grad()
def compute_laplacian_diff(img1, img2):
    C, device, dtype = img1.shape[1], img1.device, img1.dtype

    kernel = torch.tensor(
        [[0.,  1., 0.],
         [1., -4., 1.],
         [0.,  1., 0.]], dtype=dtype, device=device
    ).view(1, 1, 3, 3).expand(C, 1, 3, 3).contiguous()

    lap1 = F.conv2d(img1, kernel, padding=1, groups=C)
    lap2 = F.conv2d(img2, kernel, padding=1, groups=C)

    return (lap1 - lap2).pow(2).mean(dim=[1, 2, 3])


# Loss functions

def distillation_loss(student, teacher):
    target = teacher.detach().float()
    pred   = student.float()

    def batch_minmax(x):
        lo  = x.min()
        hi  = x.max()
        return (x - lo) / (hi - lo).clamp(min=1e-8)

    # Degenerate batch: keep zero loss but preserve grad graph
    if pred.max() == pred.min() or target.max() == target.min():
        return pred.mean() * 0.0

    return F.mse_loss(batch_minmax(pred), batch_minmax(target))


def l2_reg_loss(model, original_params):
    reg = 0.0
    for name, param in model.named_parameters():
        if name in original_params and param.requires_grad:
            reg = reg + ((param - original_params[name]) ** 2).sum()
    return reg


# Evaluation helpers

@torch.no_grad()
def collect_val_stats(simsac, loader, device):
    simsac.eval()
    all_mags, all_labels, all_is_adv = [], [], []

    for img1, img2, labels, is_advs in loader:
        img1 = img1.to(device)
        img2 = img2.to(device)

        mags = get_proxy_change_magnitude(simsac, img1, img2)

        all_mags.extend(mags.cpu().float().numpy().tolist())
        all_labels.extend(labels.numpy().astype(int).tolist())
        all_is_adv.extend(is_advs.numpy().astype(bool).tolist())

    return (np.array(all_mags,    dtype=np.float32),
            np.array(all_labels,  dtype=np.int32),
            np.array(all_is_adv,  dtype=bool))


def best_threshold_accuracy(mags, labels):
    thresholds = np.percentile(mags, np.linspace(0, 100, 200))
    best_acc, best_t = 0.0, thresholds[0]
    for t in thresholds:
        preds = (mags > t).astype(int)
        acc   = (preds == labels).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t
    return float(best_acc), float(best_t)


def change_map_separation(mags, labels):
    t = mags[labels == 1]
    c = mags[labels == 0]
    if len(t) == 0 or len(c) == 0:
        return 0.0, 0.0, 0.0
    return float(t.mean() - c.mean()), float(t.mean()), float(c.mean())


# Trainer

class LapDistillTrainer:

    def __init__(self, simsac, original_params, train_loader, val_loader,
                 config, device):
        self.simsac          = simsac
        self.original_params = original_params
        self.train_loader    = train_loader
        self.val_loader      = val_loader
        self.config          = config
        self.device          = device

        # Two param groups: pyramid (slower) vs rest (decoder, faster)
        pyramid_params = list(simsac.pyramid.parameters())
        pyramid_ids    = {id(p) for p in pyramid_params}
        other_params   = [p for p in simsac.parameters()
                          if id(p) not in pyramid_ids]

        self.optimizer = optim.Adam([
            {'params': pyramid_params, 'lr': config['lr'] * 0.1},
            {'params': other_params,   'lr': config['lr']},
        ], weight_decay=0.0)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max',
            factor=0.5, patience=3, min_lr=1e-8
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

    def train_epoch(self, epoch):
        self.simsac.train()
        total_distill = 0.0
        total_reg     = 0.0
        n = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")

        for img1, img2, labels, is_adv in pbar:
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)

            self.optimizer.zero_grad()

            # Teacher: Laplacian spatial edge difference — fixed, no grad
            lap_diff  = compute_laplacian_diff(img1, img2)

            # Student: proxy change magnitude through patched SimSAC forward
            proxy_mag = get_proxy_change_magnitude(self.simsac, img1, img2)

            # Laplacian distillation: student  match teacher (batch-normalised)
            loss_distill = distillation_loss(proxy_mag, lap_diff)

            # L2 reg toward original synthetic.pth (all params)
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
                'reg':     f"{loss_reg.item():.4f}",
                'avg':     f"{total_distill/n:.4f}",
            })

        return total_distill / n, total_reg / n

    def evaluate(self):
        mags, labels, is_adv = collect_val_stats(
            self.simsac, self.val_loader, self.device
        )

        overall_acc, best_t = best_threshold_accuracy(mags, labels)
        sep, t_mean, c_mean = change_map_separation(mags, labels)

        clean_mask = ~is_adv
        adv_mask   =  is_adv

        def split_acc(mask):
            if not mask.any():
                return overall_acc
            preds = (mags[mask] > best_t).astype(int)
            return float((preds == labels[mask]).mean())

        clean_acc = split_acc(clean_mask)
        adv_acc   = split_acc(adv_mask)

        n_clean = int(clean_mask.sum())
        n_adv   = int(adv_mask.sum())
        print(f"    [eval] clean pairs: {n_clean}  |  adv pairs: {n_adv}"
              f"  |  threshold: {best_t:.4f}"
              f"  |  sep: {sep:.4f} (tam={t_mean:.3f}, cln={c_mean:.3f})")

        return clean_acc, adv_acc, overall_acc, sep

    def train(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pyr_lr = self.config['lr'] * 0.1
        dec_lr = self.config['lr']

        print("SimSAC Laplacian Distillation Fine-tuning  (v2)")
        print(f"Epochs:           {self.config['epochs']}")
        print(f"LR pyramid:       {pyr_lr:.1e}  (conservative — VGG backbone)")
        print(f"LR decoder:       {dec_lr:.1e}")
        print(f"Lambda reg:       {self.config['lambda_reg']}")
        print(f"Baselines:        clean=84.0%  |  adversarial=70.7%")
        print(f"Output:           {output_dir}")

        for epoch in range(self.config['epochs']):

            train_distill, train_reg = self.train_epoch(epoch)
            clean_acc, adv_acc, overall_acc, sep = self.evaluate()

            combined = 0.5 * clean_acc + 0.5 * adv_acc
            self.scheduler.step(combined)
            lr_dec = self.optimizer.param_groups[1]['lr']

            self.history['train_distill_loss'].append(train_distill)
            self.history['train_reg_loss'].append(train_reg)
            self.history['val_clean_acc'].append(clean_acc)
            self.history['val_adv_acc'].append(adv_acc)
            self.history['val_combined'].append(combined)
            self.history['val_separation'].append(sep)
            self.history['learning_rate'].append(lr_dec)

            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print(f"  Train distill loss:  {train_distill:.4f}")
            print(f"  Train reg loss:      {train_reg:.4f}")
            print(f"  Val clean acc:       {clean_acc*100:.1f}%  (baseline 84.0%)")
            print(f"  Val adv acc:         {adv_acc*100:.1f}%  (baseline 70.7%)")
            print(f"  Val overall acc:     {overall_acc*100:.1f}%")
            print(f"  Val separation:      {sep:.4f}")
            print(f"  Combined (metric):   {combined:.4f}")
            print(f"  LR decoder:          {lr_dec:.2e}")

            is_best = combined > self.best_combined
            if is_best:
                self.best_combined = combined
                self.best_epoch    = epoch + 1
                self._save(epoch, output_dir, 'best.pth')
                print(f"   New best saved  (combined: {combined:.4f})")

            if (epoch + 1) % 5 == 0:
                self._save(epoch, output_dir, f'epoch_{epoch+1}.pth')
                self._plot(output_dir)

        self._save(self.config['epochs'] - 1, output_dir, 'final.pth')
        self._plot(output_dir)

        print(f"Training complete!")
        print(f"Best epoch:        {self.best_epoch}")
        print(f"Best combined acc: {self.best_combined:.4f}")
        print(f"\nNext step — verify with real change map:")
        print(f"  python src/tools/compute_similarity_scores.py \\")
        print(f"    --checkpoint {output_dir}/best.pth")
        print(f"  Then compare simsac simple_threshold to baseline 70.7%")

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

        axes[0].plot(epochs, self.history['train_distill_loss'],
                     marker='o', color='steelblue', label='Distill')
        axes[0].plot(epochs, self.history['train_reg_loss'],
                     marker='s', color='orange', linestyle='--', label='Reg')
        axes[0].set_title('Train Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, [v*100 for v in self.history['val_clean_acc']],
                     marker='o', color='green', label='Clean')
        axes[1].plot(epochs, [v*100 for v in self.history['val_adv_acc']],
                     marker='s', color='red', label='Adversarial')
        axes[1].plot(epochs, [v*100 for v in self.history['val_combined']],
                     marker='^', color='purple', linestyle='--', label='Combined')
        axes[1].axhline(84.0, color='green', linestyle=':', alpha=0.4,
                        label='Clean baseline 84%')
        axes[1].axhline(70.7, color='red',   linestyle=':', alpha=0.4,
                        label='Adv baseline 70.7%')
        axes[1].set_title('Val Accuracy (proxy change map threshold)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('%')
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, self.history['val_separation'],
                     marker='o', color='darkorange')
        axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[2].set_title('Proxy CM Separation\n(mean_mag(tampered) − mean_mag(clean))')
        axes[2].set_xlabel('Epoch')
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(epochs, self.history['learning_rate'],
                     marker='o', color='gray')
        axes[3].set_title('Decoder LR')
        axes[3].set_xlabel('Epoch')
        axes[3].set_yscale('log')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'training_progress.png', dpi=150)
        plt.close()
        print(f"  Plot saved  {output_dir}/training_progress.png")


# SimSAC loader

def load_simsac(weights_path, device):
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

    for p in simsac.parameters():
        p.requires_grad_(True)

    simsac.train()

    # --- Apply gradient proxy patch ---
    enable_simsac_gradients(simsac)

    n_total = sum(p.numel() for p in simsac.parameters())
    n_train = sum(p.numel() for p in simsac.parameters() if p.requires_grad)
    n_pyr   = sum(p.numel() for p in simsac.pyramid.parameters())
    print(f"   SimSAC loaded ({n_total/1e6:.1f}M total, "
          f"{n_train/1e6:.1f}M trainable) from {weights_path}")
    print(f"    pyramid: {n_pyr/1e6:.1f}M @ lr×0.1 | "
          f"decoder: {(n_train-n_pyr)/1e6:.1f}M @ lr")
    return simsac


# Main

def main():
    parser = argparse.ArgumentParser(
        description='SimSAC Laplacian Distillation Fine-tuning v2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--train_pairs', required=True,
                        help='train_pairs_surface_level.pkl from '
                             'contrastive_pairs_surface_wb_test')
    parser.add_argument('--val_pairs', required=True,
                        help='val_pairs_surface_level.pkl from '
                             'contrastive_pairs_surface_wb_test')
    parser.add_argument('--weights_path',
                        default='/content/tampar/src/simsac/weight/synthetic.pth')
    parser.add_argument('--epochs',      type=int,   default=15)
    parser.add_argument('--batch_size',  type=int,   default=8)
    parser.add_argument('--lr',          type=float, default=1e-5,
                        help='Decoder LR. Pyramid uses lr×0.1.')
    parser.add_argument('--lambda_reg',  type=float, default=0.5,
                        help='L2 reg toward synthetic.pth. Higher = more '
                             'conservative, less forgetting. Default 0.5 '
                             '(higher than v1 because pyramid is now trainable).')
    parser.add_argument('--grad_clip',   type=float, default=1.0)
    parser.add_argument('--num_workers', type=int,   default=2)
    parser.add_argument('--output_dir',  required=True)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])

    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print("SimSAC Laplacian Distillation Fine-tuning  (v2)")
    print(f"Train pairs:  {args.train_pairs}")
    print(f"Val pairs:    {args.val_pairs}")
    print(f"Weights:      {args.weights_path}")
    print(f"LR decoder:   {args.lr}    (pyramid: {args.lr*0.1:.1e})")
    print(f"Lambda reg:   {args.lambda_reg}")
    print(f"Device:       {device}")

    # Load + unfreeze + patch
    simsac = load_simsac(args.weights_path, device)

    # Store original weights for L2 reg (ALL params, including pyramid)
    original_params = {
        name: param.data.clone().detach()
        for name, param in simsac.named_parameters()
    }
    print(f"   Original weights stored ({len(original_params)} tensors, "
          f"all params including pyramid)")

    print(f"\nBuilding datasets")
    train_dataset = ContrastivePairsDataset(args.train_pairs)
    val_dataset   = ContrastivePairsDataset(args.val_pairs)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True,  num_workers=args.num_workers,
        pin_memory=(device == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
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
