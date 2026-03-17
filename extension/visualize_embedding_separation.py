#!/usr/bin/env python3

import argparse
from pathlib import Path
import pickle
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
import cv2
import os

# Add src to path - handle both local and Colab environments
parent_dir = Path(__file__).parent.parent

if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Set PYTHONPATH for subprocesses
os.environ['PYTHONPATH'] = f"{parent_dir}:{os.environ.get('PYTHONPATH', '')}"

from src.simsac.inference import SimSaC

# Import from extension directory
sys.path.insert(0, str(Path(__file__).parent))
from train_simsac_adversarial_robust import SimSaCRobust


def compute_embeddings(pairs, model, device='cuda'):
    model.eval()

    embeddings = []
    labels = []
    is_adversarial = []

    print("Computing embeddings")
    skipped = 0
    with torch.no_grad():
        for pair in tqdm(pairs):
            # Load images
            ref = cv2.imread(str(pair['reference_patch']))
            field = cv2.imread(str(pair['field_patch']))

            # Skip if files don't exist or can't be loaded
            if ref is None or field is None:
                skipped += 1
                continue

            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            ref = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0).to(device)

            field = cv2.cvtColor(field, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            field = torch.from_numpy(field).permute(2, 0, 1).unsqueeze(0).to(device)

            # Get embedding
            embedding, _ = model(ref, field)
            embedding = embedding.cpu().numpy()[0]

            embeddings.append(embedding)
            labels.append(pair['label'])
            is_adversarial.append(pair['is_adversarial'])

    if skipped > 0:
        print(f"Warning: Skipped {skipped} pairs due to missing image files")

    embeddings = np.array(embeddings)
    labels = np.array(labels)
    is_adversarial = np.array(is_adversarial)

    return embeddings, labels, is_adversarial


def compute_pairwise_distances(embeddings, labels):
    from scipy.spatial.distance import pdist, squareform

    # Compute all pairwise distances
    distances = squareform(pdist(embeddings, metric='euclidean'))

    # Create label matrix
    label_matrix = (labels[:, None] == labels[None, :])

    # Get upper triangle indices (avoid duplicates)
    triu_indices = np.triu_indices(len(labels), k=1)

    # Separate positive and negative pairs
    positive_mask = label_matrix[triu_indices]
    negative_mask = ~label_matrix[triu_indices]

    positive_distances = distances[triu_indices][positive_mask]
    negative_distances = distances[triu_indices][negative_mask]

    return positive_distances, negative_distances


def plot_distributions(positive_dist, negative_dist, title, ax):
    # Compute KDE for smooth curves
    if len(positive_dist) > 1:
        try:
            kde_pos = gaussian_kde(positive_dist)
            x_pos = np.linspace(positive_dist.min(), positive_dist.max(), 200)
            y_pos = kde_pos(x_pos)
        except:
            x_pos = np.linspace(positive_dist.min(), positive_dist.max(), 200)
            y_pos = np.histogram(positive_dist, bins=50, density=True)[0]

    if len(negative_dist) > 1:
        try:
            kde_neg = gaussian_kde(negative_dist)
            x_neg = np.linspace(negative_dist.min(), negative_dist.max(), 200)
            y_neg = kde_neg(x_neg)
        except:
            x_neg = np.linspace(negative_dist.min(), negative_dist.max(), 200)
            y_neg = np.histogram(negative_dist, bins=50, density=True)[0]

    # Plot histograms
    ax.hist(positive_dist, bins=30, alpha=0.5, density=True, color='blue', label='Positive (Untampered)')
    ax.hist(negative_dist, bins=30, alpha=0.5, density=True, color='red', label='Negative (Tampered)')

    # Plot KDE curves
    if len(positive_dist) > 1:
        ax.plot(x_pos, y_pos, 'b-', linewidth=2)
    if len(negative_dist) > 1:
        ax.plot(x_neg, y_neg, 'r-', linewidth=2)

    # Add vertical lines for means
    ax.axvline(positive_dist.mean(), color='blue', linestyle='--', alpha=0.7,
               label=f'Pos mean: {positive_dist.mean():.3f}')
    ax.axvline(negative_dist.mean(), color='red', linestyle='--', alpha=0.7,
               label=f'Neg mean: {negative_dist.mean():.3f}')

    # Compute separation metric (higher = better)
    separation = abs(negative_dist.mean() - positive_dist.mean())
    overlap = min(positive_dist.max(), negative_dist.max()) - max(positive_dist.min(), negative_dist.min())
    overlap_ratio = overlap / (max(positive_dist.max(), negative_dist.max()) - min(positive_dist.min(), negative_dist.min()))

    ax.set_xlabel('Embedding Distance')
    ax.set_ylabel('Density')
    ax.set_title(f'{title}\nSeparation: {separation:.3f} | Overlap: {overlap_ratio:.1%}')
    ax.legend()
    ax.grid(alpha=0.3)


def visualize_embeddings(pairs_file, checkpoint_path=None, output_path='embedding_separation.png', compare_baseline=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load pairs
    with open(pairs_file, 'rb') as f:
        pairs = pickle.load(f)

    print(f"Loaded {len(pairs)} pairs")

    # Subsample if too many pairs (for speed)
    if len(pairs) > 500:
        import random
        pairs = random.sample(pairs, 500)
        print(f"Subsampled to {len(pairs)} pairs for visualization")

    if compare_baseline:
        # Show before and after
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Before (baseline)
        print("\n=== BASELINE (Before Fine-tuning) ===")
        model_baseline = SimSaCRobust(freeze_backbone=True).to(device)
        embeddings_baseline, labels, is_adv = compute_embeddings(pairs, model_baseline, device)
        pos_dist_baseline, neg_dist_baseline = compute_pairwise_distances(embeddings_baseline, labels)

        plot_distributions(pos_dist_baseline, neg_dist_baseline, 'Before Fine-Tuning', axes[0])

        # After (fine-tuned)
        print("\n=== AFTER FINE-TUNING ===")
        model_finetuned = SimSaCRobust(freeze_backbone=True).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_finetuned.load_state_dict(checkpoint['model_state_dict'])

        embeddings_finetuned, labels, is_adv = compute_embeddings(pairs, model_finetuned, device)
        pos_dist_finetuned, neg_dist_finetuned = compute_pairwise_distances(embeddings_finetuned, labels)

        plot_distributions(pos_dist_finetuned, neg_dist_finetuned, 'After Fine-Tuning', axes[1])

        # Print improvement metrics
        sep_before = abs(neg_dist_baseline.mean() - pos_dist_baseline.mean())
        sep_after = abs(neg_dist_finetuned.mean() - pos_dist_finetuned.mean())
        improvement = ((sep_after - sep_before) / sep_before) * 100

        print(f"IMPROVEMENT METRICS")
        print(f"Separation before: {sep_before:.3f}")
        print(f"Separation after:  {sep_after:.3f}")
        print(f"Improvement:       {improvement:+.1f}%")

    else:
        # Show single plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        if checkpoint_path and checkpoint_path.lower() != 'none':
            # Fine-tuned model
            print("\n=== FINE-TUNED MODEL ===")
            model = SimSaCRobust(freeze_backbone=True).to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            title = "After Fine-Tuning"
        else:
            # Baseline model
            print("\n=== BASELINE MODEL ===")
            model = SimSaCRobust(freeze_backbone=True).to(device)
            title = "Before Fine-Tuning (Baseline)"

        embeddings, labels, is_adv = compute_embeddings(pairs, model, device)
        pos_dist, neg_dist = compute_pairwise_distances(embeddings, labels)

        plot_distributions(pos_dist, neg_dist, title, ax)

        # Print statistics
        print(f"  Mean distance: {pos_dist.mean():.3f}")
        print(f"  Std distance:  {pos_dist.std():.3f}")
        print(f"  Range: [{pos_dist.min():.3f}, {pos_dist.max():.3f}]")
        print(f"\nNegative pairs (tampered):")
        print(f"  Mean distance: {neg_dist.mean():.3f}")
        print(f"  Std distance:  {neg_dist.std():.3f}")
        print(f"  Range: [{neg_dist.min():.3f}, {neg_dist.max():.3f}]")
        print(f"\nSeparation: {abs(neg_dist.mean() - pos_dist.mean()):.3f}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n Saved visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize embedding separation')
    parser.add_argument('--pairs_file', type=str, required=True,
                       help='Path to adversarial pairs pickle file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to fine-tuned checkpoint (none/None for baseline)')
    parser.add_argument('--output', type=str, default='embedding_separation.png',
                       help='Output image path')
    parser.add_argument('--compare_baseline', action='store_true',
                       help='Show before/after comparison')

    args = parser.parse_args()

    visualize_embeddings(
        args.pairs_file,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        compare_baseline=args.compare_baseline
    )


if __name__ == '__main__':
    main()
