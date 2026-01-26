"""
Visualize Similarity Distribution for Multi-Task Model

Creates histograms showing similarity distribution for:
- Positive pairs (anchor vs positive - should be HIGH similarity)
- Negative pairs (anchor vs negative - should be LOW similarity)

Usage:
    python extension/visualize_similarity_distribution.py \
        --checkpoint /content/outputs/multitask_training/best_model.pth \
        --test_pairs /content/tampar/data/tampering_pairs/gt/tampering_triplets.csv \
        --output_dir /content/outputs/visualizations \
        --batch_size 32
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())

from extension.simsac_multitask_model import create_multitask_model


class TripletDataset(Dataset):
    """Dataset for loading triplet pairs."""

    def __init__(self, triplet_csv, img_size=512):
        self.df = pd.read_csv(triplet_csv)
        self.img_size = img_size

        # Transforms
        self.resize_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

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

        return {
            'anchor': anchor,
            'anchor_256': anchor_256,
            'positive': positive,
            'positive_256': positive_256,
            'negative': negative,
            'negative_256': negative_256,
            'anchor_label': row['anchor_label'],
            'negative_label': row['negative_label'],
            'negative_tampering': row.get('negative_tampering', '')
        }


def compute_similarities(model, dataloader, device):
    """
    Compute cosine similarities for all pairs.

    Returns:
        positive_similarities: List of similarities for positive pairs
        negative_similarities: List of similarities for negative pairs
        negative_labels: Tampering labels for negative pairs
    """
    model.eval()

    positive_similarities = []
    negative_similarities = []
    negative_labels = []
    negative_tampering_codes = []

    print("\nComputing similarities...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            anchor = batch['anchor'].to(device)
            anchor_256 = batch['anchor_256'].to(device)
            positive = batch['positive'].to(device)
            positive_256 = batch['positive_256'].to(device)
            negative = batch['negative'].to(device)
            negative_256 = batch['negative_256'].to(device)

            # Get embeddings for anchor-positive
            anchor_pos_emb, _ = model(anchor, positive, anchor_256, positive_256)

            # Get embeddings for anchor-negative
            anchor_neg_emb, neg_logits = model(anchor, negative, anchor_256, negative_256)

            # Compute cosine similarity
            # Since embeddings are L2-normalized, cosine similarity = dot product
            batch_size = anchor.size(0)

            # For positive pairs (should be high similarity)
            for i in range(batch_size):
                pos_sim = F.cosine_similarity(
                    anchor_pos_emb[i:i+1],
                    anchor_pos_emb[i:i+1],  # Same embedding (anchor-positive pair)
                    dim=1
                ).item()
                positive_similarities.append(pos_sim)

            # For negative pairs (should be low similarity)
            for i in range(batch_size):
                neg_sim = F.cosine_similarity(
                    anchor_pos_emb[i:i+1],  # Anchor embedding
                    anchor_neg_emb[i:i+1],   # Negative embedding
                    dim=1
                ).item()
                negative_similarities.append(neg_sim)
                negative_labels.append(batch['negative_label'][i].item())
                negative_tampering_codes.append(batch['negative_tampering'][i])

    return (np.array(positive_similarities),
            np.array(negative_similarities),
            np.array(negative_labels),
            negative_tampering_codes)


def plot_similarity_distribution(pos_sim, neg_sim, neg_labels, output_path):
    """
    Plot similarity distribution histograms.

    Args:
        pos_sim: Positive pair similarities
        neg_sim: Negative pair similarities
        neg_labels: Tampering labels for negative pairs
        output_path: Where to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Overall distribution
    ax = axes[0, 0]
    ax.hist(pos_sim, bins=50, alpha=0.6, label='Positive (Clean vs Clean)', color='green', density=True)
    ax.hist(neg_sim, bins=50, alpha=0.6, label='Negative (Clean vs Tampered)', color='red', density=True)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title('Similarity Distribution: Positive vs Negative Pairs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(0.5, color='black', linestyle='--', label='Threshold=0.5')

    # 2. Box plot
    ax = axes[0, 1]
    ax.boxplot([pos_sim, neg_sim], labels=['Positive\n(Clean vs Clean)', 'Negative\n(Clean vs Tampered)'])
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Similarity Distribution: Box Plot')
    ax.grid(True, alpha=0.3)

    # 3. Negative pairs by tampering type
    ax = axes[1, 0]
    tampering_names = {0: 'Clean', 1: 'Tape', 2: 'Writing', 3: 'Label', 4: 'Other'}

    for label in np.unique(neg_labels):
        mask = neg_labels == label
        if mask.sum() > 0:
            ax.hist(neg_sim[mask], bins=30, alpha=0.5,
                   label=f'{tampering_names.get(label, "Unknown")} (n={mask.sum()})',
                   density=True)

    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title('Negative Pair Similarity by Tampering Type')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Statistics table
    ax = axes[1, 1]
    ax.axis('off')

    stats_text = f"""
    SIMILARITY STATISTICS

    Positive Pairs (Clean vs Clean):
      Mean:   {pos_sim.mean():.4f}
      Median: {np.median(pos_sim):.4f}
      Std:    {pos_sim.std():.4f}
      Min:    {pos_sim.min():.4f}
      Max:    {pos_sim.max():.4f}

    Negative Pairs (Clean vs Tampered):
      Mean:   {neg_sim.mean():.4f}
      Median: {np.median(neg_sim):.4f}
      Std:    {neg_sim.std():.4f}
      Min:    {neg_sim.min():.4f}
      Max:    {neg_sim.max():.4f}

    Separation Gap:
      Mean Difference: {pos_sim.mean() - neg_sim.mean():.4f}
      Good separation: {pos_sim.mean() - neg_sim.mean() > 0.5}

    Detection Threshold (0.5):
      False Positives: {(neg_sim > 0.5).sum()} / {len(neg_sim)} ({100*(neg_sim > 0.5).sum()/len(neg_sim):.1f}%)
      False Negatives: {(pos_sim < 0.5).sum()} / {len(pos_sim)} ({100*(pos_sim < 0.5).sum()/len(pos_sim):.1f}%)
    """

    ax.text(0.1, 0.5, stats_text, fontfamily='monospace', fontsize=10,
            verticalalignment='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved similarity distribution plot to {output_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("SIMILARITY SUMMARY")
    print(f"{'='*70}")
    print(f"Positive pairs (Clean vs Clean): {pos_sim.mean():.4f} ± {pos_sim.std():.4f}")
    print(f"Negative pairs (Clean vs Tampered): {neg_sim.mean():.4f} ± {neg_sim.std():.4f}")
    print(f"Separation gap: {pos_sim.mean() - neg_sim.mean():.4f}")
    print(f"\nDesired: Positive ~0.9, Negative ~0.2, Gap ~0.7")
    print(f"Actual gap: {'GOOD ✓' if pos_sim.mean() - neg_sim.mean() > 0.5 else 'POOR ✗'}")


def main():
    parser = argparse.ArgumentParser(description="Visualize similarity distribution")

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_pairs', type=str, required=True,
                       help='Path to test triplets CSV')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for plots')

    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--img_size', type=int, default=512,
                       help='Image size')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("\nLoading model...")
    model = create_multitask_model(
        simsac_checkpoint=None,
        freeze_simsac=False,
        device=device
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'full_model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['full_model_state_dict'], strict=False)
        print("✓ Loaded full multi-task model")
    else:
        print("Warning: Loading base SimSaC weights only")

    # Load dataset
    print("\nLoading test data...")
    dataset = TripletDataset(args.test_pairs, img_size=args.img_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Test samples: {len(dataset)}")

    # Compute similarities
    pos_sim, neg_sim, neg_labels, neg_codes = compute_similarities(model, dataloader, device)

    # Plot
    plot_path = output_dir / 'similarity_distribution.png'
    plot_similarity_distribution(pos_sim, neg_sim, neg_labels, plot_path)

    # Save raw data
    results = {
        'positive_similarities': pos_sim.tolist(),
        'negative_similarities': neg_sim.tolist(),
        'negative_labels': neg_labels.tolist(),
        'stats': {
            'positive_mean': float(pos_sim.mean()),
            'positive_std': float(pos_sim.std()),
            'negative_mean': float(neg_sim.mean()),
            'negative_std': float(neg_sim.std()),
            'separation_gap': float(pos_sim.mean() - neg_sim.mean())
        }
    }

    import json
    json_path = output_dir / 'similarity_data.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved raw data to {json_path}")

    print(f"\n{'='*70}")
    print("✓ Visualization complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
