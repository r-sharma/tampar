"""
Hard Negative Mining for SimSAC Contrastive Learning

This script identifies adversarial tampered surfaces that the fine-tuned model
incorrectly gives high similarity scores to, and creates a focused dataset of
these "hard negatives" for additional training.

Usage:
    python extension/mine_hard_negatives.py \
        --checkpoint /path/to/phase2_best.pth \
        --pairs_csv /path/to/pairs.csv \
        --output_csv hard_negatives.csv \
        --threshold 0.90
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())

import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from extension.simsac_contrastive_model import SimSaCContrastive


def compute_pair_similarity(model, image1_path, image2_path, device):
    """
    Compute similarity score for a pair of images.

    Returns:
        similarity: Float similarity score (0-1)
    """
    # Load images
    img1 = Image.open(image1_path).convert('RGB')
    img2 = Image.open(image2_path).convert('RGB')

    # Convert to tensors and normalize
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    img1_tensor = transform(img1).unsqueeze(0).to(device)
    img2_tensor = transform(img2).unsqueeze(0).to(device)

    # Get embeddings
    with torch.no_grad():
        emb1 = model(img1_tensor)
        emb2 = model(img2_tensor)

        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()

    return similarity


def mine_hard_negatives(checkpoint_path, pairs_csv, output_csv, threshold=0.90, top_k=None):
    """
    Mine hard negative pairs: adversarial tampered surfaces with high similarity.

    Args:
        checkpoint_path: Path to fine-tuned SimSAC checkpoint
        pairs_csv: Path to pairs CSV file
        output_csv: Path to save hard negatives CSV
        threshold: Similarity threshold (negatives above this are "hard")
        top_k: Optional - only return top K hardest negatives
    """
    print("=" * 80)
    print("HARD NEGATIVE MINING")
    print("=" * 80)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nLoading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SimSaCContrastive(backbone='vgg16', pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ Model loaded on {device}")

    # Load pairs
    print(f"\nLoading pairs from: {pairs_csv}")
    pairs_df = pd.read_csv(pairs_csv)

    # Filter for negative pairs only
    negative_pairs = pairs_df[pairs_df['label'] == 0].copy()
    print(f"✓ Found {len(negative_pairs)} negative pairs")

    # Filter for adversarial tampered pairs
    adv_tampered = negative_pairs[
        (negative_pairs['pair_type'] == 'clean_reference_vs_adversarial_tampered') |
        (negative_pairs['image2_path'].str.contains('adv', na=False) &
         negative_pairs['pair_type'].str.contains('tampered', na=False))
    ].copy()

    print(f"✓ Found {len(adv_tampered)} adversarial tampered negative pairs")

    if len(adv_tampered) == 0:
        print("⚠ No adversarial tampered pairs found!")
        return

    # Compute similarities for all adversarial tampered pairs
    print(f"\nComputing similarities for adversarial tampered pairs...")
    similarities = []

    for idx, row in tqdm(adv_tampered.iterrows(), total=len(adv_tampered)):
        sim = compute_pair_similarity(
            model,
            row['image1_path'],
            row['image2_path'],
            device
        )
        similarities.append(sim)

    adv_tampered['similarity'] = similarities

    # Find hard negatives (high similarity on negative pairs = model failure)
    hard_negatives = adv_tampered[adv_tampered['similarity'] >= threshold].copy()
    hard_negatives = hard_negatives.sort_values('similarity', ascending=False)

    print(f"\n" + "=" * 80)
    print(f"HARD NEGATIVES FOUND")
    print("=" * 80)
    print(f"Total adversarial tampered pairs: {len(adv_tampered)}")
    print(f"Hard negatives (similarity ≥ {threshold}): {len(hard_negatives)}")
    print(f"Percentage: {100 * len(hard_negatives) / len(adv_tampered):.1f}%")

    # Statistics
    print(f"\nSimilarity statistics for adversarial tampered:")
    print(f"  Mean: {adv_tampered['similarity'].mean():.4f}")
    print(f"  Median: {adv_tampered['similarity'].median():.4f}")
    print(f"  Min: {adv_tampered['similarity'].min():.4f}")
    print(f"  Max: {adv_tampered['similarity'].max():.4f}")

    print(f"\nHardest negatives (top 10):")
    print(hard_negatives[['pair_type', 'parcel_id', 'surface_name', 'similarity']].head(10).to_string(index=False))

    # Optionally limit to top K
    if top_k and len(hard_negatives) > top_k:
        print(f"\n⚠ Limiting to top {top_k} hardest negatives")
        hard_negatives = hard_negatives.head(top_k)

    # Save
    hard_negatives.to_csv(output_csv, index=False)
    print(f"\n✓ Saved hard negatives to: {output_csv}")

    # Analysis by attack type
    if 'metadata' in hard_negatives.columns:
        print(f"\nHard negatives by attack type:")
        for attack in ['fgsm', 'pgd', 'cw']:
            count = len(hard_negatives[hard_negatives['image2_path'].str.contains(attack, na=False)])
            pct = 100 * count / len(hard_negatives) if len(hard_negatives) > 0 else 0
            print(f"  {attack.upper()}: {count} pairs ({pct:.1f}%)")

    return hard_negatives


def main():
    parser = argparse.ArgumentParser(
        description="Mine hard negative pairs for SimSAC training"
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to fine-tuned SimSAC checkpoint')
    parser.add_argument('--pairs_csv', type=str, required=True,
                       help='Path to pairs CSV file')
    parser.add_argument('--output_csv', type=str, default='hard_negatives.csv',
                       help='Output path for hard negatives CSV')
    parser.add_argument('--threshold', type=float, default=0.90,
                       help='Similarity threshold for hard negatives (default: 0.90)')
    parser.add_argument('--top_k', type=int, default=None,
                       help='Only keep top K hardest negatives (optional)')

    args = parser.parse_args()

    mine_hard_negatives(
        args.checkpoint,
        args.pairs_csv,
        args.output_csv,
        threshold=args.threshold,
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()
