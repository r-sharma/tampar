
import os
import argparse
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F

from contrastive_dataset import ContrastivePairsDataset
from simsac_contrastive_model import SimSaCContrastive


def load_model(checkpoint_path, device='cuda'):
    print("Loading Trained Model")
    print(f"Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model architecture
    weights_path = '/content/tampar/src/simsac/weight/synth_then_joint_synth_changesim.pth'

    import sys
    sys.path.insert(0, "/content/tampar")
    from src.simsac.models.our_models.SimSaC import SimSaC_Model

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

    model = SimSaCContrastive(
        simsac_model=simsac,
        projection_dim=128,
        freeze_backbone=False
    )

    simsac_state_dict = checkpoint['state_dict']

    # Add "simsac." prefix to all keys
    wrapped_state_dict = {}
    for key, value in simsac_state_dict.items():
        wrapped_state_dict[f'simsac.{key}'] = value

    # Load into the wrapped model (strict=False allows missing projection_head keys)
    model.load_state_dict(wrapped_state_dict, strict=False)

    model = model.to(device)
    model.eval()

    print(f" Model loaded from epoch {checkpoint['epoch']}")

    return model


def extract_detailed_results(model, pairs_path, device='cuda', batch_size=16):
    print("Extracting Detailed Results")

    # Load pairs
    with open(pairs_path, 'rb') as f:
        pairs = pickle.load(f)

    print(f"Loaded {len(pairs)} pairs from {pairs_path}")

    # Create dataset
    dataset = ContrastivePairsDataset(str(pairs_path))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # Extract similarities
    all_similarities = []

    model.eval()
    with torch.no_grad():
        for img1, img2, labels in tqdm(dataloader, desc="Computing similarities"):
            img1 = img1.to(device)
            img2 = img2.to(device)

            # Get embeddings
            z1, z2 = model(img1, img2)

            # Compute similarity
            similarity = F.cosine_similarity(z1, z2, dim=1)

            all_similarities.append(similarity.cpu().numpy())

    # Concatenate
    all_similarities = np.concatenate(all_similarities, axis=0)

    # Build detailed results dataframe
    results = []
    for idx, (pair, similarity) in enumerate(zip(pairs, all_similarities)):
        # Extract metadata
        label = pair['label']
        pair_type = pair.get('pair_type', 'unknown')
        parcel_id = pair.get('parcel_id', 'unknown')
        surface_name = pair.get('surface_name', 'unknown')

        metadata = pair.get('metadata', {})
        ref_file = metadata.get('ref_file', 'unknown')
        field_file = metadata.get('pred_file', metadata.get('field_file', 'unknown'))

        results.append({
            'idx': idx,
            'similarity': float(similarity),
            'label': int(label),
            'parcel_id': parcel_id,
            'surface_name': surface_name,
            'pair_type': pair_type,
            'ref_file': ref_file,
            'field_file': field_file
        })

    df = pd.DataFrame(results)

    print(f" Extracted {len(df)} pairs with metadata")

    return df, pairs


def find_outliers(df, top_n=10):
    print("Finding Outliers")

    # Split by label
    positive_df = df[df['label'] == 1].copy()
    negative_df = df[df['label'] == 0].copy()

    # Find problematic negative pairs (high similarity)
    negative_outliers = negative_df.nlargest(top_n, 'similarity')

    # Find problematic positive pairs (low similarity)
    positive_outliers = positive_df.nsmallest(top_n, 'similarity')

    print(f"Top {top_n} Negative Pairs with HIGHEST Similarity (should be low)")
    print(negative_outliers[['idx', 'similarity', 'parcel_id', 'surface_name', 'pair_type', 'field_file']].to_string(index=False))

    print(f"Top {top_n} Positive Pairs with LOWEST Similarity (should be high)")
    print(positive_outliers[['idx', 'similarity', 'parcel_id', 'surface_name', 'pair_type', 'field_file']].to_string(index=False))

    return negative_outliers, positive_outliers


def visualize_outlier_pair(pairs, idx, output_dir):
    pair = pairs[idx]

    # Get images
    surface1 = pair.get('surface1', pair.get('image1'))
    surface2 = pair.get('surface2', pair.get('image2'))

    if surface1 is None or surface2 is None:
        print(f" Cannot visualize pair {idx}: missing image data")
        return

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(surface1)
    axes[0].set_title(f"Surface 1\n{pair.get('metadata', {}).get('ref_file', 'unknown')}")
    axes[0].axis('off')

    axes[1].imshow(surface2)
    axes[1].set_title(f"Surface 2\n{pair.get('metadata', {}).get('pred_file', 'unknown')}")
    axes[1].axis('off')

    # Add metadata
    label_text = "POSITIVE" if pair['label'] == 1 else "NEGATIVE"
    fig.suptitle(f"Pair {idx} - Label: {label_text}\n"
                 f"Parcel: {pair.get('parcel_id', 'unknown')} | "
                 f"Surface: {pair.get('surface_name', 'unknown')} | "
                 f"Type: {pair.get('pair_type', 'unknown')}",
                 fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = output_dir / f'pair_{idx}_visualization.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f" Saved visualization: {output_path}")


def save_detailed_results(df, output_dir):
    csv_path = output_dir / 'detailed_pair_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n Saved detailed results: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Investigate outlier pairs from evaluation")

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--val_pairs', type=str, required=True,
                       help='Path to validation pairs pickle file')
    parser.add_argument('--top_n', type=int, default=10,
                       help='Number of top outliers to show')
    parser.add_argument('--visualize_top', type=int, default=5,
                       help='Number of top outliers to visualize')
    parser.add_argument('--output_dir', type=str, default='/content/outputs/outlier_analysis',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("SimSaC Outlier Investigation")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Val pairs: {args.val_pairs}")
    print(f"Device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Extract detailed results
    df, pairs = extract_detailed_results(
        model,
        args.val_pairs,
        device=device,
        batch_size=args.batch_size
    )

    # Save detailed results
    save_detailed_results(df, output_dir)

    # Find outliers
    negative_outliers, positive_outliers = find_outliers(df, top_n=args.top_n)

    # Save outlier subsets
    negative_outliers.to_csv(output_dir / 'negative_outliers.csv', index=False)
    positive_outliers.to_csv(output_dir / 'positive_outliers.csv', index=False)

    print(f"\n Saved outlier CSVs to {output_dir}")

    # Visualize top outliers
    print(f"Visualizing Top {args.visualize_top} Outliers")

    print("\nNegative pair outliers (high similarity):")
    for i, row in negative_outliers.head(args.visualize_top).iterrows():
        idx = row['idx']
        print(f"  Visualizing pair {idx} (similarity={row['similarity']:.4f})")
        visualize_outlier_pair(pairs, idx, output_dir)

    print("\nPositive pair outliers (low similarity):")
    for i, row in positive_outliers.head(args.visualize_top).iterrows():
        idx = row['idx']
        print(f"  Visualizing pair {idx} (similarity={row['similarity']:.4f})")
        visualize_outlier_pair(pairs, idx, output_dir)

    # Summary statistics
    print("Summary Statistics")

    pos_df = df[df['label'] == 1]
    neg_df = df[df['label'] == 0]

    print(f"\nPositive Pairs:")
    print(f"  Count: {len(pos_df)}")
    print(f"  Mean similarity: {pos_df['similarity'].mean():.4f}")
    print(f"  Min similarity: {pos_df['similarity'].min():.4f}")
    print(f"  Max similarity: {pos_df['similarity'].max():.4f}")

    print(f"\nNegative Pairs:")
    print(f"  Count: {len(neg_df)}")
    print(f"  Mean similarity: {neg_df['similarity'].mean():.4f}")
    print(f"  Min similarity: {neg_df['similarity'].min():.4f}")
    print(f"  Max similarity: {neg_df['similarity'].max():.4f}")

    print(" Outlier Investigation Complete!")
    print(f"\nAll results saved to: {output_dir}")
    print(f"  - negative_outliers.csv (top {args.top_n} negative pairs with high similarity)")
    print(f"  - positive_outliers.csv (top {args.top_n} positive pairs with low similarity)")
    print(f"  - pair_*_visualization.png (visualizations of top outliers)")


if __name__ == "__main__":
    main()
