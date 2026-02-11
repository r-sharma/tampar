"""
Task 5.7: Evaluate Fine-tuned SimSaC on Validation Set

Comprehensive evaluation including:
- Pair classification metrics (accuracy, precision, recall, F1)
- Feature similarity analysis
- t-SNE visualization of embeddings
- Confusion matrix
- Comparison of positive vs negative pair similarities

Supports both full UV map pairs and surface-level pairs.

Usage:
    # Auto-detect surface-level pairs
    python evaluate_simsac.py \
        --checkpoint /content/outputs/training/best_model.pth \
        --data_dir /content/tampar/data/tampar_sample/contrastive_pairs_surface \
        --output_dir /content/outputs/evaluation

    # Explicitly specify pair file
    python evaluate_simsac.py \
        --checkpoint /content/outputs/training/best_model.pth \
        --val_pairs /path/to/val_pairs_surface_level.pkl \
        --output_dir /content/outputs/evaluation
"""

import os
import argparse
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from contrastive_dataset import ContrastivePairsDataset
from simsac_contrastive_model import SimSaCContrastive


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    print(f"\n{'='*70}")
    print("Loading Trained Model")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model architecture (same as training)
    # We need to load the full SimSaC first
    weights_path = '/content/tampar/src/simsac/weight/synth_then_joint_synth_changesim.pth'
    
    # Import and create base model
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
    
    # Create contrastive wrapper
    model = SimSaCContrastive(
        simsac_model=simsac,
        projection_dim=128,
        freeze_backbone=False  # Load with all params
    )

    # Load trained weights (TAMPAR-compatible format)
    # The checkpoint contains only base SimSaC weights (no "simsac." prefix)
    # We need to add the prefix back to load into our wrapped model
    simsac_state_dict = checkpoint['state_dict']

    # Add "simsac." prefix to all keys
    wrapped_state_dict = {}
    for key, value in simsac_state_dict.items():
        wrapped_state_dict[f'simsac.{key}'] = value

    # Load into the wrapped model (strict=False allows missing projection_head keys)
    model.load_state_dict(wrapped_state_dict, strict=False)

    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")

    # Print training history if available (fine-tuned models)
    if 'history' in checkpoint:
        print(f"  Training loss: {checkpoint['history']['train_loss'][-1]:.4f}")
        print(f"  Val loss: {checkpoint['history']['val_loss'][-1]:.4f}")
    else:
        print(f"  Using pre-trained checkpoint (no training history)")

    return model, checkpoint


def extract_embeddings(model, dataloader, device='cuda'):
    """
    Extract embeddings and labels from dataset.
    
    Returns:
        embeddings1: First image embeddings [N, 128]
        embeddings2: Second image embeddings [N, 128]
        labels: Pair labels [N]
        similarities: Cosine similarities [N]
    """
    print(f"\n{'='*70}")
    print("Extracting Embeddings")
    print(f"{'='*70}")
    
    embeddings1_list = []
    embeddings2_list = []
    labels_list = []
    similarities_list = []
    
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Extracting"):
            # Handle both 3-item and 4-item unpacking (with/without is_adversarial)
            if len(batch_data) == 4:
                img1, img2, labels, is_adversarial = batch_data
            else:
                img1, img2, labels = batch_data

            img1 = img1.to(device)
            img2 = img2.to(device)
            
            # Get embeddings
            z1, z2 = model(img1, img2)
            
            # Compute similarity
            similarity = F.cosine_similarity(z1, z2, dim=1)
            
            # Store
            embeddings1_list.append(z1.cpu().numpy())
            embeddings2_list.append(z2.cpu().numpy())
            labels_list.append(labels.numpy())
            similarities_list.append(similarity.cpu().numpy())
    
    # Concatenate
    embeddings1 = np.concatenate(embeddings1_list, axis=0)
    embeddings2 = np.concatenate(embeddings2_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    similarities = np.concatenate(similarities_list, axis=0)
    
    print(f"✓ Extracted {len(labels)} pairs")
    print(f"  Positive pairs: {(labels == 1).sum()}")
    print(f"  Negative pairs: {(labels == 0).sum()}")
    
    return embeddings1, embeddings2, labels, similarities


def compute_metrics(similarities, labels, threshold=0.5):
    """
    Compute classification metrics.
    
    Predicts positive if similarity > threshold.
    """
    print(f"\n{'='*70}")
    print("Computing Classification Metrics")
    print(f"{'='*70}")
    print(f"Classification threshold: {threshold}")
    
    # Predict labels based on threshold
    predictions = (similarities > threshold).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    print(f"\n{'='*70}")
    print("Results")
    print(f"{'='*70}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(labels, predictions, 
                                target_names=['Negative', 'Positive'],
                                digits=4))
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': predictions,
        'threshold': threshold
    }
    
    return metrics


def analyze_similarities(similarities, labels, output_dir):
    """Analyze similarity distributions."""
    print(f"\n{'='*70}")
    print("Analyzing Similarity Distributions")
    print(f"{'='*70}")
    
    pos_similarities = similarities[labels == 1]
    neg_similarities = similarities[labels == 0]
    
    print(f"\nPositive Pairs (should be high similarity):")
    print(f"  Mean: {pos_similarities.mean():.4f}")
    print(f"  Std:  {pos_similarities.std():.4f}")
    print(f"  Min:  {pos_similarities.min():.4f}")
    print(f"  Max:  {pos_similarities.max():.4f}")
    
    print(f"\nNegative Pairs (should be low similarity):")
    print(f"  Mean: {neg_similarities.mean():.4f}")
    print(f"  Std:  {neg_similarities.std():.4f}")
    print(f"  Min:  {neg_similarities.min():.4f}")
    print(f"  Max:  {neg_similarities.max():.4f}")
    
    # Separation
    separation = pos_similarities.mean() - neg_similarities.mean()
    print(f"\nSeparation (higher is better): {separation:.4f}")
    
    # Plot distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(pos_similarities, bins=30, alpha=0.7, label='Positive', color='green')
    axes[0].hist(neg_similarities, bins=30, alpha=0.7, label='Negative', color='red')
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Similarity Distributions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data = [pos_similarities, neg_similarities]
    axes[1].boxplot(data, labels=['Positive', 'Negative'])
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title('Similarity Box Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'similarity_distributions.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: {plot_path}")
    
    return {
        'pos_mean': pos_similarities.mean(),
        'pos_std': pos_similarities.std(),
        'neg_mean': neg_similarities.mean(),
        'neg_std': neg_similarities.std(),
        'separation': separation
    }


def plot_confusion_matrix(labels, predictions, output_dir):
    """Plot confusion matrix."""
    print(f"\nPlotting Confusion Matrix...")
    
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Pair Classification')
    
    plot_path = output_dir / 'confusion_matrix.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {plot_path}")


def visualize_tsne(embeddings1, embeddings2, labels, output_dir):
    """Create t-SNE visualization of embeddings."""
    print(f"\n{'='*70}")
    print("Creating t-SNE Visualization")
    print(f"{'='*70}")
    
    # Combine all embeddings
    all_embeddings = np.concatenate([embeddings1, embeddings2], axis=0)
    
    # Create labels for visualization
    # 0: negative pair img1, 1: negative pair img2, 2: positive pair img1, 3: positive pair img2
    viz_labels = np.zeros(len(all_embeddings))
    n_pairs = len(labels)
    
    for i in range(n_pairs):
        if labels[i] == 0:  # Negative pair
            viz_labels[i] = 0  # img1
            viz_labels[i + n_pairs] = 1  # img2
        else:  # Positive pair
            viz_labels[i] = 2  # img1
            viz_labels[i + n_pairs] = 3  # img2
    
    print(f"Computing t-SNE for {len(all_embeddings)} embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: By pair type
    colors = ['red', 'pink', 'green', 'lightgreen']
    labels_text = ['Neg-Img1', 'Neg-Img2', 'Pos-Img1', 'Pos-Img2']
    
    for i, (color, label) in enumerate(zip(colors, labels_text)):
        mask = viz_labels == i
        axes[0].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=color, label=label, alpha=0.6, s=50)
    
    axes[0].set_xlabel('t-SNE Dimension 1')
    axes[0].set_ylabel('t-SNE Dimension 2')
    axes[0].set_title('t-SNE: Embedding Space (Colored by Pair Type)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Positive vs Negative only
    pos_mask = (viz_labels == 2) | (viz_labels == 3)
    neg_mask = (viz_labels == 0) | (viz_labels == 1)
    
    axes[1].scatter(embeddings_2d[neg_mask, 0], embeddings_2d[neg_mask, 1],
                   c='red', label='Negative Pairs', alpha=0.6, s=50)
    axes[1].scatter(embeddings_2d[pos_mask, 0], embeddings_2d[pos_mask, 1],
                   c='green', label='Positive Pairs', alpha=0.6, s=50)
    
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')
    axes[1].set_title('t-SNE: Positive vs Negative Pairs')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'tsne_visualization.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {plot_path}")


def save_results(metrics, similarity_stats, output_dir):
    """Save all results to JSON."""
    import json
    
    results = {
        'classification_metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1']),
            'threshold': float(metrics['threshold'])
        },
        'similarity_statistics': {
            'positive_mean': float(similarity_stats['pos_mean']),
            'positive_std': float(similarity_stats['pos_std']),
            'negative_mean': float(similarity_stats['neg_mean']),
            'negative_std': float(similarity_stats['neg_std']),
            'separation': float(similarity_stats['separation'])
        }
    }
    
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned SimSaC")

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory with pair files (auto-detects surface-level or full UV map pairs)')
    parser.add_argument('--val_pairs', type=str, default=None,
                       help='Path to validation pairs file (auto-detected if not specified)')
    parser.add_argument('--output_dir', type=str, default='/content/outputs/evaluation',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Similarity threshold for classification')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*70}")
    print("SimSaC Contrastive Learning - Evaluation (Task 5.7)")
    print(f"{'='*70}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Threshold: {args.threshold}")
    print(f"Device: {device}")

    # Load model
    model, checkpoint = load_model(args.checkpoint, device)

    # Auto-detect validation pairs if not specified
    if args.val_pairs is None:
        if args.data_dir is None:
            raise ValueError("Must specify either --data_dir or --val_pairs")

        data_dir = Path(args.data_dir)
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

    print(f"Data: {val_path}")
    print(f"Output directory: {output_dir}")

    # Load validation data
    val_dataset = ContrastivePairsDataset(str(val_path))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Extract embeddings
    embeddings1, embeddings2, labels, similarities = extract_embeddings(
        model, val_loader, device
    )
    
    # Compute metrics
    metrics = compute_metrics(similarities, labels, threshold=args.threshold)
    
    # Analyze similarities
    similarity_stats = analyze_similarities(similarities, labels, output_dir)
    
    # Plot confusion matrix
    plot_confusion_matrix(labels, metrics['predictions'], output_dir)
    
    # t-SNE visualization
    visualize_tsne(embeddings1, embeddings2, labels, output_dir)
    
    # Save results
    save_results(metrics, similarity_stats, output_dir)
    
    print(f"\n{'='*70}")
    print("✓ Evaluation Complete!")
    print(f"{'='*70}")
    print(f"\nAll results saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - evaluation_results.json")
    print(f"  - similarity_distributions.png")
    print(f"  - confusion_matrix.png")
    print(f"  - tsne_visualization.png")


if __name__ == "__main__":
    main()
