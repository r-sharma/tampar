"""
Task 5.8: Extract Improved Similarity Features for Classification

Extract features from fine-tuned SimSaC for downstream tampering detection.
Compares baseline TAMPAR features vs contrastive-learned features.

Usage:
    python extract_features_for_tampering.py \
        --checkpoint /content/outputs/training/best_model.pth \
        --data_root /content/tampar/data/tampar_sample \
        --output_dir /content/outputs/features
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pickle
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from simsac_contrastive_model import SimSaCContrastive


def load_model(checkpoint_path, device='cuda'):
    """Load trained contrastive model."""
    print(f"\n{'='*70}")
    print("Loading Fine-tuned Model")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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
    
    model = SimSaCContrastive(
        simsac_model=simsac,
        projection_dim=128,
        freeze_backbone=False
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    
    return model


def load_uv_maps(data_root, split='validation'):
    """
    Load UV maps from TAMPAR dataset.
    
    Returns:
        Dictionary mapping parcel_id -> list of UV map paths
    """
    print(f"\n{'='*70}")
    print(f"Loading UV Maps from {split.upper()}")
    print(f"{'='*70}")
    
    data_root = Path(data_root)
    split_dir = data_root / split
    uvmaps_dir = data_root / 'uvmaps'
    
    uv_maps = {}
    
    # Load reference UV maps
    if uvmaps_dir.exists():
        ref_files = list(uvmaps_dir.glob('*.png'))
        print(f"Found {len(ref_files)} reference UV maps")
        for ref_file in ref_files:
            parcel_id = ref_file.stem  # e.g., "id_01_uvmap" or just "0"
            uv_maps[f"{parcel_id}_ref"] = str(ref_file)
    
    # Load predicted UV maps
    if split_dir.exists():
        pred_files = list(split_dir.glob('**/*_uvmap_pred.png'))
        print(f"Found {len(pred_files)} predicted UV maps")
        
        for pred_file in pred_files:
            # Extract parcel ID from filename
            filename = pred_file.stem.replace('_uvmap_pred', '')
            # For sample: id_01_20230516_142710 -> id_01
            parts = filename.split('_')
            if len(parts) >= 2:
                parcel_id = f"{parts[0]}_{parts[1]}"
            else:
                parcel_id = parts[0]
            
            key = f"{parcel_id}_{pred_file.name}"
            uv_maps[key] = str(pred_file)
    
    print(f"✓ Loaded {len(uv_maps)} UV map paths")
    
    return uv_maps


def extract_contrastive_features(model, uv_maps, device='cuda'):
    """
    Extract contrastive features from UV maps.
    
    Returns:
        features_dict: Dictionary mapping UV map key -> feature vector (128-dim)
    """
    print(f"\n{'='*70}")
    print("Extracting Contrastive Features")
    print(f"{'='*70}")
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    features_dict = {}
    
    model.eval()
    with torch.no_grad():
        for key, uv_path in tqdm(uv_maps.items(), desc="Extracting features"):
            # Load image
            img = Image.open(uv_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Extract embedding
            z, _ = model(img_tensor, img_tensor)  # Pass same image twice
            z = z[0].cpu().numpy()  # Take first (they're identical)
            
            features_dict[key] = z
    
    print(f"✓ Extracted features for {len(features_dict)} UV maps")
    print(f"  Feature dimension: {list(features_dict.values())[0].shape[0]}")
    
    return features_dict


def compute_pairwise_features(features_dict, output_dir):
    """
    Compute pairwise similarity features between reference and predicted UV maps.
    
    This creates the feature vectors for tampering detection classifier.
    """
    print(f"\n{'='*70}")
    print("Computing Pairwise Similarity Features")
    print(f"{'='*70}")
    
    # Separate reference and predicted features
    ref_features = {k: v for k, v in features_dict.items() if '_ref' in k}
    pred_features = {k: v for k, v in features_dict.items() if '_ref' not in k}
    
    print(f"Reference UV maps: {len(ref_features)}")
    print(f"Predicted UV maps: {len(pred_features)}")
    
    # Create feature pairs
    feature_pairs = []
    pair_info = []
    
    for pred_key, pred_feat in pred_features.items():
        # Extract parcel ID from predicted key
        # e.g., "id_01_id_01_20230516_142710_uvmap_pred.png" -> "id_01"
        parts = pred_key.split('_')
        parcel_id = f"{parts[0]}_{parts[1]}"
        
        # Find matching reference
        ref_key = None
        for rk in ref_features.keys():
            if parcel_id in rk:
                ref_key = rk
                break
        
        if ref_key:
            ref_feat = ref_features[ref_key]
            
            # Compute similarity features
            # 1. Cosine similarity
            cos_sim = np.dot(pred_feat, ref_feat) / (
                np.linalg.norm(pred_feat) * np.linalg.norm(ref_feat)
            )
            
            # 2. L2 distance
            l2_dist = np.linalg.norm(pred_feat - ref_feat)
            
            # 3. Concatenated features (can be used directly)
            concat_feat = np.concatenate([pred_feat, ref_feat])
            
            # 4. Element-wise absolute difference
            diff_feat = np.abs(pred_feat - ref_feat)
            
            # Store features
            features = {
                'cosine_similarity': cos_sim,
                'l2_distance': l2_dist,
                'concatenated': concat_feat,  # 256-dim (128 + 128)
                'difference': diff_feat,       # 128-dim
            }
            
            feature_pairs.append(features)
            pair_info.append({
                'parcel_id': parcel_id,
                'ref_key': ref_key,
                'pred_key': pred_key
            })
    
    print(f"✓ Created {len(feature_pairs)} feature pairs")
    
    # Save features
    features_path = output_dir / 'contrastive_features.pkl'
    with open(features_path, 'wb') as f:
        pickle.dump({
            'features': feature_pairs,
            'info': pair_info
        }, f)
    
    print(f"✓ Saved: {features_path}")
    
    # Also save as numpy for easy loading
    # Extract different feature types
    cos_sims = np.array([f['cosine_similarity'] for f in feature_pairs])
    l2_dists = np.array([f['l2_distance'] for f in feature_pairs])
    
    np.savez(
        output_dir / 'contrastive_features.npz',
        cosine_similarities=cos_sims,
        l2_distances=l2_dists,
        pair_info=pair_info
    )
    
    print(f"✓ Saved: {output_dir / 'contrastive_features.npz'}")
    
    return feature_pairs, pair_info


def extract_baseline_features(data_root, output_dir):
    """
    Extract baseline TAMPAR features for comparison.
    
    These would be the original features used by TAMPAR:
    - MS-SSIM
    - CW-SSIM
    - HOG
    - Etc.
    
    For now, this is a placeholder showing the structure.
    """
    print(f"\n{'='*70}")
    print("Extracting Baseline TAMPAR Features (Placeholder)")
    print(f"{'='*70}")
    
    # TODO: Implement baseline feature extraction
    # This would require running TAMPAR's original feature extraction pipeline
    
    print("Note: Baseline feature extraction not implemented")
    print("In full implementation, would extract:")
    print("  - MS-SSIM (Multi-Scale Structural Similarity)")
    print("  - CW-SSIM (Complex Wavelet SSIM)")
    print("  - HOG (Histogram of Oriented Gradients)")
    print("  - MAE (Mean Absolute Error)")
    print("  - etc.")
    
    placeholder_path = output_dir / 'baseline_features_placeholder.txt'
    with open(placeholder_path, 'w') as f:
        f.write("Baseline TAMPAR features would be extracted here\n")
        f.write("Run TAMPAR's original feature extraction pipeline\n")
    
    print(f"✓ Saved placeholder: {placeholder_path}")


def create_summary(features_dict, feature_pairs, output_dir):
    """Create summary of extracted features."""
    print(f"\n{'='*70}")
    print("Creating Summary")
    print(f"{'='*70}")
    
    summary = {
        'total_uv_maps': len(features_dict),
        'feature_pairs': len(feature_pairs),
        'feature_dimension': 128,
        'similarity_statistics': {
            'mean_cosine_sim': float(np.mean([f['cosine_similarity'] for f in feature_pairs])),
            'std_cosine_sim': float(np.std([f['cosine_similarity'] for f in feature_pairs])),
            'mean_l2_dist': float(np.mean([f['l2_distance'] for f in feature_pairs])),
            'std_l2_dist': float(np.std([f['l2_distance'] for f in feature_pairs]))
        }
    }
    
    import json
    summary_path = output_dir / 'feature_extraction_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved: {summary_path}")
    
    print(f"\nSummary:")
    print(f"  Total UV maps: {summary['total_uv_maps']}")
    print(f"  Feature pairs: {summary['feature_pairs']}")
    print(f"  Feature dim: {summary['feature_dimension']}")
    print(f"  Mean cosine similarity: {summary['similarity_statistics']['mean_cosine_sim']:.4f}")
    print(f"  Mean L2 distance: {summary['similarity_statistics']['mean_l2_dist']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract features for tampering detection (Task 5.8)"
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to TAMPAR dataset root')
    parser.add_argument('--split', type=str, default='validation',
                       choices=['validation', 'test'],
                       help='Which split to extract features from')
    parser.add_argument('--output_dir', type=str, default='/content/outputs/features',
                       help='Output directory for features')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print("Feature Extraction for Tampering Detection (Task 5.8)")
    print(f"{'='*70}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {args.data_root}")
    print(f"Split: {args.split}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Load UV maps
    uv_maps = load_uv_maps(args.data_root, args.split)
    
    # Extract contrastive features
    features_dict = extract_contrastive_features(model, uv_maps, device)
    
    # Compute pairwise features
    feature_pairs, pair_info = compute_pairwise_features(features_dict, output_dir)
    
    # Extract baseline features (placeholder)
    extract_baseline_features(args.data_root, output_dir)
    
    # Create summary
    create_summary(features_dict, feature_pairs, output_dir)
    
    print(f"\n{'='*70}")
    print("✓ Feature Extraction Complete!")
    print(f"{'='*70}")
    print(f"\nGenerated files:")
    print(f"  - contrastive_features.pkl (full features)")
    print(f"  - contrastive_features.npz (similarity metrics)")
    print(f"  - feature_extraction_summary.json")
    print(f"\nNext steps:")
    print(f"  1. Load contrastive_features.npz")
    print(f"  2. Combine with baseline TAMPAR features")
    print(f"  3. Train decision tree classifier")
    print(f"  4. Compare performance: baseline vs contrastive vs combined")


if __name__ == "__main__":
    main()
