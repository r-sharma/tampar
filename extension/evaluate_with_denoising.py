"""
Evaluate adversarial attack effectiveness WITH denoising defense.

This script applies denoising (bilateral filter) to adversarial UV maps before
computing similarity scores, testing if denoising can recover accuracy.

Usage:
    # Test denoising defense against Strategy 1 attack
    python extension/evaluate_with_denoising.py \
        --adversarial_dir /path/to/adversarial_strategy1_carpet \
        --uvmaps_dir /path/to/uvmaps \
        --output_csv simscores_adversarial_denoised.csv \
        --denoising_method bilateral \
        --denoising_strength medium

    # Then evaluate
    python src/tools/predict_tampering_adversarial_eval.py \
        --clean_csv simscores_validation_carpet_only.csv \
        --adversarial_csv simscores_adversarial_denoised.csv \
        --gt_keypoints \
        --output_csv results_with_denoising.csv
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())

import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.tampering.compare import CompareType, compute_uvmap_similarity
from src.tampering.parcel import PATCH_ORDER


def denoise_uvmap(uvmap, method='bilateral', strength='medium'):
    """
    Apply denoising to UV map to remove adversarial perturbations.

    Args:
        uvmap: UV map as numpy array [H, W, C]
        method: Denoising method ('bilateral', 'nlm', 'median', 'gaussian')
        strength: Denoising strength ('light', 'medium', 'heavy')

    Returns:
        Denoised UV map
    """
    # Strength parameters
    strength_params = {
        'light': {'bilateral': (3, 30, 30), 'nlm': 5, 'median': 3, 'gaussian': (3, 0.5)},
        'medium': {'bilateral': (5, 50, 50), 'nlm': 10, 'median': 5, 'gaussian': (5, 1.0)},
        'heavy': {'bilateral': (7, 75, 75), 'nlm': 15, 'median': 7, 'gaussian': (7, 1.5)},
    }

    params = strength_params.get(strength, strength_params['medium'])

    if method == 'bilateral':
        # Bilateral filter: smooths while preserving edges
        d, sigmaColor, sigmaSpace = params['bilateral']
        denoised = cv2.bilateralFilter(uvmap, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

    elif method == 'nlm':
        # Non-local means denoising
        h = params['nlm']
        denoised = cv2.fastNlMeansDenoisingColored(uvmap.astype(np.uint8), h=h)

    elif method == 'median':
        # Median filter
        ksize = params['median']
        denoised = cv2.medianBlur(uvmap.astype(np.uint8), ksize)

    elif method == 'gaussian':
        # Gaussian blur
        ksize, sigma = params['gaussian']
        ksize = (ksize, ksize)
        denoised = cv2.GaussianBlur(uvmap, ksize, sigma)

    else:
        raise ValueError(f"Unknown denoising method: {method}")

    return denoised


def compute_similarity_with_denoising(
    field_path,
    reference_path,
    compare_types,
    denoising_method='bilateral',
    denoising_strength='medium'
):
    """
    Compute similarity scores with denoising preprocessing.

    Args:
        field_path: Path to field UV map (possibly adversarial)
        reference_path: Path to reference UV map
        compare_types: List of compare types to use
        denoising_method: Denoising method
        denoising_strength: Denoising strength

    Returns:
        Dictionary of similarity scores by sideface and metric
    """
    # Load images
    field_img = cv2.imread(str(field_path))
    field_img = cv2.cvtColor(field_img, cv2.COLOR_BGR2RGB)

    reference_img = cv2.imread(str(reference_path))
    reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

    # Apply denoising to field image (adversarial)
    field_img_denoised = denoise_uvmap(field_img, method=denoising_method, strength=denoising_strength)

    # Compute similarity scores for each compare type
    all_scores = {}
    for compare_type in compare_types:
        similarities = compute_uvmap_similarity(
            field_img_denoised,
            reference_img,
            output_path=None,
            compare_type=compare_type,
            visualize=False
        )
        all_scores[compare_type] = similarities

    return all_scores


def process_adversarial_dataset(
    adversarial_dir,
    uvmaps_dir,
    output_csv,
    compare_types,
    denoising_method='bilateral',
    denoising_strength='medium'
):
    """
    Process adversarial dataset with denoising and compute similarity scores.

    Args:
        adversarial_dir: Directory with adversarial UV maps
        uvmaps_dir: Directory with reference UV maps
        output_csv: Output CSV file
        compare_types: List of compare types
        denoising_method: Denoising method
        denoising_strength: Denoising strength
    """
    adversarial_path = Path(adversarial_dir)
    uvmaps_path = Path(uvmaps_dir)

    print(f"\n{'='*80}")
    print(f"Computing similarity scores WITH denoising")
    print(f"{'='*80}")
    print(f"Adversarial directory: {adversarial_dir}")
    print(f"Reference directory: {uvmaps_dir}")
    print(f"Denoising method: {denoising_method}")
    print(f"Denoising strength: {denoising_strength}")
    print(f"Compare types: {compare_types}")

    # Find all adversarial UV maps
    results = []

    # Process each background folder
    background_folders = [d for d in adversarial_path.iterdir() if d.is_dir()]

    for background_folder in background_folders:
        background_name = background_folder.name
        print(f"\nProcessing: {background_name}")

        # Find all UV maps
        uvmap_files = list(background_folder.glob("*_uvmap_*.png"))

        for uvmap_file in tqdm(uvmap_files, desc=f"  {background_name}"):
            # Extract parcel ID
            filename_parts = uvmap_file.stem.split('_')
            parcel_id = int(filename_parts[1])

            # Determine if gt or pred
            gt_keypoints = "uvmap_gt" in uvmap_file.name

            # Find reference UV map
            ref_pattern = f"id_{str(parcel_id).zfill(2)}_*.png"
            ref_matches = list(uvmaps_path.glob(ref_pattern))

            if not ref_matches:
                print(f"    Warning: No reference found for parcel {parcel_id}")
                continue

            reference_path = ref_matches[0]

            # Compute similarity scores with denoising
            try:
                all_scores = compute_similarity_with_denoising(
                    uvmap_file,
                    reference_path,
                    compare_types,
                    denoising_method=denoising_method,
                    denoising_strength=denoising_strength
                )

                # Store results
                rel_path = uvmap_file.relative_to(adversarial_path)

                for compare_type in compare_types:
                    similarities = all_scores[compare_type]

                    for sideface_name, metrics in similarities.items():
                        result = {
                            "dataset_split": "adversarial_denoised",
                            "parcel_id": parcel_id,
                            "view": rel_path.as_posix(),
                            "gt_keypoints": gt_keypoints,
                            "compare_type": compare_type,
                            "sideface_name": sideface_name,
                            "background": background_name,
                            "tampering": "",  # Will be filled from mapping
                            "tampered": False,  # Will be filled from mapping
                        }
                        result.update(metrics)
                        results.append(result)

            except Exception as e:
                print(f"    Error processing {uvmap_file.name}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Load tampering mapping if available
    tampering_csv = Path("data/tampar_sample/tampering_mapping.csv")
    if tampering_csv.exists():
        df_tampering = pd.read_csv(tampering_csv)
        # Merge tampering labels
        # This requires matching logic based on your dataset structure

    # Save results
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved results to: {output_csv}")
    print(f"Total rows: {len(df)}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute similarity scores with denoising defense",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--adversarial_dir', type=str, required=True,
                       help='Directory with adversarial UV maps')
    parser.add_argument('--uvmaps_dir', type=str, required=True,
                       help='Directory with reference UV maps')
    parser.add_argument('--output_csv', type=str, required=True,
                       help='Output CSV file for similarity scores')

    parser.add_argument('--denoising_method', type=str, default='bilateral',
                       choices=['bilateral', 'nlm', 'median', 'gaussian'],
                       help='Denoising method (default: bilateral)')
    parser.add_argument('--denoising_strength', type=str, default='medium',
                       choices=['light', 'medium', 'heavy'],
                       help='Denoising strength (default: medium)')

    parser.add_argument('--compare_types', type=str, nargs='+',
                       default=['plain', 'simsac', 'canny', 'laplacian', 'meanchannel'],
                       help='Compare types to use')

    args = parser.parse_args()

    process_adversarial_dataset(
        adversarial_dir=args.adversarial_dir,
        uvmaps_dir=args.uvmaps_dir,
        output_csv=args.output_csv,
        compare_types=args.compare_types,
        denoising_method=args.denoising_method,
        denoising_strength=args.denoising_strength
    )


if __name__ == "__main__":
    main()
