"""
Visualize how different comparison methods see clean vs adversarial UV maps.

This script shows the change detection outputs for:
1. Reference UV map (from uvmaps folder)
2. Clean field UV map (uvmap_gt)
3. Adversarial field UV map (uvmap_gt with perturbations)

Supported comparison methods:
- simsac: SimSAC neural change detection
- canny: Canny edge detection
- laplacian: Laplacian edge detection

For each parcel surface, it shows:
- Original RGB images (Reference, Clean, Adversarial)
- Change/edge maps (R, C, A) in grayscale

Usage:
    python visualize_simsac_comparison.py \
        --reference_dir /path/to/uvmaps \
        --clean_dir /path/to/validation \
        --adversarial_dir /path/to/adversarial_validation \
        --parcel_id 1 \
        --background carpet \
        --method simsac \
        --output_dir /path/to/output
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.simsac.inference import SimSaC
from src.tampering.utils import get_side_surface_patches
from src.tampering.parcel import PATCH_ORDER
from src.tampering.compare import compare_canny, compare_laplacian


def process_simsac_output(im1, im2, threshold=200, ckpt_path=None):
    """
    Process two images through SimSAC and return change maps.

    This replicates what compare_simsac() does in src/tampering/compare.py

    Args:
        im1: Reference image
        im2: Comparison image (clean or adversarial)
        threshold: Threshold for binary visualization (default 200, range 0-255)
        ckpt_path: Path to SimSAC checkpoint file (optional, uses default if None)

    Returns:
        change1_raw: Raw change map 1 (grayscale 0-255)
        change1_thresh: Thresholded change map 1 (binary 0 or 255)
    """
    simsac = SimSaC.get_instance(ckpt_path=ckpt_path)
    imgs = simsac.inference(im1.astype(np.uint8), im2.astype(np.uint8))

    # Process change1 output
    img = imgs[0]
    img = cv2.resize(img, (im1.shape[1], im1.shape[0]))
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = img_gray.astype(np.uint8)

    # Raw change map
    change1_raw = img_gray

    # Thresholded version
    img_thresh = (img_gray > threshold).astype(np.float32) * 255
    change1_thresh = img_thresh.astype(np.uint8)

    return change1_raw, change1_thresh


def process_canny_output(im1, im2, threshold=200):
    """
    Process two images through Canny edge detection.

    Args:
        im1: Reference image
        im2: Comparison image (clean or adversarial)
        threshold: Threshold for binary visualization (default 200, range 0-255)

    Returns:
        edges_gray: Edge map for im2 (grayscale 0-255)
        edges_thresh: Thresholded edge map for im2 (binary 0 or 255)
    """
    # Use compare_canny from src/tampering/compare.py
    # Returns list of 2 RGB edge maps: [edges1, edges2]
    edge_maps = compare_canny(im1, im2)

    # We only need edges from im2 (the comparison image)
    # Convert RGB edge map to grayscale
    edges_gray = cv2.cvtColor(edge_maps[1].astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Thresholded version
    edges_thresh = (edges_gray > threshold).astype(np.float32) * 255
    edges_thresh = edges_thresh.astype(np.uint8)

    return edges_gray, edges_thresh


def process_laplacian_output(im1, im2, threshold=200):
    """
    Process two images through Laplacian edge detection.

    Args:
        im1: Reference image
        im2: Comparison image (clean or adversarial)
        threshold: Threshold for binary visualization (default 200, range 0-255)

    Returns:
        edges_gray: Edge map for im2 (grayscale 0-255)
        edges_thresh: Thresholded edge map for im2 (binary 0 or 255)
    """
    # Use compare_laplacian from src/tampering/compare.py
    # Returns list of 2 RGB edge maps: [edges1, edges2]
    edge_maps = compare_laplacian(im1, im2)

    # We only need edges from im2 (the comparison image)
    # Convert RGB edge map to grayscale
    edges_gray = cv2.cvtColor(edge_maps[1].astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Thresholded version
    edges_thresh = (edges_gray > threshold).astype(np.float32) * 255
    edges_thresh = edges_thresh.astype(np.uint8)

    return edges_gray, edges_thresh


def visualize_parcel_comparison(
    reference_path,
    clean_path,
    adversarial_path,
    output_path,
    threshold=200,
    method='simsac',
    simsac_ckpt_path=None
):
    """
    Visualize comparison for all surfaces of a parcel using specified method.

    Args:
        reference_path: Path to reference UV map
        clean_path: Path to clean field UV map
        adversarial_path: Path to adversarial field UV map
        output_path: Path to save output visualization
        threshold: Not used (kept for backward compatibility)
        method: Comparison method ('simsac', 'canny', or 'laplacian')
        simsac_ckpt_path: Path to SimSAC checkpoint file (optional, only used with simsac method)

    Shows 6 columns for each surface:
    - Columns 1-3: Original RGB images (Reference, Clean, Adversarial)
    - Columns 4-6: Change/edge maps (R, C, A) - grayscale 0-255

    For each surface patch (top, left, center, right, bottom).
    """
    # Load images
    reference = cv2.imread(str(reference_path))
    reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)

    clean = cv2.imread(str(clean_path))
    clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)

    adversarial = cv2.imread(str(adversarial_path))
    adversarial = cv2.cvtColor(adversarial, cv2.COLOR_BGR2RGB)

    # Get patches for each surface
    ref_patches = list(get_side_surface_patches(reference))
    clean_patches = list(get_side_surface_patches(clean))
    adv_patches = list(get_side_surface_patches(adversarial))

    # Process each surface
    num_surfaces = len(ref_patches)

    # Create figure with GridSpec for better control
    # Each surface gets 1 row with 6 columns (RGB images + change/edge maps)
    fig = plt.figure(figsize=(18, 3 * num_surfaces))

    plot_idx = 0
    for surf_idx, (ref_patch, clean_patch, adv_patch) in enumerate(
        zip(ref_patches, clean_patches, adv_patches)
    ):
        surface_name = PATCH_ORDER[surf_idx]

        # Skip if patch is mostly white (empty)
        if np.mean(ref_patch) >= 250 or np.mean(clean_patch) >= 250:
            continue

        # Select processing function based on method
        if method == 'simsac':
            process_func = process_simsac_output
            method_label = 'SimSAC'
        elif method == 'canny':
            process_func = process_canny_output
            method_label = 'Canny'
        elif method == 'laplacian':
            process_func = process_laplacian_output
            method_label = 'Laplacian'
        else:
            raise ValueError(f"Unknown method: {method}")

        # Process through selected method (only need raw maps, ignore thresholded)
        # Reference (using reference vs reference to get baseline)
        if method == 'simsac':
            change1_ref_raw, _ = process_func(
                ref_patch, ref_patch, threshold, simsac_ckpt_path
            )
            # Reference vs Clean
            change1_clean_raw, _ = process_func(
                ref_patch, clean_patch, threshold, simsac_ckpt_path
            )
            # Reference vs Adversarial
            change1_adv_raw, _ = process_func(
                ref_patch, adv_patch, threshold, simsac_ckpt_path
            )
        else:
            # For canny and laplacian (no checkpoint needed)
            change1_ref_raw, _ = process_func(
                ref_patch, ref_patch, threshold
            )
            # Reference vs Clean
            change1_clean_raw, _ = process_func(
                ref_patch, clean_patch, threshold
            )
            # Reference vs Adversarial
            change1_adv_raw, _ = process_func(
                ref_patch, adv_patch, threshold
            )

        # Create subplot for this surface (1 row, 6 columns)
        # Columns: RGB Reference, RGB Clean, RGB Adversarial, Map(R), Map(C), Map(A)
        base_idx = surf_idx * 6

        # Column 1: Reference image
        plt.subplot(num_surfaces, 6, base_idx + 1)
        plt.imshow(ref_patch)
        plt.title(f'{surface_name}\nReference', fontsize=10)
        plt.axis('off')

        # Column 2: Clean field
        plt.subplot(num_surfaces, 6, base_idx + 2)
        plt.imshow(clean_patch)
        plt.title('Clean', fontsize=10)
        plt.axis('off')

        # Column 3: Adversarial field
        plt.subplot(num_surfaces, 6, base_idx + 3)
        plt.imshow(adv_patch)
        plt.title('Adversarial', fontsize=10)
        plt.axis('off')

        # Column 4: Change/Edge map (Reference)
        plt.subplot(num_surfaces, 6, base_idx + 4)
        plt.imshow(change1_ref_raw, cmap='gray', vmin=0, vmax=255)
        plt.title(f'{method_label}(R)', fontsize=10)
        plt.axis('off')

        # Column 5: Change/Edge map (Clean)
        plt.subplot(num_surfaces, 6, base_idx + 5)
        plt.imshow(change1_clean_raw, cmap='gray', vmin=0, vmax=255)
        plt.title(f'{method_label}(C)', fontsize=10)
        plt.axis('off')

        # Column 6: Change/Edge map (Adversarial)
        plt.subplot(num_surfaces, 6, base_idx + 6)
        plt.imshow(change1_adv_raw, cmap='gray', vmin=0, vmax=255)
        plt.title(f'{method_label}(A)', fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize change detection for clean vs adversarial UV maps"
    )

    parser.add_argument('--reference_dir', type=str, required=True,
                       help='Directory with reference UV maps (uvmaps folder)')
    parser.add_argument('--clean_dir', type=str, required=True,
                       help='Directory with clean validation data')
    parser.add_argument('--adversarial_dir', type=str, required=True,
                       help='Directory with adversarial validation data')
    parser.add_argument('--background', type=str, required=True,
                       help='Background folder (e.g., carpet, table, gravel)')
    parser.add_argument('--attack_type', type=str, default='fgsm',
                       choices=['fgsm', 'pgd', 'cw'],
                       help='Attack type to visualize (default: fgsm)')
    parser.add_argument('--parcel_id', type=str, required=True,
                       help='Parcel ID to visualize (e.g., 1, 12, 25)')
    parser.add_argument('--timestamp', type=str, default=None,
                       help='Specific timestamp (optional, will use first match if not provided)')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for visualization')
    parser.add_argument('--method', type=str, default='simsac',
                       choices=['simsac', 'canny', 'laplacian'],
                       help='Comparison method to use (default: simsac)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to SimSAC checkpoint file (optional, only used with simsac method)')
    parser.add_argument('--threshold', type=int, default=200,
                       help='Threshold value for binary change map (default: 200)')

    args = parser.parse_args()

    # Find matching files
    reference_dir = Path(args.reference_dir)
    clean_dir = Path(args.clean_dir) / args.background

    # For adversarial, the folder is named like "carpet_adv_fgsm" or "carpet_adv_pgd"
    adversarial_folder = f"{args.background}_adv_{args.attack_type}"
    adversarial_dir = Path(args.adversarial_dir) / adversarial_folder

    if not adversarial_dir.exists():
        print(f"Error: Adversarial directory not found: {adversarial_dir}")
        print(f"Looking for folders matching pattern: {args.background}_adv_*")
        # Try to find any matching folder
        parent_dir = Path(args.adversarial_dir)
        matching_dirs = list(parent_dir.glob(f"{args.background}_adv_*"))
        if matching_dirs:
            print(f"Found these directories: {[d.name for d in matching_dirs]}")
            adversarial_dir = matching_dirs[0]
            print(f"Using: {adversarial_dir}")
        else:
            return

    # Pattern: id_{parcel_id}_{timestamp}_uvmap_gt.png
    pattern = f"id_{str(args.parcel_id).zfill(2)}_*_uvmap_gt.png"

    if args.timestamp:
        pattern = f"id_{str(args.parcel_id).zfill(2)}_{args.timestamp}_uvmap_gt.png"

    # Find files
    clean_files = list(clean_dir.glob(pattern))
    if not clean_files:
        print(f"Error: No clean files found matching {pattern} in {clean_dir}")
        return

    clean_path = clean_files[0]

    # Find corresponding adversarial file
    adv_files = list(adversarial_dir.glob(pattern))
    if not adv_files:
        print(f"Error: No adversarial files found matching {pattern} in {adversarial_dir}")
        return

    adversarial_path = adv_files[0]

    # Find reference file (same ID)
    ref_pattern = f"id_{str(args.parcel_id).zfill(2)}_*.png"
    ref_files = list(reference_dir.glob(ref_pattern))
    if not ref_files:
        print(f"Error: No reference files found matching {ref_pattern} in {reference_dir}")
        return

    reference_path = ref_files[0]

    print(f"Reference: {reference_path.name}")
    print(f"Clean:     {clean_path.name}")
    print(f"Adversarial: {adversarial_path.name}")

    # Generate visualization
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"{args.method}_comparison_id{args.parcel_id}_{args.background}_{args.attack_type}.png"
    output_path = output_dir / output_name

    print(f"Method: {args.method}")
    if args.checkpoint and args.method == 'simsac':
        print(f"SimSAC Checkpoint: {args.checkpoint}")
    elif args.checkpoint and args.method != 'simsac':
        print(f"Warning: --checkpoint is only used with --method simsac, ignoring for {args.method}")
    print(f"Threshold: {args.threshold}")

    visualize_parcel_comparison(
        reference_path,
        clean_path,
        adversarial_path,
        output_path,
        threshold=args.threshold,
        method=args.method,
        simsac_ckpt_path=args.checkpoint
    )


if __name__ == "__main__":
    main()
