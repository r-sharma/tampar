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
- Original images (RGB)
- Change maps (grayscale)
- Thresholded change maps (binary black/white)

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


def process_simsac_output(im1, im2, threshold=200):
    """
    Process two images through SimSAC and return change maps.

    This replicates what compare_simsac() does in src/tampering/compare.py

    Returns:
        change1_raw: Raw change map 1 (grayscale 0-255)
        change1_thresh: Thresholded change map 1 (binary 0 or 255)
    """
    simsac = SimSaC.get_instance()
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

    Returns:
        change_raw: Raw change map (grayscale 0-255)
        change_thresh: Thresholded change map (binary 0 or 255)
    """
    # Use compare_canny from src/tampering/compare.py
    change_map = compare_canny(im1, im2)

    # change_map is already grayscale uint8
    change_raw = change_map

    # Thresholded version
    change_thresh = (change_map > threshold).astype(np.float32) * 255
    change_thresh = change_thresh.astype(np.uint8)

    return change_raw, change_thresh


def process_laplacian_output(im1, im2, threshold=200):
    """
    Process two images through Laplacian edge detection.

    Returns:
        change_raw: Raw change map (grayscale 0-255)
        change_thresh: Thresholded change map (binary 0 or 255)
    """
    # Use compare_laplacian from src/tampering/compare.py
    change_map = compare_laplacian(im1, im2)

    # change_map is already grayscale uint8
    change_raw = change_map

    # Thresholded version
    change_thresh = (change_map > threshold).astype(np.float32) * 255
    change_thresh = change_thresh.astype(np.uint8)

    return change_raw, change_thresh


def visualize_parcel_comparison(
    reference_path,
    clean_path,
    adversarial_path,
    output_path,
    threshold=200,
    method='simsac'
):
    """
    Visualize comparison for all surfaces of a parcel using specified method.

    Args:
        reference_path: Path to reference UV map
        clean_path: Path to clean field UV map
        adversarial_path: Path to adversarial field UV map
        output_path: Path to save output visualization
        threshold: Threshold value for binary change map
        method: Comparison method ('simsac', 'canny', or 'laplacian')

    Shows 3 columns:
    - Column 1: Reference vs Clean
    - Column 2: Reference vs Adversarial
    - Column 3: Difference (Adversarial - Clean)

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
    # Each surface gets 1 row with 9 columns
    fig = plt.figure(figsize=(24, 3 * num_surfaces))

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

        # Process through selected method
        # Reference vs Clean
        change1_clean_raw, change1_clean_thresh = process_func(
            ref_patch, clean_patch, threshold
        )

        # Reference vs Adversarial
        change1_adv_raw, change1_adv_thresh = process_func(
            ref_patch, adv_patch, threshold
        )

        # Compute difference in raw change maps
        change_diff = cv2.absdiff(change1_adv_raw, change1_clean_raw)

        # Compute difference in thresholded maps
        change_thresh_diff = cv2.absdiff(change1_adv_thresh, change1_clean_thresh)

        # Create subplot for this surface (1 row, 9 columns)
        # Row: Original images (3), Raw change maps (3), Thresholded maps (3)
        base_idx = surf_idx * 9

        # Column 1: Reference image
        plt.subplot(num_surfaces, 9, base_idx + 1)
        plt.imshow(ref_patch)
        plt.title(f'{surface_name}\nReference', fontsize=9)
        plt.axis('off')

        # Column 2: Clean field
        plt.subplot(num_surfaces, 9, base_idx + 2)
        plt.imshow(clean_patch)
        plt.title('Clean', fontsize=9)
        plt.axis('off')

        # Column 3: Adversarial field
        plt.subplot(num_surfaces, 9, base_idx + 3)
        plt.imshow(adv_patch)
        plt.title('Adversarial', fontsize=9)
        plt.axis('off')

        # Column 4: Raw change (Clean)
        plt.subplot(num_surfaces, 9, base_idx + 4)
        plt.imshow(change1_clean_raw, cmap='gray', vmin=0, vmax=255)
        plt.title(f'{method_label}(C)\nRaw', fontsize=9)
        plt.axis('off')

        # Column 5: Raw change (Adversarial)
        plt.subplot(num_surfaces, 9, base_idx + 5)
        plt.imshow(change1_adv_raw, cmap='gray', vmin=0, vmax=255)
        plt.title(f'{method_label}(A)\nRaw', fontsize=9)
        plt.axis('off')

        # Column 6: Raw difference
        plt.subplot(num_surfaces, 9, base_idx + 6)
        plt.imshow(change_diff, cmap='hot', vmin=0, vmax=100)
        plt.title(f'Diff\n[max={change_diff.max():.0f}]', fontsize=9)
        plt.axis('off')

        # Column 7: Thresholded change (Clean)
        plt.subplot(num_surfaces, 9, base_idx + 7)
        plt.imshow(change1_clean_thresh, cmap='gray', vmin=0, vmax=255)
        plt.title(f'Thresh(C)\n@{threshold}', fontsize=9)
        plt.axis('off')

        # Column 8: Thresholded change (Adversarial)
        plt.subplot(num_surfaces, 9, base_idx + 8)
        plt.imshow(change1_adv_thresh, cmap='gray', vmin=0, vmax=255)
        plt.title(f'Thresh(A)\n@{threshold}', fontsize=9)
        plt.axis('off')

        # Column 9: Threshold difference
        plt.subplot(num_surfaces, 9, base_idx + 9)
        plt.imshow(change_thresh_diff, cmap='gray', vmin=0, vmax=255)
        pixels_changed = np.sum(change_thresh_diff > 0)
        total_pixels = change_thresh_diff.size
        pct_changed = 100 * pixels_changed / total_pixels
        plt.title(f'Diff\n[{pct_changed:.1f}%]', fontsize=9)
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
    print(f"Threshold: {args.threshold}")

    visualize_parcel_comparison(
        reference_path,
        clean_path,
        adversarial_path,
        output_path,
        threshold=args.threshold,
        method=args.method
    )


if __name__ == "__main__":
    main()
