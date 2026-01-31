"""
Strategy 1: Similarity-Targeted Adversarial Attack

This attack directly optimizes similarity metrics (SSIM, MAE, MSSSIM) instead of
attacking SimSAC's intermediate change output.

Goal: Make tampered images appear more similar to reference UV maps by adding
small adversarial perturbations that fool the similarity metrics.

Usage:
    # Generate adversarial images for carpet only (fast testing)
    python generate_adversarial_similarity_targeted.py \
        --data_dir /path/to/validation \
        --uvmaps_dir /path/to/uvmaps \
        --output_dir /path/to/adversarial_validation_similarity \
        --folders carpet \
        --attack both \
        --epsilon 0.15 \
        --pgd_steps 10

    # Generate for all backgrounds except base
    python generate_adversarial_similarity_targeted.py \
        --data_dir /path/to/validation \
        --output_dir /path/to/adversarial_validation_similarity \
        --exclude_folders base \
        --attack both \
        --epsilon 0.2 \
        --pgd_steps 20
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())

import argparse
import shutil
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

# Import similarity metrics
from src.tampering.metrics import compute_ssim, compute_msssim, compute_mae


class SimilarityTargetedAttackGenerator:
    """
    Generate adversarial UV maps by directly targeting similarity metrics.

    Unlike SimSAC-targeted attacks that attack the change detection output,
    this attack directly optimizes SSIM, MAE, and MSSSIM to make tampered
    images appear more similar to reference UV maps.
    """

    def __init__(self, epsilon=0.15, device='cuda'):
        """
        Args:
            epsilon: Maximum perturbation magnitude (L-infinity norm)
            device: Device to use for computation
        """
        self.epsilon = epsilon
        self.device = device

    def similarity_loss(self, field_img, reference_img):
        """
        Compute loss based on similarity metrics.

        Goal: MAXIMIZE similarity to reference (hide tampering)

        Loss = -SSIM + MAE + -MSSSIM

        By minimizing this loss:
        - SSIM increases (more structurally similar)
        - MAE decreases (less pixel difference)
        - MSSSIM increases (more perceptually similar)

        Args:
            field_img: Field UV map tensor [1, C, H, W] (normalized 0-1)
            reference_img: Reference UV map tensor [1, C, H, W] (normalized 0-1)

        Returns:
            loss: Scalar loss value
        """
        # Convert to numpy for metric computation (metrics expect HWC format, 0-255)
        field_np = field_img[0].permute(1, 2, 0).cpu().detach().numpy() * 255.0
        reference_np = reference_img[0].permute(1, 2, 0).cpu().detach().numpy() * 255.0

        field_np = field_np.astype(np.float32)
        reference_np = reference_np.astype(np.float32)

        # Compute similarity metrics
        ssim_value = compute_ssim(field_np, reference_np)
        msssim_value = compute_msssim(field_np, reference_np)
        mae_value = compute_mae(field_np, reference_np)

        # Convert back to tensors
        ssim_tensor = torch.tensor(ssim_value, device=self.device, dtype=torch.float32)
        msssim_tensor = torch.tensor(msssim_value, device=self.device, dtype=torch.float32)
        mae_tensor = torch.tensor(mae_value, device=self.device, dtype=torch.float32)

        # Loss: We want to MAXIMIZE SSIM and MSSSIM, MINIMIZE MAE
        # So we negate SSIM/MSSSIM and add MAE
        loss = -ssim_tensor + mae_tensor - msssim_tensor

        return loss

    def fgsm_attack_similarity(self, field_img, reference_img):
        """
        FGSM attack targeting similarity metrics.

        Args:
            field_img: Field image tensor [1, C, H, W] (normalized 0-1)
            reference_img: Reference UV map tensor [1, C, H, W] (normalized 0-1)

        Returns:
            Adversarial field image tensor
        """
        field_img = field_img.clone().detach()
        reference_img = reference_img.clone().detach()
        field_img.requires_grad = True

        # Compute similarity loss
        loss = self.similarity_loss(field_img, reference_img)

        # Clear gradients
        if field_img.grad is not None:
            field_img.grad.zero_()

        # Compute gradient
        loss.backward()

        # Check gradient
        if field_img.grad is None:
            raise ValueError("Gradient is None after backward pass!")

        # FGSM step: add perturbation in direction that reduces loss
        with torch.no_grad():
            perturbation = self.epsilon * field_img.grad.sign()
            adv_field = field_img + perturbation
            adv_field = torch.clamp(adv_field, 0, 1)

        return adv_field

    def pgd_attack_similarity(self, field_img, reference_img, steps=10, step_size=None):
        """
        PGD attack targeting similarity metrics.

        Args:
            field_img: Field image tensor [1, C, H, W]
            reference_img: Reference UV map tensor [1, C, H, W]
            steps: Number of PGD iterations
            step_size: Step size per iteration

        Returns:
            Adversarial field image
        """
        if step_size is None:
            step_size = self.epsilon / 4

        original_field = field_img.clone().detach()
        adv_field = field_img.clone().detach()

        for step in range(steps):
            adv_field.requires_grad = True

            # Compute similarity loss
            loss = self.similarity_loss(adv_field, reference_img)

            # Compute gradient
            loss.backward()

            with torch.no_grad():
                # PGD step
                adv_field = adv_field + step_size * adv_field.grad.sign()

                # Project back to epsilon ball
                perturbation = torch.clamp(adv_field - original_field, -self.epsilon, self.epsilon)
                adv_field = torch.clamp(original_field + perturbation, 0, 1)

            adv_field = adv_field.detach()

        return adv_field

    def generate_adversarial_uvmap(self, field_path, reference_path, attack_type='fgsm', pgd_steps=10):
        """
        Generate adversarial UV map from a field and reference image.

        Args:
            field_path: Path to field UV map
            reference_path: Path to reference UV map
            attack_type: 'fgsm' or 'pgd'
            pgd_steps: Number of PGD iterations (if using PGD)

        Returns:
            adversarial_image: Numpy array (H, W, C) in 0-255 range
        """
        # Load images
        field_img = Image.open(field_path).convert('RGB')
        reference_img = Image.open(reference_path).convert('RGB')

        # Convert to tensors (normalize to 0-1)
        field_tensor = torch.from_numpy(np.array(field_img)).permute(2, 0, 1).float() / 255.0
        reference_tensor = torch.from_numpy(np.array(reference_img)).permute(2, 0, 1).float() / 255.0

        # Add batch dimension
        field_tensor = field_tensor.unsqueeze(0).to(self.device)
        reference_tensor = reference_tensor.unsqueeze(0).to(self.device)

        # Generate adversarial example
        if attack_type == 'fgsm':
            adv_tensor = self.fgsm_attack_similarity(field_tensor, reference_tensor)
        elif attack_type == 'pgd':
            adv_tensor = self.pgd_attack_similarity(field_tensor, reference_tensor, steps=pgd_steps)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # Convert back to numpy (0-255)
        adv_numpy = (adv_tensor[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

        return adv_numpy


def find_reference_uvmap(parcel_id, uvmaps_dir):
    """
    Find the reference UV map for a given parcel ID.

    Args:
        parcel_id: Parcel ID (e.g., "01", "12")
        uvmaps_dir: Path to uvmaps directory

    Returns:
        Path to reference UV map, or None if not found
    """
    uvmaps_path = Path(uvmaps_dir)
    pattern = f"id_{parcel_id}_*.png"

    matches = list(uvmaps_path.glob(pattern))
    if matches:
        return matches[0]
    return None


def generate_adversarial_dataset(
    data_dir,
    uvmaps_dir,
    output_dir,
    attack_type='fgsm',
    folders=None,
    exclude_folders=None,
    uvmap_types=['gt'],
    epsilon=0.15,
    pgd_steps=10
):
    """
    Generate adversarial dataset for validation data.

    Args:
        data_dir: Input validation directory
        uvmaps_dir: Directory with reference UV maps
        output_dir: Output directory for adversarial data
        attack_type: 'fgsm', 'pgd', or 'both'
        folders: List of specific folders to include (e.g., ['carpet', 'table'])
        exclude_folders: List of folders to exclude (e.g., ['base'])
        uvmap_types: List of UV map types to process (['gt', 'pred'])
        epsilon: Perturbation magnitude
        pgd_steps: Number of PGD iterations
    """
    data_path = Path(data_dir)
    uvmaps_path = Path(uvmaps_dir)
    output_path = Path(output_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Get list of background folders
    background_folders = [d for d in data_path.iterdir() if d.is_dir()]

    # Filter folders
    if folders:
        background_folders = [d for d in background_folders if d.name in folders]
    if exclude_folders:
        background_folders = [d for d in background_folders if d.name not in exclude_folders]

    print(f"\nProcessing backgrounds: {[d.name for d in background_folders]}")
    print(f"Attack type: {attack_type}")
    print(f"Epsilon: {epsilon}")
    print(f"PGD steps: {pgd_steps}")

    # Determine which attacks to run
    attacks_to_run = []
    if attack_type in ['fgsm', 'both']:
        attacks_to_run.append('fgsm')
    if attack_type in ['pgd', 'both']:
        attacks_to_run.append('pgd')

    for attack_method in attacks_to_run:
        print(f"\n{'='*80}")
        print(f"Generating {attack_method.upper()} adversarial examples")
        print(f"{'='*80}")

        # Create attack generator
        attacker = SimilarityTargetedAttackGenerator(epsilon=epsilon, device=device)

        for background_folder in background_folders:
            folder_name = background_folder.name
            print(f"\nProcessing folder: {folder_name}")

            # Create output folder
            output_folder_name = f"{folder_name}_adv_{attack_method}"
            output_folder = output_path / output_folder_name
            output_folder.mkdir(parents=True, exist_ok=True)

            # Find all UV map files
            uvmap_files = []
            for uvmap_type in uvmap_types:
                pattern = f"*_uvmap_{uvmap_type}.png"
                uvmap_files.extend(list(background_folder.glob(pattern)))

            print(f"  Found {len(uvmap_files)} UV map files")

            # Process each UV map
            successful = 0
            failed = 0

            progress_bar = tqdm(uvmap_files, desc=f"  {attack_method.upper()} attack")
            for uvmap_file in progress_bar:
                try:
                    # Extract parcel ID from filename
                    # Format: id_01_20230523_155225_uvmap_gt.png
                    filename_parts = uvmap_file.stem.split('_')
                    parcel_id = filename_parts[1]  # "01"

                    # Find corresponding reference UV map
                    reference_path = find_reference_uvmap(parcel_id, uvmaps_path)
                    if reference_path is None:
                        progress_bar.write(f"    Warning: No reference found for parcel {parcel_id}, skipping")
                        failed += 1
                        continue

                    # Generate adversarial image
                    adv_image = attacker.generate_adversarial_uvmap(
                        uvmap_file,
                        reference_path,
                        attack_type=attack_method,
                        pgd_steps=pgd_steps
                    )

                    # Save adversarial image
                    output_file = output_folder / uvmap_file.name
                    cv2.imwrite(str(output_file), cv2.cvtColor(adv_image, cv2.COLOR_RGB2BGR))

                    successful += 1

                except Exception as e:
                    progress_bar.write(f"    Error processing {uvmap_file.name}: {str(e)}")
                    failed += 1

            print(f"  ✓ Successful: {successful}, ✗ Failed: {failed}")

    print(f"\n{'='*80}")
    print(f"Adversarial dataset generation complete!")
    print(f"Output directory: {output_path}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate similarity-targeted adversarial UV maps (Strategy 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--data_dir', type=str, required=True,
                       help='Input validation directory')
    parser.add_argument('--uvmaps_dir', type=str, required=False, default=None,
                       help='Directory with reference UV maps (default: auto-detect from data_dir parent)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for adversarial data')

    parser.add_argument('--attack', type=str, choices=['fgsm', 'pgd', 'both'],
                       default='fgsm',
                       help='Attack type')

    parser.add_argument('--folders', type=str, nargs='+',
                       help='Specific folders to include (e.g., carpet table gravel)')
    parser.add_argument('--exclude_folders', type=str, nargs='+',
                       help='Folders to exclude (e.g., base)')

    parser.add_argument('--uvmap_types', type=str, nargs='+',
                       choices=['gt', 'pred'], default=['gt'],
                       help='UV map types to generate')

    parser.add_argument('--epsilon', type=float, default=0.15,
                       help='Perturbation magnitude. Recommended: 0.15-0.3 for Strategy 1')

    parser.add_argument('--pgd_steps', type=int, default=10,
                       help='Number of PGD iterations')

    args = parser.parse_args()

    # Auto-detect uvmaps_dir if not provided
    data_dir_path = Path(args.data_dir)
    if args.uvmaps_dir is None:
        # Assume data_dir is like /path/to/tampar/validation
        # and uvmaps_dir is /path/to/tampar/uvmaps
        uvmaps_dir = data_dir_path.parent / 'uvmaps'
        if not uvmaps_dir.exists():
            raise ValueError(f"Could not auto-detect uvmaps_dir. Expected at {uvmaps_dir}. Please provide --uvmaps_dir explicitly.")
        print(f"Auto-detected uvmaps_dir: {uvmaps_dir}")
    else:
        uvmaps_dir = args.uvmaps_dir

    generate_adversarial_dataset(
        data_dir=args.data_dir,
        uvmaps_dir=uvmaps_dir,
        output_dir=args.output_dir,
        attack_type=args.attack,
        folders=args.folders,
        exclude_folders=args.exclude_folders,
        uvmap_types=args.uvmap_types,
        epsilon=args.epsilon,
        pgd_steps=args.pgd_steps
    )


if __name__ == "__main__":
    main()
