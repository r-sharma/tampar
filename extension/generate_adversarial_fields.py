"""
Generate Adversarial 3D Field Images and UV Maps

Creates adversarial perturbations on 3D field images using FGSM/PGD attacks,
then generates UV maps from these adversarial fields to challenge tampering detection.

The goal is to create realistic-looking but adversarially perturbed UV maps that
significantly reduce detection accuracy.

Usage:
    # FGSM attack, generate both GT and Pred UV maps
    python generate_adversarial_fields.py \
        --data_dir /content/tampar/data/tampar_sample/validation \
        --output_dir /content/tampar/data/tampar_sample/adversarial_validation \
        --attack fgsm \
        --uvmap_types gt pred \
        --epsilon 0.05

    # PGD attack, only GT UV maps
    python generate_adversarial_fields.py \
        --data_dir /content/tampar/data/tampar_sample/validation \
        --output_dir /content/tampar/data/tampar_sample/adversarial_validation \
        --attack pgd \
        --uvmap_types gt \
        --epsilon 0.03 \
        --pgd_steps 10

    # Both attacks, both UV map types
    python generate_adversarial_fields.py \
        --data_dir /content/tampar/data/tampar_sample/validation \
        --output_dir /content/tampar/data/tampar_sample/adversarial_validation \
        --attack both \
        --uvmap_types gt pred \
        --epsilon 0.05
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2

# Add TAMPAR to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unwrapping.unwrapper import Unwrapper


class AdversarialFieldGenerator:
    """Generate adversarial perturbations on 3D field images."""

    def __init__(self, epsilon=0.05, device='cuda'):
        """
        Initialize adversarial generator.

        Args:
            epsilon: Maximum perturbation magnitude (L-infinity norm)
            device: Device to use
        """
        self.epsilon = epsilon
        self.device = device

    def fgsm_attack(self, image, gradient):
        """
        Fast Gradient Sign Method (FGSM) attack.

        Args:
            image: Original image tensor [C, H, W]
            gradient: Gradient of loss w.r.t. image

        Returns:
            Adversarial image
        """
        # Get sign of gradient
        sign_gradient = gradient.sign()

        # Create perturbation
        perturbation = self.epsilon * sign_gradient

        # Add perturbation
        adv_image = image + perturbation

        # Clamp to valid range [0, 1]
        adv_image = torch.clamp(adv_image, 0, 1)

        return adv_image

    def pgd_attack(self, image, loss_fn, steps=10, step_size=None):
        """
        Projected Gradient Descent (PGD) attack.

        More powerful iterative version of FGSM.

        Args:
            image: Original image tensor [C, H, W]
            loss_fn: Function that computes loss for the image
            steps: Number of PGD iterations
            step_size: Step size per iteration (default: epsilon/4)

        Returns:
            Adversarial image
        """
        if step_size is None:
            step_size = self.epsilon / 4

        # Start from original image
        adv_image = image.clone().detach()

        for _ in range(steps):
            adv_image.requires_grad = True

            # Compute loss
            loss = loss_fn(adv_image)

            # Compute gradient
            loss.backward()

            with torch.no_grad():
                # Update adversarial image
                adv_image = adv_image + step_size * adv_image.grad.sign()

                # Project back to epsilon ball around original image
                perturbation = torch.clamp(adv_image - image, -self.epsilon, self.epsilon)
                adv_image = torch.clamp(image + perturbation, 0, 1)

            adv_image = adv_image.detach()

        return adv_image

    def texture_loss(self, image):
        """
        Loss that encourages texture variations to create hallucinations.

        Maximizes high-frequency content to create artifacts.
        """
        # Compute gradients (high-frequency content)
        dx = image[:, :, 1:] - image[:, :, :-1]
        dy = image[:, 1:, :] - image[:, :-1, :]

        # Negative L2 norm (maximize variation)
        loss = -(dx.pow(2).mean() + dy.pow(2).mean())

        return loss

    def smoothness_loss(self, image):
        """
        Loss that encourages smoothness violations.

        Creates unnatural smooth regions that confuse detection.
        """
        # Compute second-order gradients
        dx = image[:, :, 1:] - image[:, :, :-1]
        dy = image[:, 1:, :] - image[:, :-1, :]

        ddx = dx[:, :, 1:] - dx[:, :, :-1]
        ddy = dy[:, 1:, :] - dy[:, :-1, :]

        # Minimize second-order gradients (create smooth regions)
        loss = ddx.pow(2).mean() + ddy.pow(2).mean()

        return loss

    def generate_adversarial_field(self, field_path, attack_type='fgsm',
                                   hallucination_strength=1.0, pgd_steps=10):
        """
        Generate adversarial version of 3D field image.

        Args:
            field_path: Path to original field image
            attack_type: 'fgsm' or 'pgd'
            hallucination_strength: Weight for hallucination losses
            pgd_steps: Number of PGD steps if using PGD

        Returns:
            Adversarial field image as numpy array
        """
        # Load field image
        field = Image.open(field_path).convert('RGB')
        field_np = np.array(field).astype(np.float32) / 255.0

        # Convert to tensor
        field_tensor = torch.from_numpy(field_np).permute(2, 0, 1).to(self.device)
        field_tensor = field_tensor.unsqueeze(0)  # [1, C, H, W]

        if attack_type == 'fgsm':
            # FGSM: Single-step attack
            field_tensor.requires_grad = True

            # Combined loss: texture variation + smoothness violation
            loss = (hallucination_strength * self.texture_loss(field_tensor[0]) +
                   hallucination_strength * self.smoothness_loss(field_tensor[0]))

            # Compute gradient
            loss.backward()

            # Generate adversarial image
            adv_field = self.fgsm_attack(field_tensor[0], field_tensor.grad[0])

        elif attack_type == 'pgd':
            # PGD: Multi-step attack
            def loss_fn(img):
                return (hallucination_strength * self.texture_loss(img) +
                       hallucination_strength * self.smoothness_loss(img))

            adv_field = self.pgd_attack(field_tensor[0], loss_fn, steps=pgd_steps)

        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # Convert back to numpy
        adv_field_np = adv_field.detach().cpu().permute(1, 2, 0).numpy()
        adv_field_np = (adv_field_np * 255).astype(np.uint8)

        return adv_field_np


def generate_adversarial_dataset(data_dir, output_dir, attack_type='fgsm',
                                uvmap_types=['gt', 'pred'], epsilon=0.05,
                                hallucination_strength=2.0, pgd_steps=10):
    """
    Generate adversarial 3D fields and corresponding UV maps.

    Args:
        data_dir: Input data directory (e.g., validation folder)
        output_dir: Output directory for adversarial data
        attack_type: 'fgsm', 'pgd', or 'both'
        uvmap_types: List of UV map types to generate ('gt', 'pred', or both)
        epsilon: Perturbation magnitude
        hallucination_strength: Strength of hallucination losses
        pgd_steps: Number of PGD iterations
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Adversarial 3D Field Generation")
    print(f"{'='*70}")
    print(f"Input directory:  {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Attack type(s):   {attack_type}")
    print(f"UV map types:     {uvmap_types}")
    print(f"Epsilon:          {epsilon}")
    print(f"Hallucination:    {hallucination_strength}")
    if 'pgd' in attack_type.lower():
        print(f"PGD steps:        {pgd_steps}")

    # Initialize generator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device:           {device}")

    generator = AdversarialFieldGenerator(epsilon=epsilon, device=device)

    # Determine attack types to run
    if attack_type == 'both':
        attacks = ['fgsm', 'pgd']
    else:
        attacks = [attack_type]

    # Find all parcels
    parcel_ids = []
    for item in data_dir.iterdir():
        if item.is_dir() and item.name.startswith('id_'):
            parcel_id = int(item.name.split('_')[1])
            parcel_ids.append(parcel_id)

    parcel_ids.sort()
    print(f"\nFound {len(parcel_ids)} parcels: {parcel_ids}")

    # Process each parcel
    for parcel_id in tqdm(parcel_ids, desc="Processing parcels"):
        parcel_dir = data_dir / f"id_{parcel_id:02d}"

        # Process each attack type
        for attack in attacks:
            # Create output directory for this attack
            attack_suffix = f"_adv_{attack}"
            output_parcel_dir = output_dir / f"id_{parcel_id:02d}{attack_suffix}"
            output_parcel_dir.mkdir(parents=True, exist_ok=True)

            # Copy metadata files
            for meta_file in ['tampering_codes.txt', 'uvmap_metadata.json']:
                if (parcel_dir / meta_file).exists():
                    shutil.copy(parcel_dir / meta_file, output_parcel_dir / meta_file)

            # Generate adversarial field image
            field_path = parcel_dir / f"id_{parcel_id:02d}_field.png"
            if not field_path.exists():
                print(f"  Warning: Field not found for parcel {parcel_id}, skipping")
                continue

            adv_field = generator.generate_adversarial_field(
                field_path,
                attack_type=attack,
                hallucination_strength=hallucination_strength,
                pgd_steps=pgd_steps
            )

            # Save adversarial field
            output_field_path = output_parcel_dir / f"id_{parcel_id:02d}_field.png"
            Image.fromarray(adv_field).save(output_field_path)

            # Generate UV maps from adversarial field
            for uvmap_type in uvmap_types:
                try:
                    # Create unwrapper
                    unwrapper = Unwrapper()

                    # Determine which GT/Pred mesh to use
                    if uvmap_type == 'gt':
                        mesh_path = parcel_dir / f"id_{parcel_id:02d}_mesh_gt.ply"
                    else:  # pred
                        mesh_path = parcel_dir / f"id_{parcel_id:02d}_mesh_pred.ply"

                    if not mesh_path.exists():
                        print(f"  Warning: Mesh {uvmap_type} not found for parcel {parcel_id}")
                        continue

                    # Unwrap adversarial field onto mesh
                    uvmap = unwrapper.unwrap(
                        str(output_field_path),
                        str(mesh_path)
                    )

                    # Save UV map
                    output_uvmap_path = output_parcel_dir / f"id_{parcel_id:02d}_uvmap_{uvmap_type}.png"
                    cv2.imwrite(str(output_uvmap_path), uvmap)

                except Exception as e:
                    print(f"  Error generating {uvmap_type} UV map for parcel {parcel_id}: {e}")

    print(f"\n{'='*70}")
    print("✓ Adversarial dataset generation complete!")
    print(f"{'='*70}")
    print(f"\nOutput directory: {output_dir}")

    # Print statistics
    total_fields = 0
    total_uvmaps = 0

    for attack in attacks:
        attack_count = 0
        uvmap_count = 0

        for parcel_id in parcel_ids:
            attack_suffix = f"_adv_{attack}"
            parcel_dir = output_dir / f"id_{parcel_id:02d}{attack_suffix}"

            if (parcel_dir / f"id_{parcel_id:02d}_field.png").exists():
                attack_count += 1

            for uvmap_type in uvmap_types:
                if (parcel_dir / f"id_{parcel_id:02d}_uvmap_{uvmap_type}.png").exists():
                    uvmap_count += 1

        total_fields += attack_count
        total_uvmaps += uvmap_count

        print(f"\n{attack.upper()} Attack:")
        print(f"  Adversarial fields: {attack_count}")
        print(f"  UV maps generated:  {uvmap_count}")

    print(f"\nTotal:")
    print(f"  Adversarial fields: {total_fields}")
    print(f"  UV maps generated:  {total_uvmaps}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate adversarial 3D fields and UV maps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--data_dir', type=str, required=True,
                       help='Input data directory (e.g., validation folder)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for adversarial data')

    parser.add_argument('--attack', type=str, choices=['fgsm', 'pgd', 'both'],
                       default='fgsm',
                       help='Attack type: fgsm (fast), pgd (strong), or both')

    parser.add_argument('--uvmap_types', type=str, nargs='+',
                       choices=['gt', 'pred'], default=['gt', 'pred'],
                       help='UV map types to generate (gt, pred, or both)')

    parser.add_argument('--epsilon', type=float, default=0.05,
                       help='Perturbation magnitude (L-inf norm). Higher = stronger attack. '
                            'Recommended: 0.03-0.1')

    parser.add_argument('--hallucination_strength', type=float, default=2.0,
                       help='Strength of hallucination losses. Higher = more artifacts. '
                            'Recommended: 1.0-3.0')

    parser.add_argument('--pgd_steps', type=int, default=10,
                       help='Number of PGD iterations (only for PGD attack). '
                            'More steps = stronger attack. Recommended: 5-20')

    args = parser.parse_args()

    generate_adversarial_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        attack_type=args.attack,
        uvmap_types=args.uvmap_types,
        epsilon=args.epsilon,
        hallucination_strength=args.hallucination_strength,
        pgd_steps=args.pgd_steps
    )


if __name__ == "__main__":
    main()
