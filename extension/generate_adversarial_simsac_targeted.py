"""
Generate SimSAC-Targeted Adversarial UV Maps

Creates adversarial perturbations that specifically target SimSAC's optical flow
correspondence matching, making them much more effective than generic attacks.

Usage:
    # SimSAC-targeted FGSM attack on carpet and table folders only
    python generate_adversarial_simsac_targeted.py \
        --data_dir /content/drive/MyDrive/TAMPAR_DATA/tampar/validation \
        --uvmaps_dir /content/drive/MyDrive/TAMPAR_DATA/tampar/uvmaps \
        --output_dir /content/drive/MyDrive/TAMPAR_DATA/tampar/adversarial_validation_simsac \
        --attack fgsm \
        --folders carpet table \
        --epsilon 0.1

    # SimSAC-targeted PGD attack, all folders except base
    python generate_adversarial_simsac_targeted.py \
        --data_dir /content/drive/MyDrive/TAMPAR_DATA/tampar/validation \
        --uvmaps_dir /content/drive/MyDrive/TAMPAR_DATA/tampar/uvmaps \
        --output_dir /content/drive/MyDrive/TAMPAR_DATA/tampar/adversarial_validation_simsac \
        --attack pgd \
        --exclude_folders base \
        --epsilon 0.1 \
        --pgd_steps 20
"""

import os
import sys
from pathlib import Path

# Add TAMPAR to path
TAMPAR_ROOT = Path(__file__).parent.parent
sys.path.append(str(TAMPAR_ROOT))

import argparse
import shutil
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2

from src.simsac.models.our_models.SimSaC import SimSaC_Model


class SimSaCTargetedAttackGenerator:
    """Generate adversarial perturbations targeting SimSAC correspondence matching."""

    def __init__(self, simsac_model, epsilon=0.1, device='cuda'):
        """
        Initialize SimSAC-targeted adversarial generator.

        Args:
            simsac_model: Loaded SimSAC model
            epsilon: Maximum perturbation magnitude (L-infinity norm)
            device: Device to use
        """
        self.simsac = simsac_model
        # Keep in train mode to allow gradients to flow
        # SimSAC has internal no_grad() in eval mode which blocks gradients
        self.simsac.train()
        self.epsilon = epsilon
        self.device = device

    def simsac_flow_loss(self, field_img, reference_img):
        """
        Compute loss based on SimSAC optical flow.

        Maximizes flow magnitude to make images appear very different.
        """
        # Resize to required resolutions
        field_512 = F.interpolate(field_img, size=(512, 512), mode='bilinear', align_corners=False)
        reference_512 = F.interpolate(reference_img, size=(512, 512), mode='bilinear', align_corners=False)
        field_256 = F.interpolate(field_img, size=(256, 256), mode='bilinear', align_corners=False)
        reference_256 = F.interpolate(reference_img, size=(256, 256), mode='bilinear', align_corners=False)

        # Forward through SimSAC
        # Model is in train mode to allow gradients
        output = self.simsac(field_512, reference_512, field_256, reference_256)

        # Extract flow based on mode
        # Train mode: dict with {"flow": ([flow4, flow3], [flow2, flow1]), ...}
        # Eval mode: tuple (flow1, change1)
        if isinstance(output, dict):
            # Train mode - nested structure
            flow_tuple = output.get('flow', None)
            if flow_tuple is not None and isinstance(flow_tuple, tuple) and len(flow_tuple) == 2:
                # flow_tuple = ([flow4, flow3], [flow2, flow1])
                # We want flow1 (finest resolution)
                flow = flow_tuple[1][1]  # Second tuple, second element
            else:
                flow = output.get('flow_est', None)
        elif isinstance(output, (list, tuple)):
            # Eval mode - (flow, change)
            flow = output[0]
        else:
            flow = output

        if flow is None:
            raise ValueError(f"SimSAC did not return flow output. Got: {type(output)}, keys: {output.keys() if isinstance(output, dict) else 'N/A'}")

        # Compute flow magnitude
        flow_mag = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2)

        # Loss: NEGATIVE flow magnitude (maximize flow = maximize difference)
        loss = -flow_mag.mean()

        return loss

    def fgsm_attack_simsac(self, field_img, reference_img):
        """
        FGSM attack targeting SimSAC optical flow.

        Args:
            field_img: Field image tensor [1, C, H, W]
            reference_img: Reference UV map tensor [1, C, H, W]

        Returns:
            Adversarial field image
        """
        field_img = field_img.clone().detach()
        reference_img = reference_img.clone().detach()  # Ensure no grad
        field_img.requires_grad = True

        # Compute SimSAC flow loss
        loss = self.simsac_flow_loss(field_img, reference_img)

        # Check if gradient exists
        if field_img.grad is not None:
            field_img.grad.zero_()

        # Compute gradient
        loss.backward()

        # Check gradient
        if field_img.grad is None:
            raise ValueError("Gradient is None after backward pass!")

        # FGSM step
        with torch.no_grad():
            perturbation = self.epsilon * field_img.grad.sign()
            adv_field = field_img + perturbation
            adv_field = torch.clamp(adv_field, 0, 1)

        return adv_field

    def pgd_attack_simsac(self, field_img, reference_img, steps=10, step_size=None):
        """
        PGD attack targeting SimSAC optical flow.

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

            # Compute SimSAC flow loss
            loss = self.simsac_flow_loss(adv_field, reference_img)

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
        Generate adversarial version of field UV map targeting SimSAC.

        Args:
            field_path: Path to field UV map
            reference_path: Path to reference UV map
            attack_type: 'fgsm' or 'pgd'
            pgd_steps: Number of PGD steps

        Returns:
            Adversarial UV map as numpy array
        """
        # Load images
        field_img = Image.open(field_path).convert('RGB')
        reference_img = Image.open(reference_path).convert('RGB')

        # Convert to tensors [1, C, H, W]
        field_tensor = torch.from_numpy(np.array(field_img)).permute(2, 0, 1).float() / 255.0
        reference_tensor = torch.from_numpy(np.array(reference_img)).permute(2, 0, 1).float() / 255.0

        field_tensor = field_tensor.unsqueeze(0).to(self.device)
        reference_tensor = reference_tensor.unsqueeze(0).to(self.device)

        # Generate adversarial image
        # Note: model is already in train mode from __init__

        if attack_type == 'fgsm':
            adv_field = self.fgsm_attack_simsac(field_tensor, reference_tensor)
        elif attack_type == 'pgd':
            adv_field = self.pgd_attack_simsac(field_tensor, reference_tensor, steps=pgd_steps)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # Convert back to numpy
        adv_uvmap_np = adv_field[0].detach().cpu().permute(1, 2, 0).numpy()
        adv_uvmap_np = (adv_uvmap_np * 255).astype(np.uint8)

        return adv_uvmap_np


def load_simsac_model(checkpoint_path, device='cuda'):
    """Load pre-trained SimSAC model."""
    print(f"Loading SimSAC model from {checkpoint_path}...")

    model = SimSaC_Model(
        evaluation=True,
        pyramid_type='VGG',
        md=4,
        cyclic_consistency=True,
        use_pac=False  # Disable PAC to avoid import errors
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("✓ SimSAC model loaded successfully")

    return model


def find_reference_uvmap(field_uvmap_path, uvmaps_dir):
    """
    Find corresponding reference UV map for a field image.

    Args:
        field_uvmap_path: Path to field UV map (e.g., id_05_20230516_142710_uvmap_gt.png)
        uvmaps_dir: Directory containing reference UV maps

    Returns:
        Path to reference UV map (e.g., id_05_uvmap.png)
    """
    # Extract parcel ID from filename
    filename = field_uvmap_path.name
    # Format: id_XX_YYYYMMDD_HHMMSS_uvmap_gt.png
    parcel_id = filename.split('_')[1]  # Get XX

    # Reference format: id_XX_uvmap.png
    reference_name = f"id_{parcel_id}_uvmap.png"
    reference_path = uvmaps_dir / reference_name

    if not reference_path.exists():
        # Try alternative patterns
        alternatives = list(uvmaps_dir.glob(f"id_{parcel_id}_*.png"))
        if alternatives:
            return alternatives[0]
        else:
            raise FileNotFoundError(f"Reference UV map not found: {reference_path}")

    return reference_path


def generate_adversarial_dataset(data_dir, uvmaps_dir, output_dir, simsac_checkpoint,
                                 attack_type='fgsm', folders=None, exclude_folders=None,
                                 uvmap_types=['gt'], epsilon=0.1, pgd_steps=10):
    """
    Generate SimSAC-targeted adversarial UV maps.

    Args:
        data_dir: Input validation directory
        uvmaps_dir: Directory with reference UV maps
        output_dir: Output directory for adversarial data
        simsac_checkpoint: Path to SimSAC checkpoint
        attack_type: 'fgsm', 'pgd', or 'both'
        folders: List of folder names to include (e.g., ['carpet', 'table'])
        exclude_folders: List of folder names to exclude (e.g., ['base'])
        uvmap_types: List of UV map types ('gt', 'pred')
        epsilon: Perturbation magnitude
        pgd_steps: Number of PGD iterations
    """
    data_dir = Path(data_dir)
    uvmaps_dir = Path(uvmaps_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("SimSAC-Targeted Adversarial UV Map Generation")
    print(f"{'='*70}")
    print(f"Input directory:     {data_dir}")
    print(f"Reference UV maps:   {uvmaps_dir}")
    print(f"Output directory:    {output_dir}")
    print(f"Attack type(s):      {attack_type}")
    print(f"UV map types:        {uvmap_types}")
    print(f"Epsilon:             {epsilon}")
    if folders:
        print(f"Include folders:     {', '.join(folders)}")
    if exclude_folders:
        print(f"Exclude folders:     {', '.join(exclude_folders)}")
    if 'pgd' in attack_type.lower():
        print(f"PGD steps:           {pgd_steps}")

    # Load SimSAC model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device:              {device}")

    simsac_model = load_simsac_model(simsac_checkpoint, device=device)

    # Initialize generator
    generator = SimSaCTargetedAttackGenerator(simsac_model, epsilon=epsilon, device=device)

    # Determine attack types
    if attack_type == 'both':
        attacks = ['fgsm', 'pgd']
    else:
        attacks = [attack_type]

    # Find all subdirectories (backgrounds)
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]

    # Filter by folders/exclude_folders
    if folders:
        subdirs = [d for d in subdirs if d.name in folders]
    if exclude_folders:
        subdirs = [d for d in subdirs if d.name not in exclude_folders]

    print(f"\nProcessing {len(subdirs)} folder(s): {', '.join([d.name for d in subdirs])}")

    # Process each folder
    total_uvmaps = 0
    for subdir in subdirs:
        print(f"\nProcessing folder: {subdir.name}")

        # Find UV map files
        uvmap_files = []
        for uvmap_type in uvmap_types:
            pattern = f"*_uvmap_{uvmap_type}.png"
            found = list(subdir.glob(pattern))
            uvmap_files.extend(found)

        print(f"  Found {len(uvmap_files)} UV map files")

        # Process each attack type
        for attack in attacks:
            attack_suffix = f"_adv_{attack}"
            output_subdir = output_dir / f"{subdir.name}{attack_suffix}"
            output_subdir.mkdir(parents=True, exist_ok=True)

            # Process each UV map
            for field_path in tqdm(uvmap_files, desc=f"  {attack.upper()} attack"):
                try:
                    # Find corresponding reference
                    reference_path = find_reference_uvmap(field_path, uvmaps_dir)

                    # Generate adversarial UV map
                    adv_uvmap = generator.generate_adversarial_uvmap(
                        field_path,
                        reference_path,
                        attack_type=attack,
                        pgd_steps=pgd_steps
                    )

                    # Save with same filename
                    output_path = output_subdir / field_path.name
                    Image.fromarray(adv_uvmap).save(output_path)
                    total_uvmaps += 1

                except Exception as e:
                    print(f"    Error processing {field_path.name}: {e}")

    print(f"\n{'='*70}")
    print("✓ SimSAC-targeted adversarial generation complete!")
    print(f"{'='*70}")
    print(f"Total adversarial UV maps generated: {total_uvmaps}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate SimSAC-targeted adversarial UV maps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--data_dir', type=str, required=True,
                       help='Input validation directory')
    parser.add_argument('--uvmaps_dir', type=str, required=True,
                       help='Directory with reference UV maps')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for adversarial data')
    parser.add_argument('--simsac_checkpoint', type=str,
                       default='src/simsac/weight/synthetic.pth',
                       help='Path to SimSAC checkpoint')

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

    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Perturbation magnitude. Recommended: 0.1-0.15')

    parser.add_argument('--pgd_steps', type=int, default=10,
                       help='Number of PGD iterations')

    args = parser.parse_args()

    generate_adversarial_dataset(
        data_dir=args.data_dir,
        uvmaps_dir=args.uvmaps_dir,
        output_dir=args.output_dir,
        simsac_checkpoint=args.simsac_checkpoint,
        attack_type=args.attack,
        folders=args.folders,
        exclude_folders=args.exclude_folders,
        uvmap_types=args.uvmap_types,
        epsilon=args.epsilon,
        pgd_steps=args.pgd_steps
    )


if __name__ == "__main__":
    main()
