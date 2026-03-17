
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


class AdversarialUVMapGenerator:

    def __init__(self, epsilon=0.05, device='cuda'):
        self.epsilon = epsilon
        self.device = device

    def fgsm_attack(self, image, gradient):
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

    def cw_attack(self, image, loss_fn, steps=100, c=1.0, kappa=0, learning_rate=0.01):
        # Initialize perturbation in tanh space for automatic clamping
        # w is optimized in unbounded space, tanh(w) maps to [0,1]
        w = torch.zeros_like(image, requires_grad=True, device=self.device)

        # Initialize to original image in tanh space
        # inverse tanh: atanh(2*x - 1) maps [0,1] -> unbounded
        w.data = torch.atanh(2 * image - 1)

        # Use Adam optimizer for smoother convergence
        optimizer = torch.optim.Adam([w], lr=learning_rate)

        best_adv = image.clone()
        best_loss = float('inf')

        for step in range(steps):
            optimizer.zero_grad()

            # Transform to valid image range [0,1]
            adv_image = (torch.tanh(w) + 1) / 2

            # Compute adversarial loss
            adv_loss = loss_fn(adv_image)

            # Compute L2 perturbation
            l2_perturbation = torch.norm(adv_image - image, p=2)

            # C&W objective: minimize perturbation while maximizing adversarial loss
            # Note: We negate adv_loss because we want to maximize it
            total_loss = l2_perturbation + c * (-adv_loss)

            total_loss.backward()
            optimizer.step()

            # Track best adversarial example
            if adv_loss.item() > best_loss:
                best_loss = adv_loss.item()
                best_adv = adv_image.detach().clone()

        # Ensure perturbation is within epsilon ball (L-inf constraint)
        perturbation = torch.clamp(best_adv - image, -self.epsilon, self.epsilon)
        adv_image = torch.clamp(image + perturbation, 0, 1)

        return adv_image

    def texture_loss(self, image):
        # Compute gradients (high-frequency content)
        dx = image[:, :, 1:] - image[:, :, :-1]
        dy = image[:, 1:, :] - image[:, :-1, :]

        # Negative L2 norm (maximize variation)
        loss = -(dx.pow(2).mean() + dy.pow(2).mean())

        return loss

    def smoothness_loss(self, image):
        # Compute second-order gradients
        dx = image[:, :, 1:] - image[:, :, :-1]
        dy = image[:, 1:, :] - image[:, :-1, :]

        ddx = dx[:, :, 1:] - dx[:, :, :-1]
        ddy = dy[:, 1:, :] - dy[:, :-1, :]

        # Minimize second-order gradients (create smooth regions)
        loss = ddx.pow(2).mean() + ddy.pow(2).mean()

        return loss

    def generate_adversarial_uvmap(self, uvmap_path, attack_type='fgsm',
                                   texture_weight=1.0, smoothness_weight=1.0,
                                   pgd_steps=10, cw_steps=100, cw_c=1.0, cw_lr=0.01):
        # Load UV map
        uvmap = Image.open(uvmap_path).convert('RGB')
        uvmap_np = np.array(uvmap).astype(np.float32) / 255.0

        # Convert to tensor
        uvmap_tensor = torch.from_numpy(uvmap_np).permute(2, 0, 1).to(self.device)
        uvmap_tensor = uvmap_tensor.unsqueeze(0)

        if attack_type == 'fgsm':
            # FGSM: Single-step attack
            uvmap_tensor.requires_grad = True

            # Combined loss: texture variation + smoothness
            loss = (texture_weight * self.texture_loss(uvmap_tensor[0]) +
                   smoothness_weight * self.smoothness_loss(uvmap_tensor[0]))

            # Compute gradient
            loss.backward()

            # Generate adversarial image
            adv_uvmap = self.fgsm_attack(uvmap_tensor[0], uvmap_tensor.grad[0])

        elif attack_type == 'pgd':
            # PGD: Multi-step attack
            def loss_fn(img):
                return (texture_weight * self.texture_loss(img) +
                       smoothness_weight * self.smoothness_loss(img))

            adv_uvmap = self.pgd_attack(uvmap_tensor[0], loss_fn, steps=pgd_steps)

        elif attack_type == 'cw':
            # C&W: Optimization-based attack
            def loss_fn(img):
                return (texture_weight * self.texture_loss(img) +
                       smoothness_weight * self.smoothness_loss(img))

            adv_uvmap = self.cw_attack(uvmap_tensor[0], loss_fn, steps=cw_steps,
                                      c=cw_c, learning_rate=cw_lr)

        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # Convert back to numpy
        adv_uvmap_np = adv_uvmap.detach().cpu().permute(1, 2, 0).numpy()
        adv_uvmap_np = (adv_uvmap_np * 255).astype(np.uint8)

        return adv_uvmap_np


def generate_adversarial_dataset(data_dir, output_dir, attack_type='fgsm',
                                uvmap_types=['gt', 'pred'], epsilon=0.05,
                                texture_weight=1.0, smoothness_weight=1.0,
                                pgd_steps=10, cw_steps=100, cw_c=1.0, cw_lr=0.01,
                                filter_folders=None):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Adversarial UV Map Generation")
    print(f"Input directory:  {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Attack type(s):   {attack_type}")
    print(f"UV map types:     {uvmap_types}")
    print(f"Epsilon:          {epsilon}")
    print(f"Texture weight:   {texture_weight}")
    print(f"Smoothness weight: {smoothness_weight}")
    if 'pgd' in attack_type.lower():
        print(f"PGD steps:        {pgd_steps}")

    # Initialize generator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device:           {device}")

    generator = AdversarialUVMapGenerator(epsilon=epsilon, device=device)

    # Determine attack types to run
    if attack_type == 'all':
        attacks = ['fgsm', 'pgd', 'cw']
    elif attack_type == 'both':
        attacks = ['fgsm', 'pgd']
    else:
        attacks = [attack_type]

    # Find all UV map files recursively
    uvmap_files = []

    # Search for uvmap_gt.png and uvmap_pred.png files
    for uvmap_type in uvmap_types:
        pattern = f"*_uvmap_{uvmap_type}.png"
        found_files = list(data_dir.rglob(pattern))
        uvmap_files.extend(found_files)

    print(f"\nFound {len(uvmap_files)} UV map files across all subdirectories")

    # Group by parent directory (surface type)
    from collections import defaultdict
    files_by_dir = defaultdict(list)
    for f in uvmap_files:
        files_by_dir[f.parent].append(f)

    # Filter by folder names if specified
    if filter_folders:
        filtered_files_by_dir = {}
        for dir_path, files in files_by_dir.items():
            rel_path = dir_path.relative_to(data_dir)
            # Check if any part of the path matches the filter
            path_parts = rel_path.parts
            if any(folder_name in path_parts for folder_name in filter_folders):
                filtered_files_by_dir[dir_path] = files
        files_by_dir = filtered_files_by_dir
        print(f"\nFiltered to folders: {filter_folders}")

    print(f"Organized into {len(files_by_dir)} subdirectories:")
    for dir_path in sorted(files_by_dir.keys()):
        rel_path = dir_path.relative_to(data_dir)
        print(f"  {rel_path}: {len(files_by_dir[dir_path])} files")

    # Process each subdirectory
    total_uvmaps = 0
    for subdir_path in tqdm(sorted(files_by_dir.keys()), desc="Processing subdirectories"):
        subdir_name = subdir_path.relative_to(data_dir)

        # Process each attack type
        for attack in attacks:
            # Create output directory mirroring input structure
            attack_suffix = f"_adv_{attack}"
            output_subdir = output_dir / f"{subdir_name}{attack_suffix}"
            output_subdir.mkdir(parents=True, exist_ok=True)

            # Process each UV map file in this directory
            for uvmap_path in files_by_dir[subdir_path]:
                # Determine uvmap type from filename
                if '_uvmap_gt.png' in uvmap_path.name:
                    uvmap_type = 'gt'
                elif '_uvmap_pred.png' in uvmap_path.name:
                    uvmap_type = 'pred'
                else:
                    continue

                # Skip if this type is not requested
                if uvmap_type not in uvmap_types:
                    continue

                try:
                    # Generate adversarial UV map
                    adv_uvmap = generator.generate_adversarial_uvmap(
                        uvmap_path,
                        attack_type=attack,
                        texture_weight=texture_weight,
                        smoothness_weight=smoothness_weight,
                        pgd_steps=pgd_steps,
                        cw_steps=cw_steps,
                        cw_c=cw_c,
                        cw_lr=cw_lr
                    )

                    # Save adversarial UV map with same filename
                    output_uvmap_path = output_subdir / uvmap_path.name
                    Image.fromarray(adv_uvmap).save(output_uvmap_path)
                    total_uvmaps += 1

                except Exception as e:
                    print(f"  Error generating {uvmap_path.name}: {e}")

    print("✓ Adversarial dataset generation complete!")
    print(f"\nOutput directory: {output_dir}")

    # Print statistics
    for attack in attacks:
        uvmap_count = 0

        # Count files in attack-specific subdirectories
        attack_suffix = f"_adv_{attack}"
        for subdir_name in files_by_dir.keys():
            output_subdir = output_dir / f"{subdir_name.relative_to(data_dir)}{attack_suffix}"
            if output_subdir.exists():
                uvmap_count += len(list(output_subdir.glob("*_uvmap_*.png")))

        print(f"\n{attack.upper()} Attack:")
        print(f"  Adversarial UV maps: {uvmap_count}")

    print(f"\nTotal adversarial UV maps generated: {total_uvmaps}")


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

    parser.add_argument('--attack', type=str, choices=['fgsm', 'pgd', 'cw', 'both', 'all'],
                       default='fgsm',
                       help='Attack type: fgsm (fast), pgd (iterative), cw (optimization), both (fgsm+pgd), or all (fgsm+pgd+cw)')

    parser.add_argument('--uvmap_types', type=str, nargs='+',
                       choices=['gt', 'pred'], default=['gt'],
                       help='UV map types to generate (gt, pred, or both). Default: gt only')

    parser.add_argument('--epsilon', type=float, default=0.05,
                       help='Perturbation magnitude (L-inf norm). Higher = stronger attack. '
                            'Recommended: 0.03-0.1')

    parser.add_argument('--texture_weight', type=float, default=1.0,
                       help='Weight for texture loss (maximizes high-frequency content). '
                            '1.0=default, 0=disable, negative=reduce texture. '
                            'Recommended: 0 (texture only), 1.0 (balanced), or 2.0 (more texture)')

    parser.add_argument('--smoothness_weight', type=float, default=0.0,
                       help='Weight for smoothness loss (minimizes curvature). '
                            '0=disable (recommended), positive=smooth surfaces, negative=rough surfaces. '
                            'Recommended: 0 (disable), 0.1 (slight smooth), or -1.0 (maximize roughness)')

    parser.add_argument('--pgd_steps', type=int, default=10,
                       help='Number of PGD iterations (only for PGD attack). '
                            'More steps = stronger attack. Recommended: 5-20')

    parser.add_argument('--cw_steps', type=int, default=100,
                       help='Number of C&W optimization steps (only for C&W attack). '
                            'More steps = better convergence. Recommended: 50-200')

    parser.add_argument('--cw_c', type=float, default=1.0,
                       help='C&W regularization constant balancing perturbation vs attack. '
                            'Higher = prioritize attack success. Recommended: 0.1-10')

    parser.add_argument('--cw_lr', type=float, default=0.01,
                       help='C&W learning rate for Adam optimizer. '
                            'Recommended: 0.001-0.05')

    parser.add_argument('--filter_folders', type=str, nargs='+', default=None,
                       help='Only process specific folders (e.g., carpet, gravel, table). '
                            'If not specified, all folders are processed.')

    args = parser.parse_args()

    generate_adversarial_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        attack_type=args.attack,
        uvmap_types=args.uvmap_types,
        epsilon=args.epsilon,
        texture_weight=args.texture_weight,
        smoothness_weight=args.smoothness_weight,
        pgd_steps=args.pgd_steps,
        cw_steps=args.cw_steps,
        cw_c=args.cw_c,
        cw_lr=args.cw_lr,
        filter_folders=args.filter_folders
    )


if __name__ == "__main__":
    main()
