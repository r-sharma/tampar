
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
import math


def pytorch_ssim(img1, img2, window_size=11):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    gauss = gauss / gauss.sum()

    # Create 2D Gaussian kernel
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(img1.size(1), 1, window_size, window_size).contiguous()
    window = window.to(img1.device)

    # Compute means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2

    # Compute SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def pytorch_mae(img1, img2):
    return torch.mean(torch.abs(img1 - img2))


def pytorch_mse(img1, img2):
    return torch.mean((img1 - img2) ** 2)


class SimilarityTargetedAttackGenerator:

    def __init__(self, epsilon=0.15, device='cuda'):
        self.epsilon = epsilon
        self.device = device

    def similarity_loss(self, field_img, reference_img):
        # Compute differentiable similarity metrics
        ssim_value = pytorch_ssim(field_img, reference_img)
        mae_value = pytorch_mae(field_img, reference_img)
        mse_value = pytorch_mse(field_img, reference_img)

        loss = -ssim_value + 10.0 * mae_value + mse_value

        return loss

    def fgsm_attack_similarity(self, field_img, reference_img):
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
    print(f"PGD iterations: {pgd_steps}")

    # Determine which attacks to run
    attacks_to_run = []
    if attack_type in ['fgsm', 'both']:
        attacks_to_run.append('fgsm')
    if attack_type in ['pgd', 'both']:
        attacks_to_run.append('pgd')

    for attack_method in attacks_to_run:
        print(f"Generating {attack_method.upper()} adversarial examples")

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
                    filename_parts = uvmap_file.stem.split('_')
                    parcel_id = filename_parts[1]

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

            print(f"   Successful: {successful},  Failed: {failed}")

    print(f"Adversarial dataset generation complete!")
    print(f"Output directory: {output_path}")


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
