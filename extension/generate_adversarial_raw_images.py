
import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.simsac.models.our_models.SimSaC import SimSaC_Model


# SimSAC model loading (same as generate_adversarial_simsac_targeted.py)

def load_simsac_model(checkpoint_path, device='cuda'):
    print(f"Loading SimSAC model from {checkpoint_path}")

    model = SimSaC_Model(
        evaluation=False,
        pyramid_type='VGG',
        md=4,
        cyclic_consistency=True,
        use_pac=False
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.train()

    # Monkey-patch to remove no_grad() wrapper so gradients can flow
    print("  Enabling gradient flow through SimSAC (removing no_grad wrapper)")

    def forward_with_grad(self, im_target, im_source, im_target_256, im_source_256, disable_flow=None):
        b, _, h_full, w_full = im_target.size()

        im1_pyr = self.pyramid(im_target, eigth_resolution=True)
        im2_pyr = self.pyramid(im_source, eigth_resolution=True)
        c11 = im1_pyr[-2]
        c21 = im2_pyr[-2]
        c12 = im1_pyr[-1]
        c22 = im2_pyr[-1]

        im1_pyr_256 = self.pyramid(im_target_256)
        im2_pyr_256 = self.pyramid(im_source_256)
        c13 = im1_pyr_256[-4]
        c23 = im2_pyr_256[-4]
        c14 = im1_pyr_256[-3]
        c24 = im2_pyr_256[-3]

        flow4, corr4 = self.coarsest_resolution_flow(c14, c24, h_256=256, w_256=256, return_corr=True)

        return {
            "flow": ([flow4], [flow4]),
            "change": ([corr4], [corr4])
        }

    import types
    model.forward_sigle_ref = types.MethodType(forward_with_grad, model)

    print(" SimSAC model loaded with gradient flow enabled")
    return model


# Attack generator targeting SimSAC change map

class RawImageAttackGenerator:

    def __init__(self, simsac_model, epsilon=0.05, device='cuda'):
        self.simsac = simsac_model
        self.simsac.train()
        self.epsilon = epsilon
        self.device = device

    def simsac_change_loss(self, field_img, reference_img):
        field_512 = F.interpolate(field_img, size=(512, 512), mode='bilinear', align_corners=False)
        reference_512 = F.interpolate(reference_img, size=(512, 512), mode='bilinear', align_corners=False)
        field_256 = F.interpolate(field_img, size=(256, 256), mode='bilinear', align_corners=False)
        reference_256 = F.interpolate(reference_img, size=(256, 256), mode='bilinear', align_corners=False)

        output = self.simsac(field_512, reference_512, field_256, reference_256)

        change = None
        if isinstance(output, dict):
            change_tuple = output.get('change', None)
            if change_tuple is not None and isinstance(change_tuple, tuple) and len(change_tuple) == 2:
                if len(change_tuple[1]) > 0:
                    change = change_tuple[1][-1]
                elif len(change_tuple[0]) > 0:
                    change = change_tuple[0][-1]
            else:
                flow_tuple = output.get('flow', None)
                if flow_tuple is not None:
                    change = flow_tuple[1][-1] if len(flow_tuple[1]) > 0 else flow_tuple[0][-1]
        elif isinstance(output, (list, tuple)) and len(output) >= 2:
            change = output[1]

        if change is None:
            raise ValueError(f"SimSAC did not return change output. Got: {type(output)}")

        change_magnitude = torch.sqrt(change[:, 0] ** 2 + change[:, 1] ** 2)
        loss = change_magnitude.mean()
        return loss

    def fgsm_attack(self, field_img, reference_img):
        field_img = field_img.clone().detach()
        reference_img = reference_img.clone().detach()
        field_img.requires_grad = True

        loss = self.simsac_change_loss(field_img, reference_img)

        if field_img.grad is not None:
            field_img.grad.zero_()
        loss.backward()

        if field_img.grad is None:
            raise ValueError("Gradient is None after backward pass!")

        with torch.no_grad():
            perturbation = self.epsilon * field_img.grad.sign()
            adv_field = field_img + perturbation
            adv_field = torch.clamp(adv_field, 0, 1)

        return adv_field

    def pgd_attack(self, field_img, reference_img, steps=10, step_size=None):
        if step_size is None:
            step_size = self.epsilon / 4

        original_field = field_img.clone().detach()
        adv_field = field_img.clone().detach()

        for step in range(steps):
            adv_field.requires_grad = True
            loss = self.simsac_change_loss(adv_field, reference_img)
            loss.backward()

            with torch.no_grad():
                adv_field = adv_field + step_size * adv_field.grad.sign()
                perturbation = torch.clamp(adv_field - original_field, -self.epsilon, self.epsilon)
                adv_field = torch.clamp(original_field + perturbation, 0, 1)

            adv_field = adv_field.detach()

        return adv_field

    def attack_image(self, field_path, reference_path, attack_type='fgsm', pgd_steps=10):
        # Load field image (JPEG)
        field_img = Image.open(field_path).convert('RGB')
        original_size = field_img.size

        # Load reference UV map
        reference_img = Image.open(reference_path).convert('RGB')

        # Convert to tensors [1, C, H, W] in [0, 1]
        field_tensor = torch.from_numpy(np.array(field_img)).permute(2, 0, 1).float() / 255.0
        reference_tensor = torch.from_numpy(np.array(reference_img)).permute(2, 0, 1).float() / 255.0

        field_tensor = field_tensor.unsqueeze(0).to(self.device)
        reference_tensor = reference_tensor.unsqueeze(0).to(self.device)

        # Run attack
        if attack_type == 'fgsm':
            adv_tensor = self.fgsm_attack(field_tensor, reference_tensor)
        elif attack_type == 'pgd':
            adv_tensor = self.pgd_attack(field_tensor, reference_tensor, steps=pgd_steps)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}. Use 'fgsm' or 'pgd'.")

        # Convert back to numpy uint8
        adv_np = adv_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
        adv_np = (adv_np * 255).astype(np.uint8)

        # Resize back to original image dimensions (in case model resized internally)
        adv_pil = Image.fromarray(adv_np)
        if adv_pil.size != original_size:
            adv_pil = adv_pil.resize(original_size, Image.LANCZOS)
            adv_np = np.array(adv_pil)

        return adv_np


# Reference UV map lookup

def find_reference_uvmap(field_image_path, uvmaps_dir):
    filename = field_image_path.stem
    parcel_id = filename.split('_')[1]

    reference_name = f"id_{parcel_id}_uvmap.png"
    reference_path = uvmaps_dir / reference_name

    if not reference_path.exists():
        # Try alternative patterns
        alternatives = list(uvmaps_dir.glob(f"id_{parcel_id}_*.png"))
        if alternatives:
            return alternatives[0]
        raise FileNotFoundError(
            f"Reference UV map not found for parcel {parcel_id}. "
            f"Expected: {reference_path}"
        )

    return reference_path


# Main dataset generation logic

def generate_adversarial_raw_dataset(
    data_dir,
    output_dir,
    simsac_checkpoint,
    uvmaps_dir=None,
    attack_type='fgsm',
    folders=None,
    exclude_folders=None,
    epsilon=0.05,
    pgd_steps=10,
    copy_originals=True,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect uvmaps_dir
    if uvmaps_dir is None:
        uvmaps_dir = data_dir.parent / 'uvmaps'
        if not uvmaps_dir.exists():
            raise ValueError(
                f"Could not auto-detect uvmaps_dir. Expected at {uvmaps_dir}. "
                f"Please provide --uvmaps_dir explicitly."
            )
        print(f"Auto-detected uvmaps_dir: {uvmaps_dir}")
    else:
        uvmaps_dir = Path(uvmaps_dir)

    # Determine attack types
    attacks = ['fgsm', 'pgd'] if attack_type == 'both' else [attack_type]

    print("Adversarial Image Generation (SimSAC-Targeted)")
    print(f"Input directory:     {data_dir}")
    print(f"Reference UV maps:   {uvmaps_dir}")
    print(f"Output directory:    {output_dir}")
    print(f"In-place mode:       {data_dir.resolve() == output_dir.resolve()}")
    print(f"Attack type(s):      {attacks}")
    print(f"Epsilon:             {epsilon}")
    if 'pgd' in attacks:
        print(f"PGD steps:           {pgd_steps}")

    # Discover and filter subdirectories
    subdirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if folders:
        subdirs = [d for d in subdirs if d.name in folders]
    if exclude_folders:
        subdirs = [d for d in subdirs if d.name not in exclude_folders]

    print(f"\nProcessing {len(subdirs)} folder(s): {', '.join(d.name for d in subdirs)}")

    # Load device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device:              {device}")

    # Load SimSAC model
    simsac_model = load_simsac_model(simsac_checkpoint, device=device)
    generator = RawImageAttackGenerator(simsac_model, epsilon=epsilon, device=device)

    in_place = data_dir.resolve() == output_dir.resolve()

    stats = {attack: 0 for attack in attacks}
    errors = 0

    for subdir in subdirs:
        print(f"\n--- Folder: {subdir.name} ---")

        raw_images = []
        for f in sorted(subdir.glob("*.jpg")):
            stem = f.stem
            parts = stem.split('_')
            if len(parts) == 4 and parts[0] == 'id':
                raw_images.append(f)

        print(f"  Found {len(raw_images)} raw field images")

        if not raw_images:
            print(f"  Skipping (no raw images found)")
            continue

        # Create output subfolder
        out_subdir = output_dir / subdir.name
        out_subdir.mkdir(parents=True, exist_ok=True)

        # Copy originals if needed (and not in-place)
        if copy_originals and not in_place:
            import shutil
            for f in subdir.iterdir():
                dest = out_subdir / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)
            print(f"  Copied original files to output folder")

        # Attack each raw image
        for field_path in tqdm(raw_images, desc=f"  Attacking {subdir.name}"):
            try:
                reference_path = find_reference_uvmap(field_path, uvmaps_dir)
            except FileNotFoundError as e:
                print(f"   {e}")
                errors += 1
                continue

            for attack in attacks:
                # Output filename: id_01_20230516_142710_fgsm.jpg
                out_filename = f"{field_path.stem}_{attack}.jpg"
                out_path = out_subdir / out_filename

                if out_path.exists():
                    print(f"  Skipping (already exists): {out_filename}")
                    stats[attack] += 1
                    continue

                try:
                    adv_np = generator.attack_image(
                        field_path,
                        reference_path,
                        attack_type=attack,
                        pgd_steps=pgd_steps,
                    )
                    # Save as JPEG (quality=95 to preserve detail)
                    Image.fromarray(adv_np).save(out_path, quality=95)
                    stats[attack] += 1

                except Exception as e:
                    print(f"   Error on {field_path.name} ({attack}): {e}")
                    errors += 1

    print(" Adversarial raw image generation complete!")
    for attack, count in stats.items():
        print(f"  {attack.upper()} adversarial images: {count}")
    print(f"  Errors / skipped:        {errors}")
    print(f"\nNext step: Run UV map generation (MaskRCNN pipeline) on {output_dir}")
    print("  to produce *_fgsm_uvmap_gt.png / *_pgd_uvmap_gt.png files.")


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Attack raw field images (before UV map generation) using SimSAC-targeted FGSM/PGD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Input directory containing background subfolders with .jpg field images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory. Can be same as data_dir to write in-place.')

    # Optional
    parser.add_argument('--uvmaps_dir', type=str, default=None,
                        help='Directory with reference UV maps (id_XX_uvmap.png). '
                             'Auto-detected as data_dir/../uvmaps if not provided.')
    parser.add_argument('--simsac_checkpoint', type=str,
                        default='src/simsac/weight/synthetic.pth',
                        help='Path to SimSAC checkpoint .pth file (default: synthetic.pth)')

    # Attack settings
    parser.add_argument('--attack', type=str, choices=['fgsm', 'pgd', 'both'],
                        default='fgsm',
                        help='Attack type: fgsm, pgd, or both (default: fgsm)')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Perturbation magnitude in [0,1] pixel range (default: 0.05)')
    parser.add_argument('--pgd_steps', type=int, default=10,
                        help='Number of PGD iterations (default: 10)')

    # Folder filtering
    parser.add_argument('--folders', type=str, nargs='+', default=None,
                        help='Only process these subfolders (e.g., --folders carpet gravel table). '
                             'If not set, all folders are processed.')
    parser.add_argument('--exclude_folders', type=str, nargs='+', default=None,
                        help='Exclude these subfolders (e.g., --exclude_folders base). '
                             'Supports exact names; base_adv_* style folders are matched by prefix.')

    # Misc
    parser.add_argument('--no_copy_originals', action='store_true', default=False,
                        help='When output_dir != data_dir, skip copying original files. '
                             'Default: originals are copied so output_dir is self-contained.')

    args = parser.parse_args()


    generate_adversarial_raw_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        simsac_checkpoint=args.simsac_checkpoint,
        uvmaps_dir=args.uvmaps_dir,
        attack_type=args.attack,
        folders=args.folders,
        exclude_folders=args.exclude_folders,
        epsilon=args.epsilon,
        pgd_steps=args.pgd_steps,
        copy_originals=not args.no_copy_originals,
    )


if __name__ == "__main__":
    main()
