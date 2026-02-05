"""
Surface-Level Contrastive Pair Creation for TAMPAR Dataset
Based on the contrastive learning strategy table.

This version creates pairs at the SURFACE level (top, left, center, right, bottom)
and uses tampering_mapping.csv to ensure positive pairs only use untampered surfaces.
"""

import os
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import pickle
import cv2
import sys

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from src.tampering.utils import get_side_surface_patches
from src.tampering.parcel import PATCH_ORDER


class TamperingMapping:
    """Load and query tampering information from tampering_mapping.csv"""

    def __init__(self, csv_path):
        """
        Load tampering mapping.

        Args:
            csv_path: Path to tampering_mapping.csv
        """
        self.mapping = {}
        self.surface_names = ['center', 'top', 'bottom', 'left', 'right']

        if not Path(csv_path).exists():
            print(f"⚠ Warning: tampering_mapping.csv not found at {csv_path}")
            return

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                parcel_id = int(row['id'])
                self.mapping[parcel_id] = {
                    'center': row['center'].strip(),
                    'top': row['top'].strip(),
                    'bottom': row['bottom'].strip(),
                    'left': row['left'].strip(),
                    'right': row['right'].strip()
                }

        print(f"✓ Loaded tampering mapping for {len(self.mapping)} parcels")

    def is_surface_tampered(self, parcel_id, surface_name):
        """
        Check if a surface is tampered.

        Args:
            parcel_id: Parcel ID (int)
            surface_name: Surface name ('center', 'top', 'bottom', 'left', 'right')

        Returns:
            True if tampered, False if clean
        """
        if parcel_id not in self.mapping:
            return False  # Assume clean if no mapping

        tampering_code = self.mapping[parcel_id].get(surface_name, '')
        return tampering_code != ''  # Empty string means no tampering

    def get_tampering_code(self, parcel_id, surface_name):
        """Get tampering code for a surface."""
        if parcel_id not in self.mapping:
            return ''
        return self.mapping[parcel_id].get(surface_name, '')

    def get_clean_surfaces(self, parcel_id):
        """Get list of clean surface names for a parcel."""
        if parcel_id not in self.mapping:
            return self.surface_names

        clean = []
        for surf in self.surface_names:
            if not self.is_surface_tampered(parcel_id, surf):
                clean.append(surf)
        return clean

    def get_tampered_surfaces(self, parcel_id):
        """Get list of tampered surface names for a parcel."""
        if parcel_id not in self.mapping:
            return []

        tampered = []
        for surf in self.surface_names:
            if self.is_surface_tampered(parcel_id, surf):
                tampered.append(surf)
        return tampered


class SurfaceExtractor:
    """Extract individual surfaces from UV map images."""

    @staticmethod
    def extract_surfaces(uv_map_image):
        """
        Extract individual surfaces from UV map.

        Args:
            uv_map_image: PIL Image or numpy array of UV map

        Returns:
            Dictionary mapping surface_name -> surface_image (numpy array)
        """
        # Convert PIL to numpy if needed
        if isinstance(uv_map_image, Image.Image):
            uv_map_array = np.array(uv_map_image)
        else:
            uv_map_array = uv_map_image

        # Extract patches using TAMPAR's utility function
        patches = get_side_surface_patches(uv_map_array, grid_size=3)

        # Map patches to surface names using PATCH_ORDER
        surfaces = {}
        for i, (name, patch) in enumerate(zip(PATCH_ORDER, patches)):
            if name != "":  # Skip empty slots in PATCH_ORDER
                # Check if patch is not mostly white (mean < 250)
                if np.mean(patch) < 250:
                    surfaces[name] = patch

        return surfaces


class AugmentationPipeline:
    """Augmentation for creating positive pairs."""

    def __init__(self, rotation_range=5, brightness_range=0.1,
                 contrast_range=0.1, noise_std=0.02):
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std

    def augment(self, image):
        """Apply augmentation to a surface patch."""
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        # Random rotation
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        image = image.rotate(angle, resample=Image.BILINEAR)

        # Random brightness
        brightness_factor = 1.0 + random.uniform(-self.brightness_range, self.brightness_range)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)

        # Random contrast
        contrast_factor = 1.0 + random.uniform(-self.contrast_range, self.contrast_range)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)

        # Add Gaussian noise
        img_array = np.array(image).astype(np.float32) / 255.0
        noise = np.random.normal(0, self.noise_std, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 1)
        image = Image.fromarray((img_array * 255).astype(np.uint8))

        return np.array(image)


class SurfaceLevelPairCreator:
    """Create surface-level contrastive pairs according to the strategy table."""

    def __init__(self, data_root, adversarial_root=None, tampering_csv_path=None, random_seed=42):
        """
        Initialize pair creator.

        Args:
            data_root: Path to TAMPAR dataset
            adversarial_root: Path to adversarial TAMPAR dataset (optional)
            tampering_csv_path: Path to tampering_mapping.csv
            random_seed: Random seed
        """
        self.data_root = Path(data_root)
        self.adversarial_root = Path(adversarial_root) if adversarial_root else None

        # Check if data_root ends with a split name (validation/test)
        # If so, get parent directory for uvmaps
        if self.data_root.name in ['validation', 'test']:
            self.uvmaps_dir = self.data_root.parent / 'uvmaps'
        else:
            self.uvmaps_dir = self.data_root / 'uvmaps'

        # Load tampering mapping
        if tampering_csv_path is None:
            tampering_csv_path = ROOT / 'src' / 'tampering' / 'tampering_mapping.csv'
        self.tampering_map = TamperingMapping(tampering_csv_path)

        # Initialize augmentor
        self.augmentor = AugmentationPipeline()

        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Storage for pairs
        self.positive_pairs = []
        self.negative_pairs = []

        # Statistics
        self.pair_stats = defaultdict(int)

    def load_data(self, split='validation'):
        """
        Load UV maps and extract surfaces.

        Args:
            split: 'validation' or 'test'

        Returns:
            Dictionary with reference and split surface data
        """
        print(f"\n{'='*70}")
        print(f"Loading Surface Data - {split.upper()}")
        print(f"{'='*70}")

        data = {
            'reference': {},  # parcel_id -> {surface_name -> surface_image}
            split: {}         # parcel_id -> list of {surface_name -> surface_image, filename, type}
        }

        # Load reference UV maps from uvmaps/ folder
        if self.uvmaps_dir.exists():
            print("\nLoading reference UV maps...")
            ref_files = sorted(self.uvmaps_dir.glob('id_*_uvmap.png'))

            for ref_file in tqdm(ref_files, desc="Reference UVs"):
                # Parse: id_01_uvmap.png -> parcel_id = 1
                stem = ref_file.stem.replace('_uvmap', '')
                parcel_id = int(stem.split('_')[1])

                # Load and extract surfaces
                uv_map = Image.open(ref_file).convert('RGB')
                surfaces = SurfaceExtractor.extract_surfaces(uv_map)

                data['reference'][parcel_id] = surfaces

            print(f"✓ Loaded {len(data['reference'])} reference UV maps")

        # Load split UV maps
        # If data_root already points to a split directory, use it directly
        if self.data_root.name == split:
            split_dir = self.data_root
        else:
            split_dir = self.data_root / split

        if split_dir.exists():
            print(f"\nLoading {split} UV maps...")

            # Find all UV map files recursively (both _gt and _pred)
            # Search in background subfolders (carpet, gravel, table, etc.)
            # This will match both normal and adversarial UV maps:
            # - carpet/id_01_20230516_142710_uvmap_gt.png (normal)
            # - carpet/id_01_20230516_142710_fgsm_uvmap_gt.png (adversarial)
            # - table/id_01_20230516_142710_pgd_uvmap_pred.png (adversarial)
            uv_files = list(split_dir.glob('**/id_*_uvmap_gt.png')) + \
                      list(split_dir.glob('**/id_*_uvmap_pred.png'))

            # Filter out base folder
            uv_files_filtered = []
            for uv_file in uv_files:
                # Check if 'base' is in the path
                if 'base' not in str(uv_file.relative_to(split_dir)).lower():
                    uv_files_filtered.append(uv_file)
                else:
                    # Skip base folder files
                    continue

            print(f"  Found {len(uv_files)} UV maps total")
            print(f"  Excluded {len(uv_files) - len(uv_files_filtered)} UV maps from 'base' folder")
            print(f"  Processing {len(uv_files_filtered)} UV maps")

            uv_files = uv_files_filtered

            for uv_file in tqdm(uv_files, desc=f"{split} UVs"):
                # Parse: id_01_20230516_142710_uvmap_gt.png
                filename = uv_file.stem
                parts = filename.split('_')
                parcel_id = int(parts[1])
                uv_type = 'gt' if 'uvmap_gt' in filename else 'pred'

                # Load and extract surfaces
                uv_map = Image.open(uv_file).convert('RGB')
                surfaces = SurfaceExtractor.extract_surfaces(uv_map)

                if parcel_id not in data[split]:
                    data[split][parcel_id] = []

                data[split][parcel_id].append({
                    'surfaces': surfaces,
                    'filename': uv_file.name,
                    'type': uv_type,
                    'path': uv_file,
                    'is_adversarial': False  # Clean data
                })

            print(f"✓ Loaded UV maps for {len(data[split])} parcels in {split}")

        # Load adversarial UV maps from separate path (if provided)
        if self.adversarial_root is not None:
            # If adversarial_root already contains adversarial subfolders (like carpet_adv_fgsm),
            # use it directly; otherwise append split name
            # Check if any subdirectory has 'adv' in its name
            has_adv_subdirs = any('adv' in d.name.lower() for d in self.adversarial_root.iterdir() if d.is_dir())

            if has_adv_subdirs:
                # Already points to the split directory with adversarial folders
                adv_split_dir = self.adversarial_root
            else:
                # Need to append split name
                adv_split_dir = self.adversarial_root / split

            if adv_split_dir.exists():
                print(f"\nLoading ADVERSARIAL {split} UV maps...")

                # Find all adversarial UV map files recursively
                # Search in background subfolders (carpet_adv_fgsm, carpet_adv_pgd, etc.)
                adv_uv_files = list(adv_split_dir.glob('**/id_*_uvmap_gt.png')) + \
                              list(adv_split_dir.glob('**/id_*_uvmap_pred.png'))

                # Filter out base_adv_* folders
                adv_uv_files_filtered = []
                for uv_file in adv_uv_files:
                    # Check if 'base' is in the path
                    if 'base' not in str(uv_file.relative_to(adv_split_dir)).lower():
                        adv_uv_files_filtered.append(uv_file)
                    else:
                        # Skip base_adv_* folder files
                        continue

                print(f"  Found {len(adv_uv_files)} adversarial UV maps total")
                print(f"  Excluded {len(adv_uv_files) - len(adv_uv_files_filtered)} UV maps from 'base_adv_*' folders")
                print(f"  Processing {len(adv_uv_files_filtered)} adversarial UV maps")

                adv_uv_files = adv_uv_files_filtered

                for uv_file in tqdm(adv_uv_files, desc=f"Adversarial {split} UVs"):
                    # Parse: id_01_20230516_142710_uvmap_gt.png
                    filename = uv_file.stem
                    parts = filename.split('_')
                    parcel_id = int(parts[1])
                    uv_type = 'gt' if 'uvmap_gt' in filename else 'pred'

                    # Detect attack type from folder name
                    folder_path = str(uv_file.relative_to(adv_split_dir))
                    attack_type = None
                    if 'fgsm' in folder_path.lower():
                        attack_type = 'fgsm'
                    elif 'pgd' in folder_path.lower():
                        attack_type = 'pgd'
                    elif 'cw' in folder_path.lower():
                        attack_type = 'cw'

                    # Load and extract surfaces
                    uv_map = Image.open(uv_file).convert('RGB')
                    surfaces = SurfaceExtractor.extract_surfaces(uv_map)

                    if parcel_id not in data[split]:
                        data[split][parcel_id] = []

                    data[split][parcel_id].append({
                        'surfaces': surfaces,
                        'filename': uv_file.name,
                        'type': uv_type,
                        'path': uv_file,
                        'is_adversarial': True,
                        'attack_type': attack_type  # Store attack type
                    })

                print(f"✓ Loaded adversarial UV maps for {len([p for p in data[split].keys() if any(c.get('is_adversarial', False) for c in data[split][p])])} parcels")

        self.data = data
        return data

    def create_positive_pairs(self, split='validation'):
        """
        Create positive pairs according to the NEW strategy.

        Positive pair types:
        1. Clean reference vs clean uvmap_pred (all angles)
        2. Clean reference vs clean uvmap_gt (all angles)
        3. Clean reference vs augmented clean reference

        Only uses surfaces that are CLEAN (untampered) in field images.
        """
        print(f"\n{'='*70}")
        print("Creating POSITIVE Pairs (Surface-Level)")
        print(f"{'='*70}")

        pairs = []

        # Type 1: Clean reference vs clean uvmap_pred
        print("\n1. Clean Reference vs Clean uvmap_pred...")
        for parcel_id in self.data[split].keys():
            if parcel_id not in self.data['reference']:
                continue

            # Get clean surfaces (untampered in field as per tampering_mapping.csv)
            clean_surfaces = self.tampering_map.get_clean_surfaces(parcel_id)

            for surf_name in clean_surfaces:
                # Check if reference has this surface
                if surf_name not in self.data['reference'][parcel_id]:
                    continue

                ref_surface = self.data['reference'][parcel_id][surf_name]

                # Pair with clean uvmap_pred captures (EXCLUDE adversarial)
                for cap in self.data[split][parcel_id]:
                    if cap.get('is_adversarial', False):
                        continue
                    if cap['type'] != 'pred':
                        continue
                    if surf_name not in cap['surfaces']:
                        continue

                    pairs.append({
                        'image1': ref_surface,
                        'surface2': cap['surfaces'][surf_name],
                        'label': 1,
                        'pair_type': 'reference_vs_pred',
                        'parcel_id': parcel_id,
                        'surface_name': surf_name,
                        'metadata': {
                            'ref_file': f"id_{parcel_id:02d}_uvmap.png",
                            'pred_file': cap['filename']
                        }
                    })
                    self.pair_stats['positive_ref_vs_pred'] += 1

        print(f"   Created {self.pair_stats['positive_ref_vs_pred']} pairs")

        # Type 2: Clean reference vs clean uvmap_gt
        print("\n2. Clean Reference vs Clean uvmap_gt...")
        for parcel_id in self.data[split].keys():
            if parcel_id not in self.data['reference']:
                continue

            clean_surfaces = self.tampering_map.get_clean_surfaces(parcel_id)

            for surf_name in clean_surfaces:
                if surf_name not in self.data['reference'][parcel_id]:
                    continue

                ref_surface = self.data['reference'][parcel_id][surf_name]

                # Pair with clean uvmap_gt captures (EXCLUDE adversarial)
                for cap in self.data[split][parcel_id]:
                    if cap.get('is_adversarial', False):
                        continue
                    if cap['type'] != 'gt':
                        continue
                    if surf_name not in cap['surfaces']:
                        continue

                    pairs.append({
                        'image1': ref_surface,
                        'surface2': cap['surfaces'][surf_name],
                        'label': 1,
                        'pair_type': 'reference_vs_gt',
                        'parcel_id': parcel_id,
                        'surface_name': surf_name,
                        'metadata': {
                            'ref_file': f"id_{parcel_id:02d}_uvmap.png",
                            'gt_file': cap['filename']
                        }
                    })
                    self.pair_stats['positive_ref_vs_gt'] += 1

        print(f"   Created {self.pair_stats['positive_ref_vs_gt']} pairs")

        # Type 3: Clean reference vs augmented clean reference
        print("\n3. Clean Reference vs Augmented Clean Reference...")
        num_augmented = 0
        num_variants = 2  # Number of augmented versions per surface

        for parcel_id in self.data['reference'].keys():
            # All reference surfaces are clean (no tampering in reference UV maps)
            # But we only use surfaces that are ALSO clean in field images
            clean_surfaces = self.tampering_map.get_clean_surfaces(parcel_id)

            for surf_name in clean_surfaces:
                if surf_name not in self.data['reference'][parcel_id]:
                    continue

                ref_surface = self.data['reference'][parcel_id][surf_name]

                # Create augmented versions of the reference surface
                for variant_idx in range(num_variants):
                    aug_surface = self.augmentor.augment(ref_surface)

                    pairs.append({
                        'image1': ref_surface,
                        'surface2': aug_surface,
                        'label': 1,
                        'pair_type': 'reference_vs_augmented_reference',
                        'parcel_id': parcel_id,
                        'surface_name': surf_name,
                        'metadata': {
                            'ref_file': f"id_{parcel_id:02d}_uvmap.png",
                            'variant': variant_idx
                        }
                    })
                    num_augmented += 1

        self.pair_stats['positive_augmented'] = num_augmented
        print(f"   Created {num_augmented} pairs")

        # Type 4: Clean reference vs adversarial clean surfaces
        print("\n4. Clean Reference vs Adversarial Clean Surfaces...")
        num_adv_clean = 0

        for parcel_id in self.data[split].keys():
            if parcel_id not in self.data['reference']:
                continue

            # Get CLEAN surfaces (both in reference AND in field)
            clean_surfaces = self.tampering_map.get_clean_surfaces(parcel_id)

            # Get adversarial captures
            adv_captures = [c for c in self.data[split][parcel_id] if c.get('is_adversarial', False)]

            for surf_name in clean_surfaces:
                # IMPORTANT: Only use surfaces that are CLEAN (untampered)
                if surf_name not in self.data['reference'][parcel_id]:
                    continue

                ref_surface = self.data['reference'][parcel_id][surf_name]

                # Pair with adversarial versions of CLEAN surfaces
                # These should be POSITIVE because they're the same clean surface
                # just with adversarial perturbations - we want the model to be robust
                for adv_cap in adv_captures:
                    if surf_name not in adv_cap['surfaces']:
                        continue

                    pairs.append({
                        'image1': ref_surface,
                        'surface2': adv_cap['surfaces'][surf_name],
                        'label': 1,  # POSITIVE - same clean surface despite adversarial noise
                        'pair_type': 'clean_reference_vs_adversarial_clean',
                        'parcel_id': parcel_id,
                        'surface_name': surf_name,
                        'metadata': {
                            'ref_file': f"id_{parcel_id:02d}_uvmap.png",
                            'adversarial_file': adv_cap['filename'],
                            'attack_type': adv_cap.get('attack_type', 'unknown'),
                            'surface_status': 'clean'
                        }
                    })
                    num_adv_clean += 1

        self.pair_stats['positive_adv_clean'] = num_adv_clean
        print(f"   Created {num_adv_clean} pairs")

        self.positive_pairs = pairs
        print(f"\n✓ Total POSITIVE pairs: {len(pairs)}")

        return pairs

    def create_negative_pairs(self, split='validation'):
        """
        Create negative pairs according to the NEW strategy.

        Negative pair types:
        1. Clean reference vs tampered uvmap_gt (all angles)
        2. Clean reference vs tampered uvmap_pred (all angles)
        3. Any reference vs adversarial clean (fgsm/pgd on clean surfaces)
        4. Any reference vs adversarial tampered (fgsm/pgd on tampered surfaces)
        5. Different parcels, same surface type
        """
        print(f"\n{'='*70}")
        print("Creating NEGATIVE Pairs (Surface-Level)")
        print(f"{'='*70}")

        pairs = []

        # Type 1: Clean reference vs tampered uvmap_gt
        print("\n1. Clean Reference vs Tampered uvmap_gt...")
        for parcel_id in self.data[split].keys():
            if parcel_id not in self.data['reference']:
                continue

            # Get tampered surfaces (surfaces that ARE tampered in field)
            tampered_surfaces = self.tampering_map.get_tampered_surfaces(parcel_id)

            for surf_name in tampered_surfaces:
                if surf_name not in self.data['reference'][parcel_id]:
                    continue

                ref_surface = self.data['reference'][parcel_id][surf_name]

                # Pair with tampered uvmap_gt (EXCLUDE adversarial)
                for cap in self.data[split][parcel_id]:
                    if cap.get('is_adversarial', False):
                        continue
                    if cap['type'] != 'gt':
                        continue
                    if surf_name not in cap['surfaces']:
                        continue

                    pairs.append({
                        'image1': ref_surface,
                        'surface2': cap['surfaces'][surf_name],
                        'label': 0,
                        'pair_type': 'reference_vs_tampered_gt',
                        'parcel_id': parcel_id,
                        'surface_name': surf_name,
                        'metadata': {
                            'ref_file': f"id_{parcel_id:02d}_uvmap.png",
                            'tampered_file': cap['filename'],
                            'tampering_type': self.tampering_map.get_tampering_code(parcel_id, surf_name)
                        }
                    })
                    self.pair_stats['negative_ref_vs_tampered_gt'] += 1

        print(f"   Created {self.pair_stats['negative_ref_vs_tampered_gt']} pairs")

        # Type 2: Clean reference vs tampered uvmap_pred
        print("\n2. Clean Reference vs Tampered uvmap_pred...")
        for parcel_id in self.data[split].keys():
            if parcel_id not in self.data['reference']:
                continue

            tampered_surfaces = self.tampering_map.get_tampered_surfaces(parcel_id)

            for surf_name in tampered_surfaces:
                if surf_name not in self.data['reference'][parcel_id]:
                    continue

                ref_surface = self.data['reference'][parcel_id][surf_name]

                # Pair with tampered uvmap_pred (EXCLUDE adversarial)
                for cap in self.data[split][parcel_id]:
                    if cap.get('is_adversarial', False):
                        continue
                    if cap['type'] != 'pred':
                        continue
                    if surf_name not in cap['surfaces']:
                        continue

                    pairs.append({
                        'image1': ref_surface,
                        'surface2': cap['surfaces'][surf_name],
                        'label': 0,
                        'pair_type': 'reference_vs_tampered_pred',
                        'parcel_id': parcel_id,
                        'surface_name': surf_name,
                        'metadata': {
                            'ref_file': f"id_{parcel_id:02d}_uvmap.png",
                            'tampered_file': cap['filename'],
                            'tampering_type': self.tampering_map.get_tampering_code(parcel_id, surf_name)
                        }
                    })
                    self.pair_stats['negative_ref_vs_tampered_pred'] += 1

        print(f"   Created {self.pair_stats['negative_ref_vs_tampered_pred']} pairs")

        # Type 3: CLEAN reference vs adversarial (clean surfaces) - MOVED TO POSITIVE PAIRS
        print("\n3. CLEAN Reference vs Adversarial (Clean Surfaces) - MOVED TO POSITIVE")
        print("   This pair type is now a POSITIVE pair (see positive pair type #4)")
        print("   Reason: Same clean surface with adversarial noise should match")
        self.pair_stats['negative_ref_vs_adv_clean'] = 0
        print(f"   Created 0 pairs (moved to positive)")

        # Type 4: CLEAN reference vs adversarial (tampered surfaces)
        print("\n4. CLEAN Reference vs Adversarial (Tampered Surfaces)...")
        for parcel_id in self.data[split].keys():
            if parcel_id not in self.data['reference']:
                continue

            # Get TAMPERED surfaces (tampered in field, but reference is always clean)
            tampered_surfaces = self.tampering_map.get_tampered_surfaces(parcel_id)

            # Get adversarial captures
            adv_captures = [c for c in self.data[split][parcel_id] if c.get('is_adversarial', False)]

            for surf_name in tampered_surfaces:
                # IMPORTANT: Reference is CLEAN (untampered), but field surface is TAMPERED
                if surf_name not in self.data['reference'][parcel_id]:
                    continue

                ref_surface = self.data['reference'][parcel_id][surf_name]

                # Pair with adversarial versions of TAMPERED field surfaces
                for adv_cap in adv_captures:
                    if surf_name not in adv_cap['surfaces']:
                        continue

                    pairs.append({
                        'image1': ref_surface,
                        'surface2': adv_cap['surfaces'][surf_name],
                        'label': 0,
                        'pair_type': 'clean_reference_vs_adversarial_tampered',
                        'parcel_id': parcel_id,
                        'surface_name': surf_name,
                        'metadata': {
                            'ref_file': f"id_{parcel_id:02d}_uvmap.png",
                            'adversarial_file': adv_cap['filename'],
                            'attack_type': adv_cap.get('attack_type', 'unknown'),
                            'surface_status': 'tampered',
                            'tampering_type': self.tampering_map.get_tampering_code(parcel_id, surf_name)
                        }
                    })
                    self.pair_stats['negative_ref_vs_adv_tampered'] += 1

        print(f"   Created {self.pair_stats['negative_ref_vs_adv_tampered']} pairs")

        # Type 5: Different parcels - same surface type (DISABLED)
        # print("\n5. Different Parcels - Same Surface Type...")
        diff_parcel_count = 0

        # DISABLED: This creates pairs from different parcels which may confuse the model
        # # Use reference UV maps for different parcel pairs
        # ref_parcel_ids = list(self.data['reference'].keys())
        #
        # if len(ref_parcel_ids) >= 2:
        #     # Target number of pairs to balance dataset
        #     target_pairs = max(
        #         self.pair_stats.get('negative_ref_vs_tampered_gt', 0),
        #         self.pair_stats.get('negative_ref_vs_tampered_pred', 0),
        #         self.pair_stats.get('negative_ref_vs_adv_clean', 0),
        #         self.pair_stats.get('negative_ref_vs_adv_tampered', 0),
        #         100  # minimum
        #     )
        #
        #     attempts = 0
        #     max_attempts = target_pairs * 10
        #
        #     while diff_parcel_count < target_pairs and attempts < max_attempts:
        #         attempts += 1
        #
        #         # Select two different parcels
        #         p1, p2 = random.sample(ref_parcel_ids, 2)
        #
        #         # Select a common surface type
        #         surf_types = ['top', 'left', 'center', 'right', 'bottom']
        #         surf_name = random.choice(surf_types)
        #
        #         # Check if both parcels have this surface in reference
        #         if surf_name in self.data['reference'][p1] and \
        #            surf_name in self.data['reference'][p2]:
        #
        #             pairs.append({
        #                 'image1': self.data['reference'][p1][surf_name],
        #                 'surface2': self.data['reference'][p2][surf_name],
        #                 'label': 0,
        #                 'pair_type': 'different_parcels_same_surface',
        #                 'parcel_id': f"{p1}_vs_{p2}",
        #                 'surface_name': surf_name,
        #                 'metadata': {
        #                     'parcel1': p1,
        #                     'parcel2': p2,
        #                     'file1': f"id_{p1:02d}_uvmap.png",
        #                     'file2': f"id_{p2:02d}_uvmap.png"
        #                 }
        #             })
        #             diff_parcel_count += 1

        self.pair_stats['negative_diff_parcels'] = diff_parcel_count
        # print(f"   Created {diff_parcel_count} pairs")  # Will always be 0 now

        self.negative_pairs = pairs
        print(f"\n✓ Total NEGATIVE pairs: {len(pairs)}")

        return pairs

    def print_statistics(self):
        """Print detailed statistics about created pairs."""
        print(f"\n{'='*70}")
        print("PAIR CREATION STATISTICS")
        print(f"{'='*70}")

        print("\n📊 POSITIVE PAIRS:")
        print(f"  Reference vs uvmap_pred (clean):     {self.pair_stats['positive_ref_vs_pred']:>6}")
        print(f"  Reference vs uvmap_gt (clean):       {self.pair_stats['positive_ref_vs_gt']:>6}")
        print(f"  Reference vs Augmented Reference:    {self.pair_stats['positive_augmented']:>6}")
        print(f"  Reference vs Adversarial (clean):    {self.pair_stats.get('positive_adv_clean', 0):>6}")
        print(f"  {'─'*60}")
        print(f"  TOTAL POSITIVE:                      {len(self.positive_pairs):>6}")

        print("\n📊 NEGATIVE PAIRS:")
        print(f"  Reference vs Tampered uvmap_gt:      {self.pair_stats['negative_ref_vs_tampered_gt']:>6}")
        print(f"  Reference vs Tampered uvmap_pred:    {self.pair_stats['negative_ref_vs_tampered_pred']:>6}")
        print(f"  Reference vs Adversarial (clean):    {self.pair_stats['negative_ref_vs_adv_clean']:>6} [MOVED TO POSITIVE]")
        print(f"  Reference vs Adversarial (tampered): {self.pair_stats['negative_ref_vs_adv_tampered']:>6}")
        print(f"  Different Parcels - Same Surface:    {self.pair_stats['negative_diff_parcels']:>6}")
        print(f"  {'─'*60}")
        print(f"  TOTAL NEGATIVE:                      {len(self.negative_pairs):>6}")

        print(f"\n{'═'*70}")
        print(f"  GRAND TOTAL:                         {len(self.positive_pairs) + len(self.negative_pairs):>6}")
        print(f"{'═'*70}")

        # Create summary table
        return {
            'positive': {
                'reference_vs_pred': self.pair_stats['positive_ref_vs_pred'],
                'reference_vs_gt': self.pair_stats['positive_ref_vs_gt'],
                'reference_vs_augmented_reference': self.pair_stats['positive_augmented'],
                'reference_vs_adversarial_clean': self.pair_stats.get('positive_adv_clean', 0),
                'total': len(self.positive_pairs)
            },
            'negative': {
                'reference_vs_tampered_gt': self.pair_stats['negative_ref_vs_tampered_gt'],
                'reference_vs_tampered_pred': self.pair_stats['negative_ref_vs_tampered_pred'],
                'reference_vs_adversarial_clean': self.pair_stats['negative_ref_vs_adv_clean'],  # Should be 0
                'reference_vs_adversarial_tampered': self.pair_stats['negative_ref_vs_adv_tampered'],
                'different_parcels_same_surface': self.pair_stats['negative_diff_parcels'],
                'total': len(self.negative_pairs)
            },
            'grand_total': len(self.positive_pairs) + len(self.negative_pairs)
        }

    def visualize_pairs_for_parcel(self, parcel_id, output_dir=None, max_pairs_per_type=5):
        """
        Visualize all pairs created for a specific parcel ID.

        Args:
            parcel_id: Parcel ID to visualize (e.g., 1, 2, 7, 9)
            output_dir: Directory to save visualizations (default: data_root)
            max_pairs_per_type: Maximum pairs to show per pair type
        """
        if output_dir is None:
            output_dir = self.data_root / f'pair_visualizations_parcel_{parcel_id:02d}'
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Visualizing Pairs for Parcel {parcel_id:02d}")
        print(f"{'='*70}")

        # Filter pairs for this parcel
        pos_pairs_parcel = [p for p in self.positive_pairs if p['parcel_id'] == parcel_id]
        neg_pairs_parcel = [p for p in self.negative_pairs
                           if (isinstance(p['parcel_id'], int) and p['parcel_id'] == parcel_id)]

        print(f"\nFound {len(pos_pairs_parcel)} positive pairs")
        print(f"Found {len(neg_pairs_parcel)} negative pairs")

        # Group by pair type
        pos_by_type = defaultdict(list)
        for pair in pos_pairs_parcel:
            pos_by_type[pair['pair_type']].append(pair)

        neg_by_type = defaultdict(list)
        for pair in neg_pairs_parcel:
            neg_by_type[pair['pair_type']].append(pair)

        # Visualize each type
        print(f"\nCreating visualizations...")

        # Positive pairs
        for pair_type, pairs in pos_by_type.items():
            self._visualize_pair_type(
                pairs, pair_type, parcel_id, output_dir,
                max_pairs=max_pairs_per_type, is_positive=True
            )

        # Negative pairs
        for pair_type, pairs in neg_by_type.items():
            self._visualize_pair_type(
                pairs, pair_type, parcel_id, output_dir,
                max_pairs=max_pairs_per_type, is_positive=False
            )

        print(f"\n✓ Visualizations saved to: {output_dir}")
        return output_dir

    def _visualize_pair_type(self, pairs, pair_type, parcel_id, output_dir,
                            max_pairs=5, is_positive=True):
        """Visualize a specific pair type."""
        if not pairs:
            return

        num_pairs = min(len(pairs), max_pairs)
        sample_pairs = random.sample(pairs, num_pairs)

        fig, axes = plt.subplots(num_pairs, 2, figsize=(10, 4 * num_pairs))
        if num_pairs == 1:
            axes = axes.reshape(1, -1)

        label_str = "POSITIVE" if is_positive else "NEGATIVE"
        fig.suptitle(f"Parcel {parcel_id:02d} - {label_str}: {pair_type}\n"
                    f"Showing {num_pairs} of {len(pairs)} pairs",
                    fontsize=14, fontweight='bold')

        for idx, pair in enumerate(sample_pairs):
            # Surface 1
            surf1 = pair['image1']
            axes[idx, 0].imshow(surf1.astype(np.uint8))

            title1 = f"Surface 1 - {pair['surface_name']}\n"
            if 'file1' in pair.get('metadata', {}):
                title1 += f"{pair['metadata']['file1'][:40]}\n"
            elif 'ref' in pair.get('metadata', {}):
                title1 += f"{pair['metadata']['ref']}\n"
            elif 'clean_file' in pair.get('metadata', {}):
                title1 += f"{pair['metadata']['clean_file'][:40]}\n"
            elif 'file' in pair.get('metadata', {}):
                title1 += f"{pair['metadata']['file'][:40]}\n"

            axes[idx, 0].set_title(title1, fontsize=9)
            axes[idx, 0].axis('off')

            # Surface 2
            surf2 = pair['surface2']
            axes[idx, 1].imshow(surf2.astype(np.uint8))

            title2 = f"Surface 2 - {pair['surface_name']}\n"
            if 'file2' in pair.get('metadata', {}):
                title2 += f"{pair['metadata']['file2'][:40]}\n"
            elif 'pred_file' in pair.get('metadata', {}):
                title2 += f"{pair['metadata']['pred_file'][:40]}\n"
            elif 'field' in pair.get('metadata', {}):
                title2 += f"{pair['metadata']['field'][:40]}\n"
            elif 'tampered_file' in pair.get('metadata', {}):
                title2 += f"{pair['metadata']['tampered_file'][:40]}\n"
                title2 += f"Tampering: {pair['metadata'].get('tampering_type', 'unknown')}\n"
            elif 'adversarial_file' in pair.get('metadata', {}):
                title2 += f"{pair['metadata']['adversarial_file'][:40]}\n"
                title2 += f"Attack: {pair['metadata'].get('attack_type', 'unknown')}\n"
            elif pair['pair_type'] == 'same_surface_augmented':
                title2 += f"AUGMENTED (variant {pair['metadata'].get('variant', '?')})\n"

            axes[idx, 1].set_title(title2, fontsize=9)
            axes[idx, 1].axis('off')

        plt.tight_layout()

        # Save
        safe_type_name = pair_type.replace('_', '-')
        output_file = output_dir / f"{label_str.lower()}_{safe_type_name}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ {label_str}: {pair_type} ({num_pairs} pairs) -> {output_file.name}")

    def visualize_all_pairs_summary(self, output_dir=None, samples_per_type=3):
        """
        Create a summary visualization showing examples of all pair types.

        Args:
            output_dir: Directory to save visualization
            samples_per_type: Number of examples per pair type
        """
        if output_dir is None:
            output_dir = self.data_root / 'pair_visualizations_summary'
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print("Creating Summary Visualization of All Pair Types")
        print(f"{'='*70}")

        # Group positive pairs by type
        pos_by_type = defaultdict(list)
        for pair in self.positive_pairs:
            pos_by_type[pair['pair_type']].append(pair)

        # Group negative pairs by type
        neg_by_type = defaultdict(list)
        for pair in self.negative_pairs:
            neg_by_type[pair['pair_type']].append(pair)

        # Create visualizations for each type
        all_types = list(pos_by_type.keys()) + list(neg_by_type.keys())

        for pair_type in all_types:
            if pair_type in pos_by_type:
                pairs = pos_by_type[pair_type]
                is_positive = True
            else:
                pairs = neg_by_type[pair_type]
                is_positive = False

            num_samples = min(samples_per_type, len(pairs))
            if num_samples == 0:
                continue

            sample_pairs = random.sample(pairs, num_samples)

            fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3.5 * num_samples))
            if num_samples == 1:
                axes = axes.reshape(1, -1)

            label_str = "POSITIVE" if is_positive else "NEGATIVE"
            fig.suptitle(f"{label_str}: {pair_type}\n"
                        f"Showing {num_samples} of {len(pairs)} total pairs",
                        fontsize=12, fontweight='bold')

            for idx, pair in enumerate(sample_pairs):
                # Surface 1
                axes[idx, 0].imshow(pair['image1'].astype(np.uint8))
                title1 = f"Parcel {pair['parcel_id']}\n{pair['surface_name']}"
                axes[idx, 0].set_title(title1, fontsize=9)
                axes[idx, 0].axis('off')

                # Surface 2
                axes[idx, 1].imshow(pair['surface2'].astype(np.uint8))
                title2 = f"Parcel {pair['parcel_id']}\n{pair['surface_name']}"
                if 'tampering_type' in pair.get('metadata', {}):
                    title2 += f"\n[{pair['metadata']['tampering_type']}]"
                if 'attack_type' in pair.get('metadata', {}):
                    title2 += f"\n[{pair['metadata']['attack_type']}]"
                axes[idx, 1].set_title(title2, fontsize=9)
                axes[idx, 1].axis('off')

            plt.tight_layout()

            safe_type_name = pair_type.replace('_', '-')
            output_file = output_dir / f"summary_{label_str.lower()}_{safe_type_name}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  ✓ {label_str}: {pair_type} -> {output_file.name}")

        print(f"\n✓ Summary visualizations saved to: {output_dir}")
        return output_dir

    def save_pairs(self, output_dir, train_split=0.8):
        """Save pairs as PyTorch-ready dataset."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print("Saving Pairs Dataset")
        print(f"{'='*70}")

        # Combine and shuffle
        all_pairs = self.positive_pairs + self.negative_pairs
        random.shuffle(all_pairs)

        # Split
        split_idx = int(len(all_pairs) * train_split)
        train_pairs = all_pairs[:split_idx]
        val_pairs = all_pairs[split_idx:]

        # Save
        train_path = output_dir / 'train_pairs_surface_level.pkl'
        val_path = output_dir / 'val_pairs_surface_level.pkl'

        with open(train_path, 'wb') as f:
            pickle.dump(train_pairs, f)

        with open(val_path, 'wb') as f:
            pickle.dump(val_pairs, f)

        print(f"\n✓ Saved train pairs: {train_path}")
        print(f"✓ Saved val pairs: {val_path}")

        # Save statistics
        stats = self.print_statistics()
        stats_path = output_dir / 'pair_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved statistics: {stats_path}")

        return train_path, val_path


def main():
    parser = argparse.ArgumentParser(
        description="Create surface-level contrastive pairs for TAMPAR"
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default=str(ROOT / 'data' / 'tampar_sample'),
        help='Path to TAMPAR dataset (clean data)'
    )
    parser.add_argument(
        '--adversarial_root',
        type=str,
        default=None,
        help='Path to adversarial TAMPAR dataset (e.g., /path/to/tampar/adversarial_validation)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='validation',
        choices=['test', 'validation'],
        help='Which split to use'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: data_root/contrastive_pairs_surface)'
    )
    parser.add_argument(
        '--visualize_parcel',
        type=int,
        default=None,
        help='Visualize pairs for a specific parcel ID (e.g., 1, 2, 7, 9)'
    )
    parser.add_argument(
        '--visualize_summary',
        action='store_true',
        help='Create summary visualization of all pair types'
    )
    parser.add_argument(
        '--max_pairs_viz',
        type=int,
        default=5,
        help='Maximum pairs to show per type in visualizations'
    )

    args = parser.parse_args()

    # Initialize creator
    creator = SurfaceLevelPairCreator(
        data_root=args.data_root,
        adversarial_root=args.adversarial_root
    )

    # Load data
    creator.load_data(split=args.split)

    # Create pairs
    creator.create_positive_pairs(split=args.split)
    creator.create_negative_pairs(split=args.split)

    # Print statistics
    creator.print_statistics()

    # Save pairs
    if args.output_dir is None:
        args.output_dir = Path(args.data_root) / 'contrastive_pairs_surface'

    creator.save_pairs(args.output_dir)

    # Visualizations
    if args.visualize_parcel is not None:
        print(f"\n{'='*70}")
        print(f"Creating Visualizations for Parcel {args.visualize_parcel:02d}")
        print(f"{'='*70}")
        creator.visualize_pairs_for_parcel(
            args.visualize_parcel,
            max_pairs_per_type=args.max_pairs_viz
        )

    if args.visualize_summary:
        print(f"\n{'='*70}")
        print("Creating Summary Visualizations")
        print(f"{'='*70}")
        creator.visualize_all_pairs_summary(samples_per_type=args.max_pairs_viz)

    print(f"\n{'='*70}")
    print("✓ Surface-Level Pair Creation Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
