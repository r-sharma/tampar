"""
Create Tampering-Aware Pairs for Multi-Task Learning

This script creates training pairs specifically designed for tampering detection:

1. **Positive Pairs** (similar, should have high similarity):
   - Clean surface vs Clean surface (same parcel)
   - Clean surface vs Clean surface (different parcel, same surface type)

2. **Hard Negative Pairs** (different, should have low similarity):
   - Clean vs Tampered (same parcel, same surface) - CRITICAL for learning tampering
   - Clean vs Tampered (different parcel, same surface type)
   - Tampered vs Different Tampering (same parcel, same surface)

3. **Triplets** (anchor, positive, negative):
   - Anchor: Clean surface
   - Positive: Same clean surface or similar clean surface
   - Negative: Tampered version of the surface

4. **Quadruplets** (anchor, positive, negative1, negative2):
   - Anchor: Clean surface
   - Positive: Similar clean surface
   - Negative1: Tampered surface (same tampering type)
   - Negative2: Tampered surface (different tampering type)

Usage:
    python extension/create_tampering_aware_pairs.py \
        --data_root /content/tampar/data/tampar_sample/train \
        --output_dir /content/tampar/data/tampering_pairs \
        --pair_types triplet quadruplet \
        --min_pairs_per_parcel 10

Output:
    - tampering_triplets.csv: Triplet pairs (anchor, positive, negative, tampering_label)
    - tampering_quadruplets.csv: Quadruplet pairs
    - tampering_pairs_metadata.json: Statistics and metadata
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())


class TamperingPairGenerator:
    """Generate tampering-aware pairs for contrastive learning."""

    def __init__(self, data_root, surface_types=None, seed=42):
        """
        Initialize pair generator.

        Args:
            data_root: Root directory containing parcels
            surface_types: List of surface types to use (None = auto-detect)
            seed: Random seed for reproducibility
        """
        self.data_root = Path(data_root)
        self.seed = seed
        random.seed(seed)

        # Surface types
        if surface_types is None:
            surface_types = ['base', 'carpet', 'gravel', 'table']
        self.surface_types = surface_types

        # Tampering type mapping
        self.tampering_map = {
            'clean': 0,
            'T': 1,  # Tape
            'W': 2,  # Writing
            'F': 3,  # Fold
            'other': 4  # Other tampering
        }

        # Load data structure
        self.parcels = self._load_parcels()

        print(f"\nLoaded {len(self.parcels)} parcels from {data_root}")
        print(f"Surface types: {self.surface_types}")

    def _load_parcels(self):
        """
        Load parcel structure from data directory.

        Returns:
            parcels: Dict mapping parcel_id -> {surface_type -> {gt/pred -> uvmap_path}}
        """
        parcels = defaultdict(lambda: defaultdict(dict))

        # Search recursively for UV maps
        for uvmap_path in self.data_root.rglob("*_uvmap_*.png"):
            # Extract parcel ID and surface type from path
            # Assuming structure: data_root / surface_type / parcel_id / file.png
            # Or: data_root / parcel_id / file.png

            relative_path = uvmap_path.relative_to(self.data_root)
            parts = relative_path.parts

            # Determine surface type and parcel ID
            if len(parts) >= 3:
                # surface_type / parcel_id / file.png
                surface_type = parts[0]
                parcel_id = parts[1]
            elif len(parts) == 2:
                # parcel_id / file.png (no surface type subfolder)
                surface_type = self._infer_surface_type(uvmap_path)
                parcel_id = parts[0]
            else:
                continue

            # Extract GT/Pred type from filename
            if '_uvmap_gt.png' in uvmap_path.name:
                map_type = 'gt'
            elif '_uvmap_pred.png' in uvmap_path.name:
                map_type = 'pred'
            else:
                continue

            # Store path
            parcels[parcel_id][surface_type][map_type] = uvmap_path

            # Load tampering codes if available
            tampering_file = uvmap_path.parent / 'tampering_codes.txt'
            if tampering_file.exists() and 'tampering_code' not in parcels[parcel_id][surface_type]:
                with open(tampering_file, 'r') as f:
                    code = f.read().strip()
                    parcels[parcel_id][surface_type]['tampering_code'] = code
                    parcels[parcel_id][surface_type]['tampering_label'] = self._parse_tampering_code(code)

        return dict(parcels)

    def _infer_surface_type(self, path):
        """Infer surface type from path or filename."""
        path_str = str(path).lower()
        for surface_type in self.surface_types:
            if surface_type in path_str:
                return surface_type
        return 'unknown'

    def _parse_tampering_code(self, code):
        """
        Parse tampering code to label.

        Args:
            code: Tampering code (e.g., "T1W2", "CLEAN", "F3")

        Returns:
            label: Integer label (0=clean, 1=tape, 2=writing, 3=fold, 4=other)
        """
        if not code or code.upper() == 'CLEAN':
            return 0

        # Check for specific tampering types
        if 'T' in code.upper():
            return 1  # Tape
        elif 'W' in code.upper():
            return 2  # Writing
        elif 'F' in code.upper():
            return 3  # Fold
        else:
            return 4  # Other

    def generate_triplets(self, uvmap_type='gt', min_pairs_per_parcel=5):
        """
        Generate triplet pairs: (anchor, positive, negative).

        Strategy:
        - Anchor: Clean surface
        - Positive: Same clean surface or similar clean surface
        - Negative: Tampered version (hard negative)

        Args:
            uvmap_type: Use 'gt' or 'pred' UV maps
            min_pairs_per_parcel: Minimum triplets per parcel

        Returns:
            triplets: List of (anchor, positive, negative, label) tuples
        """
        triplets = []

        # Group parcels by surface type
        parcels_by_surface = defaultdict(list)
        for parcel_id, surfaces in self.parcels.items():
            for surface_type, data in surfaces.items():
                if uvmap_type in data and 'tampering_label' in data:
                    parcels_by_surface[surface_type].append({
                        'parcel_id': parcel_id,
                        'surface_type': surface_type,
                        'uvmap_path': data[uvmap_type],
                        'tampering_label': data['tampering_label'],
                        'tampering_code': data.get('tampering_code', 'CLEAN')
                    })

        print(f"\nGenerating triplets ({uvmap_type} UV maps)...")

        for surface_type, parcel_list in parcels_by_surface.items():
            # Group by parcel ID first - we only want to compare surfaces from SAME parcel
            parcels_by_id = defaultdict(list)
            for p in parcel_list:
                parcels_by_id[p['parcel_id']].append(p)

            print(f"  {surface_type}: {len(parcels_by_id)} unique parcels")

            # Create triplets within SAME parcel only
            for parcel_id, surfaces in parcels_by_id.items():
                # Separate clean and tampered for THIS parcel
                clean_surfaces = [s for s in surfaces if s['tampering_label'] == 0]
                tampered_surfaces = [s for s in surfaces if s['tampering_label'] > 0]

                if len(clean_surfaces) == 0 or len(tampered_surfaces) == 0:
                    continue

                num_triplets = 0

                # Anchor and Positive: SAME parcel, clean surfaces
                for clean_anchor in clean_surfaces:
                    # Positive: Same clean surface (can use duplicates or self)
                    for clean_positive in clean_surfaces:
                        # Negative: Tampered surface from SAME parcel (hard negative!)
                        for tampered_negative in tampered_surfaces:
                            triplets.append({
                                'anchor': str(clean_anchor['uvmap_path']),
                                'positive': str(clean_positive['uvmap_path']),
                                'negative': str(tampered_negative['uvmap_path']),
                                'anchor_label': clean_anchor['tampering_label'],
                                'positive_label': clean_positive['tampering_label'],
                                'negative_label': tampered_negative['tampering_label'],
                                'surface_type': surface_type,
                                'parcel_id': parcel_id,
                                'negative_tampering': tampered_negative['tampering_code']
                            })
                            num_triplets += 1

                            if num_triplets >= min_pairs_per_parcel:
                                break

                        if num_triplets >= min_pairs_per_parcel:
                            break

                    if num_triplets >= min_pairs_per_parcel:
                        break

        print(f"Generated {len(triplets)} triplets")
        return triplets

    def generate_quadruplets(self, uvmap_type='gt', min_pairs_per_parcel=5):
        """
        Generate quadruplet pairs: (anchor, positive, negative1, negative2).

        Strategy:
        - Anchor: Clean surface
        - Positive: Similar clean surface
        - Negative1: Tampered surface (one type)
        - Negative2: Tampered surface (different type)

        Args:
            uvmap_type: Use 'gt' or 'pred' UV maps
            min_pairs_per_parcel: Minimum quadruplets per parcel

        Returns:
            quadruplets: List of (anchor, positive, neg1, neg2, labels) tuples
        """
        quadruplets = []

        # Group parcels by surface type
        parcels_by_surface = defaultdict(list)
        for parcel_id, surfaces in self.parcels.items():
            for surface_type, data in surfaces.items():
                if uvmap_type in data and 'tampering_label' in data:
                    parcels_by_surface[surface_type].append({
                        'parcel_id': parcel_id,
                        'surface_type': surface_type,
                        'uvmap_path': data[uvmap_type],
                        'tampering_label': data['tampering_label'],
                        'tampering_code': data.get('tampering_code', 'CLEAN')
                    })

        print(f"\nGenerating quadruplets ({uvmap_type} UV maps)...")

        for surface_type, parcel_list in parcels_by_surface.items():
            # Group by parcel ID first - we only want to compare surfaces from SAME parcel
            parcels_by_id = defaultdict(list)
            for p in parcel_list:
                parcels_by_id[p['parcel_id']].append(p)

            print(f"  {surface_type}: {len(parcels_by_id)} unique parcels")

            # Create quadruplets within SAME parcel only
            for parcel_id, surfaces in parcels_by_id.items():
                # Separate clean for THIS parcel
                clean_surfaces = [s for s in surfaces if s['tampering_label'] == 0]

                # Group tampered by type for THIS parcel
                tampered_by_type = defaultdict(list)
                for s in surfaces:
                    if s['tampering_label'] > 0:
                        tampered_by_type[s['tampering_label']].append(s)

                if len(clean_surfaces) == 0 or len(tampered_by_type) < 2:
                    continue

                num_quads = 0
                tampering_types = list(tampered_by_type.keys())

                # Anchor and Positive: SAME parcel, clean surfaces
                for clean_anchor in clean_surfaces:
                    # Positive: Same clean surface
                    for clean_positive in clean_surfaces:
                        # Negative1 and Negative2: Different tampering types from SAME parcel
                        for i, type1 in enumerate(tampering_types):
                            for type2 in tampering_types[i+1:]:
                                if len(tampered_by_type[type1]) == 0 or len(tampered_by_type[type2]) == 0:
                                    continue

                                neg1 = random.choice(tampered_by_type[type1])
                                neg2 = random.choice(tampered_by_type[type2])

                                quadruplets.append({
                                    'anchor': str(clean_anchor['uvmap_path']),
                                    'positive': str(clean_positive['uvmap_path']),
                                    'negative1': str(neg1['uvmap_path']),
                                    'negative2': str(neg2['uvmap_path']),
                                    'anchor_label': clean_anchor['tampering_label'],
                                    'positive_label': clean_positive['tampering_label'],
                                    'negative1_label': neg1['tampering_label'],
                                    'negative2_label': neg2['tampering_label'],
                                    'surface_type': surface_type,
                                    'parcel_id': parcel_id,
                                    'negative1_tampering': neg1['tampering_code'],
                                    'negative2_tampering': neg2['tampering_code']
                                })
                                num_quads += 1

                                if num_quads >= min_pairs_per_parcel:
                                    break

                            if num_quads >= min_pairs_per_parcel:
                                break

                        if num_quads >= min_pairs_per_parcel:
                            break

                    if num_quads >= min_pairs_per_parcel:
                        break

        print(f"Generated {len(quadruplets)} quadruplets")
        return quadruplets

    def save_pairs(self, triplets, quadruplets, output_dir):
        """
        Save generated pairs to CSV files.

        Args:
            triplets: List of triplet dicts
            quadruplets: List of quadruplet dicts
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save triplets
        if triplets:
            triplet_df = pd.DataFrame(triplets)
            triplet_path = output_dir / 'tampering_triplets.csv'
            triplet_df.to_csv(triplet_path, index=False)
            print(f"\n✓ Saved {len(triplets)} triplets to {triplet_path}")

        # Save quadruplets
        if quadruplets:
            quadruplet_df = pd.DataFrame(quadruplets)
            quadruplet_path = output_dir / 'tampering_quadruplets.csv'
            quadruplet_df.to_csv(quadruplet_path, index=False)
            print(f"✓ Saved {len(quadruplets)} quadruplets to {quadruplet_path}")

        # Save metadata
        metadata = {
            'data_root': str(self.data_root),
            'num_parcels': len(self.parcels),
            'num_triplets': len(triplets),
            'num_quadruplets': len(quadruplets),
            'surface_types': self.surface_types,
            'tampering_map': self.tampering_map,
            'seed': self.seed
        }

        metadata_path = output_dir / 'tampering_pairs_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate tampering-aware pairs for multi-task learning"
    )

    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing parcels')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for pair files')

    parser.add_argument('--pair_types', type=str, nargs='+',
                       choices=['triplet', 'quadruplet', 'both'],
                       default=['both'],
                       help='Types of pairs to generate')

    parser.add_argument('--uvmap_type', type=str, choices=['gt', 'pred', 'both'],
                       default='gt',
                       help='UV map type to use')

    parser.add_argument('--min_pairs_per_parcel', type=int, default=10,
                       help='Minimum pairs per parcel')

    parser.add_argument('--surface_types', type=str, nargs='+',
                       default=None,
                       help='Surface types to include (default: auto-detect)')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("Tampering-Aware Pair Generation")
    print(f"{'='*70}")
    print(f"Data root:        {args.data_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Pair types:       {args.pair_types}")
    print(f"UV map type:      {args.uvmap_type}")
    print(f"Min pairs/parcel: {args.min_pairs_per_parcel}")

    # Generate pairs
    generator = TamperingPairGenerator(
        data_root=args.data_root,
        surface_types=args.surface_types,
        seed=args.seed
    )

    # Determine which UV map types to use
    uvmap_types = ['gt', 'pred'] if args.uvmap_type == 'both' else [args.uvmap_type]

    for uvmap_type in uvmap_types:
        triplets = []
        quadruplets = []

        # Generate pairs
        if 'triplet' in args.pair_types or 'both' in args.pair_types:
            triplets = generator.generate_triplets(
                uvmap_type=uvmap_type,
                min_pairs_per_parcel=args.min_pairs_per_parcel
            )

        if 'quadruplet' in args.pair_types or 'both' in args.pair_types:
            quadruplets = generator.generate_quadruplets(
                uvmap_type=uvmap_type,
                min_pairs_per_parcel=args.min_pairs_per_parcel
            )

        # Save pairs
        output_subdir = Path(args.output_dir) / uvmap_type
        generator.save_pairs(triplets, quadruplets, output_subdir)

    print(f"\n{'='*70}")
    print("✓ Pair generation complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
