import os
import sys
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())


class TamperingPairGenerator:

    def __init__(self, data_root, surface_types=None, tampering_mapping_csv=None, seed=42):
        self.data_root = Path(data_root)
        self.seed = seed
        random.seed(seed)

        # surface types
        if surface_types is None:
            surface_types = ['center', 'top', 'bottom', 'left', 'right']
        self.surface_types = surface_types

        # tampering type mapping
        self.tampering_map = {
            '': 0,
            'T': 1,
            'W': 2,
            'L': 3,
            'other': 4
        }

        if tampering_mapping_csv is None:
            tampar_root = Path(__file__).parent.parent
            tampering_mapping_csv = tampar_root / "src" / "tampering" / "tampering_mapping.csv"

        self.tampering_mapping_csv = Path(tampering_mapping_csv)
        if not self.tampering_mapping_csv.exists():
            raise FileNotFoundError(f"tampering_mapping.csv not found at {self.tampering_mapping_csv}")

        self.tampering_mapping = self._load_tampering_mapping()

        self.parcels = self._load_parcels()

        print(f"\nLoaded {len(self.parcels)} parcels from {data_root}")
        print(f"Surface types: {self.surface_types}")
        print(f"Tampering labels: {len(self.tampering_mapping)} parcels")

    def _load_tampering_mapping(self):
        import pandas as pd

        df = pd.read_csv(self.tampering_mapping_csv)
        df.fillna('', inplace=True)

        tampering_mapping = {}
        for _, row in df.iterrows():
            parcel_id = f"id_{str(row['id']).zfill(2)}"
            tampering_mapping[parcel_id] = {
                'center': row['center'],
                'top': row['top'],
                'bottom': row['bottom'],
                'left': row['left'],
                'right': row['right']
            }

        return tampering_mapping

    def _load_parcels(self):
        parcels = defaultdict(lambda: defaultdict(dict))

        # search recursively for uvmaps
        for uvmap_path in self.data_root.rglob("*_uvmap_*.png"):
        
            relative_path = uvmap_path.relative_to(self.data_root)
            parts = relative_path.parts

            if len(parts) >= 3:
                surface_type = parts[0]
                parcel_id = parts[1]
            elif len(parts) == 2:
                surface_type = self._infer_surface_type(uvmap_path)
                parcel_id = parts[0]
            else:
                continue

            if '_uvmap_gt.png' in uvmap_path.name:
                map_type = 'gt'
            elif '_uvmap_pred.png' in uvmap_path.name:
                map_type = 'pred'
            else:
                continue

            parcels[parcel_id][surface_type][map_type] = uvmap_path

            if parcel_id in self.tampering_mapping and 'tampering_code' not in parcels[parcel_id][surface_type]:
                code = self.tampering_mapping[parcel_id].get(surface_type, '')
                parcels[parcel_id][surface_type]['tampering_code'] = code
                parcels[parcel_id][surface_type]['tampering_label'] = self._parse_tampering_code(code)

        return dict(parcels)

    def _infer_surface_type(self, path):
        path_str = str(path).lower()
        for surface_type in self.surface_types:
            if surface_type in path_str:
                return surface_type
        return 'unknown'

    def _parse_tampering_code(self, code):
        if not code or code == '':
            return 0

        code_lower = code.lower()
        first_char = code_lower[0] if code_lower else ''

        # map tampering types
        if first_char == 't':
            return 1 
        elif first_char == 'w':
            return 2
        elif first_char == 'l':
            return 3
        else:
            return 4

    def generate_triplets(self, uvmap_type='gt', min_pairs_per_parcel=5):
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

        print(f"\nGenerating triplets ({uvmap_type} UV maps)")

        for surface_type, parcel_list in parcels_by_surface.items():
            parcels_by_id = defaultdict(list)
            for p in parcel_list:
                parcels_by_id[p['parcel_id']].append(p)

            print(f"  {surface_type}: {len(parcels_by_id)} unique parcels")

            for parcel_id, surfaces in parcels_by_id.items():
                clean_surfaces = [s for s in surfaces if s['tampering_label'] == 0]
                tampered_surfaces = [s for s in surfaces if s['tampering_label'] > 0]

                if len(clean_surfaces) == 0 or len(tampered_surfaces) == 0:
                    continue

                num_triplets = 0

                for clean_anchor in clean_surfaces:
                    for clean_positive in clean_surfaces:
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

        print(f"\nGenerating quadruplets ({uvmap_type} UV maps)")

        for surface_type, parcel_list in parcels_by_surface.items():
            # group by parcel id first
            parcels_by_id = defaultdict(list)
            for p in parcel_list:
                parcels_by_id[p['parcel_id']].append(p)

            print(f"  {surface_type}: {len(parcels_by_id)} unique parcels")

            for parcel_id, surfaces in parcels_by_id.items():
                clean_surfaces = [s for s in surfaces if s['tampering_label'] == 0]

                tampered_by_type = defaultdict(list)
                for s in surfaces:
                    if s['tampering_label'] > 0:
                        tampered_by_type[s['tampering_label']].append(s)

                if len(clean_surfaces) == 0 or len(tampered_by_type) < 2:
                    continue

                num_quads = 0
                tampering_types = list(tampered_by_type.keys())

                for clean_anchor in clean_surfaces:
                    for clean_positive in clean_surfaces:
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
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # save triplets
        if triplets:
            triplet_df = pd.DataFrame(triplets)
            triplet_path = output_dir / 'tampering_triplets.csv'
            triplet_df.to_csv(triplet_path, index=False)
            print(f"\n Saved {len(triplets)} triplets to {triplet_path}")

        # save quadruplets
        if quadruplets:
            quadruplet_df = pd.DataFrame(quadruplets)
            quadruplet_path = output_dir / 'tampering_quadruplets.csv'
            quadruplet_df.to_csv(quadruplet_path, index=False)
            print(f" Saved {len(quadruplets)} quadruplets to {quadruplet_path}")

        # save metadata
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
        print(f" Saved metadata to {metadata_path}")


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

    print("\nTampering-Aware Pair Generation")
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

    uvmap_types = ['gt', 'pred'] if args.uvmap_type == 'both' else [args.uvmap_type]

    for uvmap_type in uvmap_types:
        triplets = []
        quadruplets = []

        # generate pairs
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

        # save pairs
        output_subdir = Path(args.output_dir) / uvmap_type
        generator.save_pairs(triplets, quadruplets, output_subdir)

    print("\nPair generation complete!")


if __name__ == "__main__":
    main()
