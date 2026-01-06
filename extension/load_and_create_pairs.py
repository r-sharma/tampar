"""
Task 4: Load Existing UV Maps and Create Contrastive Pairs
Development Version - Using tampar_sample dataset

This script:
1. Loads existing UV maps from TAMPAR sample dataset
2. Explores the data structure
3. Creates positive and negative pairs
4. Saves pairs for contrastive training

Usage:
    python load_and_create_pairs.py --data_root src/data/tampar_sample
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


class TAMPARDatasetLoader:
    """
    Load and explore existing UV maps from TAMPAR dataset.
    """
    
    def __init__(self, data_root):
        """
        Initialize dataset loader.
        
        Args:
            data_root: Path to TAMPAR dataset (e.g., 'src/data/tampar_sample' or 'tampar')
        """
        self.data_root = Path(data_root)
        self.uvmaps_dir = self.data_root / 'uvmaps'
        
        # Check if this is full dataset or sample
        self.is_sample = 'sample' in str(data_root)
        
        print(f"\n{'='*70}")
        print(f"TAMPAR Dataset Loader")
        print(f"{'='*70}")
        print(f"Data root: {self.data_root}")
        print(f"Is sample dataset: {self.is_sample}")
        print(f"UV maps directory: {self.uvmaps_dir}")
        
        self.uv_maps = {}
        self.metadata = {}
        
    def explore_structure(self):
        """Explore and print dataset structure."""
        print(f"\n{'='*70}")
        print("Exploring Dataset Structure")
        print(f"{'='*70}")
        
        # Check what exists
        print("\nDirectories found:")
        for item in sorted(self.data_root.iterdir()):
            if item.is_dir():
                print(f"  📁 {item.name}/")
                # Count files in directory
                num_files = len(list(item.glob('**/*.*')))
                print(f"      {num_files} files")
        
        print("\nFiles in root:")
        for item in sorted(self.data_root.iterdir()):
            if item.is_file():
                print(f"  📄 {item.name}")
        
        # Check for UV maps
        if self.uvmaps_dir.exists():
            uv_files = list(self.uvmaps_dir.glob('*.png'))
            print(f"\n✓ Found {len(uv_files)} reference UV maps in uvmaps/")
            if uv_files:
                print(f"  Example: {uv_files[0].name}")
        else:
            print(f"\n✗ No uvmaps/ directory found")
        
        # Check for test/validation splits
        for split in ['test', 'validation']:
            split_dir = self.data_root / split
            if split_dir.exists():
                print(f"\n✓ Found {split}/ directory")
                
                # Check backgrounds
                backgrounds = [d for d in split_dir.iterdir() if d.is_dir()]
                print(f"  Backgrounds: {len(backgrounds)}")
                
                for bg in backgrounds[:3]:  # Show first 3
                    uv_pred_files = list(bg.glob('*_uvmap_pred.png'))
                    uv_gt_files = list(bg.glob('*_uvmap_gt.png'))
                    print(f"    {bg.name}:")
                    print(f"      {len(uv_pred_files)} predicted UV maps")
                    print(f"      {len(uv_gt_files)} ground truth UV maps")
                    if uv_pred_files:
                        print(f"      Example: {uv_pred_files[0].name}")
            else:
                print(f"\n✗ No {split}/ directory found")
        
        # Check for metadata
        for metadata_file in ['tampar_sample_validation.json', 'metadata_validation.json']:
            metadata_path = self.data_root / metadata_file
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                print(f"\n✓ Found {metadata_file}")
                print(f"  Entries: {len(data) if isinstance(data, list) else 'dict structure'}")
            else:
                print(f"\n✗ No {metadata_file} found")
    
    def load_reference_uvmaps(self):
        """
        Load reference UV maps from uvmaps/ folder.
        
        Returns:
            Dictionary mapping parcel_id -> UV map image
        """
        print(f"\n{'='*70}")
        print("Loading Reference UV Maps")
        print(f"{'='*70}")
        
        if not self.uvmaps_dir.exists():
            print("✗ uvmaps/ directory not found")
            return {}
        
        uv_maps = {}
        uv_files = sorted(self.uvmaps_dir.glob('*.png'))
        
        for uv_file in tqdm(uv_files, desc="Loading reference UV maps"):
            # Filename is typically: <parcel_id>.png (e.g., "0.png", "1.png")
            parcel_id = uv_file.stem
            
            try:
                uv_map = Image.open(uv_file).convert('RGB')
                uv_maps[parcel_id] = {
                    'image': uv_map,
                    'path': uv_file,
                    'size': uv_map.size
                }
            except Exception as e:
                print(f"\n✗ Error loading {uv_file}: {e}")
        
        print(f"\n✓ Loaded {len(uv_maps)} reference UV maps")
        if uv_maps:
            # Show example
            example_id = list(uv_maps.keys())[0]
            example = uv_maps[example_id]
            print(f"  Example - Parcel {example_id}:")
            print(f"    Size: {example['size']}")
            print(f"    Path: {example['path']}")
        
        self.uv_maps['reference'] = uv_maps
        return uv_maps
    
    def load_split_uvmaps(self, split='validation'):
        """
        Load predicted and ground truth UV maps from test/validation split.
        
        Args:
            split: 'test' or 'validation'
        
        Returns:
            Dictionary with UV map data organized by parcel_id
        """
        print(f"\n{'='*70}")
        print(f"Loading {split.upper()} UV Maps")
        print(f"{'='*70}")
        
        split_dir = self.data_root / split
        if not split_dir.exists():
            print(f"✗ {split}/ directory not found")
            return {}
        
        uv_data = defaultdict(lambda: defaultdict(list))
        
        # Find all backgrounds
        backgrounds = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        print(f"Found {len(backgrounds)} background(s): {[b.name for b in backgrounds]}")
        
        for background in backgrounds:
            print(f"\nProcessing {background.name}...")
            
            # Find all UV map files
            uv_pred_files = sorted(background.glob('*_uvmap_pred.png'))
            uv_gt_files = sorted(background.glob('*_uvmap_gt.png'))
            
            print(f"  Predicted UV maps: {len(uv_pred_files)}")
            print(f"  Ground truth UV maps: {len(uv_gt_files)}")
            
            # Process predicted UV maps
            for uv_file in tqdm(uv_pred_files, desc=f"  Loading {background.name}", leave=False):
                # Parse filename: <parcel_id>_<bg_id>_<img_id>_uvmap_pred.png
                # Example: "0_0_0_uvmap_pred.png" -> parcel=0, bg=0, img=0
                parts = uv_file.stem.split('_')
                
                if len(parts) >= 3:
                    parcel_id = parts[0]
                    bg_id = parts[1]
                    img_id = parts[2]
                    
                    try:
                        uv_map = Image.open(uv_file).convert('RGB')
                        
                        uv_data[parcel_id]['predicted'].append({
                            'image': uv_map,
                            'path': uv_file,
                            'background': background.name,
                            'bg_id': bg_id,
                            'img_id': img_id,
                            'size': uv_map.size
                        })
                    except Exception as e:
                        print(f"\n✗ Error loading {uv_file}: {e}")
            
            # Process ground truth UV maps
            for uv_file in uv_gt_files:
                parts = uv_file.stem.split('_')
                
                if len(parts) >= 3:
                    parcel_id = parts[0]
                    bg_id = parts[1]
                    img_id = parts[2]
                    
                    try:
                        uv_map = Image.open(uv_file).convert('RGB')
                        
                        uv_data[parcel_id]['ground_truth'].append({
                            'image': uv_map,
                            'path': uv_file,
                            'background': background.name,
                            'bg_id': bg_id,
                            'img_id': img_id,
                            'size': uv_map.size
                        })
                    except Exception as e:
                        print(f"\n✗ Error loading {uv_file}: {e}")
        
        # Convert defaultdict to regular dict
        uv_data = {k: dict(v) for k, v in uv_data.items()}
        
        # Summary
        print(f"\n{'='*70}")
        print(f"Summary - {split.upper()} Split")
        print(f"{'='*70}")
        print(f"Total unique parcels: {len(uv_data)}")
        
        for parcel_id, data in sorted(uv_data.items())[:5]:  # Show first 5
            print(f"\nParcel {parcel_id}:")
            print(f"  Predicted UV maps: {len(data.get('predicted', []))}")
            print(f"  Ground truth UV maps: {len(data.get('ground_truth', []))}")
        
        if len(uv_data) > 5:
            print(f"\n... and {len(uv_data) - 5} more parcels")
        
        self.uv_maps[split] = uv_data
        return uv_data
    
    def visualize_samples(self, num_samples=3):
        """Visualize sample UV maps."""
        print(f"\n{'='*70}")
        print(f"Visualizing {num_samples} Sample UV Maps")
        print(f"{'='*70}")
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        # Get some parcel IDs
        if 'reference' in self.uv_maps and self.uv_maps['reference']:
            parcel_ids = list(self.uv_maps['reference'].keys())[:num_samples]
            
            for idx, parcel_id in enumerate(parcel_ids):
                # Reference UV
                ref_uv = self.uv_maps['reference'][parcel_id]['image']
                axes[idx, 0].imshow(ref_uv)
                axes[idx, 0].set_title(f"Parcel {parcel_id}\nReference UV")
                axes[idx, 0].axis('off')
                
                # Predicted UV (if available)
                split_key = 'validation' if 'validation' in self.uv_maps else 'test'
                if split_key in self.uv_maps and parcel_id in self.uv_maps[split_key]:
                    if 'predicted' in self.uv_maps[split_key][parcel_id]:
                        pred_uvs = self.uv_maps[split_key][parcel_id]['predicted']
                        if pred_uvs:
                            axes[idx, 1].imshow(pred_uvs[0]['image'])
                            axes[idx, 1].set_title(f"Predicted UV\n{pred_uvs[0]['background']}")
                            axes[idx, 1].axis('off')
                    
                    if 'ground_truth' in self.uv_maps[split_key][parcel_id]:
                        gt_uvs = self.uv_maps[split_key][parcel_id]['ground_truth']
                        if gt_uvs:
                            axes[idx, 2].imshow(gt_uvs[0]['image'])
                            axes[idx, 2].set_title(f"Ground Truth UV\n{gt_uvs[0]['background']}")
                            axes[idx, 2].axis('off')
        
        plt.tight_layout()
        output_path = self.data_root / 'sample_uvmaps_visualization.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_path}")
        plt.close()


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Load and explore TAMPAR UV maps"
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='src/data/tampar_sample',
        help='Path to TAMPAR dataset'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='validation',
        choices=['test', 'validation'],
        help='Which split to load'
    )
    
    args = parser.parse_args()
    
    # Initialize loader
    loader = TAMPARDatasetLoader(args.data_root)
    
    # Explore structure
    loader.explore_structure()
    
    # Load reference UV maps
    loader.load_reference_uvmaps()
    
    # Load split UV maps
    loader.load_split_uvmaps(args.split)
    
    # Visualize samples
    if loader.uv_maps:
        loader.visualize_samples(num_samples=min(3, len(loader.uv_maps.get('reference', {}))))
    
    print(f"\n{'='*70}")
    print("✓ Dataset Loading Complete!")
    print(f"{'='*70}")
    print(f"\nNext step: Create contrastive pairs from loaded UV maps")


if __name__ == "__main__":
    main()
