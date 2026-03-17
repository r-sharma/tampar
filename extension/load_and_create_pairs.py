
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import pickle
import torch
from torchvision import transforms


class TAMPARDatasetLoader:
    
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.uvmaps_dir = self.data_root / 'uvmaps'
        
        # Check if this is full dataset or sample
        self.is_sample = 'sample' in str(data_root)
        
        print(f"TAMPAR Dataset Loader (v2 - Fixed for Sample Dataset)")
        print(f"Data root: {self.data_root}")
        print(f"Is sample dataset: {self.is_sample}")
        print(f"UV maps directory: {self.uvmaps_dir}")
        
        self.uv_maps = {}
        self.metadata = {}
        
    def explore_structure(self):
        print("Exploring Dataset Structure")
        
        # Check what exists
        print("\nDirectories found:")
        for item in sorted(self.data_root.iterdir()):
            if item.is_dir():
                print(f"   {item.name}/")
                # Count files in directory
                num_files = len(list(item.glob('*.*')))
                print(f"      {num_files} files")
        
        print("\nFiles in root:")
        for item in sorted(self.data_root.iterdir()):
            if item.is_file():
                print(f"  📄 {item.name}")
        
        # Check for UV maps
        if self.uvmaps_dir.exists():
            uv_files = list(self.uvmaps_dir.glob('*.png'))
            print(f"\n Found {len(uv_files)} reference UV maps in uvmaps/")
            if uv_files:
                print(f"  Examples:")
                for f in uv_files[:3]:
                    print(f"    - {f.name}")
        else:
            print(f"\n No uvmaps/ directory found")
        
        # Check for test/validation splits
        for split in ['test', 'validation']:
            split_dir = self.data_root / split
            if split_dir.exists():
                print(f"\n Found {split}/ directory")
                
                # Check if files are directly in split dir or in background subdirs
                direct_files = list(split_dir.glob('*.png'))
                subdirs = [d for d in split_dir.iterdir() if d.is_dir()]
                
                if direct_files:
                    print(f"  Structure: Files directly in {split}/ (sample dataset)")
                    uv_pred = [f for f in direct_files if 'uvmap_pred' in f.name]
                    uv_gt = [f for f in direct_files if 'uvmap_gt' in f.name]
                    orig_imgs = [f for f in split_dir.glob('*.jpg')]
                    print(f"    {len(uv_pred)} predicted UV maps")
                    print(f"    {len(uv_gt)} ground truth UV maps")
                    print(f"    {len(orig_imgs)} original images")
                    if uv_pred:
                        print(f"    Example: {uv_pred[0].name}")
                elif subdirs:
                    print(f"  Structure: Background subdirectories (full dataset)")
                    print(f"  Backgrounds: {len(subdirs)}")
                    for bg in subdirs[:2]:
                        uv_pred_files = list(bg.glob('*_uvmap_pred.png'))
                        print(f"    {bg.name}: {len(uv_pred_files)} UV maps")
            else:
                print(f"\n No {split}/ directory found")
        
        # Check for metadata (handle different naming)
        metadata_patterns = [
            'metadata_test.json',
            'metadata_validation.json',
            'tampar_sample_validation.json',
            'tampar_sample_test.json'
        ]
        
        for metadata_file in metadata_patterns:
            metadata_path = self.data_root / metadata_file
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                print(f"\n Found {metadata_file}")
                if isinstance(data, list):
                    print(f"  Entries: {len(data)}")
                elif isinstance(data, dict):
                    print(f"  Keys: {list(data.keys())[:5]}")
    
    def load_reference_uvmaps(self):
        print("Loading Reference UV Maps")
        
        if not self.uvmaps_dir.exists():
            print(" uvmaps/ directory not found")
            return {}
        
        uv_maps = {}
        uv_files = sorted(self.uvmaps_dir.glob('*.png'))
        
        for uv_file in tqdm(uv_files, desc="Loading reference UV maps"):
            # Filename can be: <parcel_id>.png OR id_01_uvmap.png
            parcel_id = uv_file.stem
            
            try:
                uv_map = Image.open(uv_file).convert('RGB')
                uv_maps[parcel_id] = {
                    'image': uv_map,
                    'path': uv_file,
                    'size': uv_map.size,
                    'filename': uv_file.name
                }
            except Exception as e:
                print(f"\n Error loading {uv_file}: {e}")
        
        print(f"\n Loaded {len(uv_maps)} reference UV maps")
        if uv_maps:
            # Show all examples
            print(f"  Parcel IDs found:")
            for parcel_id in sorted(uv_maps.keys()):
                print(f"    - {parcel_id} (size: {uv_maps[parcel_id]['size']})")
        
        self.uv_maps['reference'] = uv_maps
        return uv_maps
    
    def load_split_uvmaps(self, split='validation'):
        print(f"Loading {split.upper()} UV Maps")
        
        split_dir = self.data_root / split
        if not split_dir.exists():
            print(f" {split}/ directory not found")
            return {}
        
        uv_data = defaultdict(lambda: defaultdict(list))
        
        # Check structure: flat (sample) or subdirs (full dataset)
        direct_files = list(split_dir.glob('*.png'))
        subdirs = [d for d in split_dir.iterdir() if d.is_dir()]
        
        if direct_files:
            # Sample dataset structure: files directly in split/
            print(f"Detected sample dataset structure (flat)")
            self._load_flat_structure(split_dir, uv_data)
        elif subdirs:
            # Full dataset structure: background subdirectories
            print(f"Detected full dataset structure (backgrounds)")
            self._load_background_structure(split_dir, subdirs, uv_data)
        else:
            print(f" No UV map files found in {split_dir}")
        
        # Convert defaultdict to regular dict
        uv_data = {k: dict(v) for k, v in uv_data.items()}
        
        # Summary
        print(f"Summary - {split.upper()} Split")
        print(f"Total unique parcels: {len(uv_data)}")
        
        for parcel_id, data in sorted(uv_data.items()):
            print(f"\nParcel {parcel_id}:")
            print(f"  Predicted UV maps: {len(data.get('predicted', []))}")
            print(f"  Ground truth UV maps: {len(data.get('ground_truth', []))}")
            if data.get('predicted'):
                print(f"    Example: {data['predicted'][0]['filename']}")
        
        self.uv_maps[split] = uv_data
        return uv_data
    
    def _load_flat_structure(self, split_dir, uv_data):
        # Find all UV map files
        uv_pred_files = sorted(split_dir.glob('*_uvmap_pred.png'))
        uv_gt_files = sorted(split_dir.glob('*_uvmap_gt.png'))
        
        print(f"  Predicted UV maps: {len(uv_pred_files)}")
        print(f"  Ground truth UV maps: {len(uv_gt_files)}")
        
        # Process predicted UV maps
        for uv_file in tqdm(uv_pred_files, desc="  Loading predicted UVs"):
            filename = uv_file.stem.replace('_uvmap_pred', '')
            parts = filename.split('_')
            
            if len(parts) >= 2:
                # parcel_id = id_01, timestamp = 20230516_142710
                parcel_id = f"{parts[0]}_{parts[1]}"
                timestamp = '_'.join(parts[2:]) if len(parts) > 2 else "unknown"
                
                try:
                    uv_map = Image.open(uv_file).convert('RGB')
                    
                    uv_data[parcel_id]['predicted'].append({
                        'image': uv_map,
                        'path': uv_file,
                        'filename': uv_file.name,
                        'timestamp': timestamp,
                        'size': uv_map.size
                    })
                except Exception as e:
                    print(f"\n Error loading {uv_file}: {e}")
        
        # Process ground truth UV maps
        for uv_file in tqdm(uv_gt_files, desc="  Loading GT UVs"):
            filename = uv_file.stem.replace('_uvmap_gt', '')
            parts = filename.split('_')
            
            if len(parts) >= 2:
                parcel_id = f"{parts[0]}_{parts[1]}"
                timestamp = '_'.join(parts[2:]) if len(parts) > 2 else "unknown"
                
                try:
                    uv_map = Image.open(uv_file).convert('RGB')
                    
                    uv_data[parcel_id]['ground_truth'].append({
                        'image': uv_map,
                        'path': uv_file,
                        'filename': uv_file.name,
                        'timestamp': timestamp,
                        'size': uv_map.size
                    })
                except Exception as e:
                    print(f"\n Error loading {uv_file}: {e}")
    
    def _load_background_structure(self, split_dir, subdirs, uv_data):
        print(f"Found {len(subdirs)} background(s): {[b.name for b in subdirs]}")
        
        for background in subdirs:
            print(f"\nProcessing {background.name}")
            
            uv_pred_files = sorted(background.glob('*_uvmap_pred.png'))
            uv_gt_files = sorted(background.glob('*_uvmap_gt.png'))
            
            print(f"  Predicted: {len(uv_pred_files)}, GT: {len(uv_gt_files)}")
            
            # Process predicted UV maps
            for uv_file in tqdm(uv_pred_files, desc=f"  {background.name}", leave=False):
                # Parse: 0_0_0_uvmap_pred.png -> parcel=0, bg=0, img=0
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
                            'filename': uv_file.name,
                            'background': background.name,
                            'bg_id': bg_id,
                            'img_id': img_id,
                            'size': uv_map.size
                        })
                    except Exception as e:
                        print(f"\n Error loading {uv_file}: {e}")
            
            # Similar for GT files
            for uv_file in uv_gt_files:
                parts = uv_file.stem.split('_')
                if len(parts) >= 3:
                    parcel_id = parts[0]
                    try:
                        uv_map = Image.open(uv_file).convert('RGB')
                        uv_data[parcel_id]['ground_truth'].append({
                            'image': uv_map,
                            'path': uv_file,
                            'filename': uv_file.name,
                            'size': uv_map.size
                        })
                    except Exception as e:
                        print(f"\n Error loading {uv_file}: {e}")
    
    def visualize_samples(self, num_samples=None):
        
        # Determine how many samples to show
        if 'reference' in self.uv_maps and self.uv_maps['reference']:
            total_parcels = len(self.uv_maps['reference'])
            if num_samples is None:
                num_samples = total_parcels
            else:
                num_samples = min(num_samples, total_parcels)
        else:
            print(" No reference UV maps to visualize")
            return
        
        print(f"Visualizing {num_samples} UV Maps (Total available: {total_parcels})")
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        # Get parcel IDs
        parcel_ids = sorted(self.uv_maps['reference'].keys())[:num_samples]
        
        for idx, parcel_id in enumerate(parcel_ids):
            # Reference UV
            ref_uv = self.uv_maps['reference'][parcel_id]['image']
            axes[idx, 0].imshow(ref_uv)
            axes[idx, 0].set_title(f"{parcel_id}\nReference UV\n{ref_uv.size}")
            axes[idx, 0].axis('off')
            
            split_key = 'validation' if 'validation' in self.uv_maps else 'test'
            
            # Find matching parcel in split data
            matched = False
            if split_key in self.uv_maps:
                # Try exact match first
                if parcel_id in self.uv_maps[split_key]:
                    matched_parcel = parcel_id
                    matched = True
                else:
                    # Try partial match (e.g., id_01_uvmap matches id_01)
                    for split_parcel in self.uv_maps[split_key].keys():
                        if parcel_id.replace('_uvmap', '') in split_parcel:
                            matched_parcel = split_parcel
                            matched = True
                            break
                
                if matched:
                    # Predicted UV
                    if 'predicted' in self.uv_maps[split_key][matched_parcel]:
                        pred_uvs = self.uv_maps[split_key][matched_parcel]['predicted']
                        if pred_uvs:
                            pred_img = pred_uvs[0]['image']
                            axes[idx, 1].imshow(pred_img)
                            axes[idx, 1].set_title(f"Predicted UV\n{pred_uvs[0]['filename']}\n{pred_img.size}")
                            axes[idx, 1].axis('off')
                    
                    # Ground truth UV
                    if 'ground_truth' in self.uv_maps[split_key][matched_parcel]:
                        gt_uvs = self.uv_maps[split_key][matched_parcel]['ground_truth']
                        if gt_uvs:
                            gt_img = gt_uvs[0]['image']
                            axes[idx, 2].imshow(gt_img)
                            axes[idx, 2].set_title(f"Ground Truth UV\n{gt_uvs[0]['filename']}\n{gt_img.size}")
                            axes[idx, 2].axis('off')
            
            if not matched:
                axes[idx, 1].text(0.5, 0.5, 'No predicted\nUV found', 
                                 ha='center', va='center', transform=axes[idx, 1].transAxes)
                axes[idx, 1].axis('off')
                axes[idx, 2].text(0.5, 0.5, 'No GT\nUV found', 
                                 ha='center', va='center', transform=axes[idx, 2].transAxes)
                axes[idx, 2].axis('off')
        
        plt.tight_layout()
        output_path = self.data_root / 'sample_uvmaps_visualization_all.png'
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"\n Visualization saved to: {output_path}")
        plt.close()
        
        return str(output_path)


class AugmentationPipeline:
    
    def __init__(self, rotation_range=5, brightness_range=0.1, 
                 contrast_range=0.1, noise_std=0.02):
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
    
    def augment(self, image):
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
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
        
        return image


class ContrastivePairCreator:
    
    def __init__(self, loader, augmentor=None, random_seed=42):
        self.loader = loader
        self.augmentor = augmentor or AugmentationPipeline()
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        self.positive_pairs = []
        self.negative_pairs = []
    
    def create_positive_pairs(self, strategies=['same_parcel', 'ref_vs_pred', 'augmented'], 
                             num_pairs=100):
        print("Creating Positive Pairs")
        print(f"Strategies: {strategies}")
        print(f"Target pairs: {num_pairs}")
        
        positive_pairs = []
        split = 'validation' if 'validation' in self.loader.uv_maps else 'test'
        
        # Strategy 1: Same parcel, different captures
        if 'same_parcel' in strategies:
            print("\n1. Same parcel, different captures")
            pairs = self._create_same_parcel_pairs(split)
            positive_pairs.extend(pairs)
            print(f"   Created {len(pairs)} pairs")
        
        # Strategy 2: Reference vs Predicted
        if 'ref_vs_pred' in strategies:
            print("\n2. Reference vs Predicted UV")
            pairs = self._create_ref_vs_pred_pairs(split)
            positive_pairs.extend(pairs)
            print(f"   Created {len(pairs)} pairs")
        
        # Strategy 3: Augmented pairs
        if 'augmented' in strategies:
            print("\n3. Original vs Augmented")
            pairs = self._create_augmented_pairs(split, num_variants=2)
            positive_pairs.extend(pairs)
            print(f"   Created {len(pairs)} pairs")
        
        # Limit to target number
        if len(positive_pairs) > num_pairs:
            positive_pairs = random.sample(positive_pairs, num_pairs)
        
        self.positive_pairs = positive_pairs
        
        print(f"\n Total positive pairs created: {len(positive_pairs)}")
        return positive_pairs
    
    def _create_same_parcel_pairs(self, split):
        pairs = []
        
        if split not in self.loader.uv_maps:
            return pairs
        
        for parcel_id, data in self.loader.uv_maps[split].items():
            if 'predicted' not in data:
                continue
            
            pred_uvs = data['predicted']
            
            # If multiple captures exist, pair them
            if len(pred_uvs) >= 2:
                # Create all possible pairs
                for i in range(len(pred_uvs)):
                    for j in range(i + 1, len(pred_uvs)):
                        pairs.append({
                            'image1': pred_uvs[i]['image'],
                            'image2': pred_uvs[j]['image'],
                            'label': 1,
                            'parcel_id': parcel_id,
                            'type': 'same_parcel_different_capture',
                            'metadata': {
                                'file1': pred_uvs[i]['filename'],
                                'file2': pred_uvs[j]['filename']
                            }
                        })
        
        return pairs
    
    def _create_ref_vs_pred_pairs(self, split):
        pairs = []
        
        if 'reference' not in self.loader.uv_maps or split not in self.loader.uv_maps:
            return pairs
        
        for parcel_id, split_data in self.loader.uv_maps[split].items():
            if 'predicted' not in split_data:
                continue
            
            matched_ref = None
            for ref_id in self.loader.uv_maps['reference'].keys():
                if parcel_id in ref_id or ref_id.replace('_uvmap', '') == parcel_id:
                    matched_ref = ref_id
                    break
            
            if matched_ref:
                ref_uv = self.loader.uv_maps['reference'][matched_ref]['image']
                
                # Pair reference with each predicted UV
                for pred_data in split_data['predicted']:
                    pairs.append({
                        'image1': ref_uv,
                        'image2': pred_data['image'],
                        'label': 1,
                        'parcel_id': parcel_id,
                        'type': 'reference_vs_predicted',
                        'metadata': {
                            'ref_id': matched_ref,
                            'pred_file': pred_data['filename']
                        }
                    })
        
        return pairs
    
    def _create_augmented_pairs(self, split, num_variants=2):
        pairs = []
        
        if split not in self.loader.uv_maps:
            return pairs
        
        for parcel_id, data in self.loader.uv_maps[split].items():
            if 'predicted' not in data:
                continue
            
            # For each predicted UV, create augmented versions
            for pred_data in data['predicted']:
                orig_image = pred_data['image']
                
                for variant_idx in range(num_variants):
                    aug_image = self.augmentor.augment(orig_image)
                    
                    pairs.append({
                        'image1': orig_image,
                        'image2': aug_image,
                        'label': 1,
                        'parcel_id': parcel_id,
                        'type': 'original_vs_augmented',
                        'metadata': {
                            'orig_file': pred_data['filename'],
                            'variant': variant_idx
                        }
                    })
        
        return pairs
    
    def create_negative_pairs(self, strategies=['different_parcels'], num_pairs=100):
        print("Creating Negative Pairs")
        print(f"Strategies: {strategies}")
        print(f"Target pairs: {num_pairs}")
        
        negative_pairs = []
        split = 'validation' if 'validation' in self.loader.uv_maps else 'test'
        
        # Strategy 1: Different parcels
        if 'different_parcels' in strategies:
            print("\n1. Different parcels")
            pairs = self._create_different_parcel_pairs(split, num_pairs)
            negative_pairs.extend(pairs)
            print(f"   Created {len(pairs)} pairs")
        
        # Strategy 2: Clean vs Tampered (if tampering data available)
        if 'clean_vs_tampered' in strategies:
            print("\n2. Clean vs Tampered")
            # TODO: Implement if tampering_mapping.csv is available
            print("   (Skipped - tampering data not available in sample)")
        
        self.negative_pairs = negative_pairs
        
        print(f"\n Total negative pairs created: {len(negative_pairs)}")
        return negative_pairs
    
    def _create_different_parcel_pairs(self, split, num_pairs):
        pairs = []
        
        if split not in self.loader.uv_maps:
            return pairs
        
        # Get list of all parcel IDs
        parcel_ids = list(self.loader.uv_maps[split].keys())
        
        if len(parcel_ids) < 2:
            print("    Need at least 2 parcels for negative pairs")
            return pairs
        
        # Create random pairs of different parcels
        attempts = 0
        max_attempts = num_pairs * 10
        
        while len(pairs) < num_pairs and attempts < max_attempts:
            attempts += 1
            
            # Randomly select two different parcels
            parcel1_id, parcel2_id = random.sample(parcel_ids, 2)
            
            # Get random UV from each
            data1 = self.loader.uv_maps[split][parcel1_id]
            data2 = self.loader.uv_maps[split][parcel2_id]
            
            if 'predicted' in data1 and 'predicted' in data2:
                uv1 = random.choice(data1['predicted'])
                uv2 = random.choice(data2['predicted'])
                
                pairs.append({
                    'image1': uv1['image'],
                    'image2': uv2['image'],
                    'label': 0,
                    'parcel_id': f"{parcel1_id}_vs_{parcel2_id}",
                    'type': 'different_parcels',
                    'metadata': {
                        'parcel1': parcel1_id,
                        'parcel2': parcel2_id,
                        'file1': uv1['filename'],
                        'file2': uv2['filename']
                    }
                })
        
        return pairs
    
    def visualize_pairs(self, num_examples=20, output_dir=None):
        if output_dir is None:
            output_dir = self.loader.data_root
        
        print(f"Visualizing {num_examples} Example Pairs")
        
        # Visualize positive pairs
        if self.positive_pairs:
            fig, axes = plt.subplots(num_examples, 2, figsize=(10, 3*num_examples))
            if num_examples == 1:
                axes = axes.reshape(1, -1)
            
            examples = random.sample(self.positive_pairs, min(num_examples, len(self.positive_pairs)))
            
            for idx, pair in enumerate(examples):
                # Image 1
                axes[idx, 0].imshow(pair['image1'])
                
                # Build title with details
                title1 = f"Positive Pair {idx+1} - Image 1\n"
                title1 += f"Type: {pair['type']}\n"
                
                if pair['type'] == 'original_vs_augmented':
                    title1 += f"Original\n"
                    if 'metadata' in pair and 'orig_file' in pair['metadata']:
                        title1 += f"{pair['metadata']['orig_file'][:30]}..."
                elif 'metadata' in pair and 'file1' in pair['metadata']:
                    title1 += f"{pair['metadata']['file1'][:30]}..."
                
                axes[idx, 0].set_title(title1, fontsize=8)
                axes[idx, 0].axis('off')
                
                # Image 2
                axes[idx, 1].imshow(pair['image2'])
                
                title2 = f"Image 2\n"
                title2 += f"Parcel: {pair['parcel_id']}\n"
                
                if pair['type'] == 'original_vs_augmented':
                    title2 += f"AUGMENTED\n"
                    if 'metadata' in pair and 'variant' in pair['metadata']:
                        title2 += f"Variant #{pair['metadata']['variant']}\n"
                    # Show augmentation applied
                    title2 += f"(rotation, brightness,\ncontrast, noise)"
                elif pair['type'] == 'same_parcel_different_capture':
                    title2 += "Different capture"
                    if 'metadata' in pair and 'file2' in pair['metadata']:
                        title2 += f"\n{pair['metadata']['file2'][:30]}..."
                elif pair['type'] == 'reference_vs_predicted':
                    title2 += "Predicted UV"
                    if 'metadata' in pair and 'pred_file' in pair['metadata']:
                        title2 += f"\n{pair['metadata']['pred_file'][:30]}..."
                
                axes[idx, 1].set_title(title2, fontsize=8)
                axes[idx, 1].axis('off')
            
            plt.tight_layout()
            pos_path = Path(output_dir) / 'positive_pairs_examples.png'
            plt.savefig(pos_path, dpi=200, bbox_inches='tight')
            print(f" Positive pairs visualization: {pos_path}")
            plt.close()
        
        # Visualize negative pairs
        if self.negative_pairs:
            fig, axes = plt.subplots(num_examples, 2, figsize=(10, 3*num_examples))
            if num_examples == 1:
                axes = axes.reshape(1, -1)
            
            examples = random.sample(self.negative_pairs, min(num_examples, len(self.negative_pairs)))
            
            for idx, pair in enumerate(examples):
                # Image 1
                axes[idx, 0].imshow(pair['image1'])
                
                title1 = f"Negative Pair {idx+1} - Image 1\n"
                title1 += f"Parcel: {pair['metadata']['parcel1']}\n"
                if 'file1' in pair['metadata']:
                    title1 += f"{pair['metadata']['file1'][:30]}..."
                
                axes[idx, 0].set_title(title1, fontsize=8)
                axes[idx, 0].axis('off')
                
                # Image 2
                axes[idx, 1].imshow(pair['image2'])
                
                title2 = f"Image 2\n"
                title2 += f"Parcel: {pair['metadata']['parcel2']}\n"
                title2 += "DIFFERENT PARCEL ❌\n"
                if 'file2' in pair['metadata']:
                    title2 += f"{pair['metadata']['file2'][:30]}..."
                
                axes[idx, 1].set_title(title2, fontsize=8)
                axes[idx, 1].axis('off')
            
            plt.tight_layout()
            neg_path = Path(output_dir) / 'negative_pairs_examples.png'
            plt.savefig(neg_path, dpi=200, bbox_inches='tight')
            print(f" Negative pairs visualization: {neg_path}")
            plt.close()
    
    def save_pairs(self, output_dir, train_split=0.8):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Saving Pairs Dataset")
        print(f"Output directory: {output_dir}")
        
        # Combine all pairs
        all_pairs = self.positive_pairs + self.negative_pairs
        random.shuffle(all_pairs)
        
        # Split train/val
        split_idx = int(len(all_pairs) * train_split)
        train_pairs = all_pairs[:split_idx]
        val_pairs = all_pairs[split_idx:]
        
        print(f"\nTotal pairs: {len(all_pairs)}")
        print(f"  Positive: {len(self.positive_pairs)}")
        print(f"  Negative: {len(self.negative_pairs)}")
        print(f"\nSplit:")
        print(f"  Train: {len(train_pairs)} ({train_split*100:.0f}%)")
        print(f"  Val: {len(val_pairs)} ({(1-train_split)*100:.0f}%)")
        
        # Save as pickle
        train_path = output_dir / 'train_pairs.pkl'
        val_path = output_dir / 'val_pairs.pkl'
        
        with open(train_path, 'wb') as f:
            pickle.dump(train_pairs, f)
        print(f"\n Saved train pairs: {train_path}")
        
        with open(val_path, 'wb') as f:
            pickle.dump(val_pairs, f)
        print(f" Saved val pairs: {val_path}")
        
        # Save summary
        summary = {
            'total_pairs': len(all_pairs),
            'positive_pairs': len(self.positive_pairs),
            'negative_pairs': len(self.negative_pairs),
            'train_pairs': len(train_pairs),
            'val_pairs': len(val_pairs),
            'train_split': train_split
        }
        
        summary_path = output_dir / 'dataset_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f" Saved summary: {summary_path}")
        
        return train_path, val_path


def main():
    parser = argparse.ArgumentParser(
        description="Load TAMPAR UV maps and create contrastive pairs (v2)"
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='/content/tampar/data/tampar_sample',
        help='Path to TAMPAR dataset'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='validation',
        choices=['test', 'validation'],
        help='Which split to load'
    )
    parser.add_argument(
        '--show_all',
        action='store_true',
        help='Visualize ALL parcels instead of just 3'
    )
    parser.add_argument(
        '--create_pairs',
        action='store_true',
        help='Create contrastive pairs'
    )
    parser.add_argument(
        '--num_positive',
        type=int,
        default=50,
        help='Number of positive pairs to create'
    )
    parser.add_argument(
        '--num_negative',
        type=int,
        default=50,
        help='Number of negative pairs to create'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for pairs (default: data_root/contrastive_pairs)'
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
        num_to_show = None if args.show_all else 3
        vis_path = loader.visualize_samples(num_samples=num_to_show)
        
        # Display in Colab if running there
        try:
            from IPython.display import Image as IPImage, display
            print(f"\nDisplaying UV maps visualization:")
            display(IPImage(vis_path))
        except ImportError:
            print(f"\n(Run in Colab to auto-display the image)")
    
    # Create pairs if requested
    if args.create_pairs:
        print("CREATING CONTRASTIVE PAIRS")
        
        # Initialize pair creator
        augmentor = AugmentationPipeline(
            rotation_range=5,
            brightness_range=0.1,
            contrast_range=0.1,
            noise_std=0.02
        )
        
        creator = ContrastivePairCreator(loader, augmentor)
        
        # Create positive pairs
        positive_pairs = creator.create_positive_pairs(
            strategies=['same_parcel', 'ref_vs_pred', 'augmented'],
            num_pairs=args.num_positive
        )
        
        # Create negative pairs
        negative_pairs = creator.create_negative_pairs(
            strategies=['different_parcels'],
            num_pairs=args.num_negative
        )
        
        # Visualize examples
        creator.visualize_pairs(num_examples=20, output_dir=args.data_root)
        
        # Display visualizations in Colab
        try:
            from IPython.display import Image as IPImage, display
            
            pos_viz = Path(args.data_root) / 'positive_pairs_examples.png'
            neg_viz = Path(args.data_root) / 'negative_pairs_examples.png'
            
            if pos_viz.exists():
                print(f"\nDisplaying positive pairs:")
                display(IPImage(str(pos_viz)))
            
            if neg_viz.exists():
                print(f"\nDisplaying negative pairs:")
                display(IPImage(str(neg_viz)))
        except ImportError:
            pass
        
        # Save pairs
        if args.output_dir is None:
            args.output_dir = Path(args.data_root) / 'contrastive_pairs'
        
        train_path, val_path = creator.save_pairs(
            output_dir=args.output_dir,
            train_split=0.8
        )
        
        print(f"  Train: {train_path}")
        print(f"  Val: {val_path}")
    
    print(f"  Reference UV maps: {len(loader.uv_maps.get('reference', {}))}")
    if args.split in loader.uv_maps:
        print(f"  {args.split.capitalize()} parcels: {len(loader.uv_maps[args.split])}")
    
    if args.create_pairs:
        print(f"\nPairs created:")
        print(f"  Positive: {len(creator.positive_pairs)}")
        print(f"  Negative: {len(creator.negative_pairs)}")
        print(f"\nNext step: Train SimSaC with contrastive learning using these pairs")
    else:
        print(f"\nNext step: Run with --create_pairs to generate training pairs")


if __name__ == "__main__":
    main()
