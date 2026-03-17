
import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())

from extension.simsac_multitask_model import create_multitask_model


class EvaluationDataset(Dataset):

    def __init__(self, pairs_csv, transform=None, img_size=512):
        self.df = pd.read_csv(pairs_csv)
        self.img_size = img_size

        # Resize transform for 256x256 version
        self.resize_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        # Full resolution transform
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load anchor and positive/negative
        img1_path = row.get('anchor', row.get('img1'))
        img2_path = row.get('negative', row.get('positive', row.get('img2')))

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        # Apply transforms
        img1_full = self.transform(img1)
        img1_256 = self.resize_transform(img1)

        img2_full = self.transform(img2)
        img2_256 = self.resize_transform(img2)

        # Labels
        label1 = torch.tensor(row.get('anchor_label', 0), dtype=torch.long)
        label2 = torch.tensor(row.get('negative_label', row.get('positive_label', 0)), dtype=torch.long)

        return {
            'img1': img1_full,
            'img1_256': img1_256,
            'img2': img2_full,
            'img2_256': img2_256,
            'label1': label1,
            'label2': label2,
            'pair_type': row.get('surface_type', 'unknown')
        }


class MultiTaskEvaluator:

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, dataloader):
        all_embeddings1 = []
        all_embeddings2 = []
        all_logits1 = []
        all_logits2 = []
        all_labels1 = []
        all_labels2 = []
        all_pair_types = []

        print("\nExtracting features and predictions")
        for batch in tqdm(dataloader):
            img1 = batch['img1'].to(self.device)
            img1_256 = batch['img1_256'].to(self.device)
            img2 = batch['img2'].to(self.device)
            img2_256 = batch['img2_256'].to(self.device)

            label1 = batch['label1'].cpu().numpy()
            label2 = batch['label2'].cpu().numpy()

            # Forward pass
            embeddings, logits = self.model(img1, img2, img1_256, img2_256)

            all_embeddings2.append(embeddings.cpu().numpy())
            all_logits2.append(logits.cpu().numpy())
            all_labels1.append(label1)
            all_labels2.append(label2)
            all_pair_types.extend(batch['pair_type'])


        # Concatenate all batches
        all_embeddings2 = np.concatenate(all_embeddings2, axis=0)
        all_logits2 = np.concatenate(all_logits2, axis=0)
        all_labels1 = np.concatenate(all_labels1, axis=0)
        all_labels2 = np.concatenate(all_labels2, axis=0)

        # Compute metrics
        results = {}

        # 1. Classification metrics
        print("\nComputing classification metrics")
        pred_labels2 = np.argmax(all_logits2, axis=1)

        accuracy = accuracy_score(all_labels2, pred_labels2)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels2, pred_labels2, average='weighted', zero_division=0
        )

        results['classification'] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }

        # Per-class metrics
        class_names = ['Clean', 'Tape', 'Writing', 'Fold', 'Other']
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(all_labels2, pred_labels2, average=None, zero_division=0)

        results['classification_per_class'] = {}
        for i, name in enumerate(class_names):
            if i < len(precision_per_class):
                results['classification_per_class'][name] = {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1': float(f1_per_class[i]),
                    'support': int(support_per_class[i])
                }

        # Confusion matrix
        cm = confusion_matrix(all_labels2, pred_labels2)
        results['confusion_matrix'] = cm.tolist()


        print("\nComputing contrastive metrics")

        # Analyze by pair type (clean-clean, clean-tampered, tampered-tampered)
        pair_type_metrics = defaultdict(list)

        for i in range(len(all_labels1)):
            label1 = all_labels1[i]
            label2 = all_labels2[i]

            if label1 == 0 and label2 == 0:
                pair_type = 'clean_clean'
            elif label1 == 0 and label2 > 0:
                pair_type = 'clean_tampered'
            elif label1 > 0 and label2 == 0:
                pair_type = 'tampered_clean'
            elif label1 > 0 and label2 > 0:
                if label1 == label2:
                    pair_type = 'tampered_same'
                else:
                    pair_type = 'tampered_different'
            else:
                pair_type = 'unknown'

            # Classification confidence
            confidence = np.max(np.exp(all_logits2[i]) / np.sum(np.exp(all_logits2[i])))
            pair_type_metrics[pair_type].append({
                'confidence': confidence,
                'correct': (pred_labels2[i] == all_labels2[i])
            })

        # Aggregate pair type metrics
        results['pair_type_analysis'] = {}
        for pair_type, metrics in pair_type_metrics.items():
            avg_confidence = np.mean([m['confidence'] for m in metrics])
            accuracy = np.mean([m['correct'] for m in metrics])
            results['pair_type_analysis'][pair_type] = {
                'count': len(metrics),
                'avg_confidence': float(avg_confidence),
                'accuracy': float(accuracy)
            }

        return results

    def print_results(self, results):
        print("Evaluation Results")

        # Classification metrics
        print("\n1. Classification Metrics (Overall)")
        cls_metrics = results['classification']
        print(f"  Accuracy:  {cls_metrics['accuracy']:.4f}")
        print(f"  Precision: {cls_metrics['precision']:.4f}")
        print(f"  Recall:    {cls_metrics['recall']:.4f}")
        print(f"  F1 Score:  {cls_metrics['f1']:.4f}")

        # Per-class metrics
        print("\n2. Classification Metrics (Per Class)")
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        for class_name, metrics in results['classification_per_class'].items():
            print(f"{class_name:<15} "
                  f"{metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} "
                  f"{metrics['f1']:<12.4f} "
                  f"{metrics['support']:<10}")

        # Pair type analysis
        print("\n3. Pair Type Analysis")
        print(f"{'Pair Type':<25} {'Count':<10} {'Avg Confidence':<18} {'Accuracy':<12}")
        for pair_type, metrics in results['pair_type_analysis'].items():
            print(f"{pair_type:<25} "
                  f"{metrics['count']:<10} "
                  f"{metrics['avg_confidence']:<18.4f} "
                  f"{metrics['accuracy']:<12.4f}")

        # Confusion matrix
        print("\n4. Confusion Matrix")
        cm = np.array(results['confusion_matrix'])
        class_names = ['Clean', 'Tape', 'Writing', 'Fold', 'Other']

        # Print header
        print(f"{'True/Pred':<15}", end='')
        for name in class_names[:cm.shape[1]]:
            print(f"{name:<12}", end='')
        print()

        # Print rows
        for i, name in enumerate(class_names[:cm.shape[0]]):
            print(f"{name:<15}", end='')
            for j in range(cm.shape[1]):
                print(f"{cm[i,j]:<12}", end='')
            print()



def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-task SimSaC model")

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_pairs', type=str, required=True,
                       help='Path to test pairs CSV')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')

    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--img_size', type=int, default=512,
                       help='Image size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    parser.add_argument('--projection_dim', type=int, default=512,
                       help='Projection dimension')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of tampering classes')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\nLoading multi-task model")
    model = create_multitask_model(
        simsac_checkpoint=None,
        freeze_simsac=False,
        projection_dim=args.projection_dim,
        num_tampering_classes=args.num_classes,
        device=device
    )

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Load full model state dict (includes all heads)
    if 'full_model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['full_model_state_dict'], strict=False)
        print("   Loaded full multi-task model weights")
    else:
        # Fall back to base SimSaC weights only
        simsac_state_dict = checkpoint['state_dict']
        wrapped_state_dict = {}
        for key, value in simsac_state_dict.items():
            wrapped_state_dict[f'simsac.{key}'] = value
        model.load_state_dict(wrapped_state_dict, strict=False)
        print("   Loaded base SimSaC weights only (heads randomly initialized)")

    print(f"  Model trained for {checkpoint.get('epoch', '?')} epochs")
    print(f"  Best loss: {checkpoint.get('best_loss', '?')}")

    # Create dataset
    print("\nLoading test data")
    test_dataset = EvaluationDataset(
        args.test_pairs,
        img_size=args.img_size
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"  Test samples: {len(test_dataset)}")

    # Evaluate
    evaluator = MultiTaskEvaluator(model, device=device)
    results = evaluator.evaluate(test_loader)

    # Print results
    evaluator.print_results(results)

    # Save results
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n Saved results to {results_path}")

    # Save classification report
    print("\nGenerating detailed classification report")
    # We'd need to re-run to get all predictions, for now just save what we have

    print("Evaluation Complete!")


if __name__ == "__main__":
    main()
