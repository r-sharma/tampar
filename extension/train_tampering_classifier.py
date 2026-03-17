
import sys
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

#  ML 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed. Install with: pip install xgboost")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'extension'))


# Feature engineering from CSV

COMPARE_TYPES = ['simsac', 'plain', 'canny', 'laplacian', 'meanchannel']
METRICS       = ['msssim', 'cwssim', 'ssim', 'hog', 'mae']


def pivot_csv_to_features(df: pd.DataFrame) -> pd.DataFrame:
    # Filter to known compare types
    df = df[df['compare_type'].isin(COMPARE_TYPES)].copy()

    # Pivot: index = (parcel_id, view, sideface_name), columns = compare_type × metric
    pivot = df.pivot_table(
        index=['parcel_id', 'view', 'sideface_name', 'tampered',
               'dataset_split', 'background'],
        columns='compare_type',
        values=METRICS,
        aggfunc='first'
    )

    # Flatten multi-level columns: (metric, compare_type)  compare_type_metric
    pivot.columns = [f"{ct}_{m}" for m, ct in pivot.columns]
    pivot = pivot.reset_index()

    return pivot


def load_and_pivot(csv_path: str, tag: str = '') -> pd.DataFrame:
    print(f"\nLoading {tag} CSV: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    print(f"  Raw rows: {len(df)}, tampered: {df['tampered'].sum()}, "
          f"clean: {(~df['tampered']).sum()}")

    pivoted = pivot_csv_to_features(df)
    pivoted['source'] = tag
    print(f"  Pivoted rows: {len(pivoted)}")
    return pivoted


# Projection head cosine similarity extraction

def extract_proj_head_features(df: pd.DataFrame, checkpoint: str,
                                weights_path: str, device: str) -> pd.Series:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image

    from simsac_contrastive_model import create_simsac_contrastive

    print(f"\nLoading fine-tuned model for projection head features")
    model = create_simsac_contrastive(
        weights_path=weights_path,
        projection_dim=128,
        freeze_backbone=False,
        device=device
    )
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(
        {f'simsac.{k}': v for k, v in ckpt['state_dict'].items()},
        strict=False
    )
    model.eval()
    print(f"   Loaded checkpoint (epoch {ckpt['epoch']})")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    IMAGE_ROOT = ROOT / "data" / "tampar_sample"
    UVMAP_DIR  = IMAGE_ROOT / "uvmaps"

    # Get unique (parcel_id, view) pairs
    unique_pairs = df[['parcel_id', 'view']].drop_duplicates()
    print(f"  Extracting projection head features for {len(unique_pairs)} unique pairs")

    results = {}
    missing = 0

    with torch.no_grad():
        for _, row in unique_pairs.iterrows():
            parcel_id = int(row['parcel_id'])
            view = row['view']

            # GT UV map path
            gt_uvmap_path = UVMAP_DIR / f"id_{str(parcel_id).zfill(2)}_uvmap.png"
            # Field UV map path (view is relative to IMAGE_ROOT)
            field_uvmap_path = IMAGE_ROOT / view

            if not gt_uvmap_path.exists() or not field_uvmap_path.exists():
                missing += 1
                results[(parcel_id, view)] = np.nan
                continue

            try:
                img_gt    = transform(Image.open(gt_uvmap_path).convert('RGB')).unsqueeze(0).to(device)
                img_field = transform(Image.open(field_uvmap_path).convert('RGB')).unsqueeze(0).to(device)

                z1, z2 = model(img_gt, img_field)
                sim = F.cosine_similarity(z1, z2, dim=1).item()
                results[(parcel_id, view)] = sim

            except Exception as e:
                results[(parcel_id, view)] = np.nan
                missing += 1

    if missing > 0:
        print(f"   {missing} pairs missing UV map files — filled with NaN")

    return pd.Series(results, name='proj_head_cosine_sim')


# Classifier training + evaluation

FEATURE_COLS = [f"{ct}_{m}" for ct in COMPARE_TYPES for m in METRICS]


def get_xy(df: pd.DataFrame) -> tuple:
    available = [c for c in FEATURE_COLS if c in df.columns]
    if 'proj_head_cosine_sim' in df.columns:
        available += ['proj_head_cosine_sim']

    X = df[available].values.astype(np.float32)
    y = df['tampered'].astype(int).values

    # Replace NaN with column median
    col_medians = np.nanmedian(X, axis=0)
    nan_mask    = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    return X, y, available


def evaluate_model(name, model, X_train, y_train, X_test, y_test,
                   test_label='test'):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = (model.predict_proba(X_test)[:, 1]
              if hasattr(model, 'predict_proba') else None)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    print(f"\n  {name} on {test_label}:")
    print(f"    Accuracy:  {acc*100:.2f}%")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1:        {f1:.4f}")
    if auc is not None:
        print(f"    AUC-ROC:   {auc:.4f}")

    return {
        'name': name, 'test_label': test_label,
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'auc': auc, 'y_pred': y_pred, 'y_prob': y_prob, 'y_true': y_test
    }


def plot_results(all_results, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Accuracy bar chart
    labels   = [f"{r['name']}\n({r['test_label']})" for r in all_results]
    accs     = [r['accuracy'] * 100 for r in all_results]
    colors   = ['steelblue' if 'clean' in r['test_label'] else 'tomato'
                for r in all_results]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    bars = ax.bar(labels, accs, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(71, color='gray', linestyle='--', label='predict_tampering (adv baseline 71%)')
    ax.axhline(84, color='green', linestyle='--', label='predict_tampering (clean baseline 84%)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Classifier vs predict_tampering Baseline')
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=150)
    plt.close()
    print(f"\n Saved: {output_dir / 'accuracy_comparison.png'}")

    # Confusion matrices
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, all_results):
        cm = confusion_matrix(r['y_true'], r['y_pred'])
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Clean', 'Tampered'])
        ax.set_yticklabels(['Clean', 'Tampered'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f"{r['name']} ({r['test_label']})\n{r['accuracy']*100:.1f}%")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=150)
    plt.close()
    print(f" Saved: {output_dir / 'confusion_matrices.png'}")

    # Feature importance (RF)
    rf_results = [r for r in all_results if 'RF' in r['name'] and hasattr(r.get('model'), 'feature_importances_')]
    if rf_results:
        pass


def print_feature_importance(model, feature_names, top_n=15, output_dir=None):
    if not hasattr(model, 'feature_importances_'):
        # Pipeline — get underlying estimator
        try:
            model = model.named_steps['clf']
        except Exception:
            return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    print(f"\n  Top {top_n} features:")
    for i, idx in enumerate(indices):
        print(f"    {i+1:2d}. {feature_names[idx]:40s} {importances[idx]:.4f}")

    if output_dir:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh([feature_names[i] for i in indices[::-1]],
                [importances[i] for i in indices[::-1]],
                color='steelblue', alpha=0.8)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances (Random Forest)')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=150)
        plt.close()
        print(f" Saved: {output_dir / 'feature_importance.png'}")


# Main

def main():
    parser = argparse.ArgumentParser(
        description='Train RF/XGBoost tampering classifier on similarity features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Data
    parser.add_argument('--clean_csv', type=str, required=True,
                        help='CSV from compute_similarity_scores.py on clean images')
    parser.add_argument('--adv_csv', type=str, default=None,
                        help='CSV from compute_similarity_scores.py on adversarial images')

    # Projection head features
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Fine-tuned SimSACContrastive checkpoint for projection head features')
    parser.add_argument('--weights_path', type=str,
                        default='/content/tampar/src/simsac/weight/synthetic.pth',
                        help='Base SimSAC weights (needed to build model architecture)')
    parser.add_argument('--no_proj_head', action='store_true',
                        help='Skip projection head features (use CSV features only)')
    parser.add_argument('--image_root', type=str, default=None,
                        help='Root dir containing UV map images. Auto-detected if not set.')

    # Output
    parser.add_argument('--output_dir', type=str,
                        default='/content/classifier_results',
                        help='Output directory for results and plots')

    # Classifier settings
    parser.add_argument('--n_estimators', type=int, default=200,
                        help='Number of trees for RF/XGBoost (default: 200)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])

    args = parser.parse_args()

    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    #  Load CSVs 
    df_clean = load_and_pivot(args.clean_csv, tag='clean')
    df_adv   = load_and_pivot(args.adv_csv,   tag='adv') if args.adv_csv else None

    #  Projection head features 
    use_proj = (args.checkpoint is not None and not args.no_proj_head)

    if use_proj:
        if args.image_root:
            # monkey-patch ROOT used inside extract_proj_head_features
            import extension.train_tampering_classifier as _self
            _self.ROOT = Path(args.image_root).parent

        print(f"\nExtracting projection head cosine similarities")
        proj_series_clean = extract_proj_head_features(
            df_clean, args.checkpoint, args.weights_path, args.device
        )
        df_clean['proj_head_cosine_sim'] = df_clean.set_index(
            ['parcel_id', 'view']).index.map(proj_series_clean)

        if df_adv is not None:
            proj_series_adv = extract_proj_head_features(
                df_adv, args.checkpoint, args.weights_path, args.device
            )
            df_adv['proj_head_cosine_sim'] = df_adv.set_index(
                ['parcel_id', 'view']).index.map(proj_series_adv)
    else:
        print("\nSkipping projection head features (--no_proj_head or no --checkpoint)")

    #  Build feature matrices 
    X_clean, y_clean, feat_names = get_xy(df_clean)
    print(f"\nFeature matrix (clean):  {X_clean.shape}  "
          f"({y_clean.sum()} tampered, {(y_clean==0).sum()} clean)")
    print(f"Features used ({len(feat_names)}): {feat_names}")

    if df_adv is not None:
        X_adv, y_adv, _ = get_xy(df_adv)
        print(f"Feature matrix (adv):    {X_adv.shape}  "
              f"({y_adv.sum()} tampered, {(y_adv==0).sum()} clean)")

    #  Classifiers 
    classifiers = {
        'RF': RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=None,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    }
    if HAS_XGB:
        scale_pos = (y_clean == 0).sum() / max(y_clean.sum(), 1)
        classifiers['XGBoost'] = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            verbosity=0
        )

    #  Cross-validation on clean data 
    print("5-Fold Cross-Validation on Clean Data")
    for name, clf in classifiers.items():
        cv_scores = cross_val_score(clf, X_clean, y_clean, cv=5, scoring='accuracy')
        print(f"  {name}: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    #  Train on clean, evaluate on clean + adversarial 

    all_results = []
    trained_models = {}

    for name, clf in classifiers.items():
        # Evaluate on clean (same distribution, overfitting check via CV above)
        r_clean = evaluate_model(
            name, clf, X_clean, y_clean, X_clean, y_clean,
            test_label='clean (train=test, upper bound)'
        )
        r_clean['model'] = clf
        all_results.append(r_clean)
        trained_models[name] = clf

        if df_adv is not None:
            r_adv = evaluate_model(
                name, clf, X_clean, y_clean, X_adv, y_adv,
                test_label='adversarial'
            )
            r_adv['model'] = clf
            all_results.append(r_adv)

    #  Train on CLEAN + ADV, evaluate on adv 
    if df_adv is not None:

        X_combined = np.vstack([X_clean, X_adv])
        y_combined  = np.concatenate([y_clean, y_adv])

        for name, clf_class in [
            ('RF (combined)', RandomForestClassifier(
                n_estimators=args.n_estimators, class_weight='balanced',
                min_samples_leaf=2, random_state=42, n_jobs=-1)),
        ] + ([('XGBoost (combined)', XGBClassifier(
                n_estimators=args.n_estimators, max_depth=6,
                learning_rate=0.05, use_label_encoder=False,
                eval_metric='logloss', random_state=42, verbosity=0))]
             if HAS_XGB else []):

            r = evaluate_model(
                name, clf_class,
                X_combined, y_combined,
                X_adv, y_adv,
                test_label='adversarial (trained on combined)'
            )
            r['model'] = clf_class
            all_results.append(r)

    #  Baseline reminder 

    #  Feature importance 
    print("Feature Importance (Random Forest, trained on clean)")
    rf_model = trained_models.get('RF')
    if rf_model is not None:
        print_feature_importance(rf_model, feat_names, top_n=15,
                                 output_dir=output_dir)

    #  Plots 
    plot_results(all_results, output_dir)

    #  Save results JSON 
    summary = []
    for r in all_results:
        summary.append({
            'classifier': r['name'],
            'test_on': r['test_label'],
            'accuracy': round(r['accuracy'] * 100, 2),
            'precision': round(r['precision'], 4),
            'recall': round(r['recall'], 4),
            'f1': round(r['f1'], 4),
            'auc': round(r['auc'], 4) if r['auc'] else None,
        })
    results_path = output_dir / 'classifier_results.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n Results saved: {results_path}")

    print("Done! Generated files:")
    print(f"  {output_dir}/accuracy_comparison.png")
    print(f"  {output_dir}/confusion_matrices.png")
    print(f"  {output_dir}/feature_importance.png")
    print(f"  {output_dir}/classifier_results.json")


if __name__ == '__main__':
    main()
