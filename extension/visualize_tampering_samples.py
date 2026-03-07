"""
Tampering Detection Visualisation — XGBoost on SimSAC features

Loads a simscores CSV (output of compute_similarity_scores.py), trains
XGBoost on SimSAC compare-type features, then generates a figure showing
N correctly detected tampered surfaces and M correctly detected clean surfaces.

Usage:
    python extension/visualize_tampering_samples.py \\
        --csv /path/to/simscores_final.csv \\
        --output tampering_visualisation.png \\
        --n_tampered 2 --n_clean 1

The figure shows one panel per sample:
  - Surface identifier and ground-truth label
  - Horizontal bars for each SimSAC similarity metric
  - XGBoost tampering probability gauge
  - CORRECTLY DETECTED badge
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

METRICS      = ['msssim', 'cwssim', 'ssim', 'hog', 'mae']
METRIC_LABELS = {
    'msssim': 'MS-SSIM',
    'cwssim': 'CW-SSIM',
    'ssim':   'SSIM',
    'hog':    'HOG',
    'mae':    'MAE',
}

# Metric direction: True = lower value means MORE tampered (similarity metrics)
#                   False = higher value means MORE tampered (error metrics)
METRIC_TAMPERED_IS_LOW = {
    'msssim': True,
    'cwssim': True,
    'ssim':   True,
    'hog':    True,
    'mae':    False,   # MAE: higher error → more tampered
}

COLORS = {
    'tampered_bg':    '#FFF0F0',
    'clean_bg':       '#F0FFF4',
    'tampered_bar':   '#E74C3C',
    'clean_bar':      '#27AE60',
    'neutral_bar':    '#5B9BD5',
    'badge_tampered': '#C0392B',
    'badge_clean':    '#1E8449',
    'prob_high':      '#E74C3C',
    'prob_low':       '#27AE60',
    'border_correct': '#2ECC71',
}


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------

def load_simsac_features(csv_path):
    """
    Load simscores CSV and extract SimSAC compare-type rows.

    Handles both long format (one row per compare_type per surface)
    and already-filtered files.

    Returns DataFrame with columns:
        surface_id, tampered, msssim, cwssim, ssim, hog, mae
        + original metadata columns
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}")

    # Filter to simsac compare type if the column exists
    if 'compare_type' in df.columns:
        df = df[df['compare_type'] == 'simsac'].copy()
        print(f"After filtering to compare_type=simsac: {len(df)} rows")
    else:
        print("No compare_type column — assuming file is already simsac only")

    # Normalise tampered column to int (True/False/1/0/'True'/'False')
    df['tampered'] = df['tampered'].astype(str).str.lower().map(
        {'true': 1, 'false': 0, '1': 1, '0': 0}
    ).fillna(df['tampered'].astype(int))
    df['tampered'] = df['tampered'].astype(int)

    # Build a human-readable surface identifier
    id_parts = []
    for col in ['parcel_id', 'view', 'sideface_name']:
        if col in df.columns:
            id_parts.append(df[col].astype(str))
    if id_parts:
        df['surface_id'] = '_'.join([p.name for p in id_parts])
        # Pandas way:
        df['surface_id'] = id_parts[0]
        for part in id_parts[1:]:
            df['surface_id'] = df['surface_id'] + ' / ' + part
    else:
        df['surface_id'] = df.index.astype(str)

    # Verify metric columns are present
    missing = [m for m in METRICS if m not in df.columns]
    if missing:
        raise ValueError(f"Missing metric columns in CSV: {missing}\n"
                         f"Available columns: {list(df.columns)}")

    # Drop rows with NaN metrics
    before = len(df)
    df = df.dropna(subset=METRICS)
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with NaN metrics")

    print(f"\nDataset summary:")
    print(f"  Total surfaces:  {len(df)}")
    print(f"  Tampered:        {df['tampered'].sum()}")
    print(f"  Clean:           {(df['tampered'] == 0).sum()}")

    return df


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------

def train_xgboost(df):
    """
    Train XGBoost on SimSAC features to predict tampering.

    Trains on ALL available data — the goal here is visualisation of
    correctly detected samples, not held-out evaluation.
    """
    X = df[METRICS].values
    y = df['tampered'].values

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
    )
    clf.fit(X, y)

    probs = clf.predict_proba(X)[:, 1]   # probability of tampered
    preds = (probs >= 0.5).astype(int)
    acc   = (preds == y).mean()
    print(f"\nXGBoost (simsac features) — training accuracy: {acc*100:.1f}%")

    importances = dict(zip(METRICS, clf.feature_importances_))
    print("Feature importances:")
    for m, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {METRIC_LABELS[m]:10s}: {imp:.3f}")

    return clf, probs, preds


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def select_samples(df, probs, preds, n_tampered=2, n_clean=1):
    """
    Select the most confidently correct tampered and clean samples.

    'Most confident' = furthest from 0.5 probability threshold.
    """
    df = df.copy()
    df['prob']    = probs
    df['pred']    = preds
    df['correct'] = (preds == df['tampered'].values)

    # Confidence = distance from 0.5
    df['confidence'] = (df['prob'] - 0.5).abs()

    # Correctly predicted tampered (high prob, label=1)
    pool_tampered = df[(df['tampered'] == 1) & df['correct']].sort_values(
        'confidence', ascending=False
    )
    # Correctly predicted clean (low prob, label=0)
    pool_clean = df[(df['tampered'] == 0) & df['correct']].sort_values(
        'confidence', ascending=False
    )

    if len(pool_tampered) < n_tampered:
        print(f"Warning: only {len(pool_tampered)} correctly detected tampered surfaces "
              f"(requested {n_tampered})")
    if len(pool_clean) < n_clean:
        print(f"Warning: only {len(pool_clean)} correctly detected clean surfaces "
              f"(requested {n_clean})")

    selected_t = pool_tampered.head(n_tampered)
    selected_c = pool_clean.head(n_clean)

    # Tampered first, then clean
    selected = pd.concat([selected_t, selected_c]).reset_index(drop=True)

    print(f"\nSelected {len(selected_t)} tampered + {len(selected_c)} clean samples:")
    for _, row in selected.iterrows():
        label  = 'TAMPERED' if row['tampered'] else 'CLEAN'
        pred_l = 'TAMPERED' if row['pred']     else 'CLEAN'
        print(f"  {row['surface_id'][:50]:50s}  "
              f"GT={label:8s}  Pred={pred_l:8s}  Prob={row['prob']:.3f}")

    return selected


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_metric_bar(ax, row, importances):
    """
    Draw horizontal metric bars for one surface sample.

    Bar color encodes agreement with expected direction:
      - Similarity metrics (msssim, cwssim, ssim, hog): red if low (tampered signal)
      - Error metric (mae): red if high (tampered signal)

    Feature importance shown as bar opacity.
    """
    is_tampered = bool(row['tampered'])
    y_pos = np.arange(len(METRICS))

    for i, metric in enumerate(METRICS):
        val = float(row[metric])
        imp = importances.get(metric, 0.0)

        # Clamp to [0, 1] for display (MAE can be > 1 in raw pixel units)
        val_display = min(max(val, 0.0), 1.0)

        # Is this metric value consistent with the ground truth label?
        if METRIC_TAMPERED_IS_LOW[metric]:
            # Low = tampered signal
            signal_match = (val_display < 0.5 and is_tampered) or (val_display >= 0.5 and not is_tampered)
        else:
            # High = tampered signal (mae)
            signal_match = (val_display >= 0.5 and is_tampered) or (val_display < 0.5 and not is_tampered)

        bar_color = COLORS['tampered_bar'] if signal_match else COLORS['neutral_bar']
        alpha = 0.4 + 0.6 * imp   # brighter = more important feature

        ax.barh(i, val_display, color=bar_color, alpha=alpha, height=0.6,
                edgecolor='white', linewidth=0.5)

        # Value label — use clamped position to stay within axes bounds
        text_x = min(val_display + 0.02, 1.10)
        ax.text(text_x, i, f'{val:.3f}',
                va='center', ha='left', fontsize=9, color='#333333',
                fontweight='bold' if imp > 0.2 else 'normal')

        # Importance marker (small triangle at right)
        if imp > 0.15:
            ax.text(0.98, i, f'★', transform=ax.get_yaxis_transform(),
                    va='center', ha='right', fontsize=8,
                    color='#E67E22', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Metric Value', fontsize=9)
    ax.set_title('SimSAC Similarity Features', fontsize=10, color='#444444', pad=6)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(0.5, -0.7, '0.5', ha='center', va='top', fontsize=8, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.2)


def plot_probability_gauge(ax, prob, pred, ground_truth):
    """
    Draw a horizontal probability gauge (0 → Clean, 1 → Tampered).
    """
    # Background gradient bar (green → red)
    cmap = LinearSegmentedColormap.from_list(
        'gauge', [COLORS['prob_low'], '#F39C12', COLORS['prob_high']]
    )
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(grad, aspect='auto', cmap=cmap,
              extent=[0, 1, 0, 1], alpha=0.25)

    # Probability fill
    fill_color = COLORS['prob_high'] if prob >= 0.5 else COLORS['prob_low']
    ax.barh(0.5, prob, height=0.5, left=0,
            color=fill_color, alpha=0.75, zorder=3)

    # Threshold line
    ax.axvline(0.5, color='gray', linewidth=1.5, linestyle='--', zorder=4)

    # Probability marker
    ax.plot(prob, 0.5, 'v', color='black', markersize=10, zorder=5)
    ax.text(prob, 0.95, f'{prob*100:.1f}%',
            ha='center', va='top', fontsize=11, fontweight='bold',
            color='black', zorder=6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%\n(threshold)', '75%', '100%'], fontsize=8)
    ax.set_yticks([])
    ax.text(0.02, 0.1, 'CLEAN', fontsize=9, color=COLORS['prob_low'],
            fontweight='bold', va='bottom')
    ax.text(0.98, 0.1, 'TAMPERED', fontsize=9, color=COLORS['prob_high'],
            fontweight='bold', va='bottom', ha='right')
    ax.set_title('XGBoost Tampering Probability', fontsize=10,
                 color='#444444', pad=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)


def _load_image(path):
    """Load a single image from path as RGB numpy array, or return None."""
    import cv2 as _cv2
    try:
        img = _cv2.imread(str(path))
        if img is not None:
            return _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return None


def load_surface_images(row, uvmap_dir=None, clean_dir=None, adv_dir=None):
    """
    Load up to three UV map images for a sample row.

    ref_img  : GT reference  — uvmap_dir / id_{parcel_id:02d}_uvmap.png
    clean_img: clean test    — clean_dir / row['view']
    adv_img  : adversarial   — adv_dir   / row['view']

    Each argument is optional; returns None for any that is not provided.
    """
    ref_img = clean_img = adv_img = None

    if uvmap_dir is not None:
        try:
            parcel_id = int(row['parcel_id'])
            ref_img = _load_image(
                Path(uvmap_dir) / f'id_{parcel_id:02d}_uvmap.png'
            )
        except Exception:
            pass

    if clean_dir is not None:
        clean_img = _load_image(Path(clean_dir) / str(row['view']))

    if adv_dir is not None:
        adv_img = _load_image(Path(adv_dir) / str(row['view']))

    return ref_img, clean_img, adv_img


def _show_uv_image(ax, img, title, title_color, border_color, bg_color):
    """Display a UV map image (or a placeholder) in an axes."""
    ax.set_facecolor(bg_color)
    if img is not None:
        ax.imshow(img, aspect='auto')
    else:
        ax.text(0.5, 0.5, 'Image\nnot found', ha='center', va='center',
                fontsize=9, color='#888888', transform=ax.transAxes,
                style='italic')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, fontweight='bold',
                 color=title_color, pad=4)
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(2.0)


def build_figure(selected, importances, output_path,
                 uvmap_dir=None, clean_dir=None, adv_dir=None):
    """
    Build a figure with one column per sample.

    Layout per column (left → right):
      Header   : surface ID · ground-truth label · CORRECTLY DETECTED badge
      Images   : Reference (GT)  →  Clean  →  Adversarial  (each optional)
      Metrics  : SimSAC similarity bars
      Gauge    : XGBoost tampering probability (spans full width)

    Each image directory is optional; missing ones show a placeholder.
    """
    n = len(selected)
    n_img_cols = sum([uvmap_dir is not None,
                      clean_dir  is not None,
                      adv_dir    is not None])
    has_images = n_img_cols > 0

    # col_w grows with the number of image columns shown
    col_w = 7.0 + 3.5 * n_img_cols
    fig = plt.figure(figsize=(col_w * n, 10))
    fig.patch.set_facecolor('#F8F9FA')

    # Title — y=1.00 keeps it above the GridSpec top=0.91, no overlap
    fig.suptitle(
        'Tampering Detection — XGBoost on SimSAC Features\n'
        'Correctly Detected Samples (Adversarial Test Set)',
        fontsize=14, fontweight='bold', color='#1A1A2E', y=1.00,
    )

    # Outer grid: n sample columns
    outer = fig.add_gridspec(
        1, n,
        left=0.03, right=0.97,
        top=0.91, bottom=0.07,
        wspace=0.22,
    )

    for col_idx, (_, row) in enumerate(selected.iterrows()):
        is_tampered = bool(row['tampered'])
        prob        = float(row['prob'])
        surface_id  = str(row['surface_id'])

        bg_color    = COLORS['tampered_bg']    if is_tampered else COLORS['clean_bg']
        badge_color = COLORS['badge_tampered'] if is_tampered else COLORS['badge_clean']

        # Per-sample: 3 rows — header | content | gauge
        inner = outer[col_idx].subgridspec(
            3, 1,
            height_ratios=[1.0, 5.5, 1.2],
            hspace=0.18,
        )

        # ── Header ──────────────────────────────────────────────────────────
        ax_hdr = fig.add_subplot(inner[0])
        ax_hdr.set_facecolor(bg_color)
        ax_hdr.set_xticks([])
        ax_hdr.set_yticks([])
        for spine in ax_hdr.spines.values():
            spine.set_edgecolor(badge_color)
            spine.set_linewidth(2)

        short_id   = (surface_id[:55] + '…') if len(surface_id) > 55 else surface_id
        gt_label   = 'TAMPERED' if is_tampered else 'CLEAN'
        pred_label = 'TAMPERED' if row['pred'] else 'CLEAN'
        badge_text = '✓ TAMPERING\n  DETECTED' if is_tampered else '✓ NO TAMPERING\n  DETECTED'

        ax_hdr.text(0.5, 0.76, short_id,
                    ha='center', va='center', fontsize=9,
                    color='#333333', transform=ax_hdr.transAxes)
        ax_hdr.text(0.5, 0.26,
                    f'Ground Truth: {gt_label}   |   Predicted: {pred_label}',
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color=badge_color, transform=ax_hdr.transAxes)
        ax_hdr.text(0.98, 0.88, badge_text,
                    ha='right', va='top', fontsize=8, fontweight='bold',
                    color='white', transform=ax_hdr.transAxes,
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor=badge_color, alpha=0.9))

        # ── Content row ──────────────────────────────────────────────────────
        if has_images:
            ref_img, clean_img, adv_img = load_surface_images(
                row, uvmap_dir=uvmap_dir, clean_dir=clean_dir, adv_dir=adv_dir
            )

            # Build dynamic column spec:
            # each image slot = 2.2, each arrow = 0.25, metrics = 3.5
            img_specs  = []   # list of (image, title, title_color, border, bg)
            if uvmap_dir is not None:
                img_specs.append((ref_img, 'Reference (GT)',
                                  '#2471A3', '#2471A3', '#EBF5FB'))
            if clean_dir is not None:
                img_specs.append((clean_img, 'Clean',
                                  '#1E8449', '#1E8449', '#EAFAF1'))
            if adv_dir is not None:
                adv_title = f"Adversarial ({'TAMPERED' if is_tampered else 'CLEAN'})"
                img_specs.append((adv_img, adv_title,
                                  badge_color, badge_color, bg_color))

            n_imgs   = len(img_specs)
            n_arrows = n_imgs - 1
            # width_ratios: [img, arrow, img, arrow, ..., img, metrics]
            ratios = []
            for i in range(n_imgs):
                ratios.append(2.2)
                if i < n_arrows:
                    ratios.append(0.25)
            ratios.append(3.5)   # metrics column

            total_cols = len(ratios)
            content = inner[1].subgridspec(1, total_cols,
                                           width_ratios=ratios,
                                           wspace=0.08)

            # Place image and arrow axes
            col_cursor = 0
            for i, (img, title, t_col, b_col, b_bg) in enumerate(img_specs):
                ax_img = fig.add_subplot(content[0, col_cursor])
                _show_uv_image(ax_img, img, title=title,
                               title_color=t_col, border_color=b_col,
                               bg_color=b_bg)
                col_cursor += 1
                if i < n_arrows:
                    ax_arr = fig.add_subplot(content[0, col_cursor])
                    ax_arr.axis('off')
                    ax_arr.annotate(
                        '', xy=(0.90, 0.5), xytext=(0.10, 0.5),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='#555555',
                                        lw=2.0, mutation_scale=18),
                    )
                    col_cursor += 1

            ax_metrics = fig.add_subplot(content[0, col_cursor])
        else:
            ax_metrics = fig.add_subplot(inner[1])

        ax_metrics.set_facecolor(bg_color)
        plot_metric_bar(ax_metrics, row, importances)
        for spine in ax_metrics.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(0.8)

        # ── Gauge ────────────────────────────────────────────────────────────
        ax_gauge = fig.add_subplot(inner[2])
        ax_gauge.set_facecolor(bg_color)
        plot_probability_gauge(ax_gauge, prob, row['pred'], is_tampered)
        for spine in ax_gauge.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(0.8)

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLORS['tampered_bar'], alpha=0.75,
                       label='Metric consistent with TAMPERED label'),
        mpatches.Patch(color=COLORS['neutral_bar'],  alpha=0.75,
                       label='Metric consistent with CLEAN label'),
        mpatches.Patch(color='#E67E22', alpha=0.8,
                       label='★ High XGBoost feature importance (>15%)'),
    ]
    fig.legend(handles=legend_handles,
               loc='lower center', ncol=3,
               fontsize=8.5, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0))

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\n✓ Figure saved → {output_path}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Visualise correctly detected tampering/clean samples '
                    'using XGBoost on SimSAC similarity features.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--csv', required=True,
                        help='Path to simscores CSV from compute_similarity_scores.py')
    parser.add_argument('--output', default='tampering_visualisation.png',
                        help='Output figure path (default: tampering_visualisation.png)')
    parser.add_argument('--n_tampered', type=int, default=2,
                        help='Number of correctly detected tampered samples (default: 2)')
    parser.add_argument('--n_clean', type=int, default=1,
                        help='Number of correctly detected clean samples (default: 1)')
    parser.add_argument('--uvmap_dir', default=None,
                        help='Directory containing GT UV maps '
                             '(id_XX_uvmap.png files, e.g. tampar_sample/uvmaps/).')
    parser.add_argument('--clean_dir', default=None,
                        help='Root directory for clean UV maps '
                             '(row["view"] is appended to this path).')
    parser.add_argument('--adv_dir', default=None,
                        help='Root directory for adversarial UV maps '
                             '(row["view"] is appended to this path).')

    args = parser.parse_args()

    # 1. Load
    df = load_simsac_features(args.csv)

    # 2. Train XGBoost
    clf, probs, preds = train_xgboost(df)

    importances = dict(zip(METRICS, clf.feature_importances_))

    # 3. Select most confident correct samples
    selected = select_samples(df, probs, preds,
                              n_tampered=args.n_tampered,
                              n_clean=args.n_clean)

    # 4. Plot
    build_figure(selected, importances, output_path=args.output,
                 uvmap_dir=args.uvmap_dir,
                 clean_dir=args.clean_dir,
                 adv_dir=args.adv_dir)


if __name__ == '__main__':
    main()
