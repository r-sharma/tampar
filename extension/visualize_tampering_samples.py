"""
Tampering Detection Visualisation — XGBoost on SimSAC features

Loads a simscores CSV (output of compute_similarity_scores.py), trains
XGBoost on SimSAC compare-type features, then generates a figure showing
N correctly classified parcels with all 5 surfaces visualised.

Usage:
    python extension/visualize_tampering_samples.py \\
        --csv /path/to/simscores_final.csv \\
        --output tampering_visualisation.png \\
        --n_parcels 2 \\
        --uvmap_dir data/tampar_sample/uvmaps \\
        --adv_dir data/tampar_sample

Each parcel block shows:
  - Parcel header (parcel ID, view path, tampered/clean surface counts)
  - 5 surface rows (top, left, center, right, bottom), each containing:
      Reference patch  →  Adversarial patch  →  SimSAC metric bars  →  XGBoost probability gauge
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

METRICS       = ['msssim', 'cwssim', 'ssim', 'hog', 'mae']
METRIC_LABELS = {
    'msssim': 'MS-SSIM',
    'cwssim': 'CW-SSIM',
    'ssim':   'SSIM',
    'hog':    'HOG',
    'mae':    'MAE',
}

# True = lower value → more tampered (similarity metrics)
# False = higher value → more tampered (error metrics like MAE)
METRIC_TAMPERED_IS_LOW = {
    'msssim': True,
    'cwssim': True,
    'ssim':   True,
    'hog':    True,
    'mae':    False,
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
}

# UV map is a 3×3 grid; these are the flat indices for each named surface
SURFACE_ORDER = ['top', 'left', 'center', 'right', 'bottom']
PATCH_INDEX   = {'top': 1, 'left': 3, 'center': 4, 'right': 5, 'bottom': 7}


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------

def load_simsac_features(csv_path):
    """
    Load simscores CSV and extract SimSAC compare-type rows.

    Returns DataFrame filtered to compare_type=simsac with integer tampered column.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}")

    if 'compare_type' in df.columns:
        df = df[df['compare_type'] == 'simsac'].copy()
        print(f"After filtering to compare_type=simsac: {len(df)} rows")
    else:
        print("No compare_type column — assuming file is already simsac only")

    df['tampered'] = df['tampered'].astype(str).str.lower().map(
        {'true': 1, 'false': 0, '1': 1, '0': 0}
    ).fillna(df['tampered'].astype(int))
    df['tampered'] = df['tampered'].astype(int)

    missing = [m for m in METRICS if m not in df.columns]
    if missing:
        raise ValueError(f"Missing metric columns in CSV: {missing}\n"
                         f"Available columns: {list(df.columns)}")

    before = len(df)
    df = df.dropna(subset=METRICS)
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with NaN metrics")

    print(f"\nDataset summary:")
    print(f"  Total surfaces : {len(df)}")
    print(f"  Tampered       : {df['tampered'].sum()}")
    print(f"  Clean          : {(df['tampered'] == 0).sum()}")

    return df


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------

def train_xgboost(df):
    """Train XGBoost on SimSAC features. Returns (clf, probs, preds)."""
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

    probs = clf.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc   = (preds == y).mean()
    print(f"\nXGBoost (simsac features) — training accuracy: {acc*100:.1f}%")

    importances = dict(zip(METRICS, clf.feature_importances_))
    print("Feature importances:")
    for m, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {METRIC_LABELS[m]:10s}: {imp:.3f}")

    return clf, probs, preds


# ---------------------------------------------------------------------------
# Parcel selection
# ---------------------------------------------------------------------------

def select_parcels(df, probs, preds, n_parcels):
    """
    Select parcels where ALL 5 surfaces are correctly classified.

    Groups by (parcel_id, view) — each group = one UV map with all its surfaces.
    Returns list of (parcel_id, view, group_df) tuples, sorted by minimum
    surface confidence (most reliable parcels first), up to n_parcels entries.
    """
    df = df.copy()
    df['prob']       = probs
    df['pred']       = preds
    df['correct']    = (preds == df['tampered'].values).astype(bool)
    df['confidence'] = (df['prob'] - 0.5).abs()

    candidates = []
    for (parcel_id, view), group in df.groupby(['parcel_id', 'view']):
        # Require all 5 named surfaces to be present
        surfaces_present = set(group['sideface_name'].values)
        if not set(SURFACE_ORDER).issubset(surfaces_present):
            continue
        # All surfaces must be correctly classified
        if not group['correct'].all():
            continue
        min_conf = float(group['confidence'].min())
        candidates.append((parcel_id, view, min_conf, group))

    # Most confident first
    candidates.sort(key=lambda x: -x[2])

    result = [(pid, v, grp) for pid, v, _, grp in candidates[:n_parcels]]

    print(f"\nFound {len(candidates)} parcel-views with all surfaces correctly classified")
    print(f"Showing top {len(result)}:")
    for pid, view, grp in result:
        print(f"  Parcel {pid}  view={view}")
        for surf in SURFACE_ORDER:
            rows = grp[grp['sideface_name'] == surf]
            if len(rows):
                r = rows.iloc[0]
                label = 'TAMPERED' if r['tampered'] else 'CLEAN'
                print(f"    {surf:8s}  GT={label:8s}  Prob={r['prob']:.3f}")

    return result


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def get_surface_patch(image, sideface_name):
    """
    Extract a named surface patch from a UV map using simple 3×3 grid division.

    The UV map is divided into a 3×3 grid of equal patches.
    PATCH_INDEX maps surface names to flat grid positions (row-major):
        idx=0  idx=1(top)  idx=2
        idx=3(left)  idx=4(center)  idx=5(right)
        idx=6  idx=7(bottom)  idx=8
    """
    h, w = image.shape[:2]
    ph, pw = h // 3, w // 3
    idx    = PATCH_INDEX[sideface_name]
    row_i, col_i = divmod(idx, 3)
    return image[row_i * ph:(row_i + 1) * ph,
                 col_i * pw:(col_i + 1) * pw].copy()


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _load_image(path):
    """Load an image from path as RGB numpy array, or return None."""
    import cv2 as _cv2
    try:
        img = _cv2.imread(str(path))
        if img is not None:
            return _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return None


def _show_patch(ax, img, title, title_color, border_color, bg_color):
    """Display a patch image (or placeholder) in an axes."""
    ax.set_facecolor(bg_color)
    if img is not None:
        ax.imshow(img, aspect='auto')
    else:
        ax.text(0.5, 0.5, 'Image\nnot found',
                ha='center', va='center', fontsize=8, color='#888888',
                transform=ax.transAxes, style='italic')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8, fontweight='bold',
                 color=title_color, pad=3)
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(1.8)


def plot_metric_bar(ax, row, importances, show_title=True, show_xlabel=True):
    """
    Draw horizontal SimSAC similarity metric bars for one surface.

    Bar color encodes whether the metric is consistent with the ground-truth label.
    """
    is_tampered = bool(row['tampered'])
    y_pos = np.arange(len(METRICS))

    for i, metric in enumerate(METRICS):
        val = float(row[metric])
        imp = importances.get(metric, 0.0)

        # Clamp to [0, 1] for display (MAE can be >>1 in raw pixel units)
        val_display = min(max(val, 0.0), 1.0)

        if METRIC_TAMPERED_IS_LOW[metric]:
            signal_match = ((val_display < 0.5) == is_tampered)
        else:
            signal_match = ((val_display >= 0.5) == is_tampered)

        bar_color = COLORS['tampered_bar'] if signal_match else COLORS['neutral_bar']
        alpha = 0.4 + 0.6 * imp

        ax.barh(i, val_display, color=bar_color, alpha=alpha, height=0.6,
                edgecolor='white', linewidth=0.5)

        # Value label — keep within axes bounds
        text_x = min(val_display + 0.02, 1.10)
        ax.text(text_x, i, f'{val:.3f}',
                va='center', ha='left', fontsize=8, color='#333333',
                fontweight='bold' if imp > 0.2 else 'normal')

        if imp > 0.15:
            ax.text(0.98, i, '★', transform=ax.get_yaxis_transform(),
                    va='center', ha='right', fontsize=7,
                    color='#E67E22', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=8)
    ax.set_xlim(0, 1.15)
    if show_xlabel:
        ax.set_xlabel('Metric Value', fontsize=8)
    if show_title:
        ax.set_title('SimSAC Similarity Features', fontsize=9,
                     color='#444444', pad=4)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(0.5, -0.7, '0.5', ha='center', va='top', fontsize=7, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.2)


def plot_probability_gauge(ax, prob, pred, ground_truth,
                           show_title=True, show_labels=True):
    """Draw a horizontal XGBoost tampering probability gauge."""
    cmap = LinearSegmentedColormap.from_list(
        'gauge', [COLORS['prob_low'], '#F39C12', COLORS['prob_high']]
    )
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(grad, aspect='auto', cmap=cmap,
              extent=[0, 1, 0, 1], alpha=0.25)

    fill_color = COLORS['prob_high'] if prob >= 0.5 else COLORS['prob_low']
    ax.barh(0.5, prob, height=0.5, left=0,
            color=fill_color, alpha=0.75, zorder=3)

    ax.axvline(0.5, color='gray', linewidth=1.5, linestyle='--', zorder=4)
    ax.plot(prob, 0.5, 'v', color='black', markersize=9, zorder=5)
    ax.text(prob, 0.95, f'{prob*100:.1f}%',
            ha='center', va='top', fontsize=10, fontweight='bold',
            color='black', zorder=6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%\n(thresh)', '75%', '100%'], fontsize=7)
    ax.set_yticks([])

    if show_labels:
        ax.text(0.02, 0.1, 'CLEAN', fontsize=8, color=COLORS['prob_low'],
                fontweight='bold', va='bottom')
        ax.text(0.98, 0.1, 'TAMPERED', fontsize=8, color=COLORS['prob_high'],
                fontweight='bold', va='bottom', ha='right')
    if show_title:
        ax.set_title('XGBoost Probability', fontsize=9,
                     color='#444444', pad=4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def build_figure(parcels, importances, output_path,
                 uvmap_dir=None, adv_dir=None):
    """
    Build a multi-parcel figure, one vertical block per parcel.

    Each parcel block contains:
      - A header row (parcel ID, view, surface counts)
      - 5 surface rows (top, left, center, right, bottom), each with:
          [surface label | reference patch | → | adversarial patch | metric bars | probability gauge]

    Parameters
    ----------
    parcels : list of (parcel_id, view, group_df)
        Output of select_parcels().
    importances : dict  metric → importance float
    output_path : str | Path
    uvmap_dir : Path-like, optional
        Directory containing GT reference UV maps (id_XX_uvmap.png).
    adv_dir : Path-like, optional
        Root directory for adversarial UV maps (row['view'] is appended).
    """
    n_parcels     = len(parcels)
    n_surfaces    = len(SURFACE_ORDER)

    # Figure sizing
    HEADER_H      = 0.45   # inches for parcel header
    SURFACE_ROW_H = 1.5    # inches per surface row
    PARCEL_H      = HEADER_H + n_surfaces * SURFACE_ROW_H   # 7.95 inches
    TOP_MARGIN    = 0.65
    BOT_MARGIN    = 0.55
    fig_w         = 17.0
    fig_h         = n_parcels * PARCEL_H + TOP_MARGIN + BOT_MARGIN

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('#F8F9FA')

    fig.suptitle(
        'Tampering Detection — XGBoost on SimSAC Features\n'
        'Correctly Classified Parcels (Adversarial Test Set)',
        fontsize=14, fontweight='bold', color='#1A1A2E',
        y=1.0 - (0.1 / fig_h),
    )

    # Outer grid: one row per parcel
    outer_top    = 1.0 - TOP_MARGIN / fig_h
    outer_bottom = BOT_MARGIN / fig_h
    outer = fig.add_gridspec(
        n_parcels, 1,
        left=0.01, right=0.99,
        top=outer_top, bottom=outer_bottom,
        hspace=0.08,
    )

    # Column width ratios for surface rows
    # [name | ref_patch | arrow | adv_patch | metrics | gauge]
    col_ratios = [0.5, 2.2, 0.28, 2.2, 4.5, 2.3]

    for p_idx, (parcel_id, view, group) in enumerate(parcels):
        # Determine parcel-level colour scheme
        n_tam = int((group['tampered'] == 1).sum())
        n_cln = int((group['tampered'] == 0).sum())
        if n_tam > 0 and n_cln > 0:
            parcel_bg     = '#FFFDE7'
            parcel_border = '#B7950B'
        elif n_tam > 0:
            parcel_bg     = COLORS['tampered_bg']
            parcel_border = COLORS['badge_tampered']
        else:
            parcel_bg     = COLORS['clean_bg']
            parcel_border = COLORS['badge_clean']

        # Load UV map images once per parcel (they are the same for all surfaces)
        ref_uvmap = None
        adv_uvmap = None
        if uvmap_dir is not None:
            ref_uvmap = _load_image(
                Path(uvmap_dir) / f'id_{int(parcel_id):02d}_uvmap.png'
            )
        if adv_dir is not None:
            adv_uvmap = _load_image(Path(adv_dir) / str(view))

        # Inner grid: header row + n_surfaces surface rows
        inner = outer[p_idx].subgridspec(
            1 + n_surfaces, 1,
            height_ratios=[0.38] + [1.0] * n_surfaces,
            hspace=0.06,
        )

        # ── Parcel header ────────────────────────────────────────────────────
        ax_hdr = fig.add_subplot(inner[0])
        ax_hdr.set_facecolor(parcel_bg)
        ax_hdr.set_xticks([])
        ax_hdr.set_yticks([])
        for spine in ax_hdr.spines.values():
            spine.set_edgecolor(parcel_border)
            spine.set_linewidth(2.2)

        short_view = str(view)
        if len(short_view) > 70:
            short_view = '…' + short_view[-67:]

        ax_hdr.text(
            0.01, 0.5,
            f'Parcel  {parcel_id}   |   {short_view}',
            ha='left', va='center', fontsize=9, fontweight='bold',
            color='#333333', transform=ax_hdr.transAxes,
        )
        ax_hdr.text(
            0.99, 0.5,
            f'Tampered surfaces: {n_tam}   |   Clean surfaces: {n_cln}',
            ha='right', va='center', fontsize=9, fontweight='bold',
            color=parcel_border, transform=ax_hdr.transAxes,
        )

        # ── Surface rows ─────────────────────────────────────────────────────
        for s_idx, surface_name in enumerate(SURFACE_ORDER):
            row_idx = s_idx + 1   # +1 because inner[0] is the header

            surface_rows = group[group['sideface_name'] == surface_name]
            if len(surface_rows) == 0:
                continue
            row = surface_rows.iloc[0]

            is_tampered  = bool(row['tampered'])
            prob         = float(row['prob'])
            bg_color     = COLORS['tampered_bg']    if is_tampered else COLORS['clean_bg']
            badge_color  = COLORS['badge_tampered'] if is_tampered else COLORS['badge_clean']

            is_first_surf = (s_idx == 0)
            is_last_surf  = (s_idx == n_surfaces - 1)

            # Sub-grid for this surface row: 6 columns
            surf_gs = inner[row_idx].subgridspec(
                1, 6,
                width_ratios=col_ratios,
                wspace=0.07,
            )

            # Col 0 — Surface name + TAMPERED/CLEAN label
            ax_name = fig.add_subplot(surf_gs[0, 0])
            ax_name.set_facecolor(bg_color)
            ax_name.set_xticks([])
            ax_name.set_yticks([])
            for spine in ax_name.spines.values():
                spine.set_edgecolor(badge_color)
                spine.set_linewidth(1.5)
            ax_name.text(0.5, 0.65, surface_name.upper(),
                         ha='center', va='center', fontsize=10, fontweight='bold',
                         color=badge_color, transform=ax_name.transAxes)
            status_label = 'TAMPERED' if is_tampered else 'CLEAN'
            ax_name.text(0.5, 0.28, status_label,
                         ha='center', va='center', fontsize=7,
                         color='white', transform=ax_name.transAxes,
                         bbox=dict(boxstyle='round,pad=0.25',
                                   facecolor=badge_color, alpha=0.85,
                                   edgecolor='none'))

            # Col 1 — Reference patch
            ax_ref = fig.add_subplot(surf_gs[0, 1])
            ref_patch = get_surface_patch(ref_uvmap, surface_name) \
                if ref_uvmap is not None else None
            title_ref = 'Reference' if is_first_surf else ''
            _show_patch(ax_ref, ref_patch,
                        title=title_ref, title_color='#2471A3',
                        border_color='#2471A3', bg_color='#EBF5FB')

            # Col 2 — Arrow
            ax_arr = fig.add_subplot(surf_gs[0, 2])
            ax_arr.axis('off')
            ax_arr.annotate(
                '', xy=(0.85, 0.5), xytext=(0.15, 0.5),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#555555',
                                lw=2.0, mutation_scale=15),
            )

            # Col 3 — Adversarial patch
            ax_adv = fig.add_subplot(surf_gs[0, 3])
            adv_patch = get_surface_patch(adv_uvmap, surface_name) \
                if adv_uvmap is not None else None
            title_adv = 'Adversarial' if is_first_surf else ''
            _show_patch(ax_adv, adv_patch,
                        title=title_adv, title_color=badge_color,
                        border_color=badge_color, bg_color=bg_color)

            # Col 4 — SimSAC metric bars
            ax_metrics = fig.add_subplot(surf_gs[0, 4])
            ax_metrics.set_facecolor(bg_color)
            plot_metric_bar(ax_metrics, row, importances,
                            show_title=is_first_surf,
                            show_xlabel=is_last_surf)
            for spine in ax_metrics.spines.values():
                spine.set_edgecolor('#CCCCCC')
                spine.set_linewidth(0.8)

            # Col 5 — XGBoost probability gauge
            ax_gauge = fig.add_subplot(surf_gs[0, 5])
            ax_gauge.set_facecolor(bg_color)
            plot_probability_gauge(ax_gauge, prob, row['pred'], is_tampered,
                                   show_title=is_first_surf,
                                   show_labels=is_first_surf)
            for spine in ax_gauge.spines.values():
                spine.set_edgecolor('#CCCCCC')
                spine.set_linewidth(0.8)

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLORS['tampered_bar'], alpha=0.75,
                       label='Metric consistent with TAMPERED label'),
        mpatches.Patch(color=COLORS['neutral_bar'], alpha=0.75,
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
        description='Visualise correctly classified parcels using XGBoost on SimSAC features.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--csv', required=True,
                        help='Path to simscores CSV from compute_similarity_scores.py')
    parser.add_argument('--output', default='tampering_visualisation.png',
                        help='Output figure path (default: tampering_visualisation.png)')
    parser.add_argument('--n_parcels', type=int, default=2,
                        help='Number of correctly classified parcels to show (default: 2)')
    parser.add_argument('--uvmap_dir', default=None,
                        help='Directory containing reference UV maps '
                             '(id_XX_uvmap.png files, e.g. data/tampar_sample/uvmaps/)')
    parser.add_argument('--adv_dir', default=None,
                        help='Root directory for adversarial UV maps '
                             '(row["view"] is appended to this path, '
                             'e.g. data/tampar_sample/)')

    args = parser.parse_args()

    # 1. Load
    df = load_simsac_features(args.csv)

    # 2. Train XGBoost
    clf, probs, preds = train_xgboost(df)
    importances = dict(zip(METRICS, clf.feature_importances_))

    # 3. Select parcels where all surfaces are correctly classified
    parcels = select_parcels(df, probs, preds, n_parcels=args.n_parcels)
    if not parcels:
        print("No parcels found where all surfaces are correctly classified. "
              "Try with a larger dataset or lower n_parcels.")
        return

    # 4. Build and save figure
    build_figure(parcels, importances, output_path=args.output,
                 uvmap_dir=args.uvmap_dir,
                 adv_dir=args.adv_dir)


if __name__ == '__main__':
    main()
