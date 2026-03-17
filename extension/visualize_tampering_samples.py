
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from xgboost import XGBClassifier


# Config

METRICS       = ['msssim', 'cwssim', 'ssim', 'hog', 'mae']
METRIC_LABELS = {
    'msssim': 'MS-SSIM',
    'cwssim': 'CW-SSIM',
    'ssim':   'SSIM',
    'hog':    'HOG',
    'mae':    'MAE',
}

METRIC_TAMPERED_IS_LOW = {
    'msssim': True,
    'cwssim': True,
    'ssim':   True,
    'hog':    False,
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

# UV map 3×3 grid: flat index for each named surface
PATCH_INDEX = {'top': 1, 'left': 3, 'center': 4, 'right': 5, 'bottom': 7}


# Data loading

def load_simsac_features(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}")

    if 'compare_type' in df.columns:
        df = df[df['compare_type'] == 'simsac'].copy()
        print(f"After filtering to compare_type=simsac: {len(df)} rows")

    df['tampered'] = df['tampered'].astype(str).str.lower().map(
        {'true': 1, 'false': 0, '1': 1, '0': 0}
    ).fillna(df['tampered'].astype(int))
    df['tampered'] = df['tampered'].astype(int)

    missing = [m for m in METRICS if m not in df.columns]
    if missing:
        raise ValueError(f"Missing metric columns: {missing}")

    before = len(df)
    df = df.dropna(subset=METRICS)
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with NaN metrics")

    print(f"\nDataset summary:")
    print(f"  Total surfaces : {len(df)}")
    print(f"  Tampered       : {df['tampered'].sum()}")
    print(f"  Clean          : {(df['tampered'] == 0).sum()}")
    return df


# Metric thresholds

def compute_metric_thresholds(df):
    thresholds = {}
    for m in METRICS:
        if METRIC_TAMPERED_IS_LOW[m]:
            # Low value = tampered: use median of tampered class as the boundary
            pool = df[df['tampered'] == 1][m].dropna()
        else:
            # High value = tampered: use median of clean class as the boundary
            pool = df[df['tampered'] == 0][m].dropna()
        thresholds[m] = float(pool.median()) if len(pool) else float(df[m].median())

    print("\nMetric thresholds (data-driven):")
    for m, t in thresholds.items():
        print(f"  {METRIC_LABELS[m]:10s}: {t:.4f}")
    return thresholds


# XGBoost training

def train_xgboost(df):
    X = df[METRICS].values
    y = df['tampered'].values

    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='logloss', verbosity=0,
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


# Sample selection  (surface-level, not parcel-level)

def select_samples(df, probs, preds, n_tampered=3, n_clean=2,
                   random_state=None, top_k_factor=10):
    df = df.copy()
    df['prob']       = probs
    df['pred']       = preds
    df['correct']    = (preds == df['tampered'].values)
    df['confidence'] = (df['prob'] - 0.5).abs()

    def pick_diverse(pool, n):
        # Restrict to top confident, then shuffle for variety
        top_k = min(len(pool), max(n * top_k_factor, n))
        pool  = pool.head(top_k).sample(frac=1, random_state=random_state)

        seen_surf    = set()
        seen_parcels = set()
        picked       = []

        for allow_repeat_parcel in [False, True]:
            if len(picked) >= n:
                break
            for _, row in pool.iterrows():
                if len(picked) >= n:
                    break
                pid  = row.get('parcel_id', None)
                surf = row.get('sideface_name', None)
                key  = (pid, surf)
                if key in seen_surf:
                    continue
                if not allow_repeat_parcel and pid in seen_parcels:
                    continue
                seen_surf.add(key)
                seen_parcels.add(pid)
                picked.append(row)

        return pd.DataFrame(picked)

    pool_t = df[(df['tampered'] == 1) & df['correct']].sort_values('confidence', ascending=False)
    pool_c = df[(df['tampered'] == 0) & df['correct']].sort_values('confidence', ascending=False)

    if len(pool_t) < n_tampered:
        print(f"Warning: only {len(pool_t)} correctly detected tampered surfaces (requested {n_tampered})")
    if len(pool_c) < n_clean:
        print(f"Warning: only {len(pool_c)} correctly detected clean surfaces (requested {n_clean})")

    selected = pd.concat([pick_diverse(pool_t, n_tampered),
                          pick_diverse(pool_c, n_clean)]).reset_index(drop=True)

    print(f"\nSelected {min(n_tampered, len(pool_t))} tampered + {min(n_clean, len(pool_c))} clean surfaces:")
    for _, row in selected.iterrows():
        pid  = row.get('parcel_id', '?')
        surf = row.get('sideface_name', '?')
        lab  = 'TAMPERED' if row['tampered'] else 'CLEAN'
        print(f"  parcel={pid} surface={surf}  GT={lab}  Prob={row['prob']:.3f}")

    return selected


# Image helpers

def _load_image(path):
    import cv2 as _cv2
    try:
        img = _cv2.imread(str(path))
        if img is not None:
            return _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return None


def _load_adv_image(adv_dir, row):
    view       = str(row.get('view', ''))
    background = str(row.get('background', ''))
    filename   = Path(view).name
    base       = Path(adv_dir)

    candidates = [
        base / view,
        base / background / filename,
        base / filename,
    ]
    for p in candidates:
        img = _load_image(p)
        if img is not None:
            return img
    return None


def get_surface_patch(image, sideface_name):
    h, w   = image.shape[:2]
    ph, pw = h // 3, w // 3
    idx    = PATCH_INDEX[sideface_name]
    r, c   = divmod(idx, 3)
    return image[r*ph:(r+1)*ph, c*pw:(c+1)*pw].copy()


def _show_patch(ax, img, title, title_color, border_color, bg_color):
    ax.set_facecolor(bg_color)
    if img is not None:
        ax.imshow(img, aspect='auto')
    else:
        ax.text(0.5, 0.5, 'Image\nnot found', ha='center', va='center',
                fontsize=8, color='#888888', transform=ax.transAxes, style='italic')
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', color=title_color, pad=4)
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(1.8)


# Metric bars + probability gauge

def plot_metric_bar(ax, row, importances, metric_thresholds=None,
                    show_title=True, show_xlabel=True):
    y_pos = np.arange(len(METRICS))

    for i, metric in enumerate(METRICS):
        val         = float(row[metric])
        imp         = importances.get(metric, 0.0)
        val_display = min(max(val, 0.0), 1.0)

        # Use data-driven threshold if available, else fall back to 0.5
        threshold = metric_thresholds[metric] if metric_thresholds else 0.5

        if METRIC_TAMPERED_IS_LOW[metric]:
            tampered_signal = val < threshold
        else:
            tampered_signal = val > threshold

        bar_color = COLORS['tampered_bar'] if tampered_signal else COLORS['neutral_bar']
        alpha     = 0.4 + 0.6 * imp

        ax.barh(i, val_display, color=bar_color, alpha=alpha, height=0.6,
                edgecolor='white', linewidth=0.5)

        text_x = min(val_display + 0.02, 1.10)
        ax.text(text_x, i, f'{val:.3f}', va='center', ha='left', fontsize=8,
                color='#333333', fontweight='bold' if imp > 0.2 else 'normal')

        if imp > 0.15:
            ax.text(0.98, i, '★', transform=ax.get_yaxis_transform(),
                    va='center', ha='right', fontsize=7, color='#E67E22', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=8)
    ax.set_xlim(0, 1.15)
    if show_xlabel:
        ax.set_xlabel('Metric Value', fontsize=8)
    if show_title:
        ax.set_title('SimSAC Similarity Features', fontsize=12, color='#444444', pad=4)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(0.5, -0.7, '0.5', ha='center', va='top', fontsize=7, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.2)


def plot_probability_gauge(ax, prob, pred, ground_truth,
                           show_title=True, show_labels=True):
    cmap = LinearSegmentedColormap.from_list(
        'gauge', [COLORS['prob_low'], '#F39C12', COLORS['prob_high']]
    )
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(grad, aspect='auto', cmap=cmap, extent=[0, 1, 0, 1], alpha=0.25)

    fill_color = COLORS['prob_high'] if prob >= 0.5 else COLORS['prob_low']
    ax.barh(0.5, prob, height=0.5, left=0, color=fill_color, alpha=0.75, zorder=3)
    ax.axvline(0.5, color='gray', linewidth=1.5, linestyle='--', zorder=4)
    ax.plot(prob, 0.5, 'v', color='black', markersize=9, zorder=5)
    ax.text(prob, 0.95, f'{prob*100:.1f}%', ha='center', va='top',
            fontsize=10, fontweight='bold', color='black', zorder=6)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%\n(thresh)', '75%', '100%'], fontsize=7)
    ax.set_yticks([])
    if show_labels:
        ax.text(0.02, 0.1, 'CLEAN',    fontsize=8, color=COLORS['prob_low'],
                fontweight='bold', va='bottom')
        ax.text(0.98, 0.1, 'TAMPERED', fontsize=8, color=COLORS['prob_high'],
                fontweight='bold', va='bottom', ha='right')
    if show_title:
        ax.set_title('XGBoost Probability', fontsize=12, color='#444444', pad=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)


# Figure builder  — one row per surface sample

def build_figure(selected, importances, output_path,
                 uvmap_dir=None, adv_dir=None, metric_thresholds=None):
    n = len(selected)

    ROW_H      = 2.2
    TOP_MARGIN = 0.7
    BOT_MARGIN = 0.55
    fig_w      = 17.0
    fig_h      = n * ROW_H + TOP_MARGIN + BOT_MARGIN

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('#F8F9FA')

    fig.suptitle(
        'Tampering Detection — XGBoost on SimSAC Features\n'
        'Correctly Classified Surfaces (Adversarial Test Set)',
        fontsize=14, fontweight='bold', color='#1A1A2E',
        y=1.0 - 0.08 / fig_h,
    )

    outer_top    = 1.0 - TOP_MARGIN / fig_h
    outer_bottom = BOT_MARGIN / fig_h
    outer = fig.add_gridspec(
        n, 1,
        left=0.01, right=0.99,
        top=outer_top, bottom=outer_bottom,
        hspace=0.12,
    )

    col_ratios = [0.55, 2.2, 0.15, 2.2, 0.35, 3.0, 2.3]

    for i, (_, row) in enumerate(selected.iterrows()):
        is_tampered = bool(row['tampered'])
        prob        = float(row['prob'])
        bg_color    = COLORS['tampered_bg']    if is_tampered else COLORS['clean_bg']
        badge_color = COLORS['badge_tampered'] if is_tampered else COLORS['badge_clean']

        surf_name  = str(row.get('sideface_name', '?'))
        parcel_id  = row.get('parcel_id', None)
        view_path  = str(row.get('view', ''))

        is_first = (i == 0)
        is_last  = (i == n - 1)

        prev_tampered = bool(selected.iloc[i - 1]['tampered']) if i > 0 else None
        show_headers  = is_first or (prev_tampered is not None and is_tampered != prev_tampered)

        gs = outer[i].subgridspec(1, 7, width_ratios=col_ratios, wspace=0.10)

        # Load images for this surface
        ref_uvmap = None
        adv_uvmap = None
        if uvmap_dir is not None and parcel_id is not None:
            ref_uvmap = _load_image(Path(uvmap_dir) / f'id_{int(parcel_id):02d}_uvmap.png')
        if adv_dir is not None:
            adv_uvmap = _load_adv_image(adv_dir, row)

        ref_patch = get_surface_patch(ref_uvmap, surf_name) if ref_uvmap is not None and surf_name in PATCH_INDEX else None
        adv_patch = get_surface_patch(adv_uvmap, surf_name) if adv_uvmap is not None and surf_name in PATCH_INDEX else None

        #  Col 0: Label 
        ax_lbl = fig.add_subplot(gs[0, 0])
        ax_lbl.set_facecolor(bg_color)
        ax_lbl.set_xticks([]); ax_lbl.set_yticks([])
        for spine in ax_lbl.spines.values():
            spine.set_edgecolor(badge_color); spine.set_linewidth(1.8)

        if show_headers:
            ax_lbl.set_title('Surface /\nParcel ID', fontsize=12, fontweight='bold',
                             color='#444444', pad=4)

        gt_text = 'TAMPERED' if is_tampered else 'CLEAN'
        ax_lbl.text(0.5, 0.72, surf_name.upper(),
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color=badge_color, transform=ax_lbl.transAxes)
        ax_lbl.text(0.5, 0.45, f'Parcel {parcel_id}',
                    ha='center', va='center', fontsize=7.5, color='#555555',
                    transform=ax_lbl.transAxes)
        ax_lbl.text(0.5, 0.22, gt_text,
                    ha='center', va='center', fontsize=7.5, color='white',
                    transform=ax_lbl.transAxes,
                    bbox=dict(boxstyle='round,pad=0.22', facecolor=badge_color,
                              alpha=0.85, edgecolor='none'))

        #  Col 1: Reference patch 
        ax_ref = fig.add_subplot(gs[0, 1])
        _show_patch(ax_ref, ref_patch,
                    title='Reference' if show_headers else '',
                    title_color='#2471A3', border_color='#2471A3', bg_color='#EBF5FB')

        #  Col 2: Arrow 
        ax_arr = fig.add_subplot(gs[0, 2])
        ax_arr.axis('off')
        ax_arr.annotate('', xy=(0.85, 0.5), xytext=(0.15, 0.5),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='#555555',
                                        lw=2.0, mutation_scale=15))

        #  Col 3: Adversarial patch 
        ax_adv = fig.add_subplot(gs[0, 3])
        adv_label = 'TAMPERED' if is_tampered else 'CLEAN'
        _show_patch(ax_adv, adv_patch,
                    title=f'Adversarial ({adv_label})' if show_headers else '',
                    title_color=badge_color, border_color=badge_color, bg_color=bg_color)

        #  Col 4: Spacer 
        ax_sp = fig.add_subplot(gs[0, 4])
        ax_sp.axis('off')

        #  Col 5: SimSAC metric bars 
        ax_metrics = fig.add_subplot(gs[0, 5])
        ax_metrics.set_facecolor(bg_color)
        plot_metric_bar(ax_metrics, row, importances,
                        metric_thresholds=metric_thresholds,
                        show_title=is_first, show_xlabel=is_last)
        for spine in ax_metrics.spines.values():
            spine.set_edgecolor('#CCCCCC'); spine.set_linewidth(0.8)

        #  Col 6: Probability gauge 
        ax_gauge = fig.add_subplot(gs[0, 6])
        ax_gauge.set_facecolor(bg_color)
        plot_probability_gauge(ax_gauge, prob, row['pred'], is_tampered,
                               show_title=is_first, show_labels=show_headers)
        for spine in ax_gauge.spines.values():
            spine.set_edgecolor('#CCCCCC'); spine.set_linewidth(0.8)

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLORS['tampered_bar'], alpha=0.75,
                       label='Metric consistent with TAMPERED label'),
        mpatches.Patch(color=COLORS['neutral_bar'],  alpha=0.75,
                       label='Metric consistent with CLEAN label'),
        mpatches.Patch(color='#E67E22', alpha=0.8,
                       label='★ High XGBoost feature importance (>15%)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\n Figure saved  {output_path}")
    return fig


# Main

def main():
    parser = argparse.ArgumentParser(
        description='Visualise correctly classified surfaces using XGBoost on SimSAC features.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--csv',       required=True,
                        help='Path to simscores CSV from compute_similarity_scores.py')
    parser.add_argument('--output',    default='tampering_visualisation.png',
                        help='Output figure path (default: tampering_visualisation.png)')
    parser.add_argument('--n_tampered', type=int, default=3,
                        help='Number of correctly detected tampered surfaces to show (default: 3)')
    parser.add_argument('--n_clean',    type=int, default=2,
                        help='Number of correctly detected clean surfaces to show (default: 2)')
    parser.add_argument('--uvmap_dir', default=None,
                        help='Directory containing reference UV maps (id_XX_uvmap.png)')
    parser.add_argument('--adv_dir',   default=None,
                        help='Root directory for adversarial UV maps (row["view"] appended)')
    parser.add_argument('--random_state', type=int, default=None,
                        help='Random seed for sample selection. Omit for a different result each run.')
    parser.add_argument('--include_parcels', type=int, nargs='+', default=None,
                        metavar='ID',
                        help='Only show samples from these parcel IDs (e.g. --include_parcels 3 7 12)')
    parser.add_argument('--exclude_parcels', type=int, nargs='+', default=None,
                        metavar='ID',
                        help='Skip samples from these parcel IDs (e.g. --exclude_parcels 15 19)')

    args = parser.parse_args()

    df = load_simsac_features(args.csv)

    # Optional parcel filtering
    if args.include_parcels is not None:
        before = len(df)
        df = df[df['parcel_id'].isin(args.include_parcels)].copy()
        print(f"include_parcels={args.include_parcels}: {before}  {len(df)} rows")
    if args.exclude_parcels is not None:
        before = len(df)
        df = df[~df['parcel_id'].isin(args.exclude_parcels)].copy()
        print(f"exclude_parcels={args.exclude_parcels}: {before}  {len(df)} rows")
    if len(df) == 0:
        raise ValueError("No rows remain after parcel filtering — check --include_parcels / --exclude_parcels.")
    clf, probs, preds = train_xgboost(df)
    importances        = dict(zip(METRICS, clf.feature_importances_))
    metric_thresholds  = compute_metric_thresholds(df)

    selected = select_samples(df, probs, preds,
                              n_tampered=args.n_tampered,
                              n_clean=args.n_clean,
                              random_state=args.random_state)

    build_figure(selected, importances, output_path=args.output,
                 uvmap_dir=args.uvmap_dir,
                 adv_dir=args.adv_dir,
                 metric_thresholds=metric_thresholds)


if __name__ == '__main__':
    main()
