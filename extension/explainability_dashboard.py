
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Optional

from extension.tampering_localizer import (
    TamperingLocalizer,
    ParcelLocalizationResult,
    SurfaceLocalizationResult,
    SURFACE_NAMES,
)

# ─────────────────────────────────────────────────────
# Visual Style Constants
# ─────────────────────────────────────────────────────
COLOR_TAMPERED = '#FF4444'
COLOR_CLEAN    = '#44BB44'
COLOR_NEUTRAL  = '#888888'
METRIC_COLORS  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
FONT_TITLE     = {'fontsize': 13, 'fontweight': 'bold'}
FONT_SUBTITLE  = {'fontsize': 10, 'fontweight': 'bold'}
FONT_LABEL     = {'fontsize': 8}

# Metric display names and whether higher = more similar (True) or more different (False)
METRIC_META = {
    'msssim': ('MS-SSIM',  True,  '↑ similar'),
    'cwssim': ('CW-SSIM',  True,  '↑ similar'),
    'ssim':   ('SSIM',     True,  '↑ similar'),
    'hog':    ('HOG dist', False, '↓ similar'),
    'mae':    ('MAE',      False, '↓ similar'),
}


class ExplainabilityDashboard:

    def __init__(self, output_dir: str = "outputs/dashboard"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────
    # Main Entry Point
    # ─────────────────────────────────────────────────────

    def generate(self, result: ParcelLocalizationResult) -> Dict[str, Path]:
        print(f"\nGenerating dashboard for parcel {result.parcel_id}...")
        saved_files = {}

        # 1. Main overview dashboard
        path = self._plot_overview_dashboard(result)
        saved_files['overview'] = path
        print(f"  ✓ Overview dashboard : {path}")

        # 2. Per-surface heatmap grid
        path = self._plot_heatmap_grid(result)
        saved_files['heatmaps'] = path
        print(f"  ✓ Heatmap grid       : {path}")

        # 3. Metrics comparison chart
        path = self._plot_metrics_comparison(result)
        saved_files['metrics'] = path
        print(f"  ✓ Metrics chart      : {path}")

        # 4. Tampered region masks
        path = self._plot_tampered_masks(result)
        saved_files['masks'] = path
        print(f"  ✓ Tampered masks     : {path}")

        # 5. Verdict summary card
        path = self._plot_verdict_summary(result)
        saved_files['verdict'] = path
        print(f"  ✓ Verdict summary    : {path}")

        print(f"\n  All outputs saved to: {self.output_dir}")
        return saved_files

    # ─────────────────────────────────────────────────────
    # Plot 1: Overview Dashboard (main composite figure)
    # ─────────────────────────────────────────────────────

    def _plot_overview_dashboard(self, result: ParcelLocalizationResult) -> Path:
        surfaces = result.surfaces
        n_surfaces = len(surfaces)
        if n_surfaces == 0:
            return None

        surface_list = list(surfaces.values())
        fig = plt.figure(figsize=(4 * n_surfaces + 2, 18))
        fig.patch.set_facecolor('#1a1a2e')

        # Overall title
        verdict = 'TAMPERED' if result.overall_tampered else 'CLEAN'
        verdict_color = COLOR_TAMPERED if result.overall_tampered else COLOR_CLEAN
        fig.suptitle(
            f"Parcel {result.parcel_id} — Verdict: {verdict}  "
            f"({result.n_surfaces_tampered}/{result.n_surfaces_total} surfaces tampered, "
            f"confidence: {result.overall_confidence:.0%})",
            fontsize=15, fontweight='bold', color='white', y=0.98
        )

        gs = gridspec.GridSpec(4, n_surfaces, figure=fig, hspace=0.35, wspace=0.1)

        row_labels = [
            'Reference Surface',
            'Query Surface',
            'Tampering Heatmap\n(Red = changed)',
            'Similarity Scores',
        ]

        for col, surface in enumerate(surface_list):
            is_tampered = surface.is_tampered
            border_color = COLOR_TAMPERED if is_tampered else COLOR_CLEAN

            # ── Row 0: Reference patch ──
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(surface.patch_reference)
            ax.set_title(
                f"{surface.name.upper()}\n(Reference)",
                **FONT_SUBTITLE, color='white', pad=4
            )
            self._style_ax(ax, border_color, show_border=True)

            # ── Row 1: Query patch ──
            ax = fig.add_subplot(gs[1, col])
            ax.imshow(surface.patch_query)
            decision = 'TAMPERED' if is_tampered else 'CLEAN'
            ax.set_title(
                f"Query — {decision}\n(conf: {surface.confidence:.0%})",
                **FONT_SUBTITLE, color=border_color, pad=4
            )
            self._style_ax(ax, border_color, show_border=True)

            # ── Row 2: Heatmap overlay ──
            ax = fig.add_subplot(gs[2, col])
            ax.imshow(surface.change_map_overlay)
            ax.set_title(
                f"Tampered area: {surface.tampered_area_pct:.1f}%",
                **FONT_LABEL, color='white', pad=4
            )
            self._style_ax(ax, border_color, show_border=True)

            # ── Row 3: Metric bar chart ──
            ax = fig.add_subplot(gs[3, col])
            self._draw_metric_bars(ax, surface)
            ax.set_facecolor('#2d2d44')

        # Row labels on left
        for row_idx, label in enumerate(row_labels):
            fig.text(
                0.01, 0.82 - row_idx * 0.215, label,
                va='center', ha='left',
                fontsize=9, color='#aaaaaa',
                rotation=90
            )

        path = self.output_dir / f"parcel_{result.parcel_id:02d}_overview.png"
        plt.savefig(path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        return path

    # ─────────────────────────────────────────────────────
    # Plot 2: Heatmap Grid (detailed)
    # ─────────────────────────────────────────────────────

    def _plot_heatmap_grid(self, result: ParcelLocalizationResult) -> Path:
        surfaces = list(result.surfaces.values())
        n_surfaces = len(surfaces)
        if n_surfaces == 0:
            return None

        n_cols = 5
        col_labels = [
            'Reference', 'Query',
            'SimSAC Heatmap\n(Brighter = more change)',
            'Heatmap Overlay',
            'Pixel Diff Map'
        ]

        fig, axes = plt.subplots(
            n_surfaces, n_cols,
            figsize=(n_cols * 3.5, n_surfaces * 3.5 + 1)
        )
        fig.patch.set_facecolor('#1a1a2e')

        if n_surfaces == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(
            f"Parcel {result.parcel_id} — Surface Heatmap Detail",
            fontsize=14, fontweight='bold', color='white', y=1.01
        )

        # Column headers
        for col_idx, label in enumerate(col_labels):
            axes[0, col_idx].set_title(label, **FONT_SUBTITLE,
                                        color='#cccccc', pad=6)

        for row_idx, surface in enumerate(surfaces):
            is_tampered = surface.is_tampered
            border_color = COLOR_TAMPERED if is_tampered else COLOR_CLEAN

            images = [
                surface.patch_reference,
                surface.patch_query,
                cv2.applyColorMap(surface.change_map_raw, cv2.COLORMAP_JET),
                surface.change_map_overlay,
                cv2.applyColorMap(surface.diff_map, cv2.COLORMAP_JET),
            ]

            for col_idx, img in enumerate(images):
                ax = axes[row_idx, col_idx]
                if img.ndim == 2:
                    ax.imshow(img, cmap='hot')
                else:
                    # Convert BGR→RGB for cv2 outputs
                    if col_idx in [2, 4]:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img)
                self._style_ax(ax, border_color, show_border=(col_idx == 0))

            # Row label: surface name + verdict
            verdict = f"{'⚠ TAMPERED' if is_tampered else '✓ CLEAN'}"
            axes[row_idx, 0].set_ylabel(
                f"{surface.name.upper()}\n{verdict}",
                fontsize=9, fontweight='bold',
                color=border_color, labelpad=6
            )

        plt.tight_layout()
        path = self.output_dir / f"parcel_{result.parcel_id:02d}_heatmaps.png"
        plt.savefig(path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        return path

    # ─────────────────────────────────────────────────────
    # Plot 3: Metrics Comparison
    # ─────────────────────────────────────────────────────

    def _plot_metrics_comparison(self, result: ParcelLocalizationResult) -> Path:
        surfaces = result.surfaces
        if not surfaces:
            return None

        surface_names = list(surfaces.keys())
        metric_names = list(METRIC_META.keys())
        n_metrics = len(metric_names)
        n_surfaces = len(surface_names)

        fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 3.5, 5))
        fig.patch.set_facecolor('#1a1a2e')
        fig.suptitle(
            f"Parcel {result.parcel_id} — Per-Surface Metric Breakdown",
            fontsize=13, fontweight='bold', color='white', y=1.02
        )

        for ax_idx, metric_name in enumerate(metric_names):
            ax = axes[ax_idx]
            ax.set_facecolor('#2d2d44')

            display_name, higher_is_similar, direction_label = METRIC_META[metric_name]

            values = []
            colors = []
            for sname in surface_names:
                val = surfaces[sname].metrics.get(metric_name, 0.0)
                values.append(float(val))
                is_t = surfaces[sname].is_tampered
                colors.append(COLOR_TAMPERED if is_t else COLOR_CLEAN)

            bars = ax.barh(surface_names, values, color=colors, edgecolor='white',
                           linewidth=0.5, height=0.6)

            # Value labels on bars
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_width() + max(values) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}',
                    va='center', ha='left',
                    fontsize=8, color='white'
                )

            ax.set_title(f"{display_name}\n({direction_label})",
                         **FONT_SUBTITLE, color='white', pad=6)
            ax.tick_params(colors='white', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#555555')
            ax.set_xlabel('Value', fontsize=8, color='#aaaaaa')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_color('white')

        # Legend
        tampered_patch = mpatches.Patch(color=COLOR_TAMPERED, label='Tampered Surface')
        clean_patch    = mpatches.Patch(color=COLOR_CLEAN,    label='Clean Surface')
        fig.legend(
            handles=[tampered_patch, clean_patch],
            loc='lower center', ncol=2,
            fontsize=9, fancybox=True,
            framealpha=0.3, labelcolor='white',
            facecolor='#2d2d44', edgecolor='#555555',
            bbox_to_anchor=(0.5, -0.05)
        )

        plt.tight_layout()
        path = self.output_dir / f"parcel_{result.parcel_id:02d}_metrics.png"
        plt.savefig(path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        return path

    # ─────────────────────────────────────────────────────
    # Plot 4: Tampered Region Masks
    # ─────────────────────────────────────────────────────

    def _plot_tampered_masks(self, result: ParcelLocalizationResult) -> Path:
        surfaces = list(result.surfaces.values())
        n_surfaces = len(surfaces)
        if n_surfaces == 0:
            return None

        fig, axes = plt.subplots(2, n_surfaces, figsize=(n_surfaces * 3.5, 7))
        fig.patch.set_facecolor('#1a1a2e')
        fig.suptitle(
            f"Parcel {result.parcel_id} — Tampered Region Masks",
            fontsize=13, fontweight='bold', color='white'
        )

        if n_surfaces == 1:
            axes = axes[:, np.newaxis]

        for col, surface in enumerate(surfaces):
            is_tampered = surface.is_tampered
            border_color = COLOR_TAMPERED if is_tampered else COLOR_CLEAN

            # Row 0: Query patch with red mask overlay
            ax = axes[0, col]
            query_rgba = cv2.cvtColor(surface.patch_query, cv2.COLOR_RGB2RGBA)
            red_overlay = np.zeros_like(query_rgba)
            red_overlay[surface.tampered_region_mask > 0] = [255, 0, 0, 160]
            blended = query_rgba.copy()
            mask_bool = surface.tampered_region_mask > 0
            blended[mask_bool] = (
                query_rgba[mask_bool] * 0.5 + red_overlay[mask_bool] * 0.5
            ).astype(np.uint8)
            ax.imshow(blended)
            ax.set_title(
                f"{surface.name.upper()}\nTampered area: {surface.tampered_area_pct:.1f}%",
                **FONT_SUBTITLE, color=border_color, pad=4
            )
            self._style_ax(ax, border_color, show_border=True)

            # Row 1: Binary mask only
            ax = axes[1, col]
            ax.imshow(surface.tampered_region_mask, cmap='RdYlGn_r', vmin=0, vmax=255)
            verdict = '⚠ TAMPERED' if is_tampered else '✓ CLEAN'
            ax.set_title(
                f"Binary Mask\n{verdict} (conf: {surface.confidence:.0%})",
                **FONT_LABEL, color=border_color, pad=4
            )
            self._style_ax(ax, border_color, show_border=False)

        plt.tight_layout()
        path = self.output_dir / f"parcel_{result.parcel_id:02d}_masks.png"
        plt.savefig(path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        return path

    # ─────────────────────────────────────────────────────
    # Plot 5: Verdict Summary Card
    # ─────────────────────────────────────────────────────

    def _plot_verdict_summary(self, result: ParcelLocalizationResult) -> Path:
        surfaces = result.surfaces
        n_surfaces = len(surfaces)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5 + n_surfaces * 0.5),
                                  gridspec_kw={'width_ratios': [1, 2]})
        fig.patch.set_facecolor('#1a1a2e')

        # ── Left: Overall Verdict Box ──
        ax_verdict = axes[0]
        ax_verdict.set_facecolor('#1a1a2e')
        ax_verdict.axis('off')

        verdict = 'TAMPERED' if result.overall_tampered else 'CLEAN'
        verdict_color = COLOR_TAMPERED if result.overall_tampered else COLOR_CLEAN
        verdict_bg = '#3d1a1a' if result.overall_tampered else '#1a3d1a'

        # Verdict box
        rect = plt.Rectangle((0.05, 0.3), 0.9, 0.5,
                               facecolor=verdict_bg, edgecolor=verdict_color,
                               linewidth=3, transform=ax_verdict.transAxes)
        ax_verdict.add_patch(rect)

        ax_verdict.text(0.5, 0.62, f"Parcel {result.parcel_id}",
                        transform=ax_verdict.transAxes,
                        ha='center', va='center',
                        fontsize=14, color='white', fontweight='bold')

        ax_verdict.text(0.5, 0.5, verdict,
                        transform=ax_verdict.transAxes,
                        ha='center', va='center',
                        fontsize=28, color=verdict_color, fontweight='bold')

        ax_verdict.text(0.5, 0.38, f"Confidence: {result.overall_confidence:.0%}",
                        transform=ax_verdict.transAxes,
                        ha='center', va='center',
                        fontsize=11, color='#cccccc')

        ax_verdict.text(0.5, 0.22,
                        f"{result.n_surfaces_tampered} of {result.n_surfaces_total} "
                        f"surfaces tampered",
                        transform=ax_verdict.transAxes,
                        ha='center', va='center',
                        fontsize=10, color='#aaaaaa')

        # ── Right: Per-Surface Table ──
        ax_table = axes[1]
        ax_table.set_facecolor('#1a1a2e')
        ax_table.axis('off')
        ax_table.set_title("Per-Surface Analysis",
                           **FONT_SUBTITLE, color='white', pad=10)

        # Table headers
        headers = ['Surface', 'Decision', 'Confidence', 'Tamper Score', 'Tampered Area', 'MSSSIM', 'MAE']
        col_positions = [0.0, 0.16, 0.30, 0.44, 0.58, 0.72, 0.86]

        y_start = 0.88
        row_height = 0.12

        # Header row
        for header, x in zip(headers, col_positions):
            ax_table.text(x, y_start, header,
                          transform=ax_table.transAxes,
                          ha='left', va='top',
                          fontsize=9, color='#aaaaaa', fontweight='bold')

        # Horizontal line under header
        ax_table.plot([0, 1], [y_start - 0.04, y_start - 0.04],
                      color='#555555', linewidth=0.8,
                      transform=ax_table.transAxes, clip_on=False)

        for i, (sname, surface) in enumerate(surfaces.items()):
            y = y_start - (i + 1) * row_height - 0.04
            is_tampered = surface.is_tampered
            row_color = '#3d1a1a' if is_tampered else '#1a2d1a'
            verdict_text = '⚠ TAMPERED' if is_tampered else '✓ CLEAN'
            text_color = COLOR_TAMPERED if is_tampered else COLOR_CLEAN

            # Row background
            rect = plt.Rectangle(
                (-0.02, y - 0.01), 1.04, row_height - 0.01,
                facecolor=row_color, alpha=0.5,
                transform=ax_table.transAxes
            )
            ax_table.add_patch(rect)

            row_values = [
                sname.upper(),
                verdict_text,
                f"{surface.confidence:.0%}",
                f"{surface.tampering_score:.3f}",
                f"{surface.tampered_area_pct:.1f}%",
                f"{surface.metrics.get('msssim', 0):.3f}",
                f"{surface.metrics.get('mae', 0):.4f}",
            ]

            for j, (val, x) in enumerate(zip(row_values, col_positions)):
                color = text_color if j <= 1 else 'white'
                ax_table.text(x, y + 0.05, val,
                              transform=ax_table.transAxes,
                              ha='left', va='center',
                              fontsize=9, color=color)

        plt.tight_layout()
        path = self.output_dir / f"parcel_{result.parcel_id:02d}_verdict.png"
        plt.savefig(path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        return path

    # ─────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────

    def _style_ax(self, ax, border_color: str, show_border: bool = True):
        ax.set_xticks([])
        ax.set_yticks([])
        if show_border:
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2.5)
        else:
            for spine in ax.spines.values():
                spine.set_visible(False)
        ax.set_facecolor('#1a1a2e')

    def _draw_metric_bars(self, ax, surface: SurfaceLocalizationResult):
        metrics = surface.metrics
        metric_names = list(METRIC_META.keys())
        values = [float(metrics.get(m, 0)) for m in metric_names]
        display_names = [METRIC_META[m][0] for m in metric_names]

        colors = [METRIC_COLORS[i % len(METRIC_COLORS)] for i in range(len(metric_names))]
        bars = ax.barh(display_names, values, color=colors, height=0.6,
                       edgecolor='white', linewidth=0.3)

        ax.set_xlim(0, max(max(values) * 1.3, 1.0))
        ax.tick_params(colors='white', labelsize=7)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#555555')
        ax.set_title(
            f"Score: {surface.tampering_score:.2f}",
            fontsize=8, color='white', pad=2
        )


# ─────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate explainability dashboard for a TAMPAR parcel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--parcel_id', type=int, required=True,
                        help='Parcel ID (e.g. 1)')
    parser.add_argument('--reference_uvmap', type=str, required=True,
                        help='Path to reference UV map (uvmaps/id_XX_uvmap.png)')
    parser.add_argument('--query_uvmap', type=str, required=True,
                        help='Path to query UV map (validation/background/id_XX_*.png)')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/dashboard',
                        help='Directory to save dashboard images (default: outputs/dashboard)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to fine-tuned SimSAC checkpoint. '
                             'If not provided, uses default synthetic.pth')
    parser.add_argument('--tampering_threshold', type=float, default=0.75,
                        help='Similarity threshold for tampering decision (default: 0.75)')

    args = parser.parse_args()

    # Run localization
    localizer = TamperingLocalizer(
        simsac_ckpt_path=args.checkpoint,
        tampering_threshold=args.tampering_threshold,
    )

    result = localizer.localize(
        reference_uvmap_path=args.reference_uvmap,
        query_uvmap_path=args.query_uvmap,
        parcel_id=args.parcel_id,
    )

    # Generate dashboard
    dashboard = ExplainabilityDashboard(output_dir=args.output_dir)
    saved_files = dashboard.generate(result)

    print("\n" + "=" * 60)
    print("DASHBOARD COMPLETE")
    for name, path in saved_files.items():
        print(f"  {name:<12}: {path}")


if __name__ == "__main__":
    main()
