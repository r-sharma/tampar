
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.simsac.inference import SimSaC, img2vis
from src.tampering.compare import (
    METRICS,
    apply_homogenization,
    CompareType,
)
from src.tampering.metrics import (
    compute_msssim,
    compute_cwssim,
    compute_ssim,
    compute_hog,
    compute_mae,
)
from src.tampering.parcel import PATCH_ORDER
from src.tampering.utils import get_side_surface_patches


# Surface names in UV map grid order (9 positions, 4 are empty)
SURFACE_NAMES = ["top", "left", "center", "right", "bottom"]

# Color maps for heatmap visualization
HEATMAP_COLORMAP = cv2.COLORMAP_JET
OVERLAY_ALPHA = 0.5


@dataclass
class SurfaceLocalizationResult:
    name: str
    patch_reference: np.ndarray
    patch_query: np.ndarray
    change_map_raw: np.ndarray
    change_map_colored: np.ndarray
    change_map_overlay: np.ndarray
    diff_map: np.ndarray
    tampered_region_mask: np.ndarray
    metrics: Dict[str, float]
    tampering_score: float
    is_tampered: bool
    confidence: float
    tampered_area_pct: float


@dataclass
class ParcelLocalizationResult:
    parcel_id: int
    reference_uvmap_path: str
    query_uvmap_path: str
    surfaces: Dict[str, SurfaceLocalizationResult] = field(default_factory=dict)
    overall_tampered: bool = False
    overall_confidence: float = 0.0
    n_surfaces_tampered: int = 0
    n_surfaces_total: int = 0
    summary_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


class TamperingLocalizer:

    def __init__(
        self,
        simsac_ckpt_path: Optional[str] = None,
        tampering_threshold: float = 0.75,
        change_map_threshold: int = 200,
    ):
        self.simsac_ckpt_path = simsac_ckpt_path
        self.tampering_threshold = tampering_threshold
        self.change_map_threshold = change_map_threshold

        # Load SimSAC model
        print(f"Loading SimSAC model...")
        if simsac_ckpt_path:
            self.simsac = SimSaC.get_instance(ckpt_path=simsac_ckpt_path)
            print(f"  Using checkpoint: {simsac_ckpt_path}")
        else:
            self.simsac = SimSaC.get_instance()
            print(f"  Using default: synthetic.pth")

    def _load_uvmap(self, path: str) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _generate_change_map(
        self, patch_ref: np.ndarray, patch_query: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Run SimSAC inference
        imgs = self.simsac.inference(
            patch_ref.astype(np.uint8),
            patch_query.astype(np.uint8)
        )
        change1, change2, flow = imgs

        # change1 is already inverted (255 - change): brighter = more change detected
        change_gray = cv2.cvtColor(change1, cv2.COLOR_RGB2GRAY)
        change_gray = change_gray.astype(np.uint8)

        # Resize to match patch size
        h, w = patch_query.shape[:2]
        change_gray = cv2.resize(change_gray, (w, h))

        # Apply JET colormap for heatmap visualization
        change_colored = cv2.applyColorMap(change_gray, HEATMAP_COLORMAP)
        change_colored = cv2.cvtColor(change_colored, cv2.COLOR_BGR2RGB)

        # Blend heatmap onto query patch
        query_uint8 = patch_query.astype(np.uint8)
        overlay = cv2.addWeighted(query_uint8, 1 - OVERLAY_ALPHA, change_colored, OVERLAY_ALPHA, 0)

        return change_gray, change_colored, overlay

    def _generate_diff_map(
        self, patch_ref: np.ndarray, patch_query: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Convert to grayscale for comparison
        ref_gray = cv2.cvtColor(patch_ref.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        query_gray = cv2.cvtColor(patch_query.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Absolute difference
        diff = np.abs(ref_gray - query_gray)
        diff_norm = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)

        # Apply colormap
        diff_colored = cv2.applyColorMap(diff_norm, HEATMAP_COLORMAP)
        diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)

        return diff_norm, diff_colored

    def _compute_surface_metrics(
        self, patch_ref: np.ndarray, patch_query: np.ndarray
    ) -> Dict[str, float]:
        metrics = {}
        for metric_name in METRICS:
            compute_fn = {
                'msssim': compute_msssim,
                'cwssim': compute_cwssim,
                'ssim': compute_ssim,
                'hog': compute_hog,
                'mae': compute_mae,
            }[metric_name]
            try:
                val = compute_fn(
                    patch_ref.astype(np.float32),
                    patch_query.astype(np.float32)
                )
                metrics[metric_name] = float(val)
            except Exception as e:
                metrics[metric_name] = 0.0
        return metrics

    def _compute_tampering_score(self, metrics: Dict[str, float]) -> float:
        scores = []

        # Similarity metrics: invert (1 - score = tamper likelihood)
        if 'msssim' in metrics and metrics['msssim'] > 0:
            scores.append(1.0 - min(metrics['msssim'], 1.0))

        if 'ssim' in metrics and metrics['ssim'] > 0:
            scores.append(1.0 - min(metrics['ssim'], 1.0))

        # MAE: normalize to 0-1 (already normalized by 255 in compute_mae)
        if 'mae' in metrics:
            scores.append(min(metrics['mae'], 1.0))

        # HOG: normalize to 0-1 (typical range 0-10, cap at 1.0)
        if 'hog' in metrics:
            scores.append(min(metrics['hog'] / 10.0, 1.0))

        return float(np.mean(scores)) if scores else 0.0

    def _compute_tampered_area(
        self, change_map: np.ndarray, threshold: int = 128
    ) -> Tuple[np.ndarray, float]:
        mask = (change_map > threshold).astype(np.uint8) * 255

        # Morphological cleanup: remove noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        pct = float(np.sum(mask > 0)) / float(mask.size) * 100.0
        return mask, pct

    def localize_surface(
        self,
        patch_ref: np.ndarray,
        patch_query: np.ndarray,
        surface_name: str,
    ) -> SurfaceLocalizationResult:
        # 1. Generate SimSAC change map (primary heatmap)
        change_raw, change_colored, change_overlay = self._generate_change_map(
            patch_ref, patch_query
        )

        # 2. Generate pixel difference map
        diff_gray, diff_colored = self._generate_diff_map(patch_ref, patch_query)

        # 3. Compute all similarity metrics
        metrics = self._compute_surface_metrics(patch_ref, patch_query)

        # 4. Compute tampering score from metrics
        tampering_score = self._compute_tampering_score(metrics)

        # 5. Binary decision using threshold
        is_tampered = tampering_score > (1.0 - self.tampering_threshold)

        # 6. Confidence: distance from decision boundary (0.5)
        confidence = abs(tampering_score - 0.5) * 2.0

        # 7. Compute tampered area mask and percentage
        tampered_mask, tampered_pct = self._compute_tampered_area(change_raw)

        return SurfaceLocalizationResult(
            name=surface_name,
            patch_reference=patch_ref.astype(np.uint8),
            patch_query=patch_query.astype(np.uint8),
            change_map_raw=change_raw,
            change_map_colored=change_colored,
            change_map_overlay=change_overlay,
            diff_map=diff_gray,
            tampered_region_mask=tampered_mask,
            metrics=metrics,
            tampering_score=tampering_score,
            is_tampered=is_tampered,
            confidence=confidence,
            tampered_area_pct=tampered_pct,
        )

    def localize(
        self,
        reference_uvmap_path: str,
        query_uvmap_path: str,
        parcel_id: int = 0,
    ) -> ParcelLocalizationResult:
        print(f"\nLocalizing parcel {parcel_id}...")
        print(f"  Reference : {reference_uvmap_path}")
        print(f"  Query     : {query_uvmap_path}")

        # Load UV maps
        uvmap_ref = self._load_uvmap(reference_uvmap_path)
        uvmap_query = self._load_uvmap(query_uvmap_path)

        # Extract all surface patches from both UV maps
        patches_ref = list(get_side_surface_patches(uvmap_ref))
        patches_query = list(get_side_surface_patches(uvmap_query))

        result = ParcelLocalizationResult(
            parcel_id=parcel_id,
            reference_uvmap_path=str(reference_uvmap_path),
            query_uvmap_path=str(query_uvmap_path),
        )

        # Process each surface patch
        for i, (patch_ref, patch_query) in enumerate(zip(patches_ref, patches_query)):
            surface_name = PATCH_ORDER[i]

            # Skip empty patches (white background)
            if np.mean(patch_ref) >= 250 or np.mean(patch_query) >= 250:
                continue

            print(f"  Processing surface: {surface_name}")

            surface_result = self.localize_surface(patch_ref, patch_query, surface_name)
            result.surfaces[surface_name] = surface_result

            # Accumulate summary metrics
            result.summary_metrics[surface_name] = {
                **surface_result.metrics,
                'tampering_score': surface_result.tampering_score,
                'tampered_area_pct': surface_result.tampered_area_pct,
                'is_tampered': float(surface_result.is_tampered),
            }

        # Overall parcel decision
        result.n_surfaces_total = len(result.surfaces)
        result.n_surfaces_tampered = sum(
            1 for s in result.surfaces.values() if s.is_tampered
        )
        result.overall_tampered = result.n_surfaces_tampered > 0
        result.overall_confidence = float(np.mean([
            s.confidence for s in result.surfaces.values()
        ])) if result.surfaces else 0.0

        print(f"\n  Results:")
        print(f"    Surfaces analyzed : {result.n_surfaces_total}")
        print(f"    Surfaces tampered : {result.n_surfaces_tampered}")
        print(f"    Overall decision  : {'TAMPERED' if result.overall_tampered else 'CLEAN'}")
        print(f"    Confidence        : {result.overall_confidence:.2f}")

        return result
