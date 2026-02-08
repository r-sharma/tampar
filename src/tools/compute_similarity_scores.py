import concurrent.futures
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.append(ROOT.as_posix())


import cv2
import numpy as np
import pandas as pd
import tqdm

from src.tampering.compare import CompareType, compute_uvmap_similarity

IMAGE_ROOT = ROOT / "data" / "tampar_sample"
UVMAP_DIR = IMAGE_ROOT / "uvmaps"
OUT_IMAGES = ROOT / "out_imgs"
NUM_WORKERS = 4  # Use 4 parallel workers for faster processing


def load_tampering_mapping():
    tampering_mapping = pd.read_csv(
        ROOT / "src" / "tampering" / "tampering_mapping.csv"
    )
    tampering_mapping.fillna("", inplace=True)
    tampering_mapping = {d["id"]: d for d in tampering_mapping.to_dict("records")}
    return tampering_mapping


TAMPERING_MAPPING = load_tampering_mapping()


def compute_sidesurface_similarity_scores(
    ref_image_path: Path, gt_uvmap: np.ndarray, parcel_id: int, simsac_ckpt_path: str = None
):
    rel_path = ref_image_path.relative_to(IMAGE_ROOT)
    reference_image = cv2.imread(ref_image_path.as_posix())
    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

    results = []
    for compare_type in CompareType.SELECTION():
        similarities = compute_uvmap_similarity(
            gt_uvmap,
            reference_image,
            output_path=None,  # only if visualize False
            compare_type=compare_type,
            visualize=False,
            simsac_ckpt_path=simsac_ckpt_path,
        )
        for sideface_name, scores in similarities.items():
            results.append(
                {
                    "dataset_split": rel_path.parts[0],
                    "parcel_id": parcel_id,
                    "view": rel_path.as_posix(),
                    "gt_keypoints": "uvmap_gt" in ref_image_path.name,
                    "compare_type": compare_type,
                    "sideface_name": sideface_name,
                    "background": ref_image_path.parent.name,
                    "tampering": TAMPERING_MAPPING[parcel_id][sideface_name],
                    "tampered": TAMPERING_MAPPING[parcel_id][sideface_name] != "",
                    **scores,
                }
            )
    return results


def compute_parcel_similitary_scores(parcel_id: int, image_path: Path, parallel=True, simsac_ckpt_path: str = None, num_workers: int = None):
    parcel_results = []
    gt_uvmap_path = UVMAP_DIR / f"id_{str(parcel_id).zfill(2)}_uvmap.png"
    if not gt_uvmap_path.exists():
        return None
    gt_uvmap = cv2.imread(gt_uvmap_path.as_posix())
    gt_uvmap = cv2.cvtColor(gt_uvmap, cv2.COLOR_BGR2RGB)
    references_image_paths = [
        f
        for f in image_path.rglob(f"id_{str(parcel_id).zfill(2)}_*_uvmap_*.png")
        if (f.parent.name not in ["uvmaps", "base"])
    ]
    if len(references_image_paths) == 0:
        return None
    print(f"Parcel ID: {parcel_id} ({len(references_image_paths)})")
    futures = []
    workers = num_workers if num_workers is not None else NUM_WORKERS
    with tqdm.tqdm(total=len(references_image_paths)) as pbar:
        with ThreadPoolExecutor(max_workers=workers) as tp_executor:
            for ref_image_path in references_image_paths:
                if parallel:
                    future = tp_executor.submit(
                        compute_sidesurface_similarity_scores,
                        ref_image_path,
                        gt_uvmap,
                        parcel_id,
                        simsac_ckpt_path,
                    )
                    futures.append(future)
                else:
                    results = compute_sidesurface_similarity_scores(
                        ref_image_path, gt_uvmap, parcel_id, simsac_ckpt_path
                    )
                    parcel_results.extend(results)
                    pbar.update(1)
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    parcel_results.extend(res)
                    pbar.update(1)
                except Exception as e:
                    print(f"Error: {e}")
    return parcel_results


def main(parallel=False, simsac_ckpt_path: str = None, num_workers: int = None) -> pd.DataFrame:
    """
    Compute similarity scores using SimSAC.

    Args:
        parallel: Use parallel processing
        simsac_ckpt_path: Path to SimSAC checkpoint file (default: synthetic.pth)
        num_workers: Number of parallel workers (default: 4)
    """
    if simsac_ckpt_path:
        print(f"Using custom SimSAC checkpoint: {simsac_ckpt_path}")
    else:
        print("Using default SimSAC checkpoint: synthetic.pth")

    results = []
    for folder_name in ["validation"]:  # , "test"]:
        input_folder = IMAGE_ROOT / folder_name
        output_path = OUT_IMAGES / input_folder.name
        output_path.mkdir(exist_ok=True, parents=True)
        for parcel_id in range(30):
            parcel_results = compute_parcel_similitary_scores(
                parcel_id, input_folder, parallel=parallel, simsac_ckpt_path=simsac_ckpt_path, num_workers=num_workers
            )
            if parcel_results is not None:
                results.extend(parcel_results)

        df = pd.DataFrame(results)
        df.to_csv(OUT_IMAGES / f"simscores_{folder_name}.csv")
    df = pd.DataFrame(results)
    df.to_csv(OUT_IMAGES / "simscores_final.csv")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute similarity scores using SimSAC"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to SimSAC checkpoint file (e.g., phase2_best.pth). If not provided, uses default synthetic.pth'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        default=False,
        help='Enable parallel processing'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4). More workers = faster but more memory'
    )

    args = parser.parse_args()
    main(parallel=args.parallel, simsac_ckpt_path=args.checkpoint, num_workers=args.num_workers)
