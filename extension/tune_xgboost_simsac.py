
import sys
import json
import argparse
from pathlib import Path
from itertools import product

ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

from src.tampering.compare import METRICS, CompareType

SPLIT_STRING = "___"

# --------------------------------------------------------------------------- #
# Curated candidate configs (quick mode, no GridSearchCV)
# Designed to explore: lower depth (regularise), more trees, lower lr,
# and different regularisation levels.
# --------------------------------------------------------------------------- #
CANDIDATE_CONFIGS = [
    # Baseline (current default in predictor.py)
    dict(n_estimators=100, max_depth=5, learning_rate=0.1,
         subsample=0.8, colsample_bytree=0.8, reg_alpha=0, reg_lambda=1),
    # Shallower trees — reduce overfitting to clean distribution
    dict(n_estimators=200, max_depth=3, learning_rate=0.05,
         subsample=0.8, colsample_bytree=0.8, reg_alpha=0, reg_lambda=1),
    dict(n_estimators=300, max_depth=3, learning_rate=0.05,
         subsample=0.8, colsample_bytree=0.8, reg_alpha=0, reg_lambda=2),
    dict(n_estimators=300, max_depth=3, learning_rate=0.03,
         subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1),
    dict(n_estimators=500, max_depth=3, learning_rate=0.02,
         subsample=0.8, colsample_bytree=0.8, reg_alpha=0, reg_lambda=1),
    # Medium depth
    dict(n_estimators=200, max_depth=4, learning_rate=0.05,
         subsample=0.8, colsample_bytree=0.8, reg_alpha=0, reg_lambda=1),
    dict(n_estimators=200, max_depth=4, learning_rate=0.05,
         subsample=0.7, colsample_bytree=0.7, reg_alpha=0, reg_lambda=2),
    dict(n_estimators=300, max_depth=4, learning_rate=0.03,
         subsample=0.9, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1),
    # L1 regularisation (feature selection effect)
    dict(n_estimators=200, max_depth=4, learning_rate=0.05,
         subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1),
    dict(n_estimators=300, max_depth=3, learning_rate=0.05,
         subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1),
    # Higher subsampling
    dict(n_estimators=200, max_depth=4, learning_rate=0.05,
         subsample=0.6, colsample_bytree=0.6, reg_alpha=0, reg_lambda=1),
    # Depth 6 (same as default but more trees / lower lr)
    dict(n_estimators=300, max_depth=5, learning_rate=0.05,
         subsample=0.8, colsample_bytree=0.8, reg_alpha=0, reg_lambda=1),
    dict(n_estimators=500, max_depth=5, learning_rate=0.02,
         subsample=0.8, colsample_bytree=0.8, reg_alpha=0, reg_lambda=1),
    # min_child_weight (min samples per leaf — reduces overfitting)
    dict(n_estimators=200, max_depth=4, learning_rate=0.05,
         subsample=0.8, colsample_bytree=0.8, reg_alpha=0, reg_lambda=1,
         min_child_weight=3),
    dict(n_estimators=200, max_depth=4, learning_rate=0.05,
         subsample=0.8, colsample_bytree=0.8, reg_alpha=0, reg_lambda=1,
         min_child_weight=5),
    dict(n_estimators=300, max_depth=3, learning_rate=0.05,
         subsample=0.8, colsample_bytree=0.8, reg_alpha=0, reg_lambda=2,
         min_child_weight=3),
    # gamma (minimum loss reduction to split)
    dict(n_estimators=200, max_depth=4, learning_rate=0.05,
         subsample=0.8, colsample_bytree=0.8, gamma=0.1, reg_lambda=1),
    dict(n_estimators=200, max_depth=4, learning_rate=0.05,
         subsample=0.8, colsample_bytree=0.8, gamma=0.5, reg_lambda=1),
    # Combined regularisation
    dict(n_estimators=300, max_depth=4, learning_rate=0.03,
         subsample=0.8, colsample_bytree=0.8,
         min_child_weight=3, gamma=0.1, reg_alpha=0.1, reg_lambda=2),
    dict(n_estimators=500, max_depth=3, learning_rate=0.02,
         subsample=0.9, colsample_bytree=0.9,
         min_child_weight=3, gamma=0.0, reg_alpha=0, reg_lambda=2),
]

# GridSearchCV parameter grid (smaller but still meaningful)
GRID_SEARCH_PARAMS = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.02, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [1, 2],
    "min_child_weight": [1, 3],
}


# --------------------------------------------------------------------------- #
# Data loading (mirrors predict_tampering_adversarial_eval.py)
# --------------------------------------------------------------------------- #

def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.reset_index(inplace=True)
    df["id"] = (
        df["view"]
        + SPLIT_STRING
        + df["sideface_name"]
        + SPLIT_STRING
        + df["gt_keypoints"].astype(str)
    )
    return df


def create_pivot(df: pd.DataFrame) -> pd.DataFrame:
    duplicates = df[df.duplicated(subset=["id", "compare_type"], keep=False)]
    if len(duplicates) > 0:
        df = df.drop_duplicates(subset=["id", "compare_type"], keep="first")

    df_pivot = df.pivot_table(
        index="id",
        columns="compare_type",
        values=METRICS,
        aggfunc="first",
    )
    df_pivot.columns = [
        "score_{}_{}".format(col, method) for col, method in df_pivot.columns
    ]
    df_pivot = df_pivot.reset_index()

    df_final = pd.merge(
        df[["tampered", "tampering", "dataset_split", "gt_keypoints", "id"]].drop_duplicates(subset=["id"]),
        df_pivot,
        on="id",
    )
    df_final["tampering"] = df_final["tampering"].fillna("")
    df_final.fillna(-1, inplace=True)
    return df_final


def get_features(df: pd.DataFrame, compare_types: list[str]) -> list[str]:
    all_scores = [n for n in df.columns if n.startswith("score")]
    scores = [s for s in all_scores if s.split("_")[-1] in compare_types]
    scores = [s for s in scores if "_".join(s.split("_")[1:-1]) in METRICS]
    return scores


def build_xy(df: pd.DataFrame, feature_cols: list[str], gt_keypoints: bool = False):
    data = df[df["gt_keypoints"] == gt_keypoints]
    X = data[feature_cols].to_numpy().astype(float)
    y = data["tampered"].to_numpy().astype(int)
    return X, y


def make_base_xgb(extra_params: dict | None = None) -> XGBClassifier:
    params = dict(
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    if extra_params:
        params.update(extra_params)
    return XGBClassifier(**params)


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #

def cv_accuracy(model, X, y, n_splits: int = 5) -> float:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    return float(scores.mean())


def adv_accuracy(model, X_train, y_train, X_adv, y_adv) -> float:
    model.fit(X_train, y_train)
    preds = model.predict(X_adv)
    return float(accuracy_score(y_adv, preds))


# --------------------------------------------------------------------------- #
# Quick candidate search
# --------------------------------------------------------------------------- #

def candidate_search(X_clean, y_clean, X_adv, y_adv, n_cv_splits: int = 5):
    print(f"Candidate Search  ({len(CANDIDATE_CONFIGS)} configs, {n_cv_splits}-fold CV on clean)")

    rows = []
    for i, params in enumerate(CANDIDATE_CONFIGS):
        model = make_base_xgb(params)
        cv_acc = cv_accuracy(model, X_clean, y_clean, n_splits=n_cv_splits)
        adv_acc = adv_accuracy(make_base_xgb(params), X_clean, y_clean, X_adv, y_adv)
        label = f"cfg{i:02d}"
        row = {"config": label, "cv_clean": cv_acc, "adv_acc": adv_acc}
        row.update(params)
        rows.append(row)
        print(f"  [{i+1:2d}/{len(CANDIDATE_CONFIGS)}] cv={cv_acc:.4f}  adv={adv_acc:.4f}  {params}")

    df = pd.DataFrame(rows).sort_values("adv_acc", ascending=False)
    return df


# --------------------------------------------------------------------------- #
# GridSearchCV
# --------------------------------------------------------------------------- #

def grid_search(X_clean, y_clean, X_adv, y_adv, n_cv_splits: int = 5):
    total = 1
    for v in GRID_SEARCH_PARAMS.values():
        total *= len(v)
    print(f"GridSearchCV  ({total} configurations, {n_cv_splits}-fold CV on clean)")
    print("This may take a while...")

    base = make_base_xgb()
    gs = GridSearchCV(
        base,
        GRID_SEARCH_PARAMS,
        cv=StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    gs.fit(X_clean, y_clean)

    best_params = gs.best_params_
    best_cv = gs.best_score_
    print(f"\nBest CV accuracy (clean): {best_cv:.4f}")
    print(f"Best params: {best_params}")

    # Evaluate on adversarial
    best_model = make_base_xgb(best_params)
    adv_acc = adv_accuracy(best_model, X_clean, y_clean, X_adv, y_adv)
    print(f"Adversarial accuracy with best params: {adv_acc:.4f}")

    # Build results DataFrame from CV results
    cv_results = pd.DataFrame(gs.cv_results_)
    top = (
        cv_results[["params", "mean_test_score", "rank_test_score"]]
        .sort_values("mean_test_score", ascending=False)
        .head(10)
    )
    print("\nTop-10 by CV accuracy (clean):")
    for _, r in top.iterrows():
        print(f"  cv={r['mean_test_score']:.4f}  {r['params']}")

    return best_params, best_cv, adv_acc


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Tune XGBoost hyperparameters for tampering detection"
    )
    parser.add_argument("--clean_csv", type=str, required=True,
                        help="Path to clean similarity scores CSV")
    parser.add_argument("--adversarial_csv", type=str, required=True,
                        help="Path to adversarial similarity scores CSV")
    parser.add_argument("--compare_type", type=str, default="simsac",
                        help="Compare type to tune on. Use 'all' for all types (default: simsac)")
    parser.add_argument("--gt_keypoints", action="store_true",
                        help="Use ground truth keypoints subset")
    parser.add_argument("--exclude_base", action="store_true",
                        help="Exclude base folder from evaluation")
    parser.add_argument("--grid_search", action="store_true",
                        help="Run full GridSearchCV instead of curated candidate search")
    parser.add_argument("--cv_splits", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save best params to this JSON file")
    args = parser.parse_args()

    # Resolve compare types
    if args.compare_type == "all":
        compare_types = list(CompareType.SELECTION())
    else:
        compare_types = [args.compare_type]
    print(f"Compare type(s): {compare_types}")

    # Load data
    print("\nLoading clean CSV...")
    df_clean_raw = load_results(Path(args.clean_csv))
    if args.exclude_base:
        df_clean_raw = df_clean_raw[~df_clean_raw["view"].str.contains("/base/")]
        print(f"  After exclude_base: {len(df_clean_raw)} rows")
    df_clean = create_pivot(df_clean_raw)
    print(f"  {len(df_clean)} samples, {df_clean['tampered'].sum()} tampered")

    print("\nLoading adversarial CSV...")
    df_adv_raw = load_results(Path(args.adversarial_csv))
    if args.exclude_base:
        base_mask = df_adv_raw["view"].str.contains("/base/")
        if "background" in df_adv_raw.columns:
            base_mask = base_mask | df_adv_raw["background"].str.contains(r"base_adv_", na=False)
        df_adv_raw = df_adv_raw[~base_mask]
        print(f"  After exclude_base: {len(df_adv_raw)} rows")
    df_adv = create_pivot(df_adv_raw)
    print(f"  {len(df_adv)} samples, {df_adv['tampered'].sum()} tampered")

    # Build feature matrices
    feature_cols = get_features(df_clean, compare_types)
    if len(feature_cols) == 0:
        raise ValueError(f"No feature columns found for compare_type={compare_types}. "
                         f"Available columns: {[c for c in df_clean.columns if c.startswith('score')]}")
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

    X_clean, y_clean = build_xy(df_clean, feature_cols, gt_keypoints=args.gt_keypoints)
    X_adv,   y_adv   = build_xy(df_adv,   feature_cols, gt_keypoints=args.gt_keypoints)
    print(f"\nClean  X: {X_clean.shape}  tampered={y_clean.sum()}")
    print(f"Adv    X: {X_adv.shape}    tampered={y_adv.sum()}")

    # Default XGBoost accuracy (baseline)
    baseline_params = dict(n_estimators=100, max_depth=5, learning_rate=0.1,
                           subsample=0.8, colsample_bytree=0.8)
    baseline_adv = adv_accuracy(make_base_xgb(baseline_params),
                                X_clean, y_clean, X_adv, y_adv)
    baseline_cv  = cv_accuracy(make_base_xgb(baseline_params), X_clean, y_clean,
                               n_splits=args.cv_splits)
    print(f"\nBaseline (predictor.py defaults):")
    print(f"  CV accuracy (clean):  {baseline_cv:.4f}  ({baseline_cv*100:.1f}%)")
    print(f"  Adversarial accuracy: {baseline_adv:.4f}  ({baseline_adv*100:.1f}%)")

    best_params = baseline_params
    best_adv_acc = baseline_adv

    if args.grid_search:
        best_params_found, best_cv, best_adv = grid_search(
            X_clean, y_clean, X_adv, y_adv, n_cv_splits=args.cv_splits
        )
        if best_adv > best_adv_acc:
            best_params = best_params_found
            best_adv_acc = best_adv
    else:
        df_results = candidate_search(X_clean, y_clean, X_adv, y_adv,
                                      n_cv_splits=args.cv_splits)
        print("Results sorted by adversarial accuracy")
        display_cols = ["config", "cv_clean", "adv_acc",
                        "n_estimators", "max_depth", "learning_rate",
                        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]
        display_cols = [c for c in display_cols if c in df_results.columns]
        print(df_results[display_cols].to_string(index=False))

        best_row = df_results.iloc[0]
        best_params = {
            k: best_row[k]
            for k in best_row.index
            if k not in ("config", "cv_clean", "adv_acc")
            and not pd.isna(best_row[k])
        }
        # Convert numpy types to Python native for JSON serialisability
        best_params = {k: (int(v) if isinstance(v, (np.integer,)) else
                           float(v) if isinstance(v, (np.floating,)) else v)
                       for k, v in best_params.items()}
        best_adv_acc = float(best_row["adv_acc"])

    print("BEST RESULT")
    print(f"  Adversarial accuracy: {best_adv_acc:.4f}  ({best_adv_acc*100:.1f}%)")
    print(f"  Delta vs baseline:    {best_adv_acc - baseline_adv:+.4f}  "
          f"({(best_adv_acc - baseline_adv)*100:+.1f}pp)")
    print(f"  Best params: {best_params}")

    if args.output_json:
        out = {
            "compare_type": args.compare_type,
            "best_adversarial_accuracy": best_adv_acc,
            "baseline_adversarial_accuracy": baseline_adv,
            "delta_pp": round((best_adv_acc - baseline_adv) * 100, 2),
            "best_params": best_params,
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved best params to: {args.output_json}")
        print("To use these params, pass them to TamperingClassificator:")
        print(f"  predictor = TamperingClassificator('xgboost', model_parameters={best_params})")


if __name__ == "__main__":
    main()
