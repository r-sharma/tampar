
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# Config

BASE_METRICS    = ['msssim', 'cwssim', 'ssim', 'hog', 'mae']
CONTEXT_METRICS = ['msssim', 'ssim', 'hog', 'mae', 'log_cwssim']

# XGBoost params matching predictor.py baseline (for a fair features-only comparison)
BASELINE_PARAMS = dict(n_estimators=100, max_depth=5, learning_rate=0.1,
                       eval_metric='logloss', verbosity=1, random_state=42)

# Tuned params for best performance
TUNED_PARAMS    = dict(n_estimators=600, max_depth=5, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8,
                       min_child_weight=3, gamma=0.1,
                       eval_metric='logloss', verbosity=1, random_state=42)


# Feature engineering

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['log_cwssim']       = np.log1p(df['cwssim'])
    df['cwssim_saturated'] = (df['cwssim'] >= 999_999).astype(float)
    df['cwssim_mid']       = ((df['cwssim'] >= 499_999) & (df['cwssim'] < 999_999)).astype(float)

    #  2. Composite / interaction features 
    eps = 1e-6
    df['tam_score']        = (1 - df['msssim']) * (1 - df['ssim']) * df['mae']
    df['high_sim']         = df['msssim'] * df['ssim']
    df['mae_x_hog']        = df['mae'] * df['hog']
    df['mae_per_ssim']     = df['mae']  / (df['ssim']  + eps)
    df['hog_per_ssim']     = df['hog']  / (df['ssim']  + eps)
    df['ssim_msssim_diff'] = (df['ssim'] - df['msssim']).abs()
    df['cwssim_pct']       = df['cwssim'].rank(pct=True)

    for m in CONTEXT_METRICS:
        g = df.groupby(['parcel_id', 'view'])[m]
        df[f'grp_mean_{m}'] = g.transform('mean')
        df[f'grp_std_{m}']  = g.transform('std').fillna(0)
        df[f'grp_min_{m}']  = g.transform('min')
        df[f'grp_max_{m}']  = g.transform('max')
        df[f'delta_{m}']    = df[m] - df[f'grp_mean_{m}']
        df[f'rank_{m}']     = g.transform(lambda x: x.rank(pct=True))

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    meta = {
        'cwssim',
        'Unnamed: 0', 'dataset_split', 'parcel_id', 'view',
        'gt_keypoints', 'compare_type', 'sideface_name',
        'background', 'tampering', 'tampered',
    }
    return [c for c in df.columns if c not in meta]


# Evaluation helpers

def evaluate(model, X: np.ndarray, y: np.ndarray) -> dict:
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        'accuracy':         accuracy_score(y, preds),
        'precision_binary': precision_score(y, preds, zero_division=0),
        'recall_binary':    recall_score(y, preds, zero_division=0),
        'f1_binary':        f1_score(y, preds, zero_division=0),
        'roc_auc':          roc_auc_score(y, probs),
    }


def evaluate_cv(df_raw: pd.DataFrame, params: dict, feat_cols: list,
                n_splits: int = 5) -> dict:
    from sklearn.model_selection import GroupKFold

    # Group key per row
    group_key = df_raw['parcel_id'].astype(str) + '_' + df_raw['view'].astype(str)
    unique_groups = group_key.unique()

    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []

    for tr_grp_idx, te_grp_idx in gkf.split(
            unique_groups, groups=unique_groups):
        tr_groups = set(unique_groups[tr_grp_idx])
        te_groups = set(unique_groups[te_grp_idx])

        tr_mask = group_key.isin(tr_groups).values
        te_mask = group_key.isin(te_groups).values

        df_tr = engineer_features(df_raw[tr_mask].reset_index(drop=True))
        df_te = engineer_features(df_raw[te_mask].reset_index(drop=True))

        X_tr = df_tr[feat_cols].values.astype(float)
        y_tr = df_tr['tampered'].values
        X_te = df_te[feat_cols].values.astype(float)
        y_te = df_te['tampered'].values

        m = XGBClassifier(**params)
        m.fit(X_tr, y_tr)
        fold_results.append(evaluate(m, X_te, y_te))

    df_r = pd.DataFrame(fold_results)
    return {k: (df_r[k].mean(), df_r[k].std()) for k in df_r.columns}


def print_results(label: str, metrics: dict, baseline_acc: float = 0.8166):
    acc = metrics['accuracy']
    print(f"\n  [{label}]")
    for k, v in metrics.items():
        val = v[0] if isinstance(v, tuple) else v
        std = f" +/-{v[1]:.4f}" if isinstance(v, tuple) else ""
        print(f"    {k:<20s}: {val:.4f}{std}")
    acc_val = acc[0] if isinstance(acc, tuple) else acc
    delta = acc_val - baseline_acc
    arrow = '▲' if delta >= 0 else '▼'
    print(f"\n    vs baseline ({baseline_acc:.4f}): {arrow} {abs(delta)*100:+.2f} pp")
    target = 0.85
    if acc_val >= target:
        print(f"    Target (85%):  REACHED  ({acc_val*100:.2f}%)")
    else:
        print(f"    Target (85%):  gap = {(target - acc_val)*100:.2f} pp")


# Data loading

def load_simsac(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'compare_type' in df.columns:
        df = df[df['compare_type'] == 'simsac'].copy()
    df['tampered'] = (
        df['tampered'].astype(str).str.lower()
        .map({'true': 1, 'false': 0, '1': 1, '0': 0})
        .fillna(0).astype(int)
    )
    df = df.dropna(subset=BASE_METRICS).reset_index(drop=True)
    return df


# Main

def main():
    parser = argparse.ArgumentParser(
        description='Improved XGBoost on SimSAC features — targets >85% on adversarial data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--csv', required=True,
                        help='Adversarial simscores CSV')
    parser.add_argument('--clean_csv', default=None,
                        help='(Optional) Clean simscores CSV for adversarial-training variant')
    parser.add_argument('--output_csv', default='results_simsac_v2.csv',
                        help='Output CSV path (default: results_simsac_v2.csv)')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='K-fold splits for CV experiment (default: 5)')
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    #  Load 
    print(f"\nLoading adversarial simscores: {args.csv}")
    df_adv = load_simsac(args.csv)
    n, n_t, n_c = len(df_adv), df_adv['tampered'].sum(), (df_adv['tampered']==0).sum()
    print(f"  {n} rows  (tampered={n_t}, clean={n_c})")

    #  Feature engineering 
    print("\nEngineering features …")
    df_fe = engineer_features(df_adv)
    feat_cols = get_feature_columns(df_fe)
    X = df_fe[feat_cols].values.astype(float)
    y = df_fe['tampered'].values
    print(f"  {len(feat_cols)} features  ({', '.join(feat_cols[:6])} … +{len(feat_cols)-6} more)")

    # Raw features for baseline replication
    X_raw = df_adv[BASE_METRICS].values.astype(float)

    BASELINE_ACC = 0.8166


    #  Baseline replication 
    m_base = XGBClassifier(**BASELINE_PARAMS)
    m_base.fit(X_raw, y)
    r_base = evaluate(m_base, X_raw, y)
    print_results("Baseline — raw 5 features, same params (replication)", r_base, BASELINE_ACC)

    #  v2: same params, engineered features 
    m_same = XGBClassifier(**BASELINE_PARAMS)
    m_same.fit(X, y)
    r_same = evaluate(m_same, X, y)
    print_results("v2 — 44 engineered features, same params (100 trees)", r_same, BASELINE_ACC)

    #  v2: tuned params, engineered features 
    m_tuned = XGBClassifier(**TUNED_PARAMS)
    m_tuned.fit(X, y)
    r_tuned = evaluate(m_tuned, X, y)
    print_results("v2 — 44 engineered features, tuned (600 trees)", r_tuned, BASELINE_ACC)

    print(f"CROSS-VALIDATION ({args.n_splits}-fold group-aware) — generalization estimate")
    print(f"  Groups: unique (parcel_id, view) pairs kept together in each fold")

    # Baseline CV uses raw features (no context), so simple GroupKFold suffices
    from sklearn.model_selection import GroupKFold
    group_key = (df_adv['parcel_id'].astype(str) + '_' + df_adv['view'].astype(str)).values
    gkf = GroupKFold(n_splits=args.n_splits)
    base_cv_folds = []
    for tr_i, te_i in gkf.split(X_raw, y, groups=group_key):
        mb = XGBClassifier(**BASELINE_PARAMS)
        mb.fit(X_raw[tr_i], y[tr_i])
        base_cv_folds.append(evaluate(mb, X_raw[te_i], y[te_i]))
    r_cv_base = {k: (pd.DataFrame(base_cv_folds)[k].mean(),
                     pd.DataFrame(base_cv_folds)[k].std())
                 for k in base_cv_folds[0]}
    print_results("Baseline CV (raw 5 features, group-aware)", r_cv_base, BASELINE_ACC)

    r_cv_v2 = evaluate_cv(df_adv, BASELINE_PARAMS, feat_cols, args.n_splits)
    print_results("v2 CV (44 features, same params, group-aware)", r_cv_v2, BASELINE_ACC)

    r_cv_tuned = evaluate_cv(df_adv, TUNED_PARAMS, feat_cols, args.n_splits)
    print_results("v2 CV (44 features, tuned, group-aware)", r_cv_tuned, BASELINE_ACC)

    #  Adversarial training (optional) 
    r_adv_train = None
    if args.clean_csv:
        print("ADVERSARIAL TRAINING — train on clean+adv, test on adv")
        df_clean = load_simsac(args.clean_csv)
        df_clean_fe = engineer_features(df_clean)
        # Recompute context features on the combined set for consistency
        df_combined = pd.concat([df_adv, df_clean], ignore_index=True)
        df_combined_fe = engineer_features(df_combined)
        X_comb = df_combined_fe[feat_cols].values.astype(float)
        y_comb = df_combined_fe['tampered'].values

        # Train on combined, evaluate only on adversarial portion
        adv_mask = np.zeros(len(df_combined), dtype=bool)
        adv_mask[:len(df_adv)] = True
        m_at = XGBClassifier(**TUNED_PARAMS)
        m_at.fit(X_comb, y_comb)
        r_adv_train = evaluate(m_at, X_comb[adv_mask], y_comb[adv_mask])
        print_results("v2 adversarial training (test on adv portion)", r_adv_train, BASELINE_ACC)

    #  Feature importance 
    print("TOP-15 FEATURE IMPORTANCE (tuned model, full-data fit):")
    imp = dict(zip(feat_cols, m_tuned.feature_importances_))
    for feat, score in sorted(imp.items(), key=lambda x: -x[1])[:15]:
        bar = '█' * int(score * 300)
        print(f"  {feat:<30s} {score:.4f}  {bar}")

    #  Save CSV 
    rows = [
        _row('baseline_5feat_100trees',    r_base),
        _row('v2_44feat_100trees',         r_same),
        _row('v2_44feat_600trees_tuned',   r_tuned),
        _row('v2_cv_5fold_raw',            {k: v[0] for k, v in r_cv_base.items()}),
        _row('v2_cv_5fold_44feat',         {k: v[0] for k, v in r_cv_v2.items()}),
        _row('v2_cv_5fold_44feat_tuned',   {k: v[0] for k, v in r_cv_tuned.items()}),
    ]
    if r_adv_train:
        rows.append(_row('v2_adv_training_test_on_adv', r_adv_train))
    out = pd.DataFrame(rows)
    out.to_csv(args.output_csv, index=False)
    print(f"\n Results saved  {args.output_csv}")

    #  Summary 
    print(f"  v2 same params   (44 feat, 100 trees)    : {r_same['accuracy']*100:.2f}%")
    print(f"  v2 tuned         (44 feat, 600 trees)    : {r_tuned['accuracy']*100:.2f}%")
    if r_adv_train:
        print(f"  v2 adv training  (44 feat, tuned)        : {r_adv_train['accuracy']*100:.2f}%")


def _row(predictor, metrics):
    row = {'predictor': predictor}
    row.update({k: round(v, 6) for k, v in metrics.items()})
    return row


if __name__ == '__main__':
    main()
