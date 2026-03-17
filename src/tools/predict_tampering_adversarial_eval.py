
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.append(ROOT.as_posix())

import pandas as pd
import argparse

from src.tampering.compare import METRICS, CompareType
from src.tampering.evaluate import evaluate
from src.tampering.predictor import TamperingClassificator

SPLIT_STRING = "___"


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
    print(f"Total rows in CSV: {len(df)}")
    print(f"Unique IDs: {df['id'].nunique()}")

    # Check for duplicates
    duplicates = df[df.duplicated(subset=['id', 'compare_type'], keep=False)]
    if len(duplicates) > 0:
        print(f"WARNING: Found {len(duplicates)} duplicate rows! Keeping first occurrence.")
        df = df.drop_duplicates(subset=['id', 'compare_type'], keep='first')

    df_pivot = df.pivot_table(
        index="id",
        columns="compare_type",
        values=METRICS,
        aggfunc='first',
    )
    df_pivot.columns = [
        "score_{}_{}".format(col, method) for col, method in df_pivot.columns
    ]
    df_pivot = df_pivot.reset_index()

    df_final = pd.merge(
        df[["tampered", "tampering", "dataset_split", "gt_keypoints", "id"]].drop_duplicates(subset=['id']),
        df_pivot,
        on="id",
    )
    df_final["tampering"] = df_final["tampering"].fillna("")
    df_final.fillna(-1, inplace=True)
    return df_final


def get_data(df_input: pd.DataFrame, gt_keypoints: bool = False):
    if gt_keypoints:
        data = df_input[df_input["gt_keypoints"] == True]
    else:
        data = df_input[df_input["gt_keypoints"] == False]
    return data


def train_and_evaluate_predictor(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    gt_keypoints: bool = False,
    predictor_type: str = "simple_threshold",
) -> pd.DataFrame:
    SCORES = [n for n in df_train.columns if n.startswith("score")]

    # Get clean train and adversarial test data
    data_train = get_data(df_train, gt_keypoints=gt_keypoints)
    data_test = get_data(df_test, gt_keypoints=gt_keypoints)

    print(f"Training on CLEAN data: {len(data_train)} samples")
    print(f"Testing on ADVERSARIAL data: {len(data_test)} samples")
    print(f"GT Keypoints: {gt_keypoints}")

    results_performance = []
    for compare_types in [[t] for t in CompareType.SELECTION()] + [
        CompareType.SELECTION()
    ]:
        scores = [s for s in SCORES if s.split("_")[-1] in compare_types]
        scores = [s for s in scores if "_".join(s.split("_")[1:-1]) in METRICS]
        if len(scores) == 0:
            continue

        print(f"Evaluating compare_types: {compare_types}")
        print(f"Using scores: {scores[:3]}... ({len(scores)} total)")

        predictor = TamperingClassificator(predictor_type)

        # Train on CLEAN data
        X_train = data_train[scores].to_numpy().astype(float)
        y_train = data_train["tampered"].to_numpy().astype(int)
        ids_train = data_train["id"].to_numpy()

        predictor.set_data(X_train, y_train, ids_train)
        predictor.feature_names = [s.replace("score_", "") for s in scores]

        # Disable test split - we're doing manual train/test
        predictor.test_split_size = 0
        model, train_metrics, _ = predictor.train()

        # Test on ADVERSARIAL data
        X_test = data_test[scores].to_numpy().astype(float)
        y_test = data_test["tampered"].to_numpy().astype(int)
        ids_test = data_test["id"].to_numpy()

        test_metrics = evaluate(model, X_test, y_test)

        print(f"  Train accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  Test accuracy (adversarial): {test_metrics['accuracy']:.4f}")
        print(f"  Accuracy drop: {(train_metrics['accuracy'] - test_metrics['accuracy']):.4f}")

        result_dict = {
            "predictor": predictor_type,
            "compare_types": ", ".join(compare_types),
            "scores": ", ".join(
                set(["_".join(s.split("_")[1:-1]) for s in scores])
            ),
            # Training metrics (on clean data)
            "train_accuracy": train_metrics['accuracy'],
            "train_precision_binary": train_metrics.get('precision_binary', 0),
            "train_recall_binary": train_metrics.get('recall_binary', 0),
            "train_f1_binary": train_metrics.get('f1_binary', 0),
            "train_roc_auc": train_metrics.get('roc_auc', 0),
            # Test metrics (on adversarial data)
            "test_accuracy": test_metrics['accuracy'],
            "test_precision_binary": test_metrics.get('precision_binary', 0),
            "test_recall_binary": test_metrics.get('recall_binary', 0),
            "test_f1_binary": test_metrics.get('f1_binary', 0),
            "test_roc_auc": test_metrics.get('roc_auc', 0),
            # Accuracy drop
            "accuracy_drop": train_metrics['accuracy'] - test_metrics['accuracy'],
        }

        # Add feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            result_dict["feature_importance"] = {
                name: value
                for name, value in zip(
                    predictor.feature_names,
                    model.feature_importances_,
                )
                if value > 0
            }
        else:
            result_dict["feature_importance"] = "N/A (ensemble model)"

        results_performance.append(result_dict)

    df_results = pd.DataFrame(results_performance)
    return df_results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate predictor robustness against adversarial attacks'
    )
    parser.add_argument(
        '--clean_csv',
        type=str,
        default='/content/drive/MyDrive/TAMPAR_DATA/simscores_validation_clean.csv',
        help='Path to clean similarity scores CSV'
    )
    parser.add_argument(
        '--adversarial_csv',
        type=str,
        default='/content/drive/MyDrive/TAMPAR_DATA/simscores_validation_adversarial.csv',
        help='Path to adversarial similarity scores CSV'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='adversarial_evaluation_results.csv',
        help='Output CSV file for results'
    )
    parser.add_argument(
        '--gt_keypoints',
        action='store_true',
        help='Use ground truth keypoints instead of predicted'
    )
    parser.add_argument(
        '--exclude_base',
        action='store_true',
        help='Exclude base folder (noisy labels) from evaluation'
    )
    parser.add_argument(
        '--predictor_type',
        type=str,
        default='all',
        choices=['simple_threshold', 'decision_tree', 'random_forest', 'xgboost', 'ensemble', 'all'],
        help='Type of predictor to train. Use "all" to compare all classifiers (default: all)'
    )

    args = parser.parse_args()

    print("Loading clean training data...")
    df_clean = load_results(Path(args.clean_csv))

    # Filter out base folder if requested
    if args.exclude_base:
        print("Excluding base folder from clean data...")
        df_clean = df_clean[~df_clean['view'].str.contains('/base/')]
        print(f"  Remaining samples: {len(df_clean)}")

    df_train = create_pivot(df_clean)

    print("\nLoading adversarial test data...")
    df_adversarial = load_results(Path(args.adversarial_csv))

    # Filter out base folder if requested (handles both /base/ and base_adv_*)
    if args.exclude_base:
        print("Excluding base folder from adversarial data...")
        # Match '/base/' in view path
        base_mask = df_adversarial['view'].str.contains('/base/')
        # Also check for 'background' column if it exists
        if 'background' in df_adversarial.columns:
            base_mask = base_mask | df_adversarial['background'].str.contains(r'base_adv_', na=False)
        df_adversarial = df_adversarial[~base_mask]
        print(f"  Remaining samples: {len(df_adversarial)}")

    df_test = create_pivot(df_adversarial)

    # Determine which predictors to run
    if args.predictor_type == 'all':
        predictor_types = ['simple_threshold', 'decision_tree', 'random_forest', 'xgboost', 'ensemble']
    else:
        predictor_types = [args.predictor_type]

    # Run each predictor and collect results
    all_results = []
    for predictor_type in predictor_types:
        print(f"Training with predictor: {predictor_type.upper()}")

        try:
            df_results = train_and_evaluate_predictor(
                df_train,
                df_test,
                gt_keypoints=args.gt_keypoints,
                predictor_type=predictor_type
            )
            all_results.append(df_results)
        except Exception as e:
            print(f"Error with {predictor_type}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Combine all results
    if len(all_results) > 0:
        df_combined = pd.concat(all_results, ignore_index=True)
        print(f"\nSaving results to {args.output_csv}...")
        df_combined.to_csv(args.output_csv, index=False)

        print("\n" + "="*70)
        print("CLASSIFIER COMPARISON SUMMARY")

        # Show best results for each classifier
        agg_dict = {
            'train_accuracy': 'max',
            'test_accuracy': 'max',
            'accuracy_drop': 'min'
        }

        # Add optional metrics if they exist
        if 'train_f1_binary' in df_combined.columns:
            agg_dict['train_f1_binary'] = 'max'
        if 'test_f1_binary' in df_combined.columns:
            agg_dict['test_f1_binary'] = 'max'
        if 'train_roc_auc' in df_combined.columns:
            agg_dict['train_roc_auc'] = 'max'
        if 'test_roc_auc' in df_combined.columns:
            agg_dict['test_roc_auc'] = 'max'

        summary = df_combined.groupby('predictor').agg(agg_dict).round(4)
        print(summary.to_string())

        print("\n" + "="*70)
        print("BEST PERFORMING METHODS (by test accuracy)")
        df_sorted = df_combined.sort_values('test_accuracy', ascending=False)
        print(df_sorted[['predictor', 'compare_types', 'train_accuracy', 'test_accuracy', 'accuracy_drop']].head(10).to_string(index=False))
        print("\nDone!")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()
