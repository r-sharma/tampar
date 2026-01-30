"""
Adversarial Evaluation for Tampering Detection

Evaluates tampering predictor robustness against adversarial attacks.
Trains on clean validation data, tests on adversarial validation data.

Usage:
    python src/tools/predict_tampering_adversarial_eval.py \
        --clean_csv /content/drive/MyDrive/TAMPAR_DATA/simscores_validation_clean.csv \
        --adversarial_csv /content/drive/MyDrive/TAMPAR_DATA/simscores_validation_adversarial.csv \
        --output_csv adversarial_evaluation_results.csv
"""

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
        aggfunc='first',  # Use first value instead of joining with comma
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
    """Get data split by gt_keypoints."""
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
    """
    Train predictor on clean data, evaluate on adversarial data.

    Args:
        df_train: Clean training data
        df_test: Adversarial test data
        gt_keypoints: Whether to use ground truth keypoints
        predictor_type: Type of predictor

    Returns:
        DataFrame with evaluation results
    """
    SCORES = [n for n in df_train.columns if n.startswith("score")]

    # Get clean train and adversarial test data
    data_train = get_data(df_train, gt_keypoints=gt_keypoints)
    data_test = get_data(df_test, gt_keypoints=gt_keypoints)

    print(f"\n{'='*60}")
    print(f"Training on CLEAN data: {len(data_train)} samples")
    print(f"Testing on ADVERSARIAL data: {len(data_test)} samples")
    print(f"GT Keypoints: {gt_keypoints}")
    print(f"{'='*60}\n")

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

        results_performance.append(
            {
                "predictor": predictor_type,
                "compare_types": ", ".join(compare_types),
                "scores": ", ".join(
                    set(["_".join(s.split("_")[1:-1]) for s in scores])
                ),
                "feature_importance": {
                    name: value
                    for name, value in zip(
                        predictor.feature_names,
                        model.feature_importances_,
                    )
                    if value > 0
                },
                "train_accuracy": train_metrics['accuracy'],
                "train_precision": train_metrics['precision_binary'],
                "train_recall": train_metrics['recall_binary'],
                "train_f1": train_metrics['f1_binary'],
                "test_accuracy": test_metrics['accuracy'],
                "test_precision": test_metrics['precision_binary'],
                "test_recall": test_metrics['recall_binary'],
                "test_f1": test_metrics['f1_binary'],
                "accuracy_drop": train_metrics['accuracy'] - test_metrics['accuracy'],
            }
        )

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

    # Filter out base folder if requested
    if args.exclude_base:
        print("Excluding base folder from adversarial data...")
        df_adversarial = df_adversarial[~df_adversarial['view'].str.contains('/base/')]
        print(f"  Remaining samples: {len(df_adversarial)}")

    df_test = create_pivot(df_adversarial)

    print("\nTraining on clean, testing on adversarial...")
    df_results = train_and_evaluate_predictor(
        df_train,
        df_test,
        gt_keypoints=args.gt_keypoints
    )

    print(f"\nSaving results to {args.output_csv}...")
    df_results.to_csv(args.output_csv, index=False)

    print("\n" + "="*60)
    print("SUMMARY: Best performing methods")
    print("="*60)
    df_sorted = df_results.sort_values('test_accuracy', ascending=False)
    print(df_sorted[['compare_types', 'train_accuracy', 'test_accuracy', 'accuracy_drop']].head(10))
    print("\nDone!")


if __name__ == "__main__":
    main()
