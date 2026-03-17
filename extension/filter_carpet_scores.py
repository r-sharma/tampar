
import argparse
import pandas as pd


def filter_by_background(df, background_name, is_adversarial=False):
    if is_adversarial:
        # Adversarial backgrounds are named like "carpet_adv_fgsm", "carpet_adv_pgd"
        mask = df['background'].str.contains(f'{background_name}_adv')
        df_filtered = df[mask].copy()
    else:
        mask = df['view'].str.contains(f'/{background_name}/')
        df_filtered = df[mask].copy()

    return df_filtered


def main():
    parser = argparse.ArgumentParser(
        description="Filter similarity scores for a specific background"
    )

    parser.add_argument('--clean_csv', type=str, required=True,
                       help='Path to clean similarity scores CSV')
    parser.add_argument('--adversarial_csv', type=str, required=True,
                       help='Path to adversarial similarity scores CSV')
    parser.add_argument('--background', type=str, required=True,
                       help='Background to filter (e.g., carpet, table, gravel)')
    parser.add_argument('--output_clean', type=str, required=True,
                       help='Output path for filtered clean scores')
    parser.add_argument('--output_adversarial', type=str, required=True,
                       help='Output path for filtered adversarial scores')

    args = parser.parse_args()

    print(f"Filtering for background: {args.background}")

    # Load dataframes
    print(f"\nLoading clean scores from: {args.clean_csv}")
    df_clean = pd.read_csv(args.clean_csv)
    print(f"  Total rows: {len(df_clean)}")

    print(f"\nLoading adversarial scores from: {args.adversarial_csv}")
    df_adv = pd.read_csv(args.adversarial_csv)
    print(f"  Total rows: {len(df_adv)}")

    # Filter by background
    print(f"\nFiltering for background: {args.background}")

    df_clean_filtered = filter_by_background(df_clean, args.background, is_adversarial=False)
    print(f"  Clean filtered: {len(df_clean_filtered)} rows")

    df_adv_filtered = filter_by_background(df_adv, args.background, is_adversarial=True)
    print(f"  Adversarial filtered: {len(df_adv_filtered)} rows")

    # Save filtered datasets
    print(f"\nSaving filtered datasets")
    df_clean_filtered.to_csv(args.output_clean, index=False)
    print(f"  Clean: {args.output_clean}")

    df_adv_filtered.to_csv(args.output_adversarial, index=False)
    print(f"  Adversarial: {args.output_adversarial}")

    # Show statistics
    print(f"SUMMARY")
    print(f"Background: {args.background}")
    print(f"Clean samples: {len(df_clean_filtered)}")
    print(f"Adversarial samples: {len(df_adv_filtered)}")

    if 'gt_keypoints' in df_clean_filtered.columns:
        print(f"\nClean - gt_keypoints distribution:")
        print(df_clean_filtered['gt_keypoints'].value_counts())

        print(f"\nAdversarial - gt_keypoints distribution:")
        print(df_adv_filtered['gt_keypoints'].value_counts())

    if 'tampered' in df_clean_filtered.columns:
        print(f"\nClean - tampering distribution:")
        print(df_clean_filtered['tampered'].value_counts())

        print(f"\nAdversarial - tampering distribution:")
        print(df_adv_filtered['tampered'].value_counts())

    print(f"\n Filtering complete!")


if __name__ == "__main__":
    main()
