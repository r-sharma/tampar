#!/bin/bash

# Strategy 1 Testing Workflow - Carpet Only (Fast Iteration)
# This script demonstrates the complete workflow for testing Strategy 1
# on a single background (carpet) for faster iteration.

echo "========================================="
echo "Strategy 1: Carpet-Only Testing Workflow"
echo "========================================="

# Configuration
DATA_DIR="/content/drive/MyDrive/TAMPAR_DATA/tampar/validation"
UVMAPS_DIR="/content/drive/MyDrive/TAMPAR_DATA/tampar/uvmaps"
OUTPUT_DIR="/content/tampar/adversarial_validation_strategy1_carpet"
EPSILON=0.2
PGD_STEPS=10

echo ""
echo "Step 1: Generate adversarial images (carpet only)"
echo "=================================================="
python extension/generate_adversarial_similarity_targeted.py \
    --data_dir "$DATA_DIR" \
    --uvmaps_dir "$UVMAPS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --folders carpet \
    --attack both \
    --epsilon $EPSILON \
    --pgd_steps $PGD_STEPS

echo ""
echo "Step 2: Compute similarity scores for adversarial images"
echo "========================================================="
python src/tools/compute_similarity_scores.py \
    --data_dir "$OUTPUT_DIR" \
    --uvmaps_dir "$UVMAPS_DIR" \
    --output_csv data/misc/simscores_adversarial_carpet_strategy1.csv

echo ""
echo "Step 3: Filter scores to carpet only and create evaluation dataset"
echo "===================================================================="
python << 'EOF'
import pandas as pd

# Load all scores
df_clean = pd.read_csv('data/misc/simscores_validation.csv')
df_adv = pd.read_csv('data/misc/simscores_adversarial_carpet_strategy1.csv')

# Filter for carpet background only
df_clean_carpet = df_clean[df_clean['background'] == 'validation'].copy()
# For adversarial, filter for carpet_adv_* backgrounds
df_adv_carpet = df_adv[df_adv['background'].str.contains('carpet_adv')].copy()

# Save filtered datasets
df_clean_carpet.to_csv('data/misc/simscores_validation_carpet_only.csv', index=False)
df_adv_carpet.to_csv('data/misc/simscores_adversarial_carpet_only.csv', index=False)

print(f"Clean carpet samples: {len(df_clean_carpet)}")
print(f"Adversarial carpet samples: {len(df_adv_carpet)}")
EOF

echo ""
echo "Step 4: Evaluate attack effectiveness (train on clean carpet, test on adversarial carpet)"
echo "=========================================================================================="
python src/tools/predict_tampering_adversarial_eval.py \
    --clean_csv data/misc/simscores_validation_carpet_only.csv \
    --adversarial_csv data/misc/simscores_adversarial_carpet_only.csv \
    --output_csv results_strategy1_carpet.csv

echo ""
echo "========================================="
echo "Testing complete! Check results_strategy1_carpet.csv"
echo "========================================="
