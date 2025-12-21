#!/bin/bash
# cs336_alignment/scripts/run_ei_experiments.sh

# Exit on error
set -e

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Configuration
# Dataset path (train.jsonl for EI as per instructions)
DATA_PATH="data/MATH/train.jsonl"
EVAL_PATH="data/MATH/validation.jsonl"

# Base output directory
OUTPUT_BASE="/gpfs/radev/home/ax46/scratch/A5/ei"

echo "Using training data from: $DATA_PATH"
echo "Using eval data from: $EVAL_PATH"
echo "Outputting experiments to: $OUTPUT_BASE"

# ==========================================
# 1. Hyperparameter Tuning (G and Epochs)
# ==========================================
# Goal: Find the best configuration for rollouts and epochs.
# "Vary the number of rollouts G per question and the number of epochs used in the SFT step"
# We fix ei_batch_size = 512 for these tuning runs.

echo "--- Starting Hyperparameter Tuning (G and SFT Epochs) ---"

for G in 32 64 128; do
    for E in 1 3 5; do
    echo "Tuning Run: G=$G, Epochs=$E"
    uv run cs336_alignment/scripts/run_ei.py \
        --train-data-path "$DATA_PATH" \
        --eval-data-path "$EVAL_PATH" \
        --g $G \
        --sft-epochs $E \
        --ei-batch-size 512 \
        --output-dir-base "$OUTPUT_BASE"
done

# ==========================================
# SET YOUR BEST HYPERPARAMETERS HERE
# ==========================================
# Replace these with the best configuration found above.
# BEST_G=32
# BEST_EPOCHS=1

# ==========================================
# 2. Batch Size Scaling Experiments
# ==========================================
# "Vary the batch size for each expert iteration step... in {512, 1024, 2048}"
# Use the BEST_G and BEST_EPOCHS from above.

# echo "--- Starting Batch Size Scaling Experiments (G=$BEST_G, Epochs=$BEST_EPOCHS) ---"

# for BS in 512 1024 2048; do
#     echo "Running EI with Batch Size $BS..."
#     uv run cs336_alignment/scripts/run_ei.py \
#         --train-data-path "$DATA_PATH" \
#         --eval-data-path "$EVAL_PATH" \
#         --g $BEST_G \
#         --sft-epochs $BEST_EPOCHS \
#         --ei-batch-size $BS \
#         --output-dir-base "$OUTPUT_BASE"
# done

# echo "All EI experiments completed."
