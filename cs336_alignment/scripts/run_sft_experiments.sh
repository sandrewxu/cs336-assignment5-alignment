#!/bin/bash
# cs336_alignment/scripts/run_sft_experiments.sh

# Exit on error
set -e

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Configuration
# Default data path in repo: data/MATH/sft.jsonl
# You can change this to /data/a5-alignment/MATH/sft.jsonl if needed
DATA_PATH="data/MATH/sft.jsonl" 

# Base output directory
OUTPUT_BASE="/gpfs/radev/home/ax46/scratch/A5/sft"

echo "Using data from: $DATA_PATH"
echo "Outputting model to: $OUTPUT_BASE"

# ==========================================
# 1. Hyperparameter Tuning (Full Dataset)
# ==========================================
# Goal: Achieve > 15% validation accuracy.
# We will run a few configurations. Please check the logs/wandb to pick the best one.
# You can comment out this section if you already have the best params.

echo "--- Starting Hyperparameter Tuning ---"

# # Config A: LR=1e-5, Batch=64 (Default)
# echo "Tuning Run A: LR=1e-5, Batch=64"
# uv run cs336_alignment/scripts/run_sft.py \
#     --sft-data-path "$DATA_PATH" \
#     --learning-rate 1e-5 \
#     --batch-size 64 \
#     --output-dir-base "$OUTPUT_BASE"

# # Config B: LR=5e-6, Batch=64
# echo "Tuning Run B: LR=5e-6, Batch=64"
# uv run cs336_alignment/scripts/run_sft.py \
#     --sft-data-path "$DATA_PATH" \
#     --learning-rate 5e-6 \
#     --batch-size 64 \
#     --output-dir-base "$OUTPUT_BASE"

# # Config C: LR=2e-5, Batch=32
# echo "Tuning Run C: LR=2e-5, Batch=32"
# uv run cs336_alignment/scripts/run_sft.py \
#     --sft-data-path "$DATA_PATH" \
#     --learning-rate 2e-5 \
#     --batch-size 32 \
#     --output-dir-base "$OUTPUT_BASE"

# # Config D: LR=2e-5, Batch=64
# echo "Tuning Run D: LR=2e-5, Batch=64"
# uv run cs336_alignment/scripts/run_sft.py \
#     --sft-data-path "$DATA_PATH" \
#     --learning-rate 2e-5 \
#     --batch-size 64 \
#     --output-dir-base "$OUTPUT_BASE"


# # Config E: LR=5e-5, Batch=64
# echo "Tuning Run E: LR=5e-5, Batch=64"
# uv run cs336_alignment/scripts/run_sft.py \
#     --sft-data-path "$DATA_PATH" \
#     --learning-rate 5e-5 \
#     --batch-size 64 \
#     --output-dir-base "$OUTPUT_BASE"

# # Config F: LR=1e-4, Batch=64
# echo "Tuning Run F: LR=1e-4, Batch=64"
# uv run cs336_alignment/scripts/run_sft.py \
#     --sft-data-path "$DATA_PATH" \
#     --learning-rate 1e-4 \
#     --batch-size 64 \
#     --output-dir-base "$OUTPUT_BASE"

# ==========================================
# SET YOUR BEST HYPERPARAMETERS HERE
# ==========================================
# Replace these with the values that gave > 15% accuracy in the tuning step.
BEST_LR=1e-4
BEST_BS=64


# ==========================================
# 2. Dataset Size Scaling Experiments
# ==========================================
# Sizes: 128, 256, 512, 1024 (Full is covered by tuning or runs below)

echo "--- Starting Scaling Experiments (LR=$BEST_LR, BS=$BEST_BS) ---"

for N in 128 256 512 1024; do
    echo "Running SFT with $N examples..."
    uv run cs336_alignment/scripts/run_sft.py \
        --sft-data-path "$DATA_PATH" \
        --sft-examples $N \
        --learning-rate $BEST_LR \
        --batch-size $BEST_BS \
        --output-dir-base "$OUTPUT_BASE"
done

# Ensure Full Dataset is run with BEST parameters
echo "Running SFT with Full Dataset..."
uv run cs336_alignment/scripts/run_sft.py \
    --sft-data-path "$DATA_PATH" \
    --learning-rate $BEST_LR \
    --batch-size $BEST_BS \
    --output-dir-base "$OUTPUT_BASE"


# ==========================================
# 3. Filtered Dataset Experiment
# ==========================================
# Filter correct answers only, run on full filtered set.

echo "--- Starting Filtered Experiment ---"
uv run cs336_alignment/scripts/run_sft.py \
    --sft-data-path "$DATA_PATH" \
    --filter-correct \
    --learning-rate $BEST_LR \
    --batch-size $BEST_BS \
    --output-dir-base "$OUTPUT_BASE"

echo "All experiments completed."
