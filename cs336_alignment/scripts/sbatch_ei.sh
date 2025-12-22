#!/bin/bash
#SBATCH --job-name=run_ei_experiments
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h100:2
#SBATCH --time=3:00:00
#SBATCH --output=logs/slurm-%j.out

ml reset
ml load GCC/13.3.0
ml load Clang/18.1.8-GCCcore-13.3.0-CUDA-12.6.0

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

BEST_G=8
BEST_EPOCHS=3

# ==========================================
# 2. Batch Size Scaling Experiments
# ==========================================
# "Vary the batch size for each expert iteration step... in {512, 1024, 2048}"
# Use the BEST_G and BEST_EPOCHS from above.

echo "--- Starting Batch Size Scaling Experiments (G=$BEST_G, Epochs=$BEST_EPOCHS) ---"

for BS in 512 1024 2048; do
    echo "Running EI with Batch Size $BS..."
    uv run cs336_alignment/scripts/run_ei.py \
        --train-data-path "$DATA_PATH" \
        --eval-data-path "$EVAL_PATH" \
        --g $BEST_G \
        --sft-epochs $BEST_EPOCHS \
        --ei-batch-size $BS \
        --output-dir-base "$OUTPUT_BASE"
done

echo "All EI experiments completed."
