#!/bin/bash
#SBATCH --job-name=run_grpo_experiments_lr_tuning
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h100:2
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm-%j.out

ml reset
ml load GCC/13.3.0
ml load Clang/18.1.8-GCCcore-13.3.0-CUDA-12.6.0

# Exit on error
set -e

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Configuration
# Dataset path
DATA_PATH="data/MATH/train.jsonl"
EVAL_PATH="data/MATH/validation.jsonl"

# Base output directory
OUTPUT_BASE="/gpfs/radev/home/ax46/scratch/A5/rl"

echo "Using training data from: $DATA_PATH"
echo "Using eval data from: $EVAL_PATH"
echo "Outputting experiments to: $OUTPUT_BASE"

# ==========================================
# Learning Rate Tuning Experiments
# ==========================================
# "Tune based on learning rate with default hyperparameters (5e-6, 2e-5, 5e-5)"

# echo "--- Starting Learning Rate Tuning Experiments ---"

# for LR in 5e-6 2e-5 5e-5; do
#     echo "Running GRPO with Learning Rate $LR..."
#     uv run cs336_alignment/scripts/run_rl.py \
#         --train-data-path "$DATA_PATH" \
#         --eval-data-path "$EVAL_PATH" \
#         --learning-rate $LR \
#         --output-dir-base "$OUTPUT_BASE"
# done

BEST_LR=2e-5

# ==========================================
# Baselining Experiments 
# ==========================================
# "Compare `no_baseline` to `reinforce_with_baseline`"

# echo "--- Starting Loss Type Tuning Experiments ---"

# for LOSS_TYPE in "no_baseline" "reinforce_with_baseline"; do
#     echo "Running GRPO with loss type $LOSS_TYPE..."
#     uv run cs336_alignment/scripts/run_rl.py \
#         --train-data-path "$DATA_PATH" \
#         --eval-data-path "$EVAL_PATH" \
#         --learning-rate $BEST_LR \
#         --loss-type "$LOSS_TYPE" \
#         --output-dir-base "$OUTPUT_BASE"
# done

BEST_LOSS_TYPE="reinforce_with_baseline"


# ==========================================
# Length Normalization Experiments 
# ==========================================
# "Compare `masked_mean` to `masked_normalize`"

# echo "--- Starting Length Normalization Experiments ---"

# for NORM in None 1024; do
#     echo "Running GRPO with norm $NORM..."
#     uv run cs336_alignment/scripts/run_rl.py \
#         --train-data-path "$DATA_PATH" \
#         --eval-data-path "$EVAL_PATH" \
#         --learning-rate $BEST_LR \
#         --loss-type "$BEST_LOSS_TYPE" \
#         --constant-normalize-factor $NORM \
#         --output-dir-base "$OUTPUT_BASE"
# done

BEST_NORM=1024

# ==========================================
# Standard Deviation Normalization Experiments 
# ==========================================
# "Compare use_std_normalization == True and False"

# echo "--- Starting Standard Deviation Tuning Experiments ---"

# for STD_NORM in true false; do
#     echo "Running GRPO with std norm $STD_NORM..."
#     uv run cs336_alignment/scripts/run_rl.py \
#         --train-data-path "$DATA_PATH" \
#         --eval-data-path "$EVAL_PATH" \
#         --learning-rate $BEST_LR \
#         --loss-type "$BEST_LOSS_TYPE" \
#         --constant-normalize-factor $BEST_NORM \
#         --use-std-normalization $STD_NORM \
#         --output-dir-base "$OUTPUT_BASE"
# done

BEST_STD_NORM=true

# ==========================================
# Off-Policy GRPO Initial Experiments 
# ==========================================

# TEST_STEPS=25

# for EPOCHS in 1 2 4; do
#     for TRAIN_SIZE in 16 64; do
#         echo "Running GRPO-Clip for $TEST_STEPS steps with epochs=$EPOCHS and train_batch_size=$TRAIN_SIZE..."
#         uv run cs336_alignment/scripts/run_rl.py \
#             --train-data-path "$DATA_PATH" \
#             --eval-data-path "$EVAL_PATH" \
#             --n-grpo-steps $TEST_STEPS \
#             --train-batch-size $TRAIN_SIZE \
#             --gradient-accumulation-steps $((TRAIN_SIZE/2)) \
#             --epochs-per-rollout-batch $EPOCHS \
#             --learning-rate $BEST_LR \
#             --loss-type "grpo_clip" \
#             --constant-normalize-factor $BEST_NORM \
#             --use-std-normalization $BEST_STD_NORM \
#             --output-dir-base "$OUTPUT_BASE"
#     done
# done

# ==========================================
# Off-Policy GRPO Long Experiments 
# ==========================================

for EPOCHS in 1 2; do
    for TRAIN_SIZE in 16 32 64; do
        echo "Running GRPO-Clip with epochs=$EPOCHS and train_batch_size=$TRAIN_SIZE..."
        uv run cs336_alignment/scripts/run_rl.py \
            --train-data-path "$DATA_PATH" \
            --eval-data-path "$EVAL_PATH" \
            --train-batch-size $TRAIN_SIZE \
            --gradient-accumulation-steps $((TRAIN_SIZE/2)) \
            --epochs-per-rollout-batch $EPOCHS \
            --learning-rate $BEST_LR \
            --loss-type "grpo_clip" \
            --constant-normalize-factor $BEST_NORM \
            --use-std-normalization $BEST_STD_NORM \
            --output-dir-base "$OUTPUT_BASE"
    done
done

echo "All GRPO experiments completed."
