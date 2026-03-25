#!/bin/bash
# Train GPT-2 entropy models for all 6 forecasting datasets.
# Each dataset gets its own checkpoint: entropy_model_checkpoints/{DATASET}.pt
# params.json is written each run (last run wins — all use same architecture).
#
# Usage: bash scripts/train_all_entropy_models.sh

set -e
# Set correct Python executable and fix GLIBCXX version mismatch
source "$(dirname "$0")/run.sh"
mkdir -p logs/entropy_training
CHECKPOINT_DIR="./entropy_model_checkpoints"

# Shared model hyperparameters (must match params.json expected by Patcher)
N_LAYER=2
N_HEAD=4
N_EMBD=32
VOCAB_SIZE=256
SEQ_LEN=96
EPOCHS=50
BATCH_SIZE=128
PATIENCE=5

echo "======================================================"
echo "  Training entropy models for all datasets"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "======================================================"

for DATASET in ETTh1 ETTh2 ETTm1 ETTm2 Electricity weather; do
    case $DATASET in
        ETTh1)  DATA_PATH="ETTh1.csv";       FREQ="h" ;;
        ETTh2)  DATA_PATH="ETTh2.csv";       FREQ="h" ;;
        ETTm1)  DATA_PATH="ETTm1.csv";       FREQ="t" ;;
        ETTm2)  DATA_PATH="ETTm2.csv";       FREQ="t" ;;
        Electricity) DATA_PATH="electricity.csv"; FREQ="h" ;;
        weather)     DATA_PATH="weather.csv";     FREQ="h" ;;
    esac

    LOG_FILE="logs/entropy_training/${DATASET}.log"
    echo ""
    echo ">>> Training on $DATASET (data_path=$DATA_PATH, freq=$FREQ)"
    echo "    Log: $LOG_FILE"

    $PYTHON -u train_entropy_model.py \
        --dataset     "$DATASET" \
        --data_path   "$DATA_PATH" \
        --root_path   ./dataset/ \
        --freq        "$FREQ" \
        --features    M \
        --n_layer     $N_LAYER \
        --n_head      $N_HEAD \
        --n_embd      $N_EMBD \
        --vocab_size  $VOCAB_SIZE \
        --seq_len     $SEQ_LEN \
        --epochs      $EPOCHS \
        --batch_size  $BATCH_SIZE \
        --patience    $PATIENCE \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        > "$LOG_FILE" 2>&1

    echo "    Done. Checkpoint: $CHECKPOINT_DIR/${DATASET}.pt"
done

echo ""
echo "======================================================"
echo "  All entropy models trained."
echo "======================================================"
