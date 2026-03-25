#!/bin/bash
# Ablation: vocabulary size V ∈ {64, 128, 256, 512}
# For each vocab size:
#   1. Train a new entropy model with that vocab size → saves to separate dir
#   2. Run forecasting using that checkpoint
# Dataset: ETTh1, horizons: 96 and 336
#
# Usage: bash scripts/run_vocab_ablation.sh

set -e
source "$(dirname "$0")/run.sh"
mkdir -p logs/VocabAblation

DATASET=ETTh1
DATA_PATH="ETTh1.csv"
FREQ=h
ENC_IN=7

# Entropy model params (only vocab_size changes)
N_LAYER=2
N_HEAD=4
N_EMBD=32
SEQ_LEN=96
EPOCHS=50
BATCH_SIZE=128
PATIENCE=5

# Forecasting params
D_MODEL=8
N_HEADS=2
E_LAYERS=3
D_FF=256
MAX_PATCH_LENGTH=32
PATCHING_THRESHOLD=0.95
MONOTONICITY=0
DROPOUT=0.1
LEARNING_RATE=0.01
TRAIN_EPOCHS=20
FEATURES=M

PRED_LENS=(96 336)

echo "======================================================"
echo "  Vocabulary Size Ablation on $DATASET"
echo "======================================================"

for VOCAB in 64 128 256 512; do
    CKPT_DIR="entropy_model_checkpoints_vocab${VOCAB}"
    mkdir -p "$CKPT_DIR"

    echo ""
    echo ">>> Vocab=$VOCAB  Training entropy model..."
    $PYTHON -u train_entropy_model.py \
        --dataset     "$DATASET" \
        --data_path   "$DATA_PATH" \
        --root_path   ./dataset/ \
        --freq        "$FREQ" \
        --features    M \
        --n_layer     $N_LAYER \
        --n_head      $N_HEAD \
        --n_embd      $N_EMBD \
        --vocab_size  $VOCAB \
        --seq_len     $SEQ_LEN \
        --epochs      $EPOCHS \
        --batch_size  $BATCH_SIZE \
        --patience    $PATIENCE \
        --checkpoint_dir "$CKPT_DIR" \
        > "logs/VocabAblation/train_${DATASET}_vocab${VOCAB}.log" 2>&1
    echo "    Entropy model saved to $CKPT_DIR/${DATASET}.pt"

    for PRED_LEN in "${PRED_LENS[@]}"; do
        MODEL_ID="${DATASET}_vocab${VOCAB}_PL${PRED_LEN}"
        LOG_FILE="logs/VocabAblation/${DATASET}_vocab${VOCAB}_PL${PRED_LEN}.log"

        echo ">>> Vocab=$VOCAB  Forecasting  pred_len=$PRED_LEN"

        $PYTHON -u run_longExp.py \
            --is_training    1 \
            --model          EntroPE \
            --model_id       "$MODEL_ID" \
            --model_id_name  "$DATASET" \
            --data           "$DATASET" \
            --root_path      ./dataset/ \
            --data_path      "$DATA_PATH" \
            --features       $FEATURES \
            --freq           "$FREQ" \
            --enc_in         $ENC_IN \
            --seq_len        $SEQ_LEN \
            --label_len      48 \
            --pred_len       $PRED_LEN \
            --vocab_size     $VOCAB \
            --d_model        $D_MODEL \
            --n_heads        $N_HEADS \
            --e_layers       $E_LAYERS \
            --d_ff           $D_FF \
            --max_patch_length    $MAX_PATCH_LENGTH \
            --patching_threshold  $PATCHING_THRESHOLD \
            --monotonicity        $MONOTONICITY \
            --boundary_method     entropy \
            --entropy_model_checkpoint_dir "./$CKPT_DIR/" \
            --dropout        $DROPOUT \
            --learning_rate  $LEARNING_RATE \
            --batch_size     $BATCH_SIZE \
            --train_epochs   $TRAIN_EPOCHS \
            --patience       10 \
            --des            VocabAblation \
            --itr            1 \
            > "$LOG_FILE" 2>&1

        echo "    Done. See $LOG_FILE"
    done
done

echo ""
echo "======================================================"
echo "  Vocabulary ablation complete."
echo "  Results: ./results/  |  Logs: ./logs/VocabAblation/"
echo "======================================================"
