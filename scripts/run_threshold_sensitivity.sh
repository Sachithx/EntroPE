#!/bin/bash
# Extended threshold sensitivity analysis across all 6 datasets.
# Addresses Reviewer eKr6 W4 and Reviewer GmSc W4 (selective analysis concern).
# Tests threshold percentiles: 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95
# Horizon: 336 (and optionally 720)
#
# Usage: bash scripts/run_threshold_sensitivity.sh

set -e
source "$(dirname "$0")/run.sh"
mkdir -p logs/ThresholdSensitivity

D_MODEL=8
N_HEADS=2
E_LAYERS=3
D_FF=256
MAX_PATCH_LENGTH=32
MONOTONICITY=0
DROPOUT=0.1
LEARNING_RATE=0.01
TRAIN_EPOCHS=20
BATCH_SIZE=128
PATIENCE=10
SEQ_LEN=96
FEATURES=M
CHECKPOINT_DIR="./entropy_model_checkpoints"

THRESHOLDS=(0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95)
PRED_LENS=(336 720)

echo "======================================================"
echo "  Threshold Sensitivity Analysis (all 6 datasets)"
echo "======================================================"

for DATASET in ETTh1 ETTh2 ETTm1 ETTm2 Electricity weather; do
    case $DATASET in
        ETTh1)  DATA_PATH="ETTh1.csv";       FREQ="h"; ENC_IN=7 ;;
        ETTh2)  DATA_PATH="ETTh2.csv";       FREQ="h"; ENC_IN=7 ;;
        ETTm1)  DATA_PATH="ETTm1.csv";       FREQ="t"; ENC_IN=7 ;;
        ETTm2)  DATA_PATH="ETTm2.csv";       FREQ="t"; ENC_IN=7 ;;
        Electricity) DATA_PATH="electricity.csv"; FREQ="h"; ENC_IN=321 ;;
        weather)     DATA_PATH="weather.csv";     FREQ="h"; ENC_IN=21  ;;
    esac

    for PRED_LEN in "${PRED_LENS[@]}"; do
        for TH in "${THRESHOLDS[@]}"; do
            MODEL_ID="${DATASET}_PL${PRED_LEN}_TH${TH}"
            LOG_FILE="logs/ThresholdSensitivity/${DATASET}_PL${PRED_LEN}_TH${TH}.log"

            echo ">>> $DATASET  pred=$PRED_LEN  threshold=$TH"

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
                --d_model        $D_MODEL \
                --n_heads        $N_HEADS \
                --e_layers       $E_LAYERS \
                --d_ff           $D_FF \
                --max_patch_length    $MAX_PATCH_LENGTH \
                --patching_threshold  $TH \
                --monotonicity        $MONOTONICITY \
                --boundary_method     entropy \
                --entropy_model_checkpoint_dir "$CHECKPOINT_DIR" \
                --dropout        $DROPOUT \
                --learning_rate  $LEARNING_RATE \
                --batch_size     $BATCH_SIZE \
                --train_epochs   $TRAIN_EPOCHS \
                --patience       $PATIENCE \
                --des            ThreshSensitivity \
                --itr            1 \
                > "$LOG_FILE" 2>&1

            echo "    Done."
        done
    done
done

echo ""
echo "======================================================"
echo "  Threshold sensitivity complete."
echo "  Logs: ./logs/ThresholdSensitivity/"
echo "======================================================"
