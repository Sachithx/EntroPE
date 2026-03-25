#!/bin/bash
# Full forecasting experiment matrix: 6 datasets × 4 horizons
# Uses entropy boundary method (EntroPE's core method).
#
# Usage: bash scripts/run_all_forecasting.sh
# Output: logs/LongForecasting/{DATASET}_PL{HORIZON}.log
#         results/ and checkpoints/ subdirectories

set -e
source "$(dirname "$0")/run.sh"
mkdir -p logs/LongForecasting

# Shared architecture params
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
BATCH_SIZE=128
PATIENCE=10
SEQ_LEN=96
FEATURES=M
BOUNDARY_METHOD=entropy
CHECKPOINT_DIR="./entropy_model_checkpoints"

echo "======================================================"
echo "  EntroPE Full Forecasting Experiments"
echo "  Boundary method: $BOUNDARY_METHOD"
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

    for PRED_LEN in 96 192 336 720; do
        MODEL_ID="${DATASET}_SL${SEQ_LEN}_PL${PRED_LEN}"
        LOG_FILE="logs/LongForecasting/${DATASET}_PL${PRED_LEN}.log"

        echo ""
        echo ">>> $DATASET  pred_len=$PRED_LEN"
        echo "    Log: $LOG_FILE"

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
            --patching_threshold  $PATCHING_THRESHOLD \
            --monotonicity        $MONOTONICITY \
            --boundary_method     $BOUNDARY_METHOD \
            --entropy_model_checkpoint_dir "$CHECKPOINT_DIR" \
            --dropout        $DROPOUT \
            --learning_rate  $LEARNING_RATE \
            --batch_size     $BATCH_SIZE \
            --train_epochs   $TRAIN_EPOCHS \
            --patience       $PATIENCE \
            --des            Exp \
            --itr            1 \
            > "$LOG_FILE" 2>&1

        echo "    Done."
    done
done

echo ""
echo "======================================================"
echo "  All forecasting experiments complete."
echo "  Results saved in ./results/"
echo "======================================================"
