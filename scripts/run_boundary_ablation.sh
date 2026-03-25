#!/bin/bash
# Ablation: compare boundary detection methods on 4 datasets × 2 horizons.
# All methods use the same EntroPE architecture — only the boundary detection differs.
# Non-entropy methods (local_diff, variance_cp, cusum, random, static) do NOT need
# a trained entropy model, so they run even without entropy_model_checkpoints.
#
# Usage: bash scripts/run_boundary_ablation.sh
# Output: logs/BoundaryAblation/{DATASET}_PL{HORIZON}_BM{METHOD}.log

set -e
source "$(dirname "$0")/run.sh"
mkdir -p logs/BoundaryAblation

# Architecture (same as main experiments)
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
CHECKPOINT_DIR="./entropy_model_checkpoints"

# Datasets and horizons for ablation
DATASETS=(ETTh1 ETTm1 weather Electricity)
PRED_LENS=(96 336)

# All boundary methods to compare
# empirical_entropy = EAPformer-style (within-window Shannon entropy of value distribution)
# frequency_based   = spectral energy shift between adjacent short-time FFT windows
METHODS=(entropy static local_diff variance_cp cusum random empirical_entropy frequency_based)

echo "======================================================"
echo "  Boundary Method Ablation"
echo "  Datasets: ${DATASETS[*]}"
echo "  Horizons: ${PRED_LENS[*]}"
echo "  Methods:  ${METHODS[*]}"
echo "  Note: empirical_entropy = EAPformer-style (within-window Shannon entropy)"
echo "  Note: frequency_based   = spectral energy shift (short-time FFT)"
echo "======================================================"

for DATASET in "${DATASETS[@]}"; do
    case $DATASET in
        ETTh1)  DATA_PATH="ETTh1.csv";       FREQ="h"; ENC_IN=7 ;;
        ETTm1)  DATA_PATH="ETTm1.csv";       FREQ="t"; ENC_IN=7 ;;
        weather)     DATA_PATH="weather.csv";     FREQ="h"; ENC_IN=21  ;;
        Electricity) DATA_PATH="electricity.csv"; FREQ="h"; ENC_IN=321 ;;
    esac

    for PRED_LEN in "${PRED_LENS[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            MODEL_ID="${DATASET}_SL${SEQ_LEN}_PL${PRED_LEN}_BM${METHOD}"
            LOG_FILE="logs/BoundaryAblation/${DATASET}_PL${PRED_LEN}_BM${METHOD}.log"

            echo ""
            echo ">>> $DATASET  pred_len=$PRED_LEN  method=$METHOD"

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
                --boundary_method     "$METHOD" \
                --entropy_model_checkpoint_dir "$CHECKPOINT_DIR" \
                --dropout        $DROPOUT \
                --learning_rate  $LEARNING_RATE \
                --batch_size     $BATCH_SIZE \
                --train_epochs   $TRAIN_EPOCHS \
                --patience       $PATIENCE \
                --des            Ablation \
                --itr            1 \
                > "$LOG_FILE" 2>&1

            echo "    Done. See $LOG_FILE"
        done
    done
done

echo ""
echo "======================================================"
echo "  Boundary ablation complete."
echo "  Results: ./results/  |  Logs: ./logs/BoundaryAblation/"
echo "======================================================"
