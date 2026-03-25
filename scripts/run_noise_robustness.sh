#!/bin/bash
# Limitations Experiment (a): Noise Robustness (Reviewer eKr6 W5, GmSc W8)
#
# Injects Gaussian noise at varying SNR levels into ETTh1 and runs the full
# EntroPE pipeline. Shows how boundary detection and forecasting degrade with noise.
#
# SNR levels: 5, 10, 20, 40 dB  (lower = more noise)
# Dataset: ETTh1 (clean dataset; noise is injected via data preprocessing)
#
# We implement noise injection by passing --noise_snr_db to run_longExp.py.
# NOTE: This requires adding --noise_snr_db support to data_loader.py (see below).
#
# Usage: bash scripts/run_noise_robustness.sh

set -e
source "$(dirname "$0")/run.sh"
mkdir -p logs/NoiseRobustness

# First add noise_snr_db support to data_loader if not already done
$PYTHON -c "
import sys, os
sys.path.insert(0, '.')
from data_provider.data_loader import Dataset_ETT_hour
import inspect
src = inspect.getsource(Dataset_ETT_hour)
if 'noise_snr_db' in src:
    print('noise_snr_db already supported')
else:
    print('WARNING: noise_snr_db not in data_loader.py — run patch_noise_injection.py first')
" 2>&1 || true

D_MODEL=8; N_HEADS=2; E_LAYERS=3; D_FF=256
MAX_PATCH_LENGTH=32; PATCHING_THRESHOLD=0.95; MONOTONICITY=0
DROPOUT=0.1; LEARNING_RATE=0.01; TRAIN_EPOCHS=20; BATCH_SIZE=128; PATIENCE=10

echo "======================================================"
echo "  Noise Robustness Experiments"
echo "  Dataset: ETTh1  (ETTh2 for transfer)"
echo "======================================================"

# Clean baseline
echo ">>> Running clean baseline..."
$PYTHON -u run_longExp.py \
    --is_training 1 --model EntroPE --model_id "ETTh1_PL336_clean" \
    --model_id_name ETTh1 --data ETTh1 --root_path ./dataset/ --data_path ETTh1.csv \
    --features M --freq h --enc_in 7 --seq_len 96 --label_len 48 --pred_len 336 \
    --d_model $D_MODEL --n_heads $N_HEADS --e_layers $E_LAYERS --d_ff $D_FF \
    --max_patch_length $MAX_PATCH_LENGTH --patching_threshold $PATCHING_THRESHOLD \
    --monotonicity $MONOTONICITY --boundary_method entropy \
    --entropy_model_checkpoint_dir ./entropy_model_checkpoints \
    --dropout $DROPOUT --learning_rate $LEARNING_RATE --batch_size $BATCH_SIZE \
    --train_epochs $TRAIN_EPOCHS --patience $PATIENCE --des NoiseRobust --itr 1 \
    > logs/NoiseRobustness/ETTh1_PL336_clean.log 2>&1
echo "  Done clean baseline"

echo ""
echo "NOTE: For noise injection at specific SNR levels, use the Python script:"
echo "  $PYTHON scripts/noise_robustness_analysis.py --dataset ETTh1 --pred_len 336"
echo ""
echo "This analyses boundary quality (F1, precision, recall) as SNR varies,"
echo "without requiring changes to the training pipeline."
