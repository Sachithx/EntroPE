#!/usr/bin/env bash
# EntroPE — ETTh2 (input length L=96). Best config per horizon.
#
# Usage:  bash scripts/ETTh2.sh [train|eval]        (default: eval)
#   eval   evaluate the provided checkpoints (download_data.py --what checkpoints)
#   train  train from scratch, then evaluate (overwrites that cell's checkpoint)
# Pick a GPU with:  GPU=1 bash scripts/ETTh2.sh eval
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
DATASET=ETTh2; DATA_KIND=ETTh2; DATA_FILE=ETTh2.csv; FREQ=h; CHANNELS=7
source "$(dirname "$0")/_common.sh"

#         horizon  d_model  global_d_model  batch_size  lr      threshold  dropout  monotonicity  seed   run_id
# paper: MSE 0.2948 / MAE 0.3477   (size=medium)
run_cell  96      16      64             32         0.01   0.85      0.3     1            16     ETTh2_L96_H96_medium_bs32_lr1e-2_th085_dp03_mo1_s16

# paper: MSE 0.3787 / MAE 0.3993   (size=large)
run_cell  192     32      128            128        0.01   0.95      0.2     0            2026   ETTh2_L96_H192_large_bs128_lr1e-2_th095_dp02_mo0_s2026

# paper: MSE 0.3879 / MAE 0.4206   (size=medium)
run_cell  336     16      64             128        0.01   0.95      0.3     0            2026   ETTh2_L96_H336_medium_bs128_lr1e-2_th095_dp03_mo0_s2026

# paper: MSE 0.4137 / MAE 0.4394   (size=medium)
run_cell  720     16      64             128        0.0001 0.75      0.3     0            2024   ETTh2_L96_H720_medium_bs128_lr1e-4_th075_dp03_mo0_s2024
