#!/usr/bin/env bash
# EntroPE — ETTm1 (input length L=96). Best config per horizon.
#
# Usage:  bash scripts/ETTm1.sh [train|eval]        (default: eval)
#   eval   evaluate the provided checkpoints (download_data.py --what checkpoints)
#   train  train from scratch, then evaluate (overwrites that cell's checkpoint)
# Pick a GPU with:  GPU=1 bash scripts/ETTm1.sh eval
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
DATASET=ETTm1; DATA_KIND=ETTm1; DATA_FILE=ETTm1.csv; FREQ=t; CHANNELS=7
source "$(dirname "$0")/_common.sh"

#         horizon  d_model  global_d_model  batch_size  lr      threshold  dropout  monotonicity  seed   run_id
# paper: MSE 0.3148 / MAE 0.3548   (size=small)
run_cell  96      8       32             32         0.01   0.85      0.1     0            2026   ETTm1_L96_H96_small_bs32_lr1e-2_th085_dp01_mo0_s2026

# paper: MSE 0.3584 / MAE 0.3811   (size=small)
run_cell  192     8       32             128        0.01   0.95      0.1     1            68     ETTm1_L96_H192_small_bs128_lr1e-2_th095_dp01_mo1_s68

# paper: MSE 0.3863 / MAE 0.3994   (size=small)
run_cell  336     8       32             128        0.01   0.95      0.1     1            2026   ETTm1_L96_H336_small_bs128_lr1e-2_th095_dp01_mo1_s2026

# paper: MSE 0.4496 / MAE 0.4361   (size=small)
run_cell  720     8       32             32         0.01   0.85      0.1     1            2024   ETTm1_L96_H720_small_bs32_lr1e-2_th085_dp01_mo1_s2024
