#!/usr/bin/env bash
# EntroPE — ETTm2 (input length L=96). Best config per horizon.
#
# Usage:  bash scripts/ETTm2.sh [train|eval]        (default: eval)
#   eval   evaluate the provided checkpoints (download_data.py --what checkpoints)
#   train  train from scratch, then evaluate (overwrites that cell's checkpoint)
# Pick a GPU with:  GPU=1 bash scripts/ETTm2.sh eval
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
DATASET=ETTm2; DATA_KIND=ETTm2; DATA_FILE=ETTm2.csv; FREQ=t; CHANNELS=7
source "$(dirname "$0")/_common.sh"

#         horizon  d_model  global_d_model  batch_size  lr      threshold  dropout  monotonicity  seed   run_id
# paper: MSE 0.1803 / MAE 0.2679   (size=small)
run_cell  96      8       32             64         0.01   0.75      0.2     1            2026   ETTm2_L96_H96_small_bs64_lr1e-2_th075_dp02_mo1_s2026

# paper: MSE 0.2462 / MAE 0.3068   (size=medium)
run_cell  192     16      64             32         0.01   0.75      0.3     1            2026   ETTm2_L96_H192_medium_bs32_lr1e-2_th075_dp03_mo1_s2026

# paper: MSE 0.3085 / MAE 0.3485   (size=small)
run_cell  336     8       32             32         0.01   0.75      0.2     0            2026   ETTm2_L96_H336_small_bs32_lr1e-2_th075_dp02_mo0_s2026

# paper: MSE 0.4066 / MAE 0.4068   (size=large)
run_cell  720     32      128            32         0.001  0.95      0.2     0            2026   ETTm2_L96_H720_large_bs32_lr1e-3_th095_dp02_mo0_s2026
