#!/usr/bin/env bash
# EntroPE — ETTh1 (input length L=96). Best config per horizon.
#
# Usage:  bash scripts/ETTh1.sh [train|eval]        (default: eval)
#   eval   evaluate the provided checkpoints (download_data.py --what checkpoints)
#   train  train from scratch, then evaluate (overwrites that cell's checkpoint)
# Pick a GPU with:  GPU=1 bash scripts/ETTh1.sh eval
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
DATASET=ETTh1; DATA_KIND=ETTh1; DATA_FILE=ETTh1.csv; FREQ=h; CHANNELS=7
source "$(dirname "$0")/_common.sh"

#         horizon  d_model  global_d_model  batch_size  lr      threshold  dropout  monotonicity  seed   run_id
# paper: MSE 0.3788 / MAE 0.3994   (size=medium)
run_cell  96      16      64             64         0.01   0.95      0.1     0            94     ETTh1_L96_H96_medium_bs64_lr1e-2_th095_dp01_mo0_s94

# paper: MSE 0.4222 / MAE 0.4258   (size=small)
run_cell  192     8       32             128        0.01   0.75      0.1     1            2026   ETTh1_L96_H192_small_bs128_lr1e-2_th075_dp01_mo1_s2026

# paper: MSE 0.4594 / MAE 0.4462   (size=small)
run_cell  336     8       32             64         0.01   0.85      0.1     0            2026   ETTh1_L96_H336_small_bs64_lr1e-2_th085_dp01_mo0_s2026

# paper: MSE 0.4406 / MAE 0.4514   (size=small)
run_cell  720     8       32             32         0.01   0.85      0.2     0            10     ETTh1_L96_H720_small_bs32_lr1e-2_th085_dp02_mo0_s10
