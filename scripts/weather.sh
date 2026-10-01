#!/usr/bin/env bash
# EntroPE — weather (input length L=96). Best config per horizon.
#
# Usage:  bash scripts/weather.sh [train|eval]        (default: eval)
#   eval   evaluate the provided checkpoints (download_data.py --what checkpoints)
#   train  train from scratch, then evaluate (overwrites that cell's checkpoint)
# Pick a GPU with:  GPU=1 bash scripts/weather.sh eval
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
DATASET=weather; DATA_KIND=custom; DATA_FILE=weather.csv; FREQ=h; CHANNELS=21
source "$(dirname "$0")/_common.sh"

#         horizon  d_model  global_d_model  batch_size  lr      threshold  dropout  monotonicity  seed   run_id
# paper: MSE 0.1673 / MAE 0.2164   (size=small)
run_cell  96      8       32             32         0.01   0.95      0.2     1            16     weather_L96_H96_small_bs32_lr1e-2_th095_dp02_mo1_s16

# paper: MSE 0.2123 / MAE 0.2543   (size=small)
run_cell  192     8       32             32         0.01   0.85      0.1     0            2026   weather_L96_H192_small_bs32_lr1e-2_th085_dp01_mo0_s2026

# paper: MSE 0.2677 / MAE 0.2952   (size=small)
run_cell  336     8       32             32         0.01   0.95      0.2     1            2026   weather_L96_H336_small_bs32_lr1e-2_th095_dp02_mo1_s2026

# paper: MSE 0.3412 / MAE 0.3426   (size=small)
run_cell  720     8       32             128        0.01   0.85      0.1     0            2026   weather_L96_H720_small_bs128_lr1e-2_th085_dp01_mo0_s2026
