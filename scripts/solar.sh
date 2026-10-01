#!/usr/bin/env bash
# EntroPE — solar (input length L=96). Best config per horizon.
#
# Usage:  bash scripts/solar.sh [train|eval]        (default: eval)
#   eval   evaluate the provided checkpoints (download_data.py --what checkpoints)
#   train  train from scratch, then evaluate (overwrites that cell's checkpoint)
# Pick a GPU with:  GPU=1 bash scripts/solar.sh eval
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
DATASET=solar; DATA_KIND=custom; DATA_FILE=solar.csv; FREQ=t; CHANNELS=137
source "$(dirname "$0")/_common.sh"

#         horizon  d_model  global_d_model  batch_size  lr      threshold  dropout  monotonicity  seed   run_id
# paper: MSE 0.2065 / MAE 0.2544   (size=medium)
run_cell  96      16      64             64         0.01   0.95      0.1     0            44     solar_L96_H96_medium_bs64_lr1e-2_th095_dp01_mo0_s44

# paper: MSE 0.2403 / MAE 0.2760   (size=medium)
run_cell  192     16      64             32         0.01   0.85      0.1     0            94     solar_L96_H192_medium_bs32_lr1e-2_th085_dp01_mo0_s94

# paper: MSE 0.2556 / MAE 0.2861   (size=medium)
run_cell  336     16      64             64         0.01   0.95      0.1     0            10     solar_L96_H336_medium_bs64_lr1e-2_th095_dp01_mo0_s10

# paper: MSE 0.2557 / MAE 0.2838   (size=medium)
run_cell  720     16      64             32         0.01   0.85      0.1     1            10     solar_L96_H720_medium_bs32_lr1e-2_th085_dp01_mo1_s10
