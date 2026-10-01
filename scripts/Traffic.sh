#!/usr/bin/env bash
# EntroPE — Traffic (input length L=96). Best config per horizon.
#
# Usage:  bash scripts/Traffic.sh [train|eval]        (default: eval)
#   eval   evaluate the provided checkpoints (download_data.py --what checkpoints)
#   train  train from scratch, then evaluate (overwrites that cell's checkpoint)
# Pick a GPU with:  GPU=1 bash scripts/Traffic.sh eval
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
DATASET=Traffic; DATA_KIND=custom; DATA_FILE=traffic.csv; FREQ=h; CHANNELS=862
source "$(dirname "$0")/_common.sh"

#         horizon  d_model  global_d_model  batch_size  lr      threshold  dropout  monotonicity  seed   run_id
# paper: MSE 0.4856 / MAE 0.2881   (size=large)
run_cell  96      32      128            8          0.01   0.95      0.1     1            16     Traffic_L96_H96_large_bs8_lr1e-2_th095_dp01_mo1_s16

# paper: MSE 0.4881 / MAE 0.2958   (size=large)
run_cell  192     32      128            8          0.01   0.95      0.1     1            68     Traffic_L96_H192_large_bs8_lr1e-2_th095_dp01_mo1_s68

# paper: MSE 0.4947 / MAE 0.2971   (size=large)
run_cell  336     32      128            8          0.01   0.95      0.1     1            41     Traffic_L96_H336_large_bs8_lr1e-2_th095_dp01_mo1_s41

# paper: MSE 0.5254 / MAE 0.3119   (size=large)
run_cell  720     32      128            8          0.01   0.95      0.1     1            41     Traffic_L96_H720_large_bs8_lr1e-2_th095_dp01_mo1_s41
