#!/usr/bin/env bash
# EntroPE — ECL (input length L=96). Best config per horizon.
#
# Usage:  bash scripts/ECL.sh [train|eval]        (default: eval)
#   eval   evaluate the provided checkpoints (download_data.py --what checkpoints)
#   train  train from scratch, then evaluate (overwrites that cell's checkpoint)
# Pick a GPU with:  GPU=1 bash scripts/ECL.sh eval
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
DATASET=ECL; DATA_KIND=custom; DATA_FILE=electricity.csv; FREQ=h; CHANNELS=321
source "$(dirname "$0")/_common.sh"

#         horizon  d_model  global_d_model  batch_size  lr      threshold  dropout  monotonicity  seed   run_id
# paper: MSE 0.1695 / MAE 0.2626   (size=medium)
run_cell  96      16      64             64         0.01   0.95      0.1     0            29     ECL_L96_H96_medium_bs64_lr1e-2_th095_dp01_mo0_s29

# paper: MSE 0.1786 / MAE 0.2699   (size=medium)
run_cell  192     16      64             32         0.01   0.85      0.1     0            68     ECL_L96_H192_medium_bs32_lr1e-2_th085_dp01_mo0_s68

# paper: MSE 0.1940 / MAE 0.2845   (size=medium)
run_cell  336     16      64             64         0.01   0.95      0.1     0            85     ECL_L96_H336_medium_bs64_lr1e-2_th095_dp01_mo0_s85

# paper: MSE 0.2324 / MAE 0.3204   (size=medium)
run_cell  720     16      64             64         0.01   0.95      0.1     0            10     ECL_L96_H720_medium_bs64_lr1e-2_th095_dp01_mo0_s10
