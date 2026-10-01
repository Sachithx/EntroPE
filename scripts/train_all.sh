#!/usr/bin/env bash
# Train every (dataset, horizon) cell FROM SCRATCH at its best config, then
# evaluate. Each run writes checkpoints/<setting>/checkpoint.pth, overwriting
# any provided checkpoint for that cell. Assumes the stage-1 entropy models
# already exist in entropy_model_checkpoints/dm16/ (provided, or regenerated
# with scripts/train_entropy.sh).
#
# Usage:  scripts/train_all.sh [dataset]
#   optional [dataset] filter: ETTh1 | ETTh2 | ETTm1 | ETTm2 | weather
set -euo pipefail
cd "$(dirname "$0")/.."

filter="${1:-}"
tail -n +2 scripts/best_configs.tsv | while IFS=$'\t' read -r dataset data_kind data_file freq channels \
    horizon size d_model global_d_model batch_size lr threshold dropout monotonicity \
    seed mse mae run_id; do
  if [ -n "$filter" ] && [ "$filter" != "$dataset" ]; then continue; fi
  echo ">>> training $dataset H$horizon  (paper MSE/MAE: $mse / $mae)"
  scripts/run_cell.sh train "$dataset" "$data_kind" "$data_file" "$freq" "$channels" \
    "$horizon" "$size" "$d_model" "$global_d_model" "$batch_size" "$lr" \
    "$threshold" "$dropout" "$monotonicity" "$seed" "$run_id"
done
