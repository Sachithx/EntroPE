#!/usr/bin/env bash
# Reproduce every reported number by EVALUATING the provided checkpoints.
# No training. Requires checkpoints/ to be populated (see README).
#
# Usage:  scripts/eval_all.sh [dataset]
#   optional [dataset] filter: ETTh1 | ETTh2 | ETTm1 | ETTm2 | weather | ECL | Traffic | solar
set -euo pipefail
cd "$(dirname "$0")/.."

filter="${1:-}"
tsv=scripts/best_configs.tsv

printf '%-8s %5s  %-9s %-9s\n' dataset H MSE MAE
printf '%s\n' "----------------------------------------"
tail -n +2 "$tsv" | while IFS=$'\t' read -r dataset data_kind data_file freq channels \
    horizon size d_model global_d_model batch_size lr threshold dropout monotonicity \
    seed mse mae run_id; do
  if [ -n "$filter" ] && [ "$filter" != "$dataset" ]; then continue; fi
  out=$(scripts/run_cell.sh eval "$dataset" "$data_kind" "$data_file" "$freq" "$channels" \
        "$horizon" "$size" "$d_model" "$global_d_model" "$batch_size" "$lr" \
        "$threshold" "$dropout" "$monotonicity" "$seed" "$run_id" 2>/dev/null \
        | grep -E '^MSE:' | tail -1)
  got_mse=$(echo "$out" | sed -E 's/MSE: ([0-9.]+).*/\1/')
  got_mae=$(echo "$out" | sed -E 's/.*MAE: ([0-9.]+).*/\1/')
  printf '%-8s %5s  %-9.4f %-9.4f  (paper: %s / %s)\n' \
    "$dataset" "$horizon" "$got_mse" "$got_mae" "$mse" "$mae"
done
