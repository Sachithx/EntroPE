#!/usr/bin/env bash
# Stage 1 (optional): train the frozen GPT entropy models from scratch.
#
# The exact entropy checkpoints used for the paper are already provided in
# entropy_model_checkpoints/dm16/ (params.json + <dataset>.pt), so you only
# need this to regenerate them. Output dim is 16 (n_embd=16), matching the
# forecasting configs. Writes params.json + <dataset>.pt into the output dir.
#
# Usage:  scripts/train_entropy.sh [dataset]
#   optional [dataset] filter: ETTh1 | ETTh2 | ETTm1 | ETTm2 | weather
set -euo pipefail
cd "$(dirname "$0")/.."

filter="${1:-}"
OUT=entropy_model_checkpoints/dm16

# dataset  data_file     freq
rows=(
  "ETTh1    ETTh1.csv     h"
  "ETTh2    ETTh2.csv     h"
  "ETTm1    ETTm1.csv     t"
  "ETTm2    ETTm2.csv     t"
  "weather  weather.csv   h"
)

for row in "${rows[@]}"; do
  read -r dataset data_file freq <<< "$row"
  if [ -n "$filter" ] && [ "$filter" != "$dataset" ]; then continue; fi
  echo ">>> training entropy model: $dataset"
  python -u train_entropy_model.py \
    --dataset "$dataset" --data_path "$data_file" --root_path ./dataset/ --freq "$freq" \
    --seq_len 96 --n_layer 2 --n_head 4 --n_embd 16 --vocab_size 256 \
    --epochs 50 --batch_size 128 --learning_rate 1e-3 --patience 5 \
    --checkpoint_dir "$OUT"
done
