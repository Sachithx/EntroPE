#!/usr/bin/env bash
# Run ONE (dataset, horizon) cell of EntroPE at its best config.
#
# Usage:
#   scripts/run_cell.sh <mode> <dataset> <data_kind> <data_file> <freq> <channels> \
#                       <horizon> <size> <d_model> <global_d_model> <batch_size> \
#                       <lr> <threshold> <dropout> <monotonicity> <seed> <run_id>
#
#   <mode> = train  -> train from scratch, then evaluate
#          = eval   -> evaluate a provided checkpoint only (no training)
#
# This is normally called by train_all.sh / eval_all.sh, which read
# scripts/best_configs.tsv. You can also call it directly to reproduce a
# single number.
set -euo pipefail
cd "$(dirname "$0")/.."

mode="$1"; dataset="$2"; data_kind="$3"; data_file="$4"; freq="$5"; channels="$6"
horizon="$7"; size="$8"; d_model="$9"; global_d_model="${10}"; batch_size="${11}"
lr="${12}"; threshold="${13}"; dropout="${14}"; monotonicity="${15}"; seed="${16}"
run_id="${17}"

if [ "$mode" = "train" ]; then is_training=1; else is_training=0; fi

python -u run_longExp.py \
  --is_training "$is_training" --model EntroPE \
  --model_id "$run_id" --model_id_name "$dataset" --setting_suffix "$run_id" \
  --save_test_artifacts 0 \
  --data "$data_kind" --root_path ./dataset --data_path "$data_file" \
  --features M --freq "$freq" \
  --enc_in "$channels" --dec_in "$channels" --c_out "$channels" \
  --seq_len 96 --label_len 48 --pred_len "$horizon" \
  --d_model "$d_model" --global_d_model "$global_d_model" \
  --n_heads 4 --e_layers 2 --d_ff 256 \
  --dropout "$dropout" --head_dropout "$dropout" --fc_dropout "$dropout" \
  --vocab_size 256 --max_patch_length 16 \
  --boundary_method entropy --patching_threshold "$threshold" --monotonicity "$monotonicity" \
  --encoder_self_attn_within_patch 1 \
  --cross_attn_window_encoder 1 --cross_attn_window_decoder 1 \
  --local_attention_window_len 96 --cross_attn_k 1 \
  --revin 1 --affine 1 --subtract_last 0 \
  --batch_size "$batch_size" --train_epochs 50 --patience 10 \
  --learning_rate "$lr" --lradj TST --pct_start 0.3 --activation gelu --num_workers 4 \
  --random_seed "$seed" --itr 1 --gpu "${GPU:-0}" \
  --checkpoints ./checkpoints \
  --entropy_model_checkpoint_dir ./entropy_model_checkpoints/dm16
