#!/usr/bin/env bash
# EntroPE — ETTh2 (input length L=96). Best config per horizon.
#
# Usage:  bash scripts/ETTh2.sh [train|eval]        (default: eval)
#   eval   evaluate the provided checkpoints (download_data.py --what checkpoints)
#   train  train from scratch, then evaluate (overwrites that cell's checkpoint)
# Pick a GPU with:  GPU=1 bash scripts/ETTh2.sh eval

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
[ "${MODE:-eval}" = "train" ] && IS_TRAINING=1 || IS_TRAINING=0
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo; echo "----- ETTh2  H=96  (mode=${MODE:-eval}) -----"
python -u run_longExp.py \
  --is_training "$IS_TRAINING" \
  --model_id "ETTh2_L96_H96_medium_bs32_lr1e-2_th085_dp03_mo1_s16" --model_id_name "ETTh2" --setting_suffix "ETTh2_L96_H96_medium_bs32_lr1e-2_th085_dp03_mo1_s16" \
  --data "ETTh2" --data_path "ETTh2.csv" --freq "h" \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --pred_len 96 \
  --d_model 16 --global_d_model 64 \
  --batch_size 32 --learning_rate 0.01 \
  --patching_threshold 0.85 --monotonicity 1 \
  --dropout 0.3 --head_dropout 0.3 --fc_dropout 0.3 \
  --random_seed 16


set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
[ "${MODE:-eval}" = "train" ] && IS_TRAINING=1 || IS_TRAINING=0
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo; echo "----- ETTh2  H=192  (mode=${MODE:-eval}) -----"
python -u run_longExp.py \
  --is_training "$IS_TRAINING" \
  --model_id "ETTh2_L96_H192_large_bs128_lr1e-2_th095_dp02_mo0_s2026" --model_id_name "ETTh2" --setting_suffix "ETTh2_L96_H192_large_bs128_lr1e-2_th095_dp02_mo0_s2026" \
  --data "ETTh2" --data_path "ETTh2.csv" --freq "h" \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --pred_len 192 \
  --d_model 32 --global_d_model 128 \
  --batch_size 128 --learning_rate 0.01 \
  --patching_threshold 0.95 --monotonicity 0 \
  --dropout 0.2 --head_dropout 0.2 --fc_dropout 0.2 \
  --random_seed 2026


set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
[ "${MODE:-eval}" = "train" ] && IS_TRAINING=1 || IS_TRAINING=0
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo; echo "----- ETTh2  H=336  (mode=${MODE:-eval}) -----"
python -u run_longExp.py \
  --is_training "$IS_TRAINING" \
  --model_id "ETTh2_L96_H336_medium_bs128_lr1e-2_th095_dp03_mo0_s2026" --model_id_name "ETTh2" --setting_suffix "ETTh2_L96_H336_medium_bs128_lr1e-2_th095_dp03_mo0_s2026" \
  --data "ETTh2" --data_path "ETTh2.csv" --freq "h" \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --pred_len 336 \
  --d_model 16 --global_d_model 64 \
  --batch_size 128 --learning_rate 0.01 \
  --patching_threshold 0.95 --monotonicity 0 \
  --dropout 0.3 --head_dropout 0.3 --fc_dropout 0.3 \
  --random_seed 2026


set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
[ "${MODE:-eval}" = "train" ] && IS_TRAINING=1 || IS_TRAINING=0
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo; echo "----- ETTh2  H=720  (mode=${MODE:-eval}) -----"
python -u run_longExp.py \
  --is_training "$IS_TRAINING" \
  --model_id "ETTh2_L96_H720_medium_bs128_lr1e-4_th075_dp03_mo0_s2024" --model_id_name "ETTh2" --setting_suffix "ETTh2_L96_H720_medium_bs128_lr1e-4_th075_dp03_mo0_s2024" \
  --data "ETTh2" --data_path "ETTh2.csv" --freq "h" \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --pred_len 720 \
  --d_model 16 --global_d_model 64 \
  --batch_size 128 --learning_rate 0.0001 \
  --patching_threshold 0.75 --monotonicity 0 \
  --dropout 0.3 --head_dropout 0.3 --fc_dropout 0.3 \
  --random_seed 2024
