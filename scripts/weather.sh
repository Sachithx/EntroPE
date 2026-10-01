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
[ "${MODE:-eval}" = "train" ] && IS_TRAINING=1 || IS_TRAINING=0
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo; echo "----- weather  H=96  (mode=${MODE:-eval}) -----"
python -u run_longExp.py \
  --is_training "$IS_TRAINING" \
  --model_id "weather_L96_H96_small_bs32_lr1e-2_th095_dp02_mo1_s16" --model_id_name "weather" --setting_suffix "weather_L96_H96_small_bs32_lr1e-2_th095_dp02_mo1_s16" \
  --data "custom" --data_path "weather.csv" --freq "h" \
  --enc_in 21 --dec_in 21 --c_out 21 \
  --pred_len 96 \
  --d_model 8 --global_d_model 32 \
  --batch_size 32 --learning_rate 0.01 \
  --patching_threshold 0.95 --monotonicity 1 \
  --dropout 0.2 --head_dropout 0.2 --fc_dropout 0.2 \
  --random_seed 16


set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
[ "${MODE:-eval}" = "train" ] && IS_TRAINING=1 || IS_TRAINING=0
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo; echo "----- weather  H=192  (mode=${MODE:-eval}) -----"
python -u run_longExp.py \
  --is_training "$IS_TRAINING" \
  --model_id "weather_L96_H192_small_bs32_lr1e-2_th085_dp01_mo0_s2026" --model_id_name "weather" --setting_suffix "weather_L96_H192_small_bs32_lr1e-2_th085_dp01_mo0_s2026" \
  --data "custom" --data_path "weather.csv" --freq "h" \
  --enc_in 21 --dec_in 21 --c_out 21 \
  --pred_len 192 \
  --d_model 8 --global_d_model 32 \
  --batch_size 32 --learning_rate 0.01 \
  --patching_threshold 0.85 --monotonicity 0 \
  --dropout 0.1 --head_dropout 0.1 --fc_dropout 0.1 \
  --random_seed 2026


set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
[ "${MODE:-eval}" = "train" ] && IS_TRAINING=1 || IS_TRAINING=0
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo; echo "----- weather  H=336  (mode=${MODE:-eval}) -----"
python -u run_longExp.py \
  --is_training "$IS_TRAINING" \
  --model_id "weather_L96_H336_small_bs32_lr1e-2_th095_dp02_mo1_s2026" --model_id_name "weather" --setting_suffix "weather_L96_H336_small_bs32_lr1e-2_th095_dp02_mo1_s2026" \
  --data "custom" --data_path "weather.csv" --freq "h" \
  --enc_in 21 --dec_in 21 --c_out 21 \
  --pred_len 336 \
  --d_model 8 --global_d_model 32 \
  --batch_size 32 --learning_rate 0.01 \
  --patching_threshold 0.95 --monotonicity 1 \
  --dropout 0.2 --head_dropout 0.2 --fc_dropout 0.2 \
  --random_seed 2026


set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
[ "${MODE:-eval}" = "train" ] && IS_TRAINING=1 || IS_TRAINING=0
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo; echo "----- weather  H=720  (mode=${MODE:-eval}) -----"
python -u run_longExp.py \
  --is_training "$IS_TRAINING" \
  --model_id "weather_L96_H720_small_bs128_lr1e-2_th085_dp01_mo0_s2026" --model_id_name "weather" --setting_suffix "weather_L96_H720_small_bs128_lr1e-2_th085_dp01_mo0_s2026" \
  --data "custom" --data_path "weather.csv" --freq "h" \
  --enc_in 21 --dec_in 21 --c_out 21 \
  --pred_len 720 \
  --d_model 8 --global_d_model 32 \
  --batch_size 128 --learning_rate 0.01 \
  --patching_threshold 0.85 --monotonicity 0 \
  --dropout 0.1 --head_dropout 0.1 --fc_dropout 0.1 \
  --random_seed 2026
