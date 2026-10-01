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
[ "${MODE:-eval}" = "train" ] && IS_TRAINING=1 || IS_TRAINING=0
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo; echo "----- Traffic  H=96  (mode=${MODE:-eval}) -----"
python -u run_longExp.py \
  --is_training "$IS_TRAINING" \
  --model_id "Traffic_L96_H96_large_bs8_lr1e-2_th095_dp01_mo1_s16" --model_id_name "Traffic" --setting_suffix "Traffic_L96_H96_large_bs8_lr1e-2_th095_dp01_mo1_s16" \
  --data "custom" --data_path "traffic.csv" --freq "h" \
  --enc_in 862 --dec_in 862 --c_out 862 \
  --pred_len 96 \
  --d_model 32 --global_d_model 128 \
  --batch_size 8 --learning_rate 0.01 \
  --patching_threshold 0.95 --monotonicity 1 \
  --dropout 0.1 --head_dropout 0.1 --fc_dropout 0.1 \
  --random_seed 16


set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
[ "${MODE:-eval}" = "train" ] && IS_TRAINING=1 || IS_TRAINING=0
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo; echo "----- Traffic  H=192  (mode=${MODE:-eval}) -----"
python -u run_longExp.py \
  --is_training "$IS_TRAINING" \
  --model_id "Traffic_L96_H192_large_bs8_lr1e-2_th095_dp01_mo1_s68" --model_id_name "Traffic" --setting_suffix "Traffic_L96_H192_large_bs8_lr1e-2_th095_dp01_mo1_s68" \
  --data "custom" --data_path "traffic.csv" --freq "h" \
  --enc_in 862 --dec_in 862 --c_out 862 \
  --pred_len 192 \
  --d_model 32 --global_d_model 128 \
  --batch_size 8 --learning_rate 0.01 \
  --patching_threshold 0.95 --monotonicity 1 \
  --dropout 0.1 --head_dropout 0.1 --fc_dropout 0.1 \
  --random_seed 68


set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
[ "${MODE:-eval}" = "train" ] && IS_TRAINING=1 || IS_TRAINING=0
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo; echo "----- Traffic  H=336  (mode=${MODE:-eval}) -----"
python -u run_longExp.py \
  --is_training "$IS_TRAINING" \
  --model_id "Traffic_L96_H336_large_bs8_lr1e-2_th095_dp01_mo1_s41" --model_id_name "Traffic" --setting_suffix "Traffic_L96_H336_large_bs8_lr1e-2_th095_dp01_mo1_s41" \
  --data "custom" --data_path "traffic.csv" --freq "h" \
  --enc_in 862 --dec_in 862 --c_out 862 \
  --pred_len 336 \
  --d_model 32 --global_d_model 128 \
  --batch_size 8 --learning_rate 0.01 \
  --patching_threshold 0.95 --monotonicity 1 \
  --dropout 0.1 --head_dropout 0.1 --fc_dropout 0.1 \
  --random_seed 41


set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
[ "${MODE:-eval}" = "train" ] && IS_TRAINING=1 || IS_TRAINING=0
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo; echo "----- Traffic  H=720  (mode=${MODE:-eval}) -----"
python -u run_longExp.py \
  --is_training "$IS_TRAINING" \
  --model_id "Traffic_L96_H720_large_bs8_lr1e-2_th095_dp01_mo1_s41" --model_id_name "Traffic" --setting_suffix "Traffic_L96_H720_large_bs8_lr1e-2_th095_dp01_mo1_s41" \
  --data "custom" --data_path "traffic.csv" --freq "h" \
  --enc_in 862 --dec_in 862 --c_out 862 \
  --pred_len 720 \
  --d_model 32 --global_d_model 128 \
  --batch_size 8 --learning_rate 0.01 \
  --patching_threshold 0.95 --monotonicity 1 \
  --dropout 0.1 --head_dropout 0.1 --fc_dropout 0.1 \
  --random_seed 41
