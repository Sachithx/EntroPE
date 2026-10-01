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
[ "${MODE:-eval}" = "train" ] && IS_TRAINING=1 || IS_TRAINING=0
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo; echo "----- solar  H=96  (mode=${MODE:-eval}) -----"
python -u run_longExp.py \
  --is_training "$IS_TRAINING" \
  --model_id "solar_L96_H96_medium_bs64_lr1e-2_th095_dp01_mo0_s44" --model_id_name "solar" --setting_suffix "solar_L96_H96_medium_bs64_lr1e-2_th095_dp01_mo0_s44" \
  --data "custom" --data_path "solar.csv" --freq "t" \
  --enc_in 137 --dec_in 137 --c_out 137 \
  --pred_len 96 \
  --d_model 16 --global_d_model 64 \
  --batch_size 64 --learning_rate 0.01 \
  --patching_threshold 0.95 --monotonicity 0 \
  --dropout 0.1 --head_dropout 0.1 --fc_dropout 0.1 \
  --random_seed 44


set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
[ "${MODE:-eval}" = "train" ] && IS_TRAINING=1 || IS_TRAINING=0
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo; echo "----- solar  H=192  (mode=${MODE:-eval}) -----"
python -u run_longExp.py \
  --is_training "$IS_TRAINING" \
  --model_id "solar_L96_H192_medium_bs32_lr1e-2_th085_dp01_mo0_s94" --model_id_name "solar" --setting_suffix "solar_L96_H192_medium_bs32_lr1e-2_th085_dp01_mo0_s94" \
  --data "custom" --data_path "solar.csv" --freq "t" \
  --enc_in 137 --dec_in 137 --c_out 137 \
  --pred_len 192 \
  --d_model 16 --global_d_model 64 \
  --batch_size 32 --learning_rate 0.01 \
  --patching_threshold 0.85 --monotonicity 0 \
  --dropout 0.1 --head_dropout 0.1 --fc_dropout 0.1 \
  --random_seed 94


set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
[ "${MODE:-eval}" = "train" ] && IS_TRAINING=1 || IS_TRAINING=0
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo; echo "----- solar  H=336  (mode=${MODE:-eval}) -----"
python -u run_longExp.py \
  --is_training "$IS_TRAINING" \
  --model_id "solar_L96_H336_medium_bs64_lr1e-2_th095_dp01_mo0_s10" --model_id_name "solar" --setting_suffix "solar_L96_H336_medium_bs64_lr1e-2_th095_dp01_mo0_s10" \
  --data "custom" --data_path "solar.csv" --freq "t" \
  --enc_in 137 --dec_in 137 --c_out 137 \
  --pred_len 336 \
  --d_model 16 --global_d_model 64 \
  --batch_size 64 --learning_rate 0.01 \
  --patching_threshold 0.95 --monotonicity 0 \
  --dropout 0.1 --head_dropout 0.1 --fc_dropout 0.1 \
  --random_seed 10


set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
[ "${MODE:-eval}" = "train" ] && IS_TRAINING=1 || IS_TRAINING=0
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo; echo "----- solar  H=720  (mode=${MODE:-eval}) -----"
python -u run_longExp.py \
  --is_training "$IS_TRAINING" \
  --model_id "solar_L96_H720_medium_bs32_lr1e-2_th085_dp01_mo1_s10" --model_id_name "solar" --setting_suffix "solar_L96_H720_medium_bs32_lr1e-2_th085_dp01_mo1_s10" \
  --data "custom" --data_path "solar.csv" --freq "t" \
  --enc_in 137 --dec_in 137 --c_out 137 \
  --pred_len 720 \
  --d_model 16 --global_d_model 64 \
  --batch_size 32 --learning_rate 0.01 \
  --patching_threshold 0.85 --monotonicity 1 \
  --dropout 0.1 --head_dropout 0.1 --fc_dropout 0.1 \
  --random_seed 10
