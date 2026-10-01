#!/usr/bin/env bash
# EntroPE — ETTm2 (input length L=96). Best config per horizon.
#
# Usage:  bash scripts/ETTm2.sh [train|eval]        (default: eval)
#   eval   evaluate the provided checkpoints (download_data.py --what checkpoints)
#   train  train from scratch, then evaluate (overwrites that cell's checkpoint)
# Pick a GPU with:  GPU=1 bash scripts/ETTm2.sh eval

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
[ "${MODE:-eval}" = "train" ] && IS_TRAINING=1 || IS_TRAINING=0
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo; echo "----- ETTm2  H=96  (mode=${MODE:-eval}) -----"
python -u run_longExp.py \
  --is_training "$IS_TRAINING" \
  --model_id "ETTm2_L96_H96_small_bs64_lr1e-2_th075_dp02_mo1_s2026" --model_id_name "ETTm2" --setting_suffix "ETTm2_L96_H96_small_bs64_lr1e-2_th075_dp02_mo1_s2026" \
  --data "ETTm2" --data_path "ETTm2.csv" --freq "t" \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --pred_len 96 \
  --d_model 8 --global_d_model 32 \
  --batch_size 64 --learning_rate 0.01 \
  --patching_threshold 0.75 --monotonicity 1 \
  --dropout 0.2 --head_dropout 0.2 --fc_dropout 0.2 \
  --random_seed 2026


set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
[ "${MODE:-eval}" = "train" ] && IS_TRAINING=1 || IS_TRAINING=0
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo; echo "----- ETTm2  H=192  (mode=${MODE:-eval}) -----"
python -u run_longExp.py \
  --is_training "$IS_TRAINING" \
  --model_id "ETTm2_L96_H192_medium_bs32_lr1e-2_th075_dp03_mo1_s2026" --model_id_name "ETTm2" --setting_suffix "ETTm2_L96_H192_medium_bs32_lr1e-2_th075_dp03_mo1_s2026" \
  --data "ETTm2" --data_path "ETTm2.csv" --freq "t" \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --pred_len 192 \
  --d_model 16 --global_d_model 64 \
  --batch_size 32 --learning_rate 0.01 \
  --patching_threshold 0.75 --monotonicity 1 \
  --dropout 0.3 --head_dropout 0.3 --fc_dropout 0.3 \
  --random_seed 2026


set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
[ "${MODE:-eval}" = "train" ] && IS_TRAINING=1 || IS_TRAINING=0
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo; echo "----- ETTm2  H=336  (mode=${MODE:-eval}) -----"
python -u run_longExp.py \
  --is_training "$IS_TRAINING" \
  --model_id "ETTm2_L96_H336_small_bs32_lr1e-2_th075_dp02_mo0_s2026" --model_id_name "ETTm2" --setting_suffix "ETTm2_L96_H336_small_bs32_lr1e-2_th075_dp02_mo0_s2026" \
  --data "ETTm2" --data_path "ETTm2.csv" --freq "t" \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --pred_len 336 \
  --d_model 8 --global_d_model 32 \
  --batch_size 32 --learning_rate 0.01 \
  --patching_threshold 0.75 --monotonicity 0 \
  --dropout 0.2 --head_dropout 0.2 --fc_dropout 0.2 \
  --random_seed 2026


set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-eval}"
[ "${MODE:-eval}" = "train" ] && IS_TRAINING=1 || IS_TRAINING=0
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo; echo "----- ETTm2  H=720  (mode=${MODE:-eval}) -----"
python -u run_longExp.py \
  --is_training "$IS_TRAINING" \
  --model_id "ETTm2_L96_H720_large_bs32_lr1e-3_th095_dp02_mo0_s2026" --model_id_name "ETTm2" --setting_suffix "ETTm2_L96_H720_large_bs32_lr1e-3_th095_dp02_mo0_s2026" \
  --data "ETTm2" --data_path "ETTm2.csv" --freq "t" \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --pred_len 720 \
  --d_model 32 --global_d_model 128 \
  --batch_size 32 --learning_rate 0.001 \
  --patching_threshold 0.95 --monotonicity 0 \
  --dropout 0.2 --head_dropout 0.2 --fc_dropout 0.2 \
  --random_seed 2026
