#!/usr/bin/env bash
# _common.sh — shared runner sourced by each per-dataset script
# (scripts/<DATASET>.sh). Not meant to be run directly.
#
# The per-dataset script sets MODE, GPU and the dataset fields
# (DATASET / DATA_KIND / DATA_FILE / FREQ / CHANNELS), then calls `run_cell`
# once per horizon with that horizon's tunable config.
#
# Everything fixed across all cells (architecture L=96, n_heads=4, e_layers=2,
# d_ff=256, boundary_method=entropy, revin, 50 epochs, the frozen entropy-model
# dir, …) is baked into run_longExp.py's argparse defaults, so it does not need
# to be repeated here or in the dataset scripts.

[ "${MODE:-eval}" = "train" ] && IS_TRAINING=1 || IS_TRAINING=0
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo "=================================================================="
echo "  EntroPE | dataset=${DATASET} | mode=${MODE:-eval} | L=96"
echo "  eval = evaluate provided checkpoints;  train = train then evaluate"
echo "=================================================================="

# run_cell <horizon> <d_model> <global_d_model> <batch_size> <lr> \
#          <threshold> <dropout> <monotonicity> <seed> <run_id>
run_cell () {
  local horizon=$1 d_model=$2 global_d_model=$3 batch_size=$4 lr=$5 \
        threshold=$6 dropout=$7 monotonicity=$8 seed=$9 run_id=${10}

  echo; echo "----- ${DATASET}  H=${horizon}  (mode=${MODE:-eval}) -----"
  python -u run_longExp.py \
    --is_training "$IS_TRAINING" \
    --model_id "$run_id" --model_id_name "$DATASET" --setting_suffix "$run_id" \
    --data "$DATA_KIND" --data_path "$DATA_FILE" --freq "$FREQ" \
    --enc_in "$CHANNELS" --dec_in "$CHANNELS" --c_out "$CHANNELS" \
    --pred_len "$horizon" \
    --d_model "$d_model" --global_d_model "$global_d_model" \
    --batch_size "$batch_size" --learning_rate "$lr" \
    --patching_threshold "$threshold" --monotonicity "$monotonicity" \
    --dropout "$dropout" --head_dropout "$dropout" --fc_dropout "$dropout" \
    --random_seed "$seed"
}
