#!/bin/bash

mkdir -p ./logs/LongForecasting

# Common parameters
model_name=EntroPE
root_path_name=./dataset/
entropy_model_checkpoint_dir=./entropy_model_checkpoints/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
enc_in=7

# Try longer sequence length for minute-level data
seq_len=96  

# Random seeds
random_seeds="2025 2026 2027 2028 2029"

# Detect GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -eq 0 ]; then
        NUM_GPUS=1
    fi
else
    NUM_GPUS=1
fi
echo "Using $NUM_GPUS GPU(s)"

# Experiment configurations: pred_len:quant_range:dim:multiple_of:heads:layers:batch_size:lr:dropout:max_patch:patching_threshold:patching_threshold_add:pct_start:epochs:patience
configs=(
    # pred_len 96 - adjusted parameters
    "96:3:16:256:2:1:256:0.001:0.1:16:0.25:0.15:0.3:50:20"
    # pred_len 192 - lower lr, more regularization
    "192:3:16:256:2:1:256:0.0005:0.2:16:0.25:0.15:0.3:50:20"
    # pred_len 336 - even lower lr
    "336:3:8:256:2:1:256:0.0001:0.2:20:0.25:0.15:0.3:50:20"
    # pred_len 720 - smallest model, lowest lr
    "720:3:8:256:1:1:256:0.0001:0.3:24:0.25:0.15:0.3:50:20"
)

# Run experiments
gpu_idx=0
for config in "${configs[@]}"; do
    IFS=':' read -r pred_len quant_range dim multiple_of heads layers batch_size learning_rate dropout max_patch_length patching_threshold patching_threshold_add pct_start train_epochs patience <<< "$config"
    
    gpu_id=$((gpu_idx % NUM_GPUS))
    
    echo "Starting experiment on GPU $gpu_id: pred_len=$pred_len"
    
    (
        for random_seed in $random_seeds; do
            CUDA_VISIBLE_DEVICES=$gpu_id python -u run_longExp.py \
                --random_seed $random_seed \
                --is_training 1 \
                --root_path $root_path_name \
                --entropy_model_checkpoint_dir $entropy_model_checkpoint_dir \
                --data_path $data_path_name \
                --model_id $model_id_name'_'$seq_len'_'$pred_len \
                --model_id_name $model_id_name \
                --model $model_name \
                --data $data_name \
                --features M \
                --seq_len $seq_len \
                --pred_len $pred_len \
                --enc_in $enc_in \
                --vocab_size 256 \
                --quant_range $quant_range \
                --n_layers_local_encoder $layers \
                --n_layers_local_decoder $layers \
                --n_layers_global $layers \
                --dim_global $dim \
                --dim_local_encoder $dim \
                --dim_local_decoder $dim \
                --n_heads_local_encoder $heads \
                --n_heads_local_decoder $heads \
                --n_heads_global $heads \
                --cross_attn_nheads $heads \
                --dropout $dropout \
                --multiple_of $multiple_of \
                --max_patch_length $max_patch_length \
                --patching_threshold $patching_threshold \
                --patching_threshold_add $patching_threshold_add \
                --monotonicity 1 \
                --des 'Exp' \
                --train_epochs $train_epochs \
                --patience $patience \
                --lradj 'TST' \
                --pct_start $pct_start \
                --batch_size $batch_size \
                --patching_batch_size $((batch_size * enc_in)) \
                --learning_rate $learning_rate \
                >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_seed'$random_seed.log 2>&1
        done
        echo "Completed experiment on GPU $gpu_id: pred_len=$pred_len"
    ) &
    
    gpu_idx=$((gpu_idx + 1))
done

wait
echo "All experiments completed!"
