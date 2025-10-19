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
seq_len=96

# Random seeds
random_seeds="1025 2025 4025 5025 6025"

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

# Format: pred_len:quant_range:dim:multiple_of:heads:layers:batch_size:lr:dropout:max_patch:patching_threshold:patching_threshold_add:pct_start:epochs:patience:cross_attn_k:attn_window
configs=(
    # pred_len 96 
    "96:3:16:256:4:1:32:0.01:0.2:24:0.2:0.15:0.3:30:10:1:96"
    
    # pred_len 192
    "192:3:16:32:8:1:128:0.01:0.2:8:0.5:0.15:0.3:10:7:1:96"
    
    # pred_len 336
    "336:3:16:32:8:1:128:0.01:0.2:8:0.5:0.15:0.3:10:7:1:96"
    
    # pred_len 720
    "720:3:16:32:8:1:128:0.01:0.2:24:3.5:0.15:0.3:10:10:1:96"
)

# Run experiments
gpu_idx=0
for config in "${configs[@]}"; do
    IFS=':' read -r pred_len quant_range dim multiple_of heads layers batch_size learning_rate dropout max_patch_length patching_threshold patching_threshold_add pct_start train_epochs patience cross_attn_k attn_window <<< "$config"
    
    gpu_id=$((gpu_idx % NUM_GPUS))
    
    echo "Starting experiment on GPU $gpu_id: pred_len=$pred_len, dim=$dim, heads=$heads, batch_size=$batch_size"
    
    (
        for random_seed in $random_seeds; do
            log_file="logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}_seed${random_seed}.log"
            
            echo "Running seed $random_seed for pred_len=$pred_len on GPU $gpu_id"
            
            CUDA_VISIBLE_DEVICES=$gpu_id python -u run_longExp.py \
                --random_seed $random_seed \
                --is_training 1 \
                --root_path $root_path_name \
                --entropy_model_checkpoint_dir $entropy_model_checkpoint_dir \
                --data_path $data_path_name \
                --model_id ${model_id_name}_${seq_len}_${pred_len} \
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
                --cross_attn_k $cross_attn_k \
                --n_heads_local_encoder $heads \
                --n_heads_local_decoder $heads \
                --n_heads_global $heads \
                --cross_attn_nheads $heads \
                --cross_attn_window_encoder $attn_window \
                --cross_attn_window_decoder $attn_window \
                --local_attention_window_len $attn_window \
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
                --itr 1 \
                --batch_size $batch_size \
                --patching_batch_size $((batch_size * enc_in)) \
                --learning_rate $learning_rate \
                >$log_file 2>&1
            
            echo "Completed seed $random_seed for pred_len=$pred_len on GPU $gpu_id"
        done
        echo "Finished all seeds for pred_len=$pred_len on GPU $gpu_id"
    ) &
    
    gpu_idx=$((gpu_idx + 1))
done

wait
echo "All ETTm1 experiments completed!"
