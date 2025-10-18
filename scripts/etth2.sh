#!/bin/bash

# Create log directories
mkdir -p ./logs/LongForecasting/$model_id_name

# Common parameters
model_name=EntroPE
root_path_name=./dataset/
entropy_model_checkpoint_dir=./entropy_model_checkpoints/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
enc_in=7
seq_len=96
quant_range=1
multiple_of=128
heads=2
monotonicity=1
patching_threshold=0.3
patching_threshold_add=0.15
max_patch_length=24

# Random seeds for all experiments
random_seeds="2025 2024 2023 2022 2021"

# Experiment configurations: gpu_id:pred_len:dim:layers:batch_size:lr:dropout:epochs:patience:pct_start
configs=(
    "0:96:32:2:420:0.1:0.1:60:20:0.4"
    "1:192:8:1:420:0.0001:0.2:50:40:0.3"
    "2:336:8:1:512:0.0001:0.3:50:40:0.3"
    "3:720:8:1:512:0.0001:0.3:50:40:0.3"
)

# Run experiments in parallel
for config in "${configs[@]}"; do
    IFS=':' read -r gpu_id pred_len dim layers batch_size learning_rate dropout train_epochs patience pct_start <<< "$config"
    
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
                --monotonicity $monotonicity \
                --des 'Exp' \
                --train_epochs $train_epochs \
                --patience $patience \
                --lradj 'TST' \
                --pct_start $pct_start \
                --batch_size $batch_size \
                --patching_batch_size $((batch_size * enc_in)) \
                --learning_rate $learning_rate \
                >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_seed'$random_seed.log
        done
        echo "Completed experiment on GPU $gpu_id: pred_len=$pred_len"
    ) &
done

# Wait for all background processes to complete
wait

echo "All experiments completed!"
