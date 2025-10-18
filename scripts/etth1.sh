#!/bin/bash

# Create log directories
mkdir -p ./logs/LongForecasting

# Common parameters
model_name=EntroPE
root_path_name=./dataset/
entropy_model_checkpoint_dir=./entropy_model_checkpoints/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
enc_in=7
seq_len=96
monotonicity=1
patching_threshold=0.25
patching_threshold_add=0.15
train_epochs=50
patience=20

# Random seeds for all experiments
random_seeds="2025 2026 2027 2028 2029"

# Experiment configurations: gpu_id:pred_len:quant_range:dim:multiple_of:heads:layers:batch_size:lr:dropout:max_patch:pct_start
configs=(
    "0:96:3:8:256:1:1:256:0.001:0.1:8:0.5"
    "1:192:3:16:256:4:1:256:0.001:0.2:12:0.5"
    "2:336:3:8:256:2:2:256:0.0005:0.1:16:0.5"
    "3:720:3:8:256:1:1:256:0.001:0.1:8:0.5"
)

# Run experiments in parallel
for config in "${configs[@]}"; do
    IFS=':' read -r gpu_id pred_len quant_range dim multiple_of heads layers batch_size learning_rate dropout max_patch_length pct_start <<< "$config"
    
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
