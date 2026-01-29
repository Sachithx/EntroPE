#!/bin/bash

mkdir -p ./logs/LongForecasting

# =========================
# Fixed experiment metadata
# =========================
model=EntroPE
data=ETTh1
features=M
root_path=./dataset/
data_path=ETTh1.csv
model_id_name=ETTh1
enc_in=7
freq=h
seq_len=96
batch_size=128
seed=2026

# =========================
# Core architecture params
# =========================
d_model=8 
n_heads=2
e_layers=3
d_ff=256

# =========================
# Patching params
# =========================
max_patch_length=96
patching_threshold=0.25
monotonicity=0

# =========================
# Optimization params
# =========================
dropout=0.1
learning_rate=0.01
train_epochs=20
itr=1
des=Exp

# =========================
# Training mode
# =========================
is_training=1

# =========================
# Experiment loop
# =========================
for pred_len in 96; do
    
    model_id="${model_id_name}_${seq_len}preds${pred_len}"
    
    echo "Running $seq_len -> $pred_len ..."
    
    python -u run_longExp.py \
        --model $model \
        --data $data \
        --features $features \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --root_path $root_path \
        --data_path $data_path \
        --model_id_name $model_id_name \
        --model_id $model_id \
        --freq $freq \
        --enc_in $enc_in \
        --d_model $d_model \
        --n_heads $n_heads \
        --e_layers $e_layers \
        --d_ff $d_ff \
        --max_patch_length $max_patch_length \
        --patching_threshold $patching_threshold \
        --monotonicity $monotonicity \
        --dropout $dropout \
        --learning_rate $learning_rate \
        --batch_size $batch_size \
        --train_epochs $train_epochs \
        --itr $itr \
        --des $des \
        --is_training $is_training \
        --random_seed $seed \
        > logs/LongForecasting/${model}_${model_id}.log 2>&1

done

echo "All experiments finished."
