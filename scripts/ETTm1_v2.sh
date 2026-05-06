#!/bin/bash
# Train EntroPE_v2 on ETTm1 — APE pooler + fusion head + local transformer.
source "$(dirname "$0")/run.sh"

mkdir -p ./logs/LongForecasting

# ── Dataset ────────────────────────────────────────────────────────────────
data=ETTm1
root_path=/home/AD/sachith/PRECEPT_BASELINES/EntroPE/dataset/ETT-small
data_path=ETTm1.csv
model_id_name=ETTm1
enc_in=7
freq=t
seq_len=96
batch_size=128

# ── Architecture ───────────────────────────────────────────────────────────
d_model=64
n_heads=4
global_layers=3
local_layers=1

# ── Patching ───────────────────────────────────────────────────────────────
cmi_threshold=0.125
k_max_estimate=14

# ── Forecast head ──────────────────────────────────────────────────────────
fusion_token_dim=16

# ── Loss weights ───────────────────────────────────────────────────────────
lambda_count=0.0
lambda_min=0.0
lambda_div=0.0
lambda_nll=0.1

# ── Optimisation ───────────────────────────────────────────────────────────
learning_rate=0.01
dropout=0.1
train_epochs=30
patience=10

# ── Experiment loop ────────────────────────────────────────────────────────
for pred_len in 96 192 336 720; do

    model_id="${model_id_name}_${seq_len}_${pred_len}_ape_fusion_local1"

    echo "========================================================"
    echo "ETTm1  seq=${seq_len}  pred=${pred_len}"
    echo "========================================================"

    $PYTHON scripts/train_entrope_v2.py \
        --data          $data \
        --root_path     $root_path \
        --data_path     $data_path \
        --model_id_name $model_id_name \
        --model_id      $model_id \
        --seq_len       $seq_len \
        --pred_len      $pred_len \
        --enc_in        $enc_in \
        --freq          $freq \
        --d_model       $d_model \
        --n_heads       $n_heads \
        --global_layers $global_layers \
        --local_layers  $local_layers \
        --cmi_threshold $cmi_threshold \
        --k_max_estimate $k_max_estimate \
        --ape_pooler \
        --fusion_head \
        --fusion_token_dim $fusion_token_dim \
        --lambda_count  $lambda_count \
        --lambda_min    $lambda_min \
        --lambda_div    $lambda_div \
        --lambda_nll    $lambda_nll \
        --learning_rate $learning_rate \
        --dropout       $dropout \
        --train_epochs  $train_epochs \
        --patience      $patience \
        --batch_size    $batch_size \
        > logs/LongForecasting/EntroPE_v2_${model_id}.log 2>&1

    echo "  → logs/LongForecasting/EntroPE_v2_${model_id}.log"

done

echo "All ETTm1 experiments finished."
