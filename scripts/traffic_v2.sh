#!/bin/bash
# Train EntroPE_v2 on the Traffic dataset (862 channels, 17544 steps).
# Uses APE pooler + fusion head + local causal transformer (the improved v2 config).
source "$(dirname "$0")/run.sh"

mkdir -p ./logs/LongForecasting

# ── Dataset ────────────────────────────────────────────────────────────────
data=custom
root_path=/home/AD/sachith/TimeKAN/dataset/
data_path=traffic.csv
model_id_name=Traffic
enc_in=862
freq=h
seq_len=96
batch_size=16          # large channel count — keep batch small to fit GPU

# ── Architecture ───────────────────────────────────────────────────────────
d_model=64
n_heads=4
global_layers=3
local_layers=1         # shared local causal transformer before pooler & head
ape_local_layers=0     # 0: handled by local_layers above
mvg_cov_rank=4         # low-rank+diag: 862 + 862*4 = 4310 features vs 372453 full

# ── Patching ───────────────────────────────────────────────────────────────
cmi_threshold=0.125    # boundary rate → avg patch length 8
k_max_estimate=14

# ── Forecast head ──────────────────────────────────────────────────────────
fusion_token_dim=64

# ── Loss weights ───────────────────────────────────────────────────────────
lambda_count=0.0
lambda_min=0.0
lambda_div=0.0
lambda_nll=0.5

# ── Optimisation ───────────────────────────────────────────────────────────
learning_rate=0.0008
train_epochs=20
patience=5

# ── Experiment loop ────────────────────────────────────────────────────────
for pred_len in 96 192 336 720; do

    model_id="${model_id_name}_${seq_len}_${pred_len}_ape_fusion_local1_d64"

    echo "========================================================"
    echo "Traffic  seq=${seq_len}  pred=${pred_len}"
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
        --ape_local_layers $ape_local_layers \
        --cmi_threshold $cmi_threshold \
        --k_max_estimate $k_max_estimate \
        --ape_pooler \
        --fusion_head \
        --fusion_token_dim $fusion_token_dim \
        --lambda_count  $lambda_count \
        --lambda_min    $lambda_min \
        --lambda_div    $lambda_div \
        --lambda_nll    $lambda_nll \
        --mvg_cov_rank  $mvg_cov_rank \
        --learning_rate $learning_rate \
        --train_epochs  $train_epochs \
        --patience      $patience \
        --batch_size    $batch_size \
        > logs/LongForecasting/EntroPE_v2_${model_id}.log 2>&1

    echo "  → logs/LongForecasting/EntroPE_v2_${model_id}.log"

done

echo "All Traffic experiments finished."
