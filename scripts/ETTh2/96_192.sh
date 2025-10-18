if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=EntroPE
root_path_name=./dataset/
entropy_model_checkpoint_dir=./entropy_model_checkpoints/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
enc_in=7
seq_len=96

quant_range=3
dim=8
multiple_of=256
heads=1
layers=2
batch_size=256
learning_rate=0.001
dropout=0.1
monotonicity=1
patching_threshold=0.25
patching_threshold_add=0.15
max_patch_length=8
train_epochs=50
patience=20


for random_seed in 2025 2024 2023 2022 2021
do
    for pred_len in 192 
    do
        python -u run_longExp.py \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --entropy_model_checkpoint_dir $entropy_model_checkpoint_dir \
        --data_path $data_path_name \
        --model_id $model_id_name_$seq_len'_'$pred_len \
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
        --multiple_of $multiple_of\
        --max_patch_length $max_patch_length\
        --patching_threshold $patching_threshold \
        --patching_threshold_add $patching_threshold_add \
        --monotonicity $monotonicity \
        --des 'Exp' \
        --train_epochs $train_epochs \
        --patience $patience \
        --lradj 'TST'\
        --pct_start 0.3\
        --batch_size $batch_size \
        --patching_batch_size $((batch_size * enc_in)) \
        --learning_rate $learning_rate \
        >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
    done
done
