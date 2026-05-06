#!/bin/bash
# Diagnostic runs for EntroPE_v2 on ETTh1 (seq_len=96, pred_len=96).
#
# Usage:
#   bash scripts/run_diagnostics.sh [d1|d2|d3|d4|d5|all]
#
# Recommended order: d5 → d1 → d3 (instant) → d2 → d4
#
# Interpretation guide
#   D1 val_mse ~0.45 → bug is the learned patcher (variance collapse likely, see D3)
#   D1 val_mse ~1.0  → bug is encoder or forecast head, try D4
#   D2 better than baseline → rich tokens are corrupted (variance collapse)
#   D2 same as baseline    → rich tokens are fine, problem is elsewhere
#   D3 log_diag.min ≈ -7   → variance collapsed, MVG needs pretraining / freezing
#   D3 cmi.mean < 1e-4     → scorer is receiving noise, boundary signal useless
#   D4 better than baseline → AttentionPoolForecastHead is the bottleneck
#   D5 val_mse ~1.0        → data pipeline or RevIN bug (not model-specific)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

PYTHON="${PYTHON:-/home/AD/sachith/.conda/envs/entrope/bin/python}"

DATA_ROOT="./dataset/ETT-small"
DATA_PATH="ETTh1.csv"
DATA="ETTh1"
SEQ=96
PRED=96
ENC=7
EPOCHS=15          # short run — enough to see convergence trend
LOG_EVERY=20
BATCH=128
LR=1e-4

# Shared base args
BASE=(
    --data          "$DATA"
    --root_path     "$DATA_ROOT"
    --data_path     "$DATA_PATH"
    --seq_len       "$SEQ"
    --pred_len      "$PRED"
    --enc_in        "$ENC"
    --d_model       64
    --n_heads       4
    --e_layers      2
    --global_layers 3
    --mvg_layers    2
    --mvg_embd      64
    --mvg_heads     4
    --train_epochs  "$EPOCHS"
    --batch_size    "$BATCH"
    --learning_rate "$LR"
    --log_every     "$LOG_EVERY"
    --patience      10
    --gpu           0
)

mkdir -p logs/diagnostics

run_diag() {
    local tag="$1"; shift
    echo "========================================================"
    echo " Running $tag"
    echo "========================================================"
    $PYTHON scripts/train_entrope_v2.py \
        "${BASE[@]}" \
        --model_id "diag_${tag}" \
        "$@" \
        2>&1 | tee "logs/diagnostics/${tag}.log"
    echo "→ log: logs/diagnostics/${tag}.log"
}

TARGET="${1:-all}"

# ----------------------------------------------------------------
# D6 — CMI top-K patching (THE FIX)
#      cmi_threshold is interpreted as a TARGET BOUNDARY RATE (0 < r < 1):
#        K = floor(seq_len * rate) boundaries placed at highest CMI spikes.
#      Scale-invariant — unaffected by CMI drift during training.
#      (Previous absolute-threshold approach failed: patches grew 30→54.)
#
#      Rate sweep around target_avg_patch_len=8 (rate=0.125):
#        0.083 → avg patch len 12
#        0.125 → avg patch len  8  (matches D1 static)
#        0.167 → avg patch len  6
#
#      k_max_estimate = K + 2 slack = floor(seq_len*rate) + 2
# ----------------------------------------------------------------
if [[ "$TARGET" == "d6" || "$TARGET" == "all" ]]; then
    for rate in 0.083 0.125 0.167; do
        # k_max = floor(96 * rate) + 2
        k_max=$(python3 -c "import math; print(math.floor(96*${rate}) + 2)")
        run_diag "D6_cmi_topk_${rate}" \
            --cmi_threshold "$rate" \
            --flat_head \
            --k_max_estimate "$k_max" \
            --lambda_count 0.0 \
            --lambda_min 0.0 \
            --lambda_div 0.0 \
            --lambda_nll 0.1 \
            --train_epochs 30 \
            --patience 10
    done
fi

# ----------------------------------------------------------------
# D1 — Static patches (length 8), no learned boundary scorer
#      Rules out: encoder / forecast head are the problem
# ----------------------------------------------------------------
if [[ "$TARGET" == "d1" || "$TARGET" == "all" ]]; then
    run_diag "D1_static_patches" \
        --static_patch_len 8
fi

# ----------------------------------------------------------------
# D2 — No rich tokens (only raw x, drop mu/vech_L/r/cmi)
#      Rules out: rich token contamination
# ----------------------------------------------------------------
if [[ "$TARGET" == "d2" || "$TARGET" == "all" ]]; then
    run_diag "D2_no_rich_tokens" \
        --no_rich_tokens
fi

# ----------------------------------------------------------------
# D3 — Static patches + print MVG variance statistics
#      Diagnoses: variance collapse (log_diag.min ≈ -7)
#                 silent CMI (cmi.mean < 1e-4)
#      (D3 stats print automatically every --log_every steps)
# ----------------------------------------------------------------
if [[ "$TARGET" == "d3" || "$TARGET" == "all" ]]; then
    run_diag "D3_mvg_stats" \
        --static_patch_len 8 \
        --log_every 5
fi

# ----------------------------------------------------------------
# D4 — Static patches + SimpleFlattenHead (PatchTST-style)
#      Rules out: AttentionPoolForecastHead as bottleneck
# ----------------------------------------------------------------
if [[ "$TARGET" == "d4" || "$TARGET" == "all" ]]; then
    run_diag "D4_flat_head" \
        --static_patch_len 8 \
        --flat_head \
        --k_max_estimate 20
fi

# ----------------------------------------------------------------
# D5 — Old EntroPE (discrete entropy, original stack)
#      Rules out: data pipeline / RevIN bugs
# ----------------------------------------------------------------
if [[ "$TARGET" == "d5" || "$TARGET" == "all" ]]; then
    echo "========================================================"
    echo " Running D5 — old EntroPE (discrete entropy baseline)"
    echo "========================================================"
    $PYTHON run_longExp.py \
        --model      EntroPE \
        --data       "$DATA" \
        --root_path  "$DATA_ROOT" \
        --data_path  "$DATA_PATH" \
        --features   M \
        --seq_len    "$SEQ" \
        --pred_len   "$PRED" \
        --enc_in     "$ENC" \
        --freq       h \
        --d_model    64 \
        --n_heads    4 \
        --e_layers   3 \
        --d_ff       256 \
        --max_patch_length 32 \
        --patching_threshold 0.95 \
        --monotonicity 0 \
        --dropout    0.1 \
        --learning_rate 0.01 \
        --batch_size "$BATCH" \
        --train_epochs "$EPOCHS" \
        --itr        1 \
        --des        diag_D5 \
        --model_id   diag_D5_ETTh1_96_96 \
        --model_id_name ETTh1 \
        --is_training 1 \
        --random_seed 42 \
        2>&1 | tee logs/diagnostics/D5_old_entrope.log
    echo "→ log: logs/diagnostics/D5_old_entrope.log"
fi

echo ""
echo "========================================================"
echo " All requested diagnostics done."
echo " Results in logs/diagnostics/"
echo ""
echo " Key thresholds:"
echo "   D1 val_mse < 0.55  → patcher is the bug"
echo "   D1 val_mse > 0.90  → encoder/head is the bug → try D4"
echo "   D3 log_diag.min ≈ -7 → variance collapsed, pretrain MVG"
echo "   D3 cmi.mean < 1e-4   → scorer getting noise"
echo "   D5 val_mse > 0.90    → data pipeline bug"
echo "========================================================"
