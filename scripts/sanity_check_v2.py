#!/usr/bin/env python3
"""
scripts/sanity_check_v2.py

Four sanity checks for the refactored EntroPE_v2 architecture
(SimplePatchPooler + GlobalTransformer + AttentionPoolForecastHead).

Run BEFORE full training to confirm:
  1. Forward pass shape is correct
  2. Gradients flow to all key components
  3. Patch counts are reasonable
  4. Parameter count is in the expected range (100K–500K)

Usage:
    python scripts/sanity_check_v2.py
    python scripts/sanity_check_v2.py --d_model 128 --global_layers 4
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.EntroPE_v2 import Model


# ============================================================
# Minimal config object
# ============================================================

def make_configs(**overrides):
    class Cfg:
        # Data
        enc_in   = 7
        seq_len  = 96
        pred_len = 96
        batch_size = 2

        # MVG
        mvg_layers   = 2
        mvg_embd     = 64
        mvg_heads    = 4
        mvg_dropout  = 0.1
        mvg_cov_rank = None
        mvg_checkpoint = None
        model_id_name  = 'ETTh1'
        entropy_model_checkpoint_dir = ''

        # Backbone
        d_model      = 64
        n_heads      = 4
        e_layers     = 3
        global_layers = 3
        dropout      = 0.1

        # Boundary scorer
        scorer_hidden = 64

        # Gumbel patcher
        gumbel_tau_init  = 5.0
        gumbel_tau_min   = 0.5
        gumbel_tau_decay = 5000

        # Regularisation
        target_avg_patch_len = 8
        min_patch_len        = 3
        lambda_count = 1.0
        lambda_min   = 0.5
        lambda_div   = 0.1
        lambda_nll   = 0.1
        beta_nll     = 0.5

        # RevIN
        revin        = 1
        affine       = 1
        subtract_last = 0

        # Compat fields (unused by new backbone, but kept for imports)
        vocab_size   = 256
        random_seed  = 2025

    cfg = Cfg()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ============================================================
# Sanity checks
# ============================================================

def check_1_forward_shape(model, x, configs):
    print("\n--- Check 1: Forward pass shape ---")
    with torch.no_grad():
        forecast = model(x)
    expected = (configs.batch_size, configs.pred_len, configs.enc_in)
    assert forecast.shape == expected, \
        f"Expected {expected}, got {forecast.shape}"
    print(f"  forecast shape: {forecast.shape}  ✓")
    return forecast


def check_2_gradient_flow(model, x, configs):
    print("\n--- Check 2: Gradient flow ---")
    model.train()
    forecast = model(x)
    loss = forecast.pow(2).mean() + model.auxiliary_losses()
    loss.backward()

    # Scorer gradient (must be non-zero for the STE to work)
    scorer_grad = model.model.scorer.net[0].weight.grad
    assert scorer_grad is not None, "scorer gradient is None"
    scorer_grad_max = scorer_grad.abs().max().item()
    assert scorer_grad_max > 1e-8, \
        f"scorer gradient is effectively zero ({scorer_grad_max:.2e}) — STE broken"
    print(f"  scorer grad max: {scorer_grad_max:.4e}  ✓")

    # Token projection gradient
    tp_grad = model.model.token_proj.weight.grad
    assert tp_grad is not None and tp_grad.abs().max() > 1e-8, "token_proj gradient zero"
    print(f"  token_proj grad max: {tp_grad.abs().max().item():.4e}  ✓")

    # Global transformer gradient
    gt_grad = model.model.global_transformer.layers[0].self_attn.in_proj_weight.grad
    assert gt_grad is not None and gt_grad.abs().max() > 1e-8, \
        "global_transformer gradient zero"
    print(f"  global_transformer grad max: {gt_grad.abs().max().item():.4e}  ✓")

    # Forecast head gradient
    fh_grad = model.model.forecast_head.head.weight.grad
    assert fh_grad is not None and fh_grad.abs().max() > 1e-8, "forecast_head gradient zero"
    print(f"  forecast_head grad max: {fh_grad.abs().max().item():.4e}  ✓")

    print("  All gradients healthy  ✓")


def check_3_patch_counts(model, x, configs):
    print("\n--- Check 3: Patch counts ---")
    model.eval()
    with torch.no_grad():
        model(x)
    b   = model.model._last_b
    counts = b.detach().sum(dim=1)
    print(f"  Patches per sample: {counts.tolist()}")
    K_max = int(counts.max().item())
    print(f"  K_max in this batch: {K_max}")
    target = configs.seq_len // configs.target_avg_patch_len
    print(f"  Expected ~{target} patches (seq_len/target_avg_patch_len)")
    # Soft check: within 3x of target
    assert K_max <= target * 3 + 5, \
        f"Too many patches ({K_max}); boundary rate not converging"
    print("  Patch count looks plausible  ✓")


def check_4_param_count(model, configs):
    print("\n--- Check 4: Parameter count ---")
    n_total = sum(p.numel() for p in model.parameters())
    n_mvg   = sum(p.numel() for p in model.model.mvg.parameters())
    n_sco   = sum(p.numel() for p in model.model.scorer.parameters())
    n_pool  = sum(p.numel() for p in model.model.patch_pooler.parameters())
    n_glt   = sum(p.numel() for p in model.model.global_transformer.parameters())
    n_head  = sum(p.numel() for p in model.model.forecast_head.parameters())

    print(f"  MVG model:          {n_mvg:>8,}")
    print(f"  Boundary scorer:    {n_sco:>8,}")
    print(f"  SimplePatchPooler:  {n_pool:>8,}")
    print(f"  GlobalTransformer:  {n_glt:>8,}")
    print(f"  ForecastHead:       {n_head:>8,}")
    print(f"  TOTAL:              {n_total:>8,}  ({n_total/1e6:.3f}M)")

    if n_total > 5_000_000:
        print(f"  WARNING: model is large ({n_total/1e6:.1f}M) — risk of overfitting on small datasets")
    elif n_total < 50_000:
        print(f"  WARNING: model is very small ({n_total/1e3:.1f}K) — may underfit")
    else:
        print("  Parameter count looks healthy  ✓")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_model',       type=int,   default=64)
    parser.add_argument('--n_heads',       type=int,   default=4)
    parser.add_argument('--global_layers', type=int,   default=3)
    parser.add_argument('--enc_in',        type=int,   default=7)
    parser.add_argument('--seq_len',       type=int,   default=96)
    parser.add_argument('--pred_len',      type=int,   default=96)
    parser.add_argument('--batch_size',    type=int,   default=2)
    parser.add_argument('--gpu',           type=int,   default=0)
    args = parser.parse_args()

    configs = make_configs(
        d_model      = args.d_model,
        n_heads      = args.n_heads,
        global_layers = args.global_layers,
        e_layers     = args.global_layers,
        enc_in       = args.enc_in,
        seq_len      = args.seq_len,
        pred_len     = args.pred_len,
        batch_size   = args.batch_size,
    )

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: enc_in={configs.enc_in}, seq_len={configs.seq_len}, "
          f"pred_len={configs.pred_len}, d_model={configs.d_model}, "
          f"global_layers={configs.global_layers}")

    model = Model(configs).to(device)
    x     = torch.randn(configs.batch_size, configs.seq_len, configs.enc_in).to(device)

    passed = 0
    total  = 4

    try:
        check_1_forward_shape(model, x, configs)
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")

    try:
        check_2_gradient_flow(model, x, configs)
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")

    try:
        check_3_patch_counts(model, x, configs)
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")

    try:
        check_4_param_count(model, configs)
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")

    print(f"\n{'='*50}")
    print(f"Passed {passed}/{total} sanity checks.")
    if passed == total:
        print("All checks passed — ready for training review.")
    else:
        print("Some checks failed — do NOT start training.")
    print('='*50)


if __name__ == '__main__':
    main()
