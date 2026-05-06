#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Modifications and extensions:
#   - End-to-end training script for EntroPE_v2 with learned CMI patching
#
# Copyright (c) 2026 Sachith Abeywickrama
#
# Licensed under the same license as the original Meta code.

"""
scripts/train_entrope_v2.py

End-to-end training entrypoint for EntroPE_v2 (--patching_mode learned_cmi).

Workflow:
  1. Optionally load a pretrained MVG checkpoint (from scripts/pretrain_mvg.py).
  2. Joint training: forecast MSE loss + auxiliary losses.
  3. Evaluate on val/test splits, report MSE + MAE.

Usage (ETTh1, pred_len=96):
    python scripts/train_entrope_v2.py \
        --data ETTh1 \
        --root_path ./data/ETT/ \
        --data_path ETTh1.csv \
        --seq_len 96 \
        --pred_len 96 \
        --enc_in 7 \
        --d_model 64 \
        --n_heads 4 \
        --e_layers 2 \
        --mvg_checkpoint checkpoints/mvg/mvg_ETTh1.pt \
        --train_epochs 30 \
        --learning_rate 1e-4 \
        --gpu 0
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_provider.data_factory import data_provider
from models.EntroPE_v2 import Model as EntroPE_v2_Model
from utils.metrics import metric
from utils.tools import EarlyStopping


# ============================================================
# Args
# ============================================================

def get_args():
    p = argparse.ArgumentParser(description='Train EntroPE_v2 (learned CMI patching)')

    # --- Data ---
    p.add_argument('--data',        type=str,   default='ETTh1')
    p.add_argument('--root_path',   type=str,   default='./data/ETT/')
    p.add_argument('--data_path',   type=str,   default='ETTh1.csv')
    p.add_argument('--features',    type=str,   default='M')
    p.add_argument('--target',      type=str,   default='OT')
    p.add_argument('--freq',        type=str,   default='h')
    p.add_argument('--seq_len',     type=int,   default=96)
    p.add_argument('--label_len',   type=int,   default=48)
    p.add_argument('--pred_len',    type=int,   default=96)
    p.add_argument('--enc_in',      type=int,   default=7)
    p.add_argument('--num_workers', type=int,   default=4)
    p.add_argument('--model_id_name', type=str, default='ETTh1')

    # --- Model (backbone) ---
    p.add_argument('--d_model',      type=int,   default=64)
    p.add_argument('--n_heads',      type=int,   default=4)
    p.add_argument('--e_layers',     type=int,   default=2)
    p.add_argument('--global_layers',type=int,   default=3,
                   help='GlobalTransformer encoder layers (defaults to e_layers if absent)')
    p.add_argument('--static_patch_len', type=int, default=0,
                   help='Diagnostic 1: use static patches of this length (0=learned)')
    p.add_argument('--no_rich_tokens', action='store_true', default=False,
                   help='Diagnostic 2: use only raw x as token input (drop mu/vech_L/r/cmi)')
    p.add_argument('--ape_pooler', action='store_true', default=False,
                   help='Use AdaptivePatchPooler (APE cross-attn) instead of SimplePatchPooler')
    p.add_argument('--ape_local_layers', type=int, default=0,
                   help='Local causal transformer layers inside APE before cross-attn (0=disabled)')
    p.add_argument('--local_layers', type=int, default=0,
                   help='Shared local causal transformer layers applied to token_emb before pooler and head')
    p.add_argument('--fusion_head', action='store_true', default=False,
                   help='Use FusionForecastHead (token×patch cross-attn + flatten) as forecast head')
    p.add_argument('--fusion_token_dim', type=int, default=None,
                   help='Compression dim in FusionForecastHead before flatten (default d_model//4)')
    p.add_argument('--flat_head', action='store_true', default=False,
                   help='Diagnostic 4: use PatchTST-style flatten-and-project head')
    p.add_argument('--k_max_estimate', type=int, default=20,
                   help='Diagnostic 4: fixed K for SimpleFlattenHead (pad/truncate to this)')
    p.add_argument('--cmi_threshold', type=float, default=0.0,
                   help='CMI top-K patching rate (0=disabled). Interpreted as target boundary RATE '
                        '(fraction of timesteps): e.g. 0.125 = avg patch length 8. '
                        'Places exactly floor(seq_len * rate) boundaries at highest CMI spikes. '
                        'Scale-invariant — stable under CMI drift during training.')
    p.add_argument('--d_layers',     type=int,   default=1)
    p.add_argument('--d_ff',         type=int,   default=128)
    p.add_argument('--dropout',      type=float, default=0.1)
    p.add_argument('--fc_dropout',   type=float, default=0.1)
    p.add_argument('--head_dropout', type=float, default=0.0)
    p.add_argument('--activation',   type=str,   default='swiglu')
    p.add_argument('--individual',   type=int,   default=0)
    p.add_argument('--revin',        type=int,   default=1)
    p.add_argument('--affine',       type=int,   default=1)
    p.add_argument('--subtract_last',type=int,   default=0)
    p.add_argument('--max_patch_length', type=int, default=8)

    # Compat fields used by backbone arg builder
    p.add_argument('--cross_attn_k',              type=int, default=1)
    p.add_argument('--cross_attn_nheads',         type=int, default=4)
    p.add_argument('--cross_attn_window_encoder', type=int, default=96)
    p.add_argument('--cross_attn_window_decoder', type=int, default=96)
    p.add_argument('--local_attention_window_len',type=int, default=96)
    p.add_argument('--patching_threshold',        type=float, default=0.5)
    p.add_argument('--patching_batch_size',       type=int,   default=128)
    p.add_argument('--monotonicity',              type=int,   default=0)
    p.add_argument('--entropy_model_checkpoint_dir', type=str, default='./entropy_model_checkpoints/')
    p.add_argument('--vocab_size',                type=int,   default=256)
    p.add_argument('--random_seed',               type=int,   default=2025)

    # --- MVG model ---
    p.add_argument('--mvg_layers',   type=int,   default=2)
    p.add_argument('--mvg_embd',     type=int,   default=64)
    p.add_argument('--mvg_heads',    type=int,   default=4)
    p.add_argument('--mvg_dropout',  type=float, default=0.1)
    p.add_argument('--mvg_cov_rank', type=int,   default=None)
    p.add_argument('--mvg_checkpoint', type=str, default=None,
                   help='Path to pretrained MVG checkpoint (from pretrain_mvg.py)')

    # --- Boundary scorer ---
    p.add_argument('--scorer_hidden', type=int, default=64)

    # --- Gumbel patcher ---
    p.add_argument('--gumbel_tau_init',   type=float, default=5.0)
    p.add_argument('--gumbel_tau_min',    type=float, default=0.5)
    p.add_argument('--gumbel_tau_decay',  type=int,   default=5000)

    # --- Patch regularisation ---
    p.add_argument('--target_avg_patch_len', type=int,   default=8)
    p.add_argument('--min_patch_len',        type=int,   default=3)
    p.add_argument('--lambda_count',         type=float, default=10.0)
    p.add_argument('--lambda_min',           type=float, default=0.5)
    p.add_argument('--lambda_div',           type=float, default=0.1)
    p.add_argument('--lambda_nll',           type=float, default=0.1)
    p.add_argument('--beta_nll',             type=float, default=0.5)

    # --- Training ---
    p.add_argument('--train_epochs',   type=int,   default=30)
    p.add_argument('--batch_size',     type=int,   default=128)
    p.add_argument('--learning_rate',  type=float, default=1e-4)
    p.add_argument('--weight_decay',   type=float, default=0.05)
    p.add_argument('--grad_clip',      type=float, default=1.0)
    p.add_argument('--patience',       type=int,   default=10)
    p.add_argument('--pct_start',      type=float, default=0.3)
    p.add_argument('--use_amp',        action='store_true', default=False)

    # --- Checkpointing ---
    p.add_argument('--checkpoints', type=str, default='./checkpoints/')
    p.add_argument('--model_id',    type=str, default='EntroPE_v2_test')
    p.add_argument('--des',         type=str, default='exp')

    # --- GPU ---
    p.add_argument('--gpu',           type=int,  default=0)
    p.add_argument('--use_gpu',       type=bool, default=True)
    p.add_argument('--use_multi_gpu', action='store_true', default=False)
    p.add_argument('--devices',       type=str,  default='0')

    # --- Logging ---
    p.add_argument('--log_every',  type=int, default=20)
    p.add_argument('--itr',        type=int, default=1)
    p.add_argument('--lradj',      type=str, default='TST')

    return p.parse_args()


# ============================================================
# Helpers
# ============================================================

def _patch_args(args):
    """Add fields expected by data_provider and backbone arg builder."""
    args.embed   = 'timeF'
    args.dec_in  = args.enc_in
    args.c_out   = args.enc_in
    args.decomposition   = 0
    args.output_attention = False
    args.boundary_method = 'learned_cmi'
    args.patching_threshold_add = 0.2
    args.patch_size = args.max_patch_length
    args.stride = 8
    args.padding_patch = 'end'
    args.use_gpu = args.use_gpu and torch.cuda.is_available()


def evaluate(model, loader, criterion, device, pred_len, features):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            out     = model(batch_x)              # (B, pred_len, C)
            f_dim   = -1 if features == 'MS' else 0
            out     = out[:, -pred_len:, f_dim:]
            tgt     = batch_y[:, -pred_len:, f_dim:]
            preds.append(out.cpu().numpy())
            trues.append(tgt.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    mae, mse, *_ = metric(preds, trues)
    model.train()
    return mse, mae


# ============================================================
# Main
# ============================================================

def main():
    args = get_args()
    _patch_args(args)

    # ---- Device ----
    if args.use_gpu:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'[train_v2] device: cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
        print('[train_v2] device: cpu')

    # ---- Data ----
    _, train_loader = data_provider(args, 'train')
    _, val_loader   = data_provider(args, 'val')
    _, test_loader  = data_provider(args, 'test')
    print(f'[train_v2] train={len(train_loader)} val={len(val_loader)} test={len(test_loader)} batches')

    # ---- Model ----
    model = EntroPE_v2_Model(args).float().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[train_v2] EntroPE_v2 params: {n_params/1e6:.2f}M')

    # ---- Optimiser + scheduler ----
    opt   = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sched = OneCycleLR(
        opt,
        max_lr     = args.learning_rate,
        steps_per_epoch = len(train_loader),
        epochs     = args.train_epochs,
        pct_start  = args.pct_start,
    )
    criterion = nn.MSELoss()

    # ---- Checkpoint dir ----
    ckpt_dir = os.path.join(args.checkpoints, args.model_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    # EarlyStopping.save_checkpoint appends '/checkpoint.pth' to path, so pass ckpt_dir
    best_ckpt_file = os.path.join(ckpt_dir, 'checkpoint.pth')

    early_stop = EarlyStopping(patience=args.patience, verbose=True)
    scaler     = torch.cuda.amp.GradScaler() if args.use_amp else None

    # ---- Training ----
    print(f'[train_v2] training for {args.train_epochs} epochs ...')
    for epoch in range(1, args.train_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (batch_x, batch_y, _, _) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            opt.zero_grad()

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    out       = model(batch_x)
                    f_dim     = -1 if args.features == 'MS' else 0
                    out       = out[:, -args.pred_len:, f_dim:]
                    tgt       = batch_y[:, -args.pred_len:, f_dim:].to(device)
                    fc_loss   = criterion(out, tgt)
                    aux_loss  = model.auxiliary_losses()
                    loss      = fc_loss + aux_loss
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                out      = model(batch_x)
                f_dim    = -1 if args.features == 'MS' else 0
                out      = out[:, -args.pred_len:, f_dim:]
                tgt      = batch_y[:, -args.pred_len:, f_dim:].to(device)
                fc_loss  = criterion(out, tgt)
                aux_loss = model.auxiliary_losses()
                loss     = fc_loss + aux_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()

            sched.step()
            epoch_loss += fc_loss.item()

            if (step + 1) % args.log_every == 0:
                mon = model.monitoring_dict()
                print(
                    f'  ep{epoch} s{step+1}'
                    f'  fc={fc_loss.item():.4f}'
                    f'  aux={aux_loss.item():.4f}'
                    f'  patches={mon.get("avg_patches", -1):.1f}'
                    f'  b_rate={mon.get("boundary_rate", -1):.3f}'
                    f'  tau={mon.get("gumbel_tau", -1):.2f}'
                )
                # Diagnostic 3: print MVG variance stats if available
                if 'mvg_log_diag_min' in mon:
                    print(
                        f'    [D3] log_diag: min={mon["mvg_log_diag_min"]:.2f}'
                        f'  mean={mon["mvg_log_diag_mean"]:.2f}'
                        f'  max={mon["mvg_log_diag_max"]:.2f}'
                        f'  |  log_det: min={mon["mvg_log_det_min"]:.2f}'
                        f'  mean={mon["mvg_log_det_mean"]:.2f}'
                        f'  max={mon["mvg_log_det_max"]:.2f}'
                        f'  |  cmi: min={mon["mvg_cmi_min"]:.4f}'
                        f'  mean={mon["mvg_cmi_mean"]:.4f}'
                        f'  max={mon["mvg_cmi_max"]:.4f}'
                    )

        # ---- Validation ----
        val_mse, val_mae = evaluate(model, val_loader, criterion, device,
                                    args.pred_len, args.features)
        elapsed = time.time() - t0
        print(f'[epoch {epoch:3d}] train_loss={epoch_loss/len(train_loader):.4f} '
              f'val_mse={val_mse:.4f} val_mae={val_mae:.4f} ({elapsed:.0f}s)')

        early_stop(val_mse, model, ckpt_dir)
        if early_stop.early_stop:
            print('[train_v2] early stopping triggered')
            break

    # ---- Test ----
    print('[train_v2] loading best checkpoint for test evaluation ...')
    model.load_state_dict(torch.load(best_ckpt_file, map_location=device, weights_only=True))
    test_mse, test_mae = evaluate(model, test_loader, criterion, device,
                                   args.pred_len, args.features)
    print(f'[train_v2] TEST  mse={test_mse:.4f}  mae={test_mae:.4f}')

    # ---- Save results ----
    result_path = os.path.join(ckpt_dir, 'result.txt')
    with open(result_path, 'w') as f:
        f.write(f'mse:{test_mse}\nmae:{test_mae}\n')
    print(f'[train_v2] results saved → {result_path}')


if __name__ == '__main__':
    main()
