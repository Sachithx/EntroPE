#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Modifications and extensions:
#   - Pretraining script for MultivariateGaussianModel
#
# Copyright (c) 2026 Sachith Abeywickrama
#
# Licensed under the same license as the original Meta code.

"""
scripts/pretrain_mvg.py

Pretrain the MultivariateGaussianModel on a single dataset for ~300 steps
(~5-10 minutes on one GPU) before end-to-end EntroPE_v2 training.

Saves checkpoint to: checkpoints/mvg_{dataset_name}.pt

Usage:
    python scripts/pretrain_mvg.py \
        --data ETTh1 \
        --root_path ./data/ETT/ \
        --data_path ETTh1.csv \
        --seq_len 96 \
        --enc_in 7 \
        --mvg_layers 2 \
        --mvg_embd 64 \
        --mvg_heads 4 \
        --pretrain_steps 300 \
        --batch_size 128 \
        --learning_rate 1e-3 \
        --gpu 0
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW

# Allow running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_provider.data_factory import data_provider
from models.MultivariateGaussianModel import MultivariateGaussianModel, MVGConfig


# ============================================================
# Args
# ============================================================

def get_args():
    p = argparse.ArgumentParser(description='Pretrain MultivariateGaussianModel')

    # Data
    p.add_argument('--data',      type=str,   default='ETTh1')
    p.add_argument('--root_path', type=str,   default='./data/ETT/')
    p.add_argument('--data_path', type=str,   default='ETTh1.csv')
    p.add_argument('--features',  type=str,   default='M')
    p.add_argument('--target',    type=str,   default='OT')
    p.add_argument('--freq',      type=str,   default='h')
    p.add_argument('--seq_len',   type=int,   default=96)
    p.add_argument('--label_len', type=int,   default=48)
    p.add_argument('--pred_len',  type=int,   default=96)
    p.add_argument('--enc_in',    type=int,   default=7)
    p.add_argument('--num_workers', type=int, default=4)

    # MVG model
    p.add_argument('--mvg_layers',   type=int,   default=2)
    p.add_argument('--mvg_embd',     type=int,   default=64)
    p.add_argument('--mvg_heads',    type=int,   default=4)
    p.add_argument('--mvg_dropout',  type=float, default=0.1)
    p.add_argument('--mvg_cov_rank', type=int,   default=None,
                   help='None = full Cholesky; int r = low-rank+diag for large C')

    # Training
    p.add_argument('--pretrain_steps', type=int,   default=300)
    p.add_argument('--batch_size',     type=int,   default=128)
    p.add_argument('--learning_rate',  type=float, default=1e-3)
    p.add_argument('--weight_decay',   type=float, default=0.05)
    p.add_argument('--beta_nll',       type=float, default=0.5)
    p.add_argument('--grad_clip',      type=float, default=1.0)

    # Output
    p.add_argument('--checkpoint_dir', type=str, default='./checkpoints/mvg/')
    p.add_argument('--model_id_name',  type=str, default=None,
                   help='Override checkpoint name (default: --data value)')

    # GPU
    p.add_argument('--gpu',     type=int,  default=0)
    p.add_argument('--use_gpu', type=bool, default=True)

    # Logging
    p.add_argument('--log_every', type=int, default=50)

    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = get_args()

    # ---- Device ----
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f'[pretrain_mvg] device: {device}')

    # ---- Data loader ----
    # Patch args to satisfy data_provider interface
    args.embed        = 'timeF'
    args.dec_in       = args.enc_in
    args.c_out        = args.enc_in
    args.checkpoints  = './checkpoints/'
    _, train_loader   = data_provider(args, 'train')

    # ---- Model ----
    cfg = MVGConfig(
        n_channels = args.enc_in,
        block_size = args.seq_len,
        n_layer    = args.mvg_layers,
        n_head     = args.mvg_heads,
        n_embd     = args.mvg_embd,
        dropout    = args.mvg_dropout,
        cov_rank   = args.mvg_cov_rank,
        beta_nll   = args.beta_nll,
    )
    model = MultivariateGaussianModel(cfg).to(device)
    n_params = model.get_num_params()
    print(f'[pretrain_mvg] model params: {n_params/1e3:.1f}k')

    # ---- Optimiser ----
    opt = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # ---- Training loop ----
    model.train()
    step      = 0
    run_loss  = 0.0

    print(f'[pretrain_mvg] starting pretraining for {args.pretrain_steps} steps ...')
    while step < args.pretrain_steps:
        for batch_x, batch_y, _, _ in train_loader:
            if step >= args.pretrain_steps:
                break

            # batch_x: (B, seq_len, C)
            x = batch_x.float().to(device)   # (B, L, C)

            # Causal next-step prediction: input x[:, :-1], target x[:, 1:]
            x_in  = x[:, :-1, :]
            x_tgt = x[:, 1:,  :]

            mu, cov = model(x_in)
            loss    = MultivariateGaussianModel.mvg_nll_loss(
                x_tgt, mu, cov, beta=args.beta_nll
            )

            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            run_loss += loss.item()
            step     += 1

            if step % args.log_every == 0:
                avg = run_loss / args.log_every
                run_loss = 0.0
                print(f'  step {step:4d}/{args.pretrain_steps}  nll={avg:.4f}')

    # ---- Save checkpoint ----
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    name = args.model_id_name if args.model_id_name else args.data
    path = os.path.join(args.checkpoint_dir, f'mvg_{name}.pt')
    torch.save(model.state_dict(), path)
    print(f'[pretrain_mvg] saved checkpoint → {path}')

    # ---- Quick val sanity ----
    _, val_loader = data_provider(args, 'val')
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_x, _, _, _ in val_loader:
            x     = batch_x.float().to(device)
            x_in  = x[:, :-1, :]
            x_tgt = x[:, 1:,  :]
            mu, cov = model(x_in)
            loss = MultivariateGaussianModel.mvg_nll_loss(x_tgt, mu, cov, beta=args.beta_nll)
            val_losses.append(loss.item())
            if len(val_losses) >= 20:
                break
    import numpy as np
    val_nll = np.mean(val_losses)
    print(f'[pretrain_mvg] val NLL (first 20 batches) = {val_nll:.4f}')
    if val_nll > 5.0:
        print('[pretrain_mvg] WARNING: val NLL > 5.0 — check for numerical instability in Cholesky.')
    else:
        print('[pretrain_mvg] val NLL looks good (< 5.0). Ready for end-to-end training.')


if __name__ == '__main__':
    main()
