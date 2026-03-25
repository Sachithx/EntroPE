"""
Entropy Monotonicity Check (Reviewer GmSc W7).

Tests whether absolute entropy increases with timestep position t (because the causal
model conditions on longer history) and shows that the *relative* threshold (entropy
difference criterion, Eq. 8) mitigates this.

Plots:
  1. Average conditional entropy H(t) vs position t for each dataset
  2. Histogram of boundary positions to show they are NOT concentrated at the end
  3. Average entropy *differences* ΔH(t) vs t (the signal actually used for thresholding)

Usage:
    LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6 \\
    /home/AD/sachith/.conda/envs/entrope/bin/python scripts/entropy_monotonicity.py \\
        --dataset ETTh1 --n_samples 500

Output: results/entropy_monotonicity_{DATASET}.png
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_factory import data_provider
from layers.Tokenizer import build_tokenizer
from layers.Patcher import load_entropy_model, calculate_entropies


# ------------------------------------------------------------------ helpers --

def build_args(dataset):
    """Minimal args namespace for data_provider and tokenizer."""
    class A:
        pass
    a = A()
    ETT = {'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'}
    a.data = dataset if dataset in ETT else 'custom'
    path_map = {
        'ETTh1': 'ETTh1.csv', 'ETTh2': 'ETTh2.csv',
        'ETTm1': 'ETTm1.csv', 'ETTm2': 'ETTm2.csv',
        'Electricity': 'electricity.csv', 'weather': 'weather.csv',
    }
    freq_map = {'ETTm1': 't', 'ETTm2': 't'}
    a.data_path    = path_map[dataset]
    a.root_path    = './dataset/'
    a.freq         = freq_map.get(dataset, 'h')
    a.features     = 'M'
    a.target       = 'OT'
    a.embed        = 'timeF'
    a.seq_len      = 96
    a.label_len    = 95
    a.pred_len     = 1
    a.batch_size   = 64
    a.num_workers  = 2
    a.vocab_size   = 256
    return a


def collect_entropies(dataset, checkpoint_dir, n_samples, device, split='train'):
    """Collect per-position entropy values from n_samples sequences."""
    args = build_args(dataset)
    tokenizer = build_tokenizer(args)

    # Load frozen entropy model
    state_path = os.path.join(checkpoint_dir, f"{dataset}.pt")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Checkpoint not found: {state_path}")
    entropy_model, _ = load_entropy_model(checkpoint_dir, state_path, device=device)
    entropy_model = entropy_model.to(device)

    _, loader = data_provider(args, flag=split)

    all_entropies = []   # (n_samples, seq_len)
    n_collected = 0

    for batch_x, _, _, _ in loader:
        x = batch_x.float().squeeze(-1)
        x = x.permute(0, 2, 1)              # (bs, nvars, seq_len)
        bs, nvars, seq_len = x.shape
        x_flat = x.reshape(bs * nvars, seq_len)

        token_ids, _, _ = tokenizer.context_input_transform(x_flat.cpu())
        token_ids = token_ids.to(device)

        entropies, _ = calculate_entropies(
            token_ids, entropy_model, patching_batch_size=512, device=device
        )
        # entropies: (bs*nvars, seq_len)
        all_entropies.append(entropies.float().cpu().numpy())
        n_collected += entropies.shape[0]

        if n_collected >= n_samples:
            break

    all_entropies = np.concatenate(all_entropies, axis=0)[:n_samples]
    return all_entropies  # (n_samples, seq_len)


def plot_monotonicity(dataset, all_entropies, patching_threshold, out_dir):
    """Generate 3-panel monotonicity analysis plot."""
    seq_len = all_entropies.shape[1]
    positions = np.arange(seq_len)

    # 1. Mean entropy per position
    mean_H = all_entropies.mean(axis=0)
    std_H  = all_entropies.std(axis=0)

    # 2. Entropy differences ΔH(t) = H(t) - H(t-1)
    diffs = np.diff(all_entropies, axis=1)  # (n, seq_len-1)
    mean_dH = diffs.mean(axis=0)
    std_dH  = diffs.std(axis=0)

    # 3. Boundary positions under quantile threshold applied to DIFFERENCES
    threshold_val = np.quantile(diffs, patching_threshold, axis=1, keepdims=True)
    boundary_mask = diffs > threshold_val          # (n, seq_len-1)
    boundary_freq = boundary_mask.mean(axis=0)     # fraction of samples with boundary at t

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Entropy Monotonicity Analysis — {dataset}", fontsize=13)

    # Panel 1: Mean absolute entropy vs position
    ax = axes[0]
    ax.plot(positions, mean_H, color='steelblue', lw=1.5)
    ax.fill_between(positions, mean_H - std_H, mean_H + std_H, alpha=0.2, color='steelblue')
    ax.set_xlabel("Token position t")
    ax.set_ylabel("Conditional entropy H(t)")
    ax.set_title("Mean absolute entropy vs position\n(monotonicity concern)")
    # Fit linear trend
    slope = np.polyfit(positions, mean_H, 1)[0]
    ax.text(0.05, 0.92, f"Trend slope: {slope:+.4f}/step", transform=ax.transAxes,
            fontsize=9, va='top', color='darkred')

    # Panel 2: Mean entropy *difference* ΔH vs position
    ax = axes[1]
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.plot(positions[1:], mean_dH, color='darkorange', lw=1.5)
    ax.fill_between(positions[1:], mean_dH - std_dH, mean_dH + std_dH,
                    alpha=0.2, color='darkorange')
    ax.set_xlabel("Token position t")
    ax.set_ylabel("Mean ΔH(t) = H(t) − H(t−1)")
    ax.set_title("Entropy differences (actual thresholding signal)\nSteady → no positional bias")

    # Panel 3: Boundary frequency histogram
    ax = axes[2]
    ax.bar(positions[1:], boundary_freq, color='forestgreen', alpha=0.7, width=1.0)
    ax.axhline(1.0 - patching_threshold, color='red', lw=1.2, ls='--',
               label=f'Expected (threshold={patching_threshold})')
    ax.set_xlabel("Token position t")
    ax.set_ylabel("Fraction of samples with boundary at t")
    ax.set_title(f"Boundary position distribution\n(should be ~uniform, not tail-heavy)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"entropy_monotonicity_{dataset}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    # Print summary stats
    slope = np.polyfit(positions, mean_H, 1)[0]
    print(f"\n{dataset} Monotonicity Summary:")
    print(f"  Absolute entropy slope:   {slope:+.5f} per step  "
          f"({'increasing' if slope > 0 else 'decreasing'} trend)")
    print(f"  Max mean H:  {mean_H.max():.3f}  at t={mean_H.argmax()}")
    print(f"  Min mean H:  {mean_H.min():.3f}  at t={mean_H.argmin()}")
    print(f"  Boundary density (max): {boundary_freq.max():.3f}  "
          f"at t={boundary_freq.argmax()+1}")
    print(f"  Boundary density (min): {boundary_freq.min():.3f}  "
          f"at t={boundary_freq.argmin()+1}")
    tail_density = boundary_freq[-seq_len // 4:].mean()
    head_density = boundary_freq[:seq_len // 4].mean()
    print(f"  Last-quarter boundary density: {tail_density:.3f}  "
          f"(concern: should not >> head density {head_density:.3f})")


# -------------------------------------------------------------------- main --

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ETTh1',
                        choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Electricity', 'weather'])
    parser.add_argument('--checkpoint_dir', type=str, default='./entropy_model_checkpoints')
    parser.add_argument('--n_samples', type=int, default=500,
                        help='Number of time series to analyse')
    parser.add_argument('--patching_threshold', type=float, default=0.75,
                        help='Quantile threshold (used for boundary frequency plot)')
    parser.add_argument('--out_dir', type=str, default='./results/monotonicity')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Which data split to sample from (default: train)')
    parser.add_argument('--all_datasets', action='store_true',
                        help='Run for all 6 datasets')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    datasets = (['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Electricity', 'weather']
                if args.all_datasets else [args.dataset])

    for ds in datasets:
        print(f"\n{'='*50}")
        print(f"Processing: {ds}")
        print(f"{'='*50}")
        try:
            entropies = collect_entropies(ds, args.checkpoint_dir, args.n_samples, device, split=args.split)
            plot_monotonicity(ds, entropies, args.patching_threshold, args.out_dir)
        except Exception as e:
            print(f"  ERROR for {ds}: {e}")


if __name__ == '__main__':
    main()
