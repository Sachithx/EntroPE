"""
Boundary-Transition Alignment and Patch Quality Metrics (Reviewer fAiY Major 2).

Computes three metrics for dynamic (EntroPE) vs static patching:

(A) Synthetic experiment:
    Generate piecewise-stationary time series with KNOWN change points.
    Run boundary detection for each method.
    Report Precision, Recall, F1 vs ground truth (with tolerance window).

(B) Real-data patch quality metrics:
    1. Intra-patch variance ratio: avg within-patch variance / total sequence variance
       (lower = patches are more internally homogeneous → better boundaries)
    2. Intra-patch entropy reduction: avg conditional entropy within vs total
       (lower = patches are internally low-entropy → boundaries capture transitions)
    3. Boundary entropy concentration: fraction of top-k entropy positions that fall
       on detected boundaries (validates Proposition A.2)

Usage:
    LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6 \\
    /home/AD/sachith/.conda/envs/entrope/bin/python scripts/boundary_quality_metrics.py

Output:
    results/boundary_quality/synthetic_results.csv
    results/boundary_quality/realdata_metrics.csv
    results/boundary_quality/synthetic_f1_plot.png
"""

import argparse
import os
import sys
import json
import math
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_factory import data_provider
from layers.Tokenizer import build_tokenizer
from layers.Patcher import (
    load_entropy_model, calculate_entropies,
    PatchingModeEnum, PatcherArgs, Patcher,
    patch_start_mask_from_local_diff,
    patch_start_mask_from_variance_cp,
    patch_start_mask_from_cusum,
    patch_start_mask_from_empirical_entropy,
    patch_start_mask_from_frequency,
    patch_start_ids_from_patch_start_mask,
    patch_lengths_from_start_ids,
    find_entropy_patch_start_ids,
)

OUT_DIR = "./results/boundary_quality"
os.makedirs(OUT_DIR, exist_ok=True)

CHECKPOINT_DIR = "./entropy_model_checkpoints"


def build_args(dataset, seq_len=96):
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
    a.seq_len      = seq_len
    a.label_len    = seq_len - 1
    a.pred_len     = 1
    a.batch_size   = 32
    a.num_workers  = 2
    a.vocab_size   = 256
    return a
TOLERANCE = 3        # timesteps tolerance for F1 evaluation
QUANTILE_TH = 0.75   # threshold for all methods
PATCH_SIZE = 8       # static patch size for comparison


# ============================================================
# PART A: Synthetic Experiment
# ============================================================

def generate_piecewise_ar(seq_len=192, n_segments=4, seed=None):
    """Generate piecewise AR(1) with known change points."""
    rng = np.random.default_rng(seed)
    min_seg = max(8, seq_len // (n_segments * 3))  # minimum segment length
    # Place change points on a regular grid with jitter so they never collide
    step = seq_len // n_segments
    change_points = sorted(set(
        int(np.clip(i * step + rng.integers(-step // 4, step // 4 + 1),
                    min_seg, seq_len - min_seg))
        for i in range(1, n_segments)
    ))
    # Remove duplicates that are too close together
    filtered = []
    prev = 0
    for cp in change_points:
        if cp - prev >= min_seg:
            filtered.append(cp)
            prev = cp
    change_points = filtered
    boundaries = [0] + change_points + [seq_len]

    ts = np.zeros(seq_len)
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        phi = rng.uniform(0.1, 0.9) * rng.choice([-1, 1])
        length = end - start
        noise = rng.normal(0, 1, length)
        segment = np.zeros(length)
        segment[0] = noise[0]
        for t in range(1, length):
            segment[t] = phi * segment[t - 1] + noise[t]
        # Also add a mean shift
        segment += rng.uniform(-3, 3)
        ts[start:end] = segment

    return ts, change_points


def generate_freq_shift(seq_len=192, n_segments=3, seed=None):
    """Generate sine wave with frequency/amplitude changes at known points."""
    rng = np.random.default_rng(seed)
    min_seg = max(8, seq_len // (n_segments * 3))
    step = seq_len // n_segments
    change_points = sorted(set(
        int(np.clip(i * step + rng.integers(-step // 4, step // 4 + 1),
                    min_seg, seq_len - min_seg))
        for i in range(1, n_segments)
    ))
    filtered = []
    prev = 0
    for cp in change_points:
        if cp - prev >= min_seg:
            filtered.append(cp)
            prev = cp
    change_points = filtered
    boundaries = [0] + change_points + [seq_len]
    t = np.arange(seq_len) / seq_len

    ts = np.zeros(seq_len)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        freq = rng.uniform(2, 20)
        amp  = rng.uniform(0.5, 3.0)
        ts[start:end] = amp * np.sin(2 * np.pi * freq * t[start:end])

    return ts + rng.normal(0, 0.1, seq_len), change_points


def boundaries_from_mask(mask_1d: np.ndarray) -> list:
    """Return list of positions where mask is True (excluding position 0)."""
    return [t for t in range(1, len(mask_1d)) if mask_1d[t]]


def evaluate_f1(detected, ground_truth, seq_len, tolerance=3):
    """Compute Precision, Recall, F1 with tolerance window."""
    if len(ground_truth) == 0:
        return (1.0 if len(detected) == 0 else 0.0, 1.0, 1.0 if len(detected) == 0 else 0.0)

    matched_gt = set()
    tp = 0
    for d in detected:
        for g in ground_truth:
            if abs(d - g) <= tolerance and g not in matched_gt:
                tp += 1
                matched_gt.add(g)
                break

    precision = tp / len(detected) if detected else 0.0
    recall    = tp / len(ground_truth)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def run_synthetic_experiment(n_trials=200, seq_len=192, device='cpu',
                             checkpoint_dir=CHECKPOINT_DIR, dataset='ETTh1'):
    """Run synthetic change-point detection benchmark."""
    print("\n=== Part A: Synthetic Change-Point Benchmark ===")

    # Load entropy model
    state_path = os.path.join(checkpoint_dir, f"{dataset}.pt")
    if os.path.exists(state_path):
        entropy_model, _ = load_entropy_model(checkpoint_dir, state_path, device=device)
        entropy_model = entropy_model.to(device)
        has_entropy_model = True
    else:
        print(f"  WARNING: No checkpoint at {state_path}. Skipping entropy method.")
        has_entropy_model = False

    # Build tokenizer using a real dataset's config (ETTh1)
    tokenizer_args = build_args(dataset)
    tokenizer_args.seq_len = seq_len
    tokenizer = build_tokenizer(tokenizer_args)

    # Generators: (func, name, n_segments)
    generators = [
        (generate_piecewise_ar, "AR-switch",  4),
        (generate_freq_shift,   "Freq-shift", 3),
    ]

    method_results = {m: {'p': [], 'r': [], 'f1': []}
                      for m in ['entropy', 'local_diff', 'variance_cp',
                                'cusum', 'empirical_entropy', 'frequency_based',
                                'static', 'random']}

    for gen_fn, gen_name, n_seg in generators:
        print(f"\n  Generator: {gen_name}")
        for trial in range(n_trials):
            ts, gt_cps = gen_fn(seq_len=seq_len, n_segments=n_seg, seed=trial * 7 + 13)

            # Normalise and quantise to tokens
            ts_t = torch.tensor(ts, dtype=torch.float32).unsqueeze(0)  # (1, seq_len)
            tok_ids, _, _ = tokenizer.context_input_transform(ts_t.cpu())
            tok_ids = tok_ids.to(device)   # (1, seq_len)

            # --- ENTROPY ---
            if has_entropy_model:
                ents, _ = calculate_entropies(tok_ids, entropy_model, 512, device)
                from layers.Patcher import find_entropy_patch_start_ids, patch_start_ids_from_patch_start_mask
                threshold = torch.quantile(ents.float(), QUANTILE_TH, dim=1)
                mask = ents > threshold.unsqueeze(1)
                mask[:, 0] = True
                mask_np = mask[0].cpu().numpy()
                dets = boundaries_from_mask(mask_np)
                p, r, f1 = evaluate_f1(dets, gt_cps, seq_len, TOLERANCE)
                method_results['entropy']['p'].append(p)
                method_results['entropy']['r'].append(r)
                method_results['entropy']['f1'].append(f1)

            # --- OTHER METHODS ---
            method_fns = {
                'local_diff':       lambda t: patch_start_mask_from_local_diff(t, QUANTILE_TH),
                'variance_cp':      lambda t: patch_start_mask_from_variance_cp(t, 4, QUANTILE_TH),
                'cusum':            lambda t: patch_start_mask_from_cusum(t, QUANTILE_TH),
                'empirical_entropy':lambda t: patch_start_mask_from_empirical_entropy(t, 16, QUANTILE_TH),
                'frequency_based':  lambda t: patch_start_mask_from_frequency(t, 16, QUANTILE_TH),
            }
            for method_name, fn in method_fns.items():
                mask = fn(tok_ids)
                mask_np = mask[0].cpu().numpy()
                dets = boundaries_from_mask(mask_np)
                p, r, f1 = evaluate_f1(dets, gt_cps, seq_len, TOLERANCE)
                method_results[method_name]['p'].append(p)
                method_results[method_name]['r'].append(r)
                method_results[method_name]['f1'].append(f1)

            # --- STATIC ---
            static_dets = list(range(PATCH_SIZE, seq_len, PATCH_SIZE))
            p, r, f1 = evaluate_f1(static_dets, gt_cps, seq_len, TOLERANCE)
            method_results['static']['p'].append(p)
            method_results['static']['r'].append(r)
            method_results['static']['f1'].append(f1)

            # --- RANDOM ---
            rand_dets = [t for t in range(1, seq_len)
                         if np.random.rand() < 1.0 / PATCH_SIZE]
            p, r, f1 = evaluate_f1(rand_dets, gt_cps, seq_len, TOLERANCE)
            method_results['random']['p'].append(p)
            method_results['random']['r'].append(r)
            method_results['random']['f1'].append(f1)

    # Aggregate and save
    rows = []
    print(f"\n  {'Method':20s}  {'Precision':>10s}  {'Recall':>8s}  {'F1':>8s}")
    print(f"  {'-'*52}")
    for method, vals in method_results.items():
        if not vals['f1']:
            continue
        pm = np.mean(vals['p']); ps = np.std(vals['p'])
        rm = np.mean(vals['r']); rs = np.std(vals['r'])
        fm = np.mean(vals['f1']); fs = np.std(vals['f1'])
        rows.append({'method': method, 'precision': pm, 'prec_std': ps,
                     'recall': rm, 'rec_std': rs, 'f1': fm, 'f1_std': fs})
        print(f"  {method:20s}  {pm:.3f}±{ps:.3f}  {rm:.3f}±{rs:.3f}  {fm:.3f}±{fs:.3f}")

    df = pd.DataFrame(rows).sort_values('f1', ascending=False)
    csv_path = os.path.join(OUT_DIR, 'synthetic_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 4))
    methods = [r['method'] for r in rows]
    f1s = [r['f1'] for r in rows]
    f1s_std = [r['f1_std'] for r in rows]
    colors = ['steelblue' if m == 'entropy' else
              'darkorange' if m == 'empirical_entropy' else 'lightgray'
              for m in methods]
    bars = ax.bar(methods, f1s, yerr=f1s_std, capsize=4, color=colors, edgecolor='black')
    ax.set_ylabel("F1 Score (±std)")
    ax.set_title(f"Synthetic Change-Point Detection — F1 (tolerance={TOLERANCE})\n"
                 f"Blue=EntroPE, Orange=EAPformer-style, Gray=others")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'synthetic_f1_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

    return df


# ============================================================
# PART B: Real-data Patch Quality Metrics
# ============================================================

def compute_patch_quality(tokens_flat, patch_lengths, entropies=None):
    """
    Given tokens (seq_len,) and patch_lengths (n_patches,), compute:
      - intra_var_ratio: mean within-patch variance / total variance
      - entropy_reduction: mean within-patch entropy / overall entropy (if entropies given)
    """
    seq_len = tokens_flat.shape[0]
    tokens_f = tokens_flat.float()
    total_var = tokens_f.var().item() + 1e-8

    pos = 0
    patch_vars = []
    patch_ents = []
    valid_lengths = patch_lengths[patch_lengths > 0]

    for pl in valid_lengths.tolist():
        pl = int(pl)
        if pos + pl > seq_len:
            pl = seq_len - pos
        if pl <= 0:
            break
        chunk = tokens_f[pos:pos + pl]
        patch_vars.append(chunk.var().item() if pl > 1 else 0.0)
        if entropies is not None:
            patch_ents.append(entropies[pos:pos + pl].float().mean().item())
        pos += pl

    intra_var_ratio = np.mean(patch_vars) / total_var if patch_vars else 1.0

    if entropies is not None and patch_ents:
        total_ent = entropies.float().mean().item() + 1e-8
        entropy_reduction = np.mean(patch_ents) / total_ent
    else:
        entropy_reduction = None

    return intra_var_ratio, entropy_reduction


def run_realdata_metrics(dataset='ETTh1', n_samples=200, device='cpu',
                          checkpoint_dir=CHECKPOINT_DIR):
    """Compute patch quality metrics on real data for dynamic vs static patching."""
    print(f"\n=== Part B: Real-data Patch Quality — {dataset} ===")
    args = build_args(dataset)
    tokenizer = build_tokenizer(args)
    state_path = os.path.join(checkpoint_dir, f"{dataset}.pt")
    if not os.path.exists(state_path):
        print(f"  WARNING: {state_path} not found. Skipping.")
        return None

    entropy_model, _ = load_entropy_model(checkpoint_dir, state_path, device=device)
    entropy_model = entropy_model.to(device)

    _, loader = data_provider(args, flag='train')

    dynamic_var_ratios, dynamic_ent_reductions = [], []
    static_var_ratios, static_ent_reductions   = [], []
    boundary_concs = []

    n_collected = 0
    for batch_x, _, _, _ in loader:
        x = batch_x.float().permute(0, 2, 1)  # (bs, nvars, seq_len)
        bs, nvars, seq_len = x.shape
        x_flat = x.reshape(bs * nvars, seq_len)

        tok, _, _ = tokenizer.context_input_transform(x_flat.cpu())
        tok = tok.to(device)

        # Compute entropies for entire batch
        ents, _ = calculate_entropies(tok, entropy_model, 512, device)

        for i in range(tok.shape[0]):
            tok_i = tok[i]     # (seq_len,)
            ent_i = ents[i]    # (seq_len,)

            # Dynamic boundaries (EntroPE entropy method)
            threshold_i = torch.quantile(ent_i.float(), QUANTILE_TH)
            dyn_mask = ent_i > threshold_i
            dyn_mask[0] = True
            dyn_starts = patch_start_ids_from_patch_start_mask(dyn_mask.unsqueeze(0))
            dyn_lengths = patch_lengths_from_start_ids(dyn_starts, seq_len)[0]

            vr, er = compute_patch_quality(tok_i, dyn_lengths, ent_i)
            dynamic_var_ratios.append(vr)
            if er is not None:
                dynamic_ent_reductions.append(er)

            # Static boundaries
            n_static = math.ceil(seq_len / PATCH_SIZE)
            stat_lengths = torch.full((n_static,), PATCH_SIZE, device=device)
            if seq_len % PATCH_SIZE:
                stat_lengths[-1] = seq_len % PATCH_SIZE

            vr_s, er_s = compute_patch_quality(tok_i, stat_lengths, ent_i)
            static_var_ratios.append(vr_s)
            if er_s is not None:
                static_ent_reductions.append(er_s)

            # Boundary entropy concentration
            n_bounds = int(dyn_mask.sum().item())
            top_k = ent_i.float().topk(n_bounds).indices
            bound_pos = torch.where(dyn_mask)[0]
            overlap = len(set(top_k.cpu().tolist()) & set(bound_pos.cpu().tolist()))
            boundary_concs.append(overlap / max(n_bounds, 1))

            n_collected += 1
            if n_collected >= n_samples:
                break

        if n_collected >= n_samples:
            break

    results = {
        'dataset': dataset,
        'dynamic_var_ratio_mean':   np.mean(dynamic_var_ratios),
        'dynamic_var_ratio_std':    np.std(dynamic_var_ratios),
        'static_var_ratio_mean':    np.mean(static_var_ratios),
        'static_var_ratio_std':     np.std(static_var_ratios),
        'dynamic_ent_reduction_mean': np.mean(dynamic_ent_reductions) if dynamic_ent_reductions else None,
        'static_ent_reduction_mean':  np.mean(static_ent_reductions)  if static_ent_reductions  else None,
        'boundary_entropy_concentration_mean': np.mean(boundary_concs),
        'boundary_entropy_concentration_std':  np.std(boundary_concs),
    }

    print(f"\n  {'Metric':40s}  Dynamic         Static")
    print(f"  {'-'*70}")
    print(f"  {'Intra-patch variance ratio':40s}  "
          f"{results['dynamic_var_ratio_mean']:.4f}±{results['dynamic_var_ratio_std']:.4f}  "
          f"{results['static_var_ratio_mean']:.4f}±{results['static_var_ratio_std']:.4f}")
    if results['dynamic_ent_reduction_mean']:
        print(f"  {'Intra-patch entropy (relative)':40s}  "
              f"{results['dynamic_ent_reduction_mean']:.4f}  "
              f"  {results['static_ent_reduction_mean']:.4f}")
    print(f"  {'Boundary entropy concentration':40s}  "
          f"{results['boundary_entropy_concentration_mean']:.4f}±"
          f"{results['boundary_entropy_concentration_std']:.4f}  (n/a for static)")

    return results


# -------------------------------------------------------------------- main --

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ETTh1',
                        choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Electricity', 'weather'])
    parser.add_argument('--all_datasets', action='store_true')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR)
    parser.add_argument('--n_trials', type=int, default=200,
                        help='Synthetic: number of random trials per generator')
    parser.add_argument('--n_samples', type=int, default=200,
                        help='Real-data: number of time series to evaluate')
    parser.add_argument('--skip_synthetic', action='store_true')
    parser.add_argument('--skip_realdata', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if not args.skip_synthetic:
        run_synthetic_experiment(
            n_trials=args.n_trials, device=device,
            checkpoint_dir=args.checkpoint_dir, dataset=args.dataset
        )

    datasets = (['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Electricity', 'weather']
                if args.all_datasets else [args.dataset])

    if not args.skip_realdata:
        all_real = []
        for ds in datasets:
            res = run_realdata_metrics(ds, args.n_samples, device, args.checkpoint_dir)
            if res:
                all_real.append(res)

        if all_real:
            df = pd.DataFrame(all_real)
            csv_path = os.path.join(OUT_DIR, 'realdata_metrics.csv')
            df.to_csv(csv_path, index=False)
            print(f"\nReal-data metrics saved: {csv_path}")


if __name__ == '__main__':
    main()
