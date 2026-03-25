"""
Noise Robustness Analysis for EntroPE (Reviewer eKr6 W5, GmSc W8).

Injects Gaussian noise at varying SNR levels and measures how boundary detection
quality degrades. No retraining needed — uses the frozen entropy model.

Additionally tests:
  (c) Domain transfer: train entropy model on ETTh1, apply to ETTh2 and weather.

Usage:
    LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6 \\
    /home/AD/sachith/.conda/envs/entrope/bin/python scripts/noise_robustness_analysis.py

Output:
    results/noise_robustness/noise_boundary_quality.csv
    results/noise_robustness/noise_boundary_quality.png
    results/noise_robustness/domain_transfer.csv
"""

import argparse
import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_factory import data_provider
from layers.Tokenizer import build_tokenizer
from layers.Patcher import load_entropy_model, calculate_entropies, patch_start_ids_from_patch_start_mask

OUT_DIR = "./results/noise_robustness"
os.makedirs(OUT_DIR, exist_ok=True)
CHECKPOINT_DIR = "./entropy_model_checkpoints"
QUANTILE_TH = 0.75
N_SAMPLES = 300


def make_args(dataset):
    class A:
        pass
    a = A()
    ETT = {'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'}
    a.data = dataset if dataset in ETT else 'custom'
    path_map = {'ETTh1':'ETTh1.csv','ETTh2':'ETTh2.csv','ETTm1':'ETTm1.csv',
                'ETTm2':'ETTm2.csv','Electricity':'electricity.csv','weather':'weather.csv'}
    freq_map = {'ETTm1': 't', 'ETTm2': 't'}
    a.data_path = path_map[dataset]; a.root_path = './dataset/'
    a.freq = freq_map.get(dataset, 'h'); a.features = 'M'; a.target = 'OT'
    a.embed = 'timeF'; a.seq_len = 96; a.label_len = 95; a.pred_len = 1
    a.batch_size = 32; a.num_workers = 2; a.vocab_size = 256
    return a


def add_noise(tokens_f: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add Gaussian noise at given SNR (dB) to float token values."""
    signal_power = tokens_f.var(dim=-1, keepdim=True).clamp(min=1e-8)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(tokens_f) * noise_power.sqrt()
    return (tokens_f + noise).clamp(0, 255).long()


def boundary_count_consistency(ents_clean, ents_noisy, threshold):
    """
    Measure how consistently boundaries are placed between clean and noisy.
    Returns Jaccard overlap of boundary positions.
    """
    def get_boundaries(ents):
        th = torch.quantile(ents.float(), threshold, dim=1, keepdim=True)
        mask = ents > th
        mask[:, 0] = True
        return [set(torch.where(mask[i])[0].cpu().tolist()) for i in range(mask.shape[0])]

    b_clean = get_boundaries(ents_clean)
    b_noisy = get_boundaries(ents_noisy)

    jaccards = []
    for bc, bn in zip(b_clean, b_noisy):
        inter = len(bc & bn)
        union = len(bc | bn)
        jaccards.append(inter / union if union > 0 else 1.0)
    return np.mean(jaccards)


def run_noise_experiment(dataset='ETTh1', snr_levels=None, device='cpu',
                         checkpoint_dir=CHECKPOINT_DIR):
    """Measure boundary stability as SNR decreases."""
    if snr_levels is None:
        snr_levels = [40, 20, 10, 5, 2]  # dB

    print(f"\n=== Noise Robustness: {dataset} ===")
    args = make_args(dataset)
    tokenizer = build_tokenizer(args)

    state_path = os.path.join(checkpoint_dir, f"{dataset}.pt")
    if not os.path.exists(state_path):
        print(f"  Checkpoint not found: {state_path}"); return None
    entropy_model, _ = load_entropy_model(checkpoint_dir, state_path, device=device)
    entropy_model = entropy_model.to(device)

    _, loader = data_provider(args, flag='test')

    # Collect clean tokens
    all_tokens = []
    n = 0
    for batch_x, _, _, _ in loader:
        x = batch_x.float().permute(0, 2, 1)
        bs, nvars, seq_len = x.shape
        x_flat = x.reshape(bs * nvars, seq_len)
        tok, _, _ = tokenizer.context_input_transform(x_flat.cpu())
        all_tokens.append(tok.to(device))
        n += tok.shape[0]
        if n >= N_SAMPLES:
            break
    clean_tokens = torch.cat(all_tokens, dim=0)[:N_SAMPLES]

    # Get clean entropies
    ents_clean, _ = calculate_entropies(clean_tokens, entropy_model, 512, device)

    rows = [{'snr_db': 'clean', 'jaccard': 1.0, 'boundary_count_mean': 0, 'boundary_count_std': 0}]
    th = torch.quantile(ents_clean.float(), QUANTILE_TH, dim=1, keepdim=True)
    clean_count = (ents_clean > th).float().sum(dim=1).mean().item()
    rows[0]['boundary_count_mean'] = clean_count

    print(f"  SNR      Jaccard  BoundaryCount")
    print(f"  {'clean':>6}   1.000    {clean_count:.1f}")

    for snr_db in snr_levels:
        noisy_tokens = add_noise(clean_tokens.float(), snr_db).to(device)
        ents_noisy, _ = calculate_entropies(noisy_tokens, entropy_model, 512, device)

        jaccard = boundary_count_consistency(ents_clean, ents_noisy, QUANTILE_TH)
        th_n = torch.quantile(ents_noisy.float(), QUANTILE_TH, dim=1, keepdim=True)
        counts = (ents_noisy > th_n).float().sum(dim=1)
        count_m, count_s = counts.mean().item(), counts.std().item()

        rows.append({'snr_db': snr_db, 'jaccard': jaccard,
                     'boundary_count_mean': count_m, 'boundary_count_std': count_s})
        print(f"  {snr_db:>5}dB   {jaccard:.3f}    {count_m:.1f}±{count_s:.1f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, f'noise_boundary_quality_{dataset}.csv')
    df.to_csv(csv_path, index=False)
    return df


def run_domain_transfer(source_ds='ETTh1', target_datasets=None, device='cpu',
                        checkpoint_dir=CHECKPOINT_DIR):
    """Train on source, apply to targets — measure boundary count change."""
    if target_datasets is None:
        target_datasets = ['ETTh2', 'weather']

    print(f"\n=== Domain Transfer: source={source_ds} ===")
    state_path = os.path.join(checkpoint_dir, f"{source_ds}.pt")
    if not os.path.exists(state_path):
        print(f"  Source checkpoint not found: {state_path}"); return None
    source_model, _ = load_entropy_model(checkpoint_dir, state_path, device=device)
    source_model = source_model.to(device)

    rows = []
    for target in [source_ds] + target_datasets:
        print(f"\n  Target: {target}")
        args = make_args(target)
        tokenizer = build_tokenizer(args)

        # Target's own model (if available)
        target_path = os.path.join(checkpoint_dir, f"{target}.pt")
        has_target_model = os.path.exists(target_path)
        if has_target_model:
            target_model, _ = load_entropy_model(checkpoint_dir, target_path, device=device)
            target_model = target_model.to(device)

        _, loader = data_provider(args, flag='test')
        all_tokens = []
        n = 0
        for batch_x, _, _, _ in loader:
            x = batch_x.float().permute(0, 2, 1)
            bs, nvars, sl = x.shape
            x_flat = x.reshape(bs * nvars, sl)
            tok, _, _ = tokenizer.context_input_transform(x_flat.cpu())
            all_tokens.append(tok.to(device))
            n += tok.shape[0]
            if n >= N_SAMPLES:
                break
        tokens = torch.cat(all_tokens, dim=0)[:N_SAMPLES]

        # Boundaries with source model
        ents_src, _ = calculate_entropies(tokens, source_model, 512, device)
        th_src = torch.quantile(ents_src.float(), QUANTILE_TH, dim=1, keepdim=True)
        counts_src = (ents_src > th_src).float().sum(dim=1).mean().item()

        row = {'source': source_ds, 'target': target,
               'boundary_count_source_model': counts_src}

        if has_target_model:
            ents_tgt, _ = calculate_entropies(tokens, target_model, 512, device)
            th_tgt = torch.quantile(ents_tgt.float(), QUANTILE_TH, dim=1, keepdim=True)
            counts_tgt = (ents_tgt > th_tgt).float().sum(dim=1).mean().item()

            # Jaccard between source and target model boundaries
            def get_b(ents, th):
                mask = ents > th; mask[:,0] = True
                return [set(torch.where(mask[i])[0].cpu().tolist()) for i in range(mask.shape[0])]
            b_src = get_b(ents_src, th_src); b_tgt = get_b(ents_tgt, th_tgt)
            jacs = [len(a&b)/len(a|b) for a,b in zip(b_src,b_tgt) if len(a|b)>0]

            row['boundary_count_target_model'] = counts_tgt
            row['jaccard_src_vs_tgt_model'] = np.mean(jacs)
            print(f"    Source model → {counts_src:.1f} bounds | "
                  f"Target model → {counts_tgt:.1f} bounds | "
                  f"Jaccard = {np.mean(jacs):.3f}")
        else:
            print(f"    Source model → {counts_src:.1f} bounds (no target checkpoint)")

        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, 'domain_transfer.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    print(df.to_string(index=False))
    return df


# ----------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ETTh1')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR)
    parser.add_argument('--skip_noise', action='store_true')
    parser.add_argument('--skip_transfer', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if not args.skip_noise:
        df_noise = run_noise_experiment(args.dataset, device=device,
                                        checkpoint_dir=args.checkpoint_dir)
        if df_noise is not None:
            # Quick plot
            fig, ax = plt.subplots(figsize=(7, 4))
            snr_vals = [r for r in df_noise['snr_db'] if r != 'clean']
            jac_vals = [df_noise[df_noise['snr_db']==s]['jaccard'].values[0] for s in snr_vals]
            ax.plot(snr_vals, jac_vals, 'o-', color='steelblue', lw=2)
            ax.axhline(1.0, color='gray', ls='--', lw=1, label='Clean (perfect)')
            ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Boundary Jaccard Similarity")
            ax.set_title(f"Boundary Stability Under Noise — {args.dataset}")
            ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f'noise_robustness_{args.dataset}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\nPlot saved: {OUT_DIR}/noise_robustness_{args.dataset}.png")

    if not args.skip_transfer:
        run_domain_transfer(source_ds=args.dataset, device=device,
                            checkpoint_dir=args.checkpoint_dir)


if __name__ == '__main__':
    main()
