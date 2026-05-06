#!/usr/bin/env python3
"""
Hyperparameter search for EntroPE_v2, covering two backbone architectures:

  simple  — SimplePatchPooler + SimpleFlattenHead  (fast, current best 0.473)
  ape     — AdaptivePatchPooler + FusionForecastHead  (new, expressive)

Runs experiments in parallel across 2 GPUs (one per GPU per batch).
All experiments: ETTh1 96→96, CMI top-K patching, rich tokens.

Usage:
    python scripts/hparam_search.py                      # all configs
    python scripts/hparam_search.py --backbone simple    # simple only
    python scripts/hparam_search.py --backbone ape       # APE+Fusion only
    python scripts/hparam_search.py --dry-run            # print without running
    python scripts/hparam_search.py --results            # parse existing results
"""

import argparse
import math
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT        = Path(__file__).parent.parent
PYTHON      = str(Path(sys.executable))
TRAIN_SCRIPT = str(ROOT / "scripts" / "train_entrope_v2.py")
LOG_DIR     = ROOT / "logs" / "hparam_search"
CKPT_DIR    = ROOT / "checkpoints"


# ================================================================
# Search space
# ================================================================

# --- backbone='simple': SimplePatchPooler + SimpleFlattenHead ---
SIMPLE_CONFIGS = [
    # LR sweep around known-good lr=0.005 (test MSE 0.473)
    dict(backbone='simple', lr=0.005, bs=128, drop=0.10, nll=0.1, rate=0.125),
    dict(backbone='simple', lr=0.008, bs=128, drop=0.10, nll=0.1, rate=0.125),
    dict(backbone='simple', lr=0.010, bs=128, drop=0.10, nll=0.1, rate=0.125),
    dict(backbone='simple', lr=0.015, bs=128, drop=0.10, nll=0.1, rate=0.125),
    dict(backbone='simple', lr=0.020, bs=128, drop=0.10, nll=0.1, rate=0.125),

    # Batch size (at lr=0.01)
    dict(backbone='simple', lr=0.010, bs=32,  drop=0.10, nll=0.1, rate=0.125),
    dict(backbone='simple', lr=0.010, bs=64,  drop=0.10, nll=0.1, rate=0.125),
    dict(backbone='simple', lr=0.010, bs=256, drop=0.10, nll=0.1, rate=0.125),

    # Dropout (at lr=0.01, bs=128)
    dict(backbone='simple', lr=0.010, bs=128, drop=0.00, nll=0.1, rate=0.125),
    dict(backbone='simple', lr=0.010, bs=128, drop=0.05, nll=0.1, rate=0.125),
    dict(backbone='simple', lr=0.010, bs=128, drop=0.20, nll=0.1, rate=0.125),

    # lambda_nll (at lr=0.01, bs=128, drop=0.1)
    dict(backbone='simple', lr=0.010, bs=128, drop=0.10, nll=0.00, rate=0.125),
    dict(backbone='simple', lr=0.010, bs=128, drop=0.10, nll=0.05, rate=0.125),
    dict(backbone='simple', lr=0.010, bs=128, drop=0.10, nll=0.50, rate=0.125),
    dict(backbone='simple', lr=0.010, bs=128, drop=0.10, nll=1.00, rate=0.125),

    # CMI boundary rate (at lr=0.01, bs=128, drop=0.1)
    dict(backbone='simple', lr=0.010, bs=128, drop=0.10, nll=0.1, rate=0.083),
    dict(backbone='simple', lr=0.010, bs=128, drop=0.10, nll=0.1, rate=0.100),
    dict(backbone='simple', lr=0.010, bs=128, drop=0.10, nll=0.1, rate=0.167),
    dict(backbone='simple', lr=0.010, bs=128, drop=0.10, nll=0.1, rate=0.200),

    # Promising joint combos
    dict(backbone='simple', lr=0.010, bs=64,  drop=0.05, nll=0.1, rate=0.125),
    dict(backbone='simple', lr=0.010, bs=32,  drop=0.00, nll=0.1, rate=0.125),
    dict(backbone='simple', lr=0.015, bs=64,  drop=0.10, nll=0.1, rate=0.125),
    dict(backbone='simple', lr=0.010, bs=64,  drop=0.10, nll=0.1, rate=0.100),
    dict(backbone='simple', lr=0.010, bs=128, drop=0.00, nll=0.5, rate=0.125),
]

# --- backbone='ape': AdaptivePatchPooler + FusionForecastHead ---
APE_CONFIGS = [
    # LR sweep (most impactful; fix bs=128, drop=0.1, nll=0.1, rate=0.125)
    dict(backbone='ape', lr=0.003, bs=128, drop=0.10, nll=0.1, rate=0.125, local=0, tdim=None),
    dict(backbone='ape', lr=0.005, bs=128, drop=0.10, nll=0.1, rate=0.125, local=0, tdim=None),
    dict(backbone='ape', lr=0.008, bs=128, drop=0.10, nll=0.1, rate=0.125, local=0, tdim=None),
    dict(backbone='ape', lr=0.010, bs=128, drop=0.10, nll=0.1, rate=0.125, local=0, tdim=None),
    dict(backbone='ape', lr=0.015, bs=128, drop=0.10, nll=0.1, rate=0.125, local=0, tdim=None),

    # APE local transformer layers (at lr=0.01)
    dict(backbone='ape', lr=0.010, bs=128, drop=0.10, nll=0.1, rate=0.125, local=1, tdim=None),

    # FusionForecastHead token_dim (compress dim before flatten; at lr=0.01)
    dict(backbone='ape', lr=0.010, bs=128, drop=0.10, nll=0.1, rate=0.125, local=0, tdim=8),
    dict(backbone='ape', lr=0.010, bs=128, drop=0.10, nll=0.1, rate=0.125, local=0, tdim=16),
    dict(backbone='ape', lr=0.010, bs=128, drop=0.10, nll=0.1, rate=0.125, local=0, tdim=32),

    # CMI rate (at lr=0.01, local=0)
    dict(backbone='ape', lr=0.010, bs=128, drop=0.10, nll=0.1, rate=0.083, local=0, tdim=None),
    dict(backbone='ape', lr=0.010, bs=128, drop=0.10, nll=0.1, rate=0.100, local=0, tdim=None),
    dict(backbone='ape', lr=0.010, bs=128, drop=0.10, nll=0.1, rate=0.167, local=0, tdim=None),

    # Dropout (at lr=0.01)
    dict(backbone='ape', lr=0.010, bs=128, drop=0.00, nll=0.1, rate=0.125, local=0, tdim=None),
    dict(backbone='ape', lr=0.010, bs=128, drop=0.05, nll=0.1, rate=0.125, local=0, tdim=None),
    dict(backbone='ape', lr=0.010, bs=128, drop=0.20, nll=0.1, rate=0.125, local=0, tdim=None),

    # Batch size (at lr=0.01)
    dict(backbone='ape', lr=0.010, bs=64,  drop=0.10, nll=0.1, rate=0.125, local=0, tdim=None),
    dict(backbone='ape', lr=0.010, bs=256, drop=0.10, nll=0.1, rate=0.125, local=0, tdim=None),

    # lambda_nll (at lr=0.01)
    dict(backbone='ape', lr=0.010, bs=128, drop=0.10, nll=0.5, rate=0.125, local=0, tdim=None),

    # Promising joint combos
    dict(backbone='ape', lr=0.015, bs=64,  drop=0.10, nll=0.1, rate=0.125, local=0, tdim=16),
    dict(backbone='ape', lr=0.010, bs=64,  drop=0.05, nll=0.1, rate=0.100, local=1, tdim=16),
]

ALL_CONFIGS = SIMPLE_CONFIGS + APE_CONFIGS


# ================================================================
# Config helpers
# ================================================================

def config_id(cfg: dict) -> str:
    b = cfg['backbone']
    if b == 'ape':
        tdim = cfg.get('tdim') or 'auto'
        return (
            f"hp_APE_lr{cfg['lr']}_bs{cfg['bs']}"
            f"_dr{cfg['drop']}_nll{cfg['nll']}_rate{cfg['rate']}"
            f"_loc{cfg.get('local', 0)}_td{tdim}"
        )
    return (
        f"hp_lr{cfg['lr']}_bs{cfg['bs']}"
        f"_dr{cfg['drop']}_nll{cfg['nll']}_rate{cfg['rate']}"
    )


def k_max(rate: float, seq_len: int = 96) -> int:
    return math.floor(seq_len * rate) + 2


def build_cmd(cfg: dict, gpu: int, mid: str) -> list:
    cmd = [
        PYTHON, TRAIN_SCRIPT,
        "--data",       "ETTh1",
        "--root_path",  str(ROOT / "dataset" / "ETT-small"),
        "--data_path",  "ETTh1.csv",
        "--seq_len",    "96",
        "--pred_len",   "96",
        "--enc_in",     "7",
        "--cmi_threshold", str(cfg['rate']),
        "--k_max_estimate", str(k_max(cfg['rate'])),
        "--lambda_count", "0.0",
        "--lambda_min",   "0.0",
        "--lambda_div",   "0.0",
        "--lambda_nll",   str(cfg['nll']),
        "--learning_rate", str(cfg['lr']),
        "--batch_size",    str(cfg['bs']),
        "--dropout",       str(cfg['drop']),
        "--train_epochs",  "30",
        "--patience",      "10",
        "--log_every",     "9999",
        "--gpu",           str(gpu),
        "--model_id",      mid,
    ]

    if cfg['backbone'] == 'ape':
        cmd += ["--ape_pooler", "--fusion_head"]
        if cfg.get('local', 0) > 0:
            cmd += ["--ape_local_layers", str(cfg['local'])]
        if cfg.get('tdim') is not None:
            cmd += ["--fusion_token_dim", str(cfg['tdim'])]
    else:
        cmd += ["--flat_head"]

    return cmd


# ================================================================
# Run one experiment
# ================================================================

def parse_result(path: Path):
    mse = mae = float("inf")
    for line in path.read_text().strip().splitlines():
        if line.startswith("mse:"):
            mse = float(line.split(":")[1])
        elif line.startswith("mae:"):
            mae = float(line.split(":")[1])
    return mse, mae


def run_experiment(cfg: dict, gpu: int) -> dict:
    mid      = config_id(cfg)
    log_path = LOG_DIR / f"{mid}.log"
    result_path = CKPT_DIR / mid / "result.txt"

    if result_path.exists():
        mse, mae = parse_result(result_path)
        print(f"  [skip] {mid}  mse={mse:.4f}", flush=True)
        return {**cfg, 'model_id': mid, 'mse': mse, 'mae': mae, 'status': 'cached'}

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(cfg, gpu, mid)

    t0 = time.time()
    with open(log_path, "w") as fout:
        subprocess.run(cmd, stdout=fout, stderr=subprocess.STDOUT, cwd=str(ROOT))
    elapsed = time.time() - t0

    if result_path.exists():
        mse, mae = parse_result(result_path)
        status = "ok"
        label  = f"mse={mse:.4f}  mae={mae:.4f}  ({elapsed:.0f}s)"
    else:
        mse = mae = float("inf")
        status = "FAILED"
        label  = f"FAILED  ({elapsed:.0f}s)"

    backbone_tag = "[APE]" if cfg['backbone'] == 'ape' else "[SIM]"
    print(f"  {backbone_tag} {mid}  {label}", flush=True)
    return {**cfg, 'model_id': mid, 'mse': mse, 'mae': mae, 'status': status}


# ================================================================
# Results table
# ================================================================

def print_results(results: list):
    ok = [r for r in results if r['mse'] < float("inf")]
    ok.sort(key=lambda r: r['mse'])

    if not ok:
        print("No results found yet.")
        return

    # Separate by backbone
    for label, bb in [("APE+Fusion", "ape"), ("Simple", "simple")]:
        sub = [r for r in ok if r.get('backbone', 'simple') == bb]
        if not sub:
            continue
        print(f"\n{'='*80}")
        print(f"  {label} backbone  ({len(sub)} configs)")
        print(f"{'='*80}")
        print(f"  {'#':<3} {'MSE':>7} {'MAE':>7}  {'lr':>6} {'bs':>4} "
              f"{'drop':>5} {'nll':>5} {'rate':>6}", end="")
        if bb == 'ape':
            print(f"  {'loc':>3} {'tdim':>6}", end="")
        print()
        print("-" * 80)
        for i, r in enumerate(sub, 1):
            line = (f"  {i:<3} {r['mse']:7.4f} {r['mae']:7.4f}  "
                    f"{r['lr']:>6.4f} {r['bs']:>4} {r['drop']:>5.2f} "
                    f"{r['nll']:>5.2f} {r['rate']:>6.3f}")
            if bb == 'ape':
                line += f"  {r.get('local', 0):>3} {str(r.get('tdim') or 'auto'):>6}"
            if i == 1:
                line += "  ← best"
            print(line)

    print(f"\n{'='*80}")
    best = ok[0]
    bb = best.get('backbone', 'simple')
    print(f"Overall best: {bb} backbone  mse={best['mse']:.4f}")
    print(f"  --learning_rate {best['lr']}  --batch_size {best['bs']}"
          f"  --dropout {best['drop']}  --lambda_nll {best['nll']}"
          f"  --cmi_threshold {best['rate']}")
    if bb == 'ape':
        print(f"  --ape_pooler --fusion_head"
              + (f"  --ape_local_layers {best['local']}" if best.get('local') else "")
              + (f"  --fusion_token_dim {best['tdim']}" if best.get('tdim') else ""))
    else:
        print(f"  --flat_head  --k_max_estimate {k_max(best['rate'])}")


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["simple", "ape", "all"],
                        default="all", help="Which backbone configs to run")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print configs without running")
    parser.add_argument("--results",  action="store_true",
                        help="Parse and print existing results only")
    parser.add_argument("--gpus", default="0,1",
                        help="Comma-separated GPU ids (default: 0,1)")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    n_gpus  = len(gpu_ids)

    if args.backbone == "simple":
        configs = SIMPLE_CONFIGS
    elif args.backbone == "ape":
        configs = APE_CONFIGS
    else:
        configs = ALL_CONFIGS

    if args.dry_run:
        print(f"{len(configs)} configs  ({len(SIMPLE_CONFIGS)} simple, "
              f"{len(APE_CONFIGS)} ape)  {n_gpus} GPUs  "
              f"~{math.ceil(len(configs) / n_gpus) * 4} min\n")
        for cfg in configs:
            bb = "[APE]" if cfg['backbone'] == 'ape' else "[SIM]"
            print(f"  {bb} {config_id(cfg)}")
        return

    if args.results:
        results = []
        for cfg in ALL_CONFIGS:
            rp = CKPT_DIR / config_id(cfg) / "result.txt"
            if rp.exists():
                mse, mae = parse_result(rp)
                results.append({**cfg, 'mse': mse, 'mae': mae})
        print_results(results)
        return

    print(f"Running {len(configs)} configs  "
          f"({sum(c['backbone']=='simple' for c in configs)} simple, "
          f"{sum(c['backbone']=='ape' for c in configs)} ape)")
    print(f"GPUs: {gpu_ids}  |  Logs: {LOG_DIR}\n")

    gpu_cycle = [gpu_ids[i % n_gpus] for i in range(len(configs))]
    batches   = [
        list(zip(configs[i:i + n_gpus], gpu_cycle[i:i + n_gpus]))
        for i in range(0, len(configs), n_gpus)
    ]

    results = []
    for idx, batch in enumerate(batches, 1):
        bb_tags = "+".join(
            ("APE" if c['backbone'] == 'ape' else "SIM") for c, _ in batch
        )
        print(f"--- Batch {idx}/{len(batches)} [{bb_tags}] ---")
        with ProcessPoolExecutor(max_workers=n_gpus) as pool:
            futures = {pool.submit(run_experiment, cfg, gpu): cfg
                       for cfg, gpu in batch}
            for fut in as_completed(futures):
                results.append(fut.result())

    print_results(results)


if __name__ == "__main__":
    main()
