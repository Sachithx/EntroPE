"""
Parse multi-seed log files and produce mean ± std table.
Run after: bash scripts/run_multiseed.sh

Output: results/multiseed_summary.csv  and  results/multiseed_summary.txt
"""

import re
import os
import glob
import numpy as np
from collections import defaultdict

LOG_DIR = "logs/MultiSeed"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# Regex to extract MSE and MAE from log line:
#   MSE: 0.3906, MAE: 0.4024, RSE: 0.5926
MSE_RE = re.compile(r"MSE:\s*([\d.]+),\s*MAE:\s*([\d.]+)")

# Collect results keyed by (dataset, pred_len)
results = defaultdict(lambda: defaultdict(list))  # results[dataset][pred_len] = [(mse, mae), ...]

log_files = glob.glob(os.path.join(LOG_DIR, "*.log"))
if not log_files:
    print(f"No log files found in {LOG_DIR}. Run run_multiseed.sh first.")
    exit(1)

for fpath in sorted(log_files):
    fname = os.path.basename(fpath)
    # Expected: {DATASET}_PL{PRED_LEN}_seed{SEED}.log
    m = re.match(r"(.+)_PL(\d+)_seed(\d+)\.log", fname)
    if not m:
        continue
    dataset, pred_len, seed = m.group(1), int(m.group(2)), int(m.group(3))

    with open(fpath) as f:
        content = f.read()

    # Find the last MSE/MAE line (test set result)
    matches = MSE_RE.findall(content)
    if not matches:
        print(f"  WARNING: no MSE/MAE found in {fname}")
        continue

    mse, mae = float(matches[-1][0]), float(matches[-1][1])
    results[dataset][pred_len].append((mse, mae))
    print(f"  {dataset} PL={pred_len} seed={seed}: MSE={mse:.4f} MAE={mae:.4f}")

# Build summary table
datasets = sorted(results.keys())
pred_lens = [96, 192, 336, 720]

csv_rows = ["Dataset,PredLen,n_seeds,MSE_mean,MSE_std,MAE_mean,MAE_std"]
txt_lines = []
txt_lines.append(f"{'Dataset':12s} {'PL':>4s} {'n':>2s}  {'MSE mean±std':>18s}  {'MAE mean±std':>18s}")
txt_lines.append("-" * 62)

for dataset in datasets:
    for pl in pred_lens:
        vals = results[dataset].get(pl, [])
        if not vals:
            continue
        mses = [v[0] for v in vals]
        maes = [v[1] for v in vals]
        mse_m, mse_s = np.mean(mses), np.std(mses)
        mae_m, mae_s = np.mean(maes), np.std(maes)

        csv_rows.append(f"{dataset},{pl},{len(vals)},{mse_m:.4f},{mse_s:.4f},{mae_m:.4f},{mae_s:.4f}")
        txt_lines.append(
            f"{dataset:12s} {pl:>4d} {len(vals):>2d}  "
            f"{mse_m:.4f}±{mse_s:.4f}       {mae_m:.4f}±{mae_s:.4f}"
        )

csv_path = os.path.join(OUT_DIR, "multiseed_summary.csv")
txt_path = os.path.join(OUT_DIR, "multiseed_summary.txt")

with open(csv_path, "w") as f:
    f.write("\n".join(csv_rows))

with open(txt_path, "w") as f:
    f.write("\n".join(txt_lines))

print("\n" + "\n".join(txt_lines))
print(f"\nSaved: {csv_path}")
print(f"Saved: {txt_path}")
