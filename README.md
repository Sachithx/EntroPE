# EntroPE: Entropy-Guided Dynamic Patching for Time-Series Forecasting

Reference implementation for the NeurIPS 2026 paper. EntroPE places patch
boundaries at high-entropy positions using a frozen, pre-trained causal GPT
entropy model, then forecasts with a patch encoder / global transformer /
fusion decoder backbone.

This repository reproduces the long-term forecasting results on
**ETTh1, ETTh2, ETTm1, ETTm2, and Weather** (input length `L=96`, horizons
`{96, 192, 336, 720}`). You can either **evaluate the provided checkpoints**
(minutes) or **train everything from scratch**.

## 1. Setup

```bash
conda create -n entrope python=3.10 -y
conda activate entrope
pip install -r requirements.txt
```

Tested with Python 3.10 and PyTorch 2.x (CUDA 12.x). A single GPU is enough.

## 2. Data and checkpoints

- **Datasets.** The five standard benchmark CSVs are in `dataset/`
  (`ETTh1/ETTh2/ETTm1/ETTm2.csv`, `weather.csv`). Each dataset is normalized
  by a `StandardScaler` fit on its own training split; this is done
  automatically at load time, so no extra files are needed.
- **Checkpoints.** Two sets are required and are provided with the release:
  - `entropy_model_checkpoints/dm16/` — the frozen GPT entropy models
    (`params.json` + one `<dataset>.pt` per dataset). **These are loaded
    separately from the forecasting weights and are required even for
    eval-only.**
  - `checkpoints/<setting>/checkpoint.pth` — one trained forecasting model per
    (dataset, horizon) cell.

  If you cloned from GitHub (where large files are git-ignored), download the
  `checkpoints/` and `entropy_model_checkpoints/` folders from the release
  bucket and place them at the repository root, preserving their layout.

## 3. Reproduce the reported numbers (eval only, no training)

```bash
bash scripts/eval_all.sh           # all 5 datasets
bash scripts/eval_all.sh ETTh1     # one dataset
```

This loads each provided checkpoint and prints the test MSE/MAE next to the
value reported in the paper, e.g.:

```
ETTh1       96  0.3794    0.3961     (paper: 0.3794 / 0.3961)
```

MSE/MAE are computed in the standardized (z-scored) space, following the
standard long-term-forecasting protocol.

## 4. Train from scratch

**Stage 2 — forecasting (the frozen entropy models are already provided):**

```bash
bash scripts/train_all.sh          # all cells
bash scripts/train_all.sh ETTm2    # one dataset
```

Each cell trains at its best configuration and writes
`checkpoints/<setting>/checkpoint.pth` (overwriting the provided checkpoint for
that cell), then evaluates.

**Stage 1 — entropy models (optional).** The exact entropy checkpoints used in
the paper are already in `entropy_model_checkpoints/dm16/`. To regenerate them:

```bash
bash scripts/train_entropy.sh      # all datasets, writes to entropy_model_checkpoints/dm16/
```

## 5. Best configuration per cell

All best per-(dataset, horizon) configurations are in
[`scripts/best_configs.tsv`](scripts/best_configs.tsv) and drive every script
above. Fixed across all runs: `seq_len=96`, `label_len=48`, `e_layers=2`,
`n_heads=4`, `d_ff=256`, `vocab_size=256`, `max_patch_length=16`,
`boundary_method=entropy`, within-patch encoder attention, RevIN (affine),
50 epochs, patience 10, `lradj=TST`. Swept per cell: model size tier
(`small`=d_model 8/global 32, `medium`=16/64, `large`=32/128), batch size,
learning rate, entropy threshold, dropout, monotonicity, and seed.

To run a single cell directly, see `scripts/run_cell.sh` (called by the `*_all`
scripts).

## 6. Repository layout

```
run_longExp.py            Stage-2 entry point (train / eval forecasting)
train_entropy_model.py    Stage-1 entry point (train frozen GPT entropy model)
exp/                      Experiment loop (Exp_Main)
models/                   EntroPE forecasting model + GPT2 entropy model
layers/                   Patcher, encoder, global transformer, fusion decoder, RevIN, tokenizer
data_provider/            Dataset loaders and StandardScaler
utils/                    Metrics, schedulers, helpers
scripts/                  best_configs.tsv + reproduction scripts
dataset/                  Benchmark CSVs
checkpoints/              Provided forecasting checkpoints (one dir per cell)
entropy_model_checkpoints/dm16/   Provided frozen entropy models
```
