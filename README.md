# (NeurIPS'26) 🎲 EntroPE: Entropy-Guided Dynamic Patch Segmentation for Time Series Transformers

Reference implementation for the NeurIPS 2026 paper. EntroPE places patch
boundaries at high-entropy positions using a frozen, pre-trained causal GPT
entropy model, then forecasts with a patch encoder / global transformer /
fusion decoder backbone.

This repository reproduces the long-term forecasting results on
**ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL, Traffic, and Solar** (input length
`L=96`, horizons `{96, 192, 336, 720}`). You can either **evaluate the provided
checkpoints** (minutes) or **train everything from scratch**.

## 1. Setup

```bash
conda create -n entrope python=3.10 -y
conda activate entrope
pip install -r requirements.txt
```

Tested with Python 3.10 and PyTorch 2.x (CUDA 12.x). A single GPU is enough.

## 2. Data and checkpoints

The benchmark CSVs and the trained forecasting checkpoints are large, so they
are hosted on the HuggingFace Hub (not GitHub):
**https://huggingface.co/datasets/sachithabey/EntroPE**

Fetch them into the repo with one command (needs `huggingface_hub`, included in
`requirements.txt`):

```bash
python download_data.py                      # datasets + checkpoints
python download_data.py --what datasets      # just the CSVs (train from scratch)
python download_data.py --what checkpoints   # just the checkpoints (eval only)
```

This writes to the exact paths the code expects:

- **`dataset/`** — the eight benchmark CSVs (`ETTh1/ETTh2/ETTm1/ETTm2.csv`,
  `weather.csv`, `electricity.csv`, `traffic.csv`, `solar.csv`). Each is
  normalized by a `StandardScaler` fit on its own training split, automatically
  at load time.
- **`checkpoints/<setting>/checkpoint.pth`** — one trained forecasting model per
  (dataset, horizon) cell (download only if you want eval-only reproduction).

The frozen GPT entropy models in **`entropy_model_checkpoints/dm16/`**
(`params.json` + one `<dataset>.pt` per dataset) are small and **ship in this
GitHub repo** — no download needed. They are loaded separately from the
forecasting weights and are required even for eval-only runs.

## 3. Reproduce the numbers (eval only, no training)

There is one script per dataset (`scripts/<DATASET>.sh`), each holding the best
config for all four horizons (96/192/336/720) in a readable table. Pass
`eval` (default) or `train`:

```bash
bash scripts/ETTh1.sh eval         # one dataset, all 4 horizons
bash scripts/run_all.sh eval       # every dataset
GPU=1 bash scripts/ECL.sh eval     # choose a GPU
```

Datasets: `ETTh1 ETTh2 ETTm1 ETTm2 weather ECL Traffic solar`. Each horizon's
config is stated inline in the script with its paper MSE/MAE. Fixed settings
(architecture, L=96, epochs, …) are argparse defaults in `run_longExp.py`.


MSE/MAE are computed in the standardized (z-scored) space, following the
standard long-term-forecasting protocol.

## 4. Train from scratch

**Stage 2 — forecasting (the frozen entropy models are already provided):**

```bash
bash scripts/ETTm2.sh train        # one dataset, all 4 horizons
bash scripts/run_all.sh train      # every cell
```

Each cell trains at its best configuration and writes
`checkpoints/<setting>/checkpoint.pth` (overwriting the provided checkpoint for
that cell), then evaluates.

**Stage 1 — entropy models (optional).** The exact entropy checkpoints used in
the paper are already in `entropy_model_checkpoints/dm16/`. To regenerate them:

```bash
bash scripts/train_entropy.sh      # all datasets, writes to entropy_model_checkpoints/dm16/
```


## 5. Repository layout

```
run_longExp.py            Stage-2 entry point (train / eval forecasting)
train_entropy_model.py    Stage-1 entry point (train frozen GPT entropy model)
exp/                      Experiment loop (Exp_Main)
models/                   EntroPE forecasting model + GPT2 entropy model
layers/                   Patcher, encoder, global transformer, fusion decoder, RevIN, tokenizer
data_provider/            Dataset loaders and StandardScaler
utils/                    Metrics, schedulers, helpers
scripts/                  per-dataset run scripts (<DATASET>.sh, train|eval), run_all.sh, train_entropy.sh
dataset/                  Benchmark CSVs
checkpoints/              Provided forecasting checkpoints (one dir per cell)
entropy_model_checkpoints/dm16/   Provided frozen entropy models
```


## Citing

If you found this work useful for you, please consider citing it.

```bibtex
@inproceedings{sachith_entrope_26,
  title={Entropy Guided Dynamic Patch Segmentation for Time Series Transformers},
  author={Abeywickrama, Sachith and Eldele, Emadeldeen and Wu, Min and Li, Xiaoli and Yuen, Chau},
  booktitle = {Advances in Neural Information Processing Systems},
  year={2026}
}
```


## Acknowledgments

This work builds upon and is inspired by several key contributions in the field:

- **nanoGPT**: The Entropy Model GPT-2 architecture implementation partially incorporates code from Andrej Karpathy's nanoGPT implementation. We gratefully acknowledge this clean and educational codebase.
  - Repository: https://github.com/karpathy/nanoGPT

- **Byte Latent Transformer**: Our dynamic patching approach draws inspiration from advances in NLP, particularly the Byte Latent Transformer's innovative approach to variable-length tokenization.
  - Repository: https://github.com/facebookresearch/blt

---

We thank the authors of these works for their contributions to the open-source community and for advancing the state of the art in time series forecasting and transformer architectures.
