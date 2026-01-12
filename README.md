<div align="center">
<img src="assets/logo.png" alt="Main Architecture" width="300">
</div>

# Entropy-Guided Dynamic Patch Encoder for Time Series Forecasting

This repository contains the official implementation of the paper:
> **EntroPE: Entropy-Guided Dynamic Patch Encoder for Time Series Forecasting**  
> *Under review.*

---

## Overview

**EntroPE** introduces entropy-driven dynamic patching for time series Transformers. Instead of fixed-length patches, we use information-theoretic principles to place patch boundaries where predictive uncertainty is highest, creating semantically meaningful temporal segments.

<div align="center">
<img src="assets/main_architecture.png" alt="Main Architecture" width="600">
<p><em>EntroPE framework architecture: (A) Entropy-Based Dynamic Patcher calculates entropy to identify boundaries; (B) Adaptive Patch Encoder aggregates patches into fixed-size embeddings; (C) Fusion Decoder combines global and local context for forecasting.</em></p>
</div>

<div align="center">
<img src="assets/static_vs_dynamic_patch.png" alt="Static vs Dynamic Patching" width="400">
<p><em>Static patching (top) vs. our entropy-driven dynamic approach (bottom).</em></p>
</div>

---

## Key Contributions

- **Entropy-Based Dynamic Patcher (EDP)**: Uses conditional entropy to identify temporal transition points and place patch boundaries at positions with high predictive uncertainty
- **Adaptive Patch Encoder (APE)**: Employs cross-attention mechanisms to encode variable-length patches into fixed-size representations while preserving dependencies
- **Temporally-Informed Architecture**: Preserves temporal coherence by respecting natural time series structure, addressing train-inference mismatch

---

## Installation

```bash
# Create environment
conda create --name entrope python=3.10
conda activate entrope
pip install -r requirements.txt
```

---

## Experiments

### Datasets
We evaluate on 7 benchmark datasets:
- **ETT family**: ETTh1, ETTh2, ETTm1, ETTm2 (Energy)
- **Weather**: Meteorological data 
- **Electricity**: Power consumption
- **Exchange Rate**: Currency exchange rates

### Quick Start
```bash
# Run specific dataset
sh scripts/etth1.sh
```

### Results Summary
EntroPE achieves significant improvements over existing methods:
- **~20% improvement** on ETTh1 compared to PatchTST
- **~15% improvement** on Electricity dataset
- **~10% average improvement** across all benchmarks
- Better computational efficiency through dynamic patching

---

## Configuration

### Model Parameters
```yaml
# Core model settings
learning_rate: 0.001
batch_size: 64
train_epochs: 20
patience: 7
dropout: 0.1

# Architecture
dim: 8
heads: 2
layers: 1
multiple_of: 64

# Dynamic patching
patching_threshold: 0.2
patching_threshold_add: 0.01
max_patch_length: 24
monotonicity: 1
```

### Threshold Selection
- **threshold_global**: Controls patch granularity (higher = fewer patches)
- **threshold_relative**: Controls sensitivity to entropy changes
- Robust across range [0.15, 0.55] for most datasets

---

## Key Features

### Dynamic Boundary Detection
- Information-theoretic approach using conditional entropy
- Dual-threshold mechanism (global + relative thresholds)
- Respects temporal causality and predictive difficulty

### Adaptive Encoding
- Handles variable-length patches efficiently
- Cross-attention refinement preserves intra-patch dependencies
- Fixed-size output suitable for transformer processing

### Efficiency Benefits
- Reduces token count through intelligent boundary placement
- Maintains computational tractability
- Scales well with sequence length

---

## Reproducibility

All results can be reproduced using:
- Fixed random seeds (5 different seeds reported)
- Complete hyperparameter configurations provided
- Environment specifications included

**Tested Platforms** (single-GPU):
- NVIDIA RTX 4090
- NVIDIA RTX A5000  
- NVIDIA GeForce RTX 4090 D
- NVIDIA RTX 6000 Ada-16Q

**Evaluation Metrics**: MSE, MAE, MACs (Multiply-Accumulate Operations)

---

## License

This project is licensed under the Apache License - see the LICENSE file for details.

---

## Citing

If you found this work useful for you, please consider citing it.

```bibtex
@article{abeywickrama2025entrope,
  title={EntroPE: Entropy-Guided Dynamic Patch Encoder for Time Series Forecasting},
  author={Abeywickrama, Sachith and Eldele, Emadeldeen and Wu, Min and Li, Xiaoli and Yuen, Chau},
  journal={arXiv preprint arXiv:2509.26157},
  year={2025}
}
```

---

## Acknowledgments

This work builds upon and is inspired by several key contributions in the field:

### Foundational Work
- **PatchTST**: Our approach is built on the foundation of PatchTST and other patch-based time series transformers, which demonstrated the effectiveness of patch-based architectures for time series forecasting.
  - Repository: https://github.com/yuqinie98/PatchTST

- **nanoGPT**: The Entropy Model GPT-2 architecture implementation partially incorporates code from Andrej Karpathy's nanoGPT implementation. We gratefully acknowledge this clean and educational codebase.
  - Repository: https://github.com/karpathy/nanoGPT

- **Byte Latent Transformer**: Our dynamic patching approach draws inspiration from advances in NLP, particularly the Byte Latent Transformer's innovative approach to variable-length tokenization.
  - Repository: https://github.com/facebookresearch/blt

---

We thank the authors of these works for their contributions to the open-source community and for advancing the state of the art in time series forecasting and transformer architectures.
