<div align="center">
<img src="assets/logo.png" alt="Main Architecture" width="300">
</div>

# Entropy-Guided Dynamic Patch Encoder for Time Series Forecasting

This repository contains the official implementation of the paper:
> **Entropy-Guided Dynamic Patch Segmentation for Time Series Transformers**  
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
We evaluate on 6 benchmark datasets:
- **ETT family**: ETTh1, ETTh2, ETTm1, ETTm2 (Energy)
- **Weather**: Meteorological data 
- **Electricity**: Power consumption

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


### Threshold Selection
- **threshold_global**: Controls patch granularity (higher = fewer patches)
- **threshold_relative**: Controls sensitivity to entropy changes
- Robust across range [75%, 95%] for most datasets

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

**Evaluation Metrics**: MSE, MAE, MACs (Multiply-Accumulate Operations)

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

## License

This project is licensed under the Apache License - see the LICENSE file for details.

---

## Acknowledgments

This work builds upon and is inspired by several key contributions in the field:

- **PatchTST**: Our approach is built on the foundation of PatchTST and other patch-based time series transformers, which demonstrated the effectiveness of patch-based architectures for time series forecasting.
  - Repository: https://github.com/yuqinie98/PatchTST

- **nanoGPT**: The Entropy Model GPT-2 architecture implementation partially incorporates code from Andrej Karpathy's nanoGPT implementation. We gratefully acknowledge this clean and educational codebase.
  - Repository: https://github.com/karpathy/nanoGPT

- **Byte Latent Transformer**: Our dynamic patching approach draws inspiration from advances in NLP, particularly the Byte Latent Transformer's innovative approach to variable-length tokenization.
  - Repository: https://github.com/facebookresearch/blt

---

We thank the authors of these works for their contributions to the open-source community and for advancing the state of the art in time series forecasting and transformer architectures.
