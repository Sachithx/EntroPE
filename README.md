# EntroPE: Entropy-Guided Dynamic Patch Encoder for Time Series Forecasting

This repository contains the official implementation of the paper:
> **EntroPE: Entropy-Guided Dynamic Patch Encoder for Time Series Forecasting**  
> *Under review at ICLR 2026*

---

## Overview

**EntroPE** introduces a novel framework for temporally-informed dynamic patching in time series forecasting using Transformer models. Unlike conventional fixed-length patching schemes that arbitrarily segment temporal sequences, our method uses information-theoretic principles to identify natural transition points and dynamically place patch boundaries where predictive uncertainty is highest.

---

## Key Contributions

- **Entropy-Based Dynamic Patcher (EDP)**  
  Uses conditional entropy from a lightweight causal transformer to identify temporal transition points and place patch boundaries at positions with high predictive uncertainty.

- **Adaptive Patch Encoder (APE)**  
  Employs pooling and cross-attention mechanisms to encode variable-length patches into fixed-size representations while preserving intra-patch dependencies.

- **Temporally-Informed Architecture**  
  Preserves temporal coherence by respecting the natural structure of time series data, addressing train-inference mismatch and boundary fragmentation issues.

---

## Architecture

EntroPE consists of three main components:

1. **Entropy-Based Dynamic Patcher**: Computes entropy at each time point using a pre-trained lightweight transformer and applies dual-threshold boundary detection
2. **Adaptive Patch Encoder**: Converts variable-length patches into fixed-size embeddings through pooling and iterative cross-attention refinement  
3. **Fusion Decoder**: Combines global patch context with local encoder hidden states using cross-attention for accurate forecasting

---

## Installation

```bash
# Clone the repository (link will be provided upon acceptance)
git clone https://anonymous.repository/entrope
cd entrope

# Install dependencies
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate entrope
```

---
## Experiments

### Datasets

We evaluate on 8 benchmark datasets:
- **ETT family**: ETTh1, ETTh2, ETTm1, ETTm2 (Energy)
- **Weather**: Meteorological data 
- **Electricity**: Power consumption
- **Exchange Rate**: Currency exchange rates
- **Traffic**: Road occupancy rates

### Reproduction

```bash
# Run specific dataset
sh scripts/etth1.sh
```

### Results Summary

EntroPE achieves significant improvements over existing methods:
- ~20% improvement on ETTh1 compared to PatchTST
- ~15% improvement on Electricity dataset
- ~10% average improvement across all benchmarks
- Better computational efficiency through dynamic patching

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

## Configuration

### Model Parameters

```yaml
# Core architecture
seq_len: 96              # Input sequence length
pred_len: 96             # Prediction horizon
d_model: 512             # Model dimension
n_heads: 8               # Number of attention heads
e_layers: 2              # Number of encoder layers

# Dynamic patching
threshold_global: 3.0    # Global entropy threshold (θ)
threshold_relative: 0.25 # Relative entropy threshold (γ)
max_patch_len: 24        # Maximum patch length
vocab_size: 256          # Tokenization vocabulary size

# Training
learning_rate: 0.001
batch_size: 64
train_epochs: 20
patience: 7
dropout: 0.1
```

### Threshold Selection
- **threshold_global**: Controls patch granularity (higher = fewer patches)
- **threshold_relative**: Controls sensitivity to entropy changes
- Robust across range [0.15, 0.55] for most datasets

---

## Evaluation Metrics

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error  
- **Efficiency**: MACs (Multiply-Accumulate Operations)

---

## Reproducibility

All results can be reproduced using:
- Fixed random seeds (5 different seeds reported)
- Complete hyperparameter configurations provided
- Environment specifications included
- Multiple GPU platform compatibility

Tested on (Only single-GPU settings):
- NVIDIA RTX 4090
- NVIDIA RTX A5000  
- NVIDIA GeForce RTX 4090 D
- NVIDIA RTX 6000 Ada-16Q

---

## License

This project is licensed under the Apache License - see the LICENSE file for details.

---

## Acknowledgments

- Built on the foundation of PatchTST and other patch-based time series transformers
- Inspired by dynamic patching approaches in NLP (Byte Latent Transformer)
- Evaluation framework adapted from TimeKAN benchmarking suite

---

## Contact

For questions about this work, please contact: [contact information will be provided upon acceptance]

Repository will be made public upon paper acceptance.