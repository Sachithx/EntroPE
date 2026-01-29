<div align="center">
<img src="assets/logo.png" alt="Main Architecture" width="300">
</div>

# Entropy-Guided Dynamic Patch Segmentation for Time Series Transformers

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

### Quick Start
```bash
# Run specific dataset
sh scripts/ETTh1.sh
```
---

## License

This project is licensed under the Apache License - see the LICENSE file for details.

---

## Acknowledgments

This work is inspired by several key contributions in the field:

- **PatchTST**: Our approach is built on the foundation of PatchTST and other patch-based time series transformers, which demonstrated the effectiveness of patch-based architectures for time series forecasting.
  - Repository: https://github.com/yuqinie98/PatchTST

- **nanoGPT**: The Entropy Model GPT-2 architecture implementation partially incorporates code from Andrej Karpathy's nanoGPT implementation. We gratefully acknowledge this clean and educational codebase.
  - Repository: https://github.com/karpathy/nanoGPT

- **Byte Latent Transformer**: Our dynamic patching approach draws inspiration from advances in NLP, particularly the Byte Latent Transformer's innovative approach to variable-length tokenization.
  - Repository: https://github.com/facebookresearch/blt

---

We thank the authors of these works for their contributions to the open-source community and for advancing the state of the art in time series forecasting and transformer architectures.
