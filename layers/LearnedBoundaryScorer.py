# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This file contains code adapted from Meta's Byte Latent Transformer
# implementation (https://github.com/facebookresearch/blt).
#
# Modifications and extensions:
#   - Learned boundary scorer MLP for CMI-guided patching
#   - Input: raw values + MVG distribution parameters + CMI signal
#
# Copyright (c) 2026 Sachith Abeywickrama
#
# Licensed under the same license as the original Meta code.

"""
LearnedBoundaryScorer

Small MLP that consumes per-timestep features derived from the multivariate
Gaussian model and produces a boundary logit for each timestep.

Feature vector (B, L, feat_dim):
  x         (C)              raw values
  mu        (C)              MVG predicted mean
  vech_L    (C*(C+1)/2)      vectorised Cholesky  [or C + C*r for low-rank]
  r         (C)              whitened (Mahalanobis) residual
  cmi       (1)              |Delta log|Sigma_t|| CMI signal

The scorer is trained end-to-end with the forecasting loss via
Gumbel-sigmoid straight-through estimation (see GumbelSigmoidPatcher).
"""

import torch
import torch.nn as nn


class LearnedBoundaryScorer(nn.Module):
    """
    MLP boundary scorer.

    Args:
        n_channels:  C, number of time series variables
        cov_feat_dim: dimension of flattened covariance features
                      (C*(C+1)//2 for full Cholesky, C + C*r for low-rank)
        hidden:      hidden dimension of the MLP
    """

    def __init__(self, n_channels: int, cov_feat_dim: int, hidden: int = 64):
        super().__init__()
        # x + mu + vech(L) + r + cmi
        feat_dim = n_channels + n_channels + cov_feat_dim + n_channels + 1

        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),   # boundary logit per timestep
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x:      torch.Tensor,   # (B, L, C)  raw values
        mu:     torch.Tensor,   # (B, L, C)  MVG mean
        vech_L: torch.Tensor,   # (B, L, F)  flattened cov params
        cmi:    torch.Tensor,   # (B, L)     CMI signal
        r:      torch.Tensor,   # (B, L, C)  whitened residuals
    ) -> torch.Tensor:
        """Returns boundary logits of shape (B, L)."""
        feat = torch.cat(
            [x, mu, vech_L, r, cmi.unsqueeze(-1)], dim=-1
        )                                   # (B, L, feat_dim)
        return self.net(feat).squeeze(-1)   # (B, L)
