# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This file contains code adapted from Meta's Byte Latent Transformer
# implementation (https://github.com/facebookresearch/blt).
#
# Modifications and extensions:
#   - Continuous multivariate Gaussian entropy model (no discretization)
#   - Full Cholesky and low-rank+diagonal covariance parameterisations
#   - beta-NLL loss (Seitzer et al. 2022) in multivariate form
#   - CMI (Covariance Matrix Information) boundary signal
#
# Copyright (c) 2026 Sachith Abeywickrama
#
# Licensed under the same license as the original Meta code.

"""
MultivariateGaussianModel

A small causal transformer that directly models p(x_t | x_{1:t-1}) as a
multivariate Gaussian distribution over raw real-valued time series.
Replaces the discrete-token GPT-2 entropy model with a continuous density
estimator, eliminating all quantisation / discretisation.

Two covariance parameterisations:
  cov_rank=None  -> full Cholesky  L_t  (B, L, C, C)  for small C (<= 32)
  cov_rank=r     -> low-rank+diag (D_t, U_t)           for large C (e.g. 321)

Static utility methods:
  log_det(cov)                   -> per-timestep log|Sigma_t|  (B, L)
  mahalanobis_residual(x,mu,cov) -> whitened residual r_t       (B, L, C)
  cmi_signal(log_det)            -> |Delta log|Sigma_t||        (B, L)
  flatten_cov(cov)               -> vectorised cov params       (B, L, feat)
  mvg_nll_loss(target,mu,cov,b)  -> beta-NLL scalar
"""

import math
from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================
# Config
# ============================================================

@dataclass
class MVGConfig:
    n_channels: int          # C: number of time series variables
    block_size: int          # L: context (sequence) length
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.1
    bias: bool = False
    cov_rank: int | None = None   # None = full Cholesky; int r = low-rank+diag
    min_log_diag: float = -7.0
    max_log_diag: float = 3.0
    beta_nll: float = 0.5         # beta for Seitzer 2022 beta-NLL


# ============================================================
# Transformer Building Blocks (self-contained)
# ============================================================

class _LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class _CausalSelfAttention(nn.Module):
    def __init__(self, config: MVGConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class _MLP(nn.Module):
    def __init__(self, config: MVGConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class _Block(nn.Module):
    def __init__(self, config: MVGConfig):
        super().__init__()
        self.ln_1 = _LayerNorm(config.n_embd, bias=config.bias)
        self.attn = _CausalSelfAttention(config)
        self.ln_2 = _LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = _MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ============================================================
# MultivariateGaussianModel
# ============================================================

class MultivariateGaussianModel(nn.Module):
    """
    Causal transformer estimating p(x_t | x_{1:t-1}) ~ N(mu_t, Sigma_t).

    Input:  x  (B, L, C)  raw real-valued multivariate time series
    Output: (mu, cov)
        full mode:      mu (B,L,C),  L_chol (B,L,C,C) lower-triangular
        low-rank mode:  mu (B,L,C),  (D (B,L,C), U (B,L,C,r))
    """

    def __init__(self, config: MVGConfig):
        super().__init__()
        self.config = config
        C, r = config.n_channels, config.cov_rank

        self.input_proj = nn.Linear(C, config.n_embd, bias=config.bias)
        self.pos_emb    = nn.Embedding(config.block_size, config.n_embd)
        self.drop       = nn.Dropout(config.dropout)
        self.blocks     = nn.ModuleList([_Block(config) for _ in range(config.n_layer)])
        self.ln_f       = _LayerNorm(config.n_embd, bias=config.bias)

        self.mu_head    = nn.Linear(config.n_embd, C, bias=config.bias)

        if r is None:
            cov_out = C * (C + 1) // 2
        else:
            cov_out = C + C * r
        self.cov_head = nn.Linear(config.n_embd, cov_out, bias=config.bias)

        self.apply(self._init_weights)
        # Rescale output-projection weights per GPT-2 paper
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """x: (B, L, C) — one step ahead prediction target is x[:, 1:]."""
        B, L, C = x.shape
        assert L <= self.config.block_size, (
            f"Sequence length {L} exceeds block_size {self.config.block_size}"
        )
        pos = torch.arange(L, device=x.device)
        h = self.drop(self.input_proj(x) + self.pos_emb(pos))  # (B, L, n_embd)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)

        mu      = self.mu_head(h)       # (B, L, C)
        cov_raw = self.cov_head(h)      # (B, L, cov_out)

        if self.config.cov_rank is None:
            return mu, self._build_full_cholesky(cov_raw)
        else:
            return mu, self._build_low_rank(cov_raw)

    # ----------------------------------------------------------
    # Covariance builders
    # ----------------------------------------------------------

    def _build_full_cholesky(self, raw: torch.Tensor) -> torch.Tensor:
        """(B, L, C*(C+1)/2) → (B, L, C, C) lower-triangular Cholesky factor."""
        B, L = raw.shape[:2]
        C    = self.config.n_channels
        dev  = raw.device

        tril_r, tril_c = torch.tril_indices(C, C, device=dev)
        L_chol = raw.new_zeros(B, L, C, C)
        L_chol[:, :, tril_r, tril_c] = raw

        # Positive diagonal: softplus + clamp in log space
        idx    = torch.arange(C, device=dev)
        d_raw  = L_chol[:, :, idx, idx]
        d_pos  = F.softplus(d_raw) + 1e-3
        d_log  = d_pos.log().clamp(self.config.min_log_diag, self.config.max_log_diag)
        L_chol[:, :, idx, idx] = d_log.exp()
        return L_chol  # (B, L, C, C)

    def _build_low_rank(self, raw: torch.Tensor):
        """(B, L, C+C*r) → (D:(B,L,C), U:(B,L,C,r))."""
        C, r    = self.config.n_channels, self.config.cov_rank
        D_raw   = raw[..., :C]
        U       = raw[..., C:].view(*raw.shape[:2], C, r)
        d_pos   = F.softplus(D_raw) + 1e-3
        D       = d_pos.log().clamp(self.config.min_log_diag, self.config.max_log_diag).exp()
        return D, U     # U is unconstrained

    # ----------------------------------------------------------
    # Static utility methods
    # ----------------------------------------------------------

    @staticmethod
    def log_det(cov) -> torch.Tensor:
        """Per-timestep log|Sigma_t|. Returns (B, L)."""
        if isinstance(cov, tuple):
            D, U = cov          # (B,L,C), (B,L,C,r)
            r    = U.shape[-1]
            D_inv   = 1.0 / D                              # (B, L, C)
            UD_inv  = U * D_inv.unsqueeze(-1)              # (B, L, C, r)
            # M = I_r + U^T D^{-1} U
            M = torch.einsum('...ci,...cj->...ij', UD_inv, U)  # (B, L, r, r)
            M = M + torch.eye(r, device=U.device, dtype=U.dtype)
            log_det_M = torch.linalg.slogdet(M)[1]        # (B, L)
            log_det_D = D.log().sum(-1)                    # (B, L)
            return log_det_D + log_det_M
        else:
            # log det(L L^T) = 2 * sum log diag(L)
            return 2.0 * cov.diagonal(dim1=-2, dim2=-1).log().sum(-1)  # (B, L)

    @staticmethod
    def mahalanobis_residual(
        x: torch.Tensor, mu: torch.Tensor, cov
    ) -> torch.Tensor:
        """
        Whitened residual r_t = L_t^{-1}(x_t - mu_t) so that ||r_t||^2 = mahal.
        Returns (B, L, C).
        """
        res = x - mu  # (B, L, C)
        if isinstance(cov, tuple):
            return MultivariateGaussianModel._woodbury_solve(res, *cov)
        else:
            return torch.linalg.solve_triangular(
                cov, res.unsqueeze(-1), upper=False
            ).squeeze(-1)

    @staticmethod
    def _woodbury_solve(
        res: torch.Tensor, D: torch.Tensor, U: torch.Tensor
    ) -> torch.Tensor:
        """Sigma^{-1} res via Woodbury. Sigma = diag(D) + U U^T."""
        r       = U.shape[-1]
        D_inv   = 1.0 / D                                  # (B, L, C)
        v       = res * D_inv                              # D^{-1} res
        UD_inv  = U * D_inv.unsqueeze(-1)                  # (B, L, C, r)
        M = torch.einsum('...ci,...cj->...ij', UD_inv, U)  # (B, L, r, r)
        M = M + torch.eye(r, device=U.device, dtype=U.dtype)
        Utv     = torch.einsum('...ci,...c->...i', U, v)   # U^T v, (B, L, r)
        M_inv   = torch.linalg.solve(M, Utv)               # (B, L, r)
        corr    = torch.einsum('...ci,...i->...c', UD_inv, M_inv)  # D^{-1} U M^{-1} U^T v
        return v - corr                                    # Sigma^{-1} res

    @staticmethod
    def cmi_signal(log_det: torch.Tensor) -> torch.Tensor:
        """
        Covariance Matrix Information (CMI) signal: |Delta log|Sigma_t||.
        High value -> sharp distributional shift -> likely patch boundary.
        Returns (B, L) with cmi[:, 0] = 0.
        """
        cmi = torch.zeros_like(log_det)
        if log_det.shape[1] > 1:
            cmi[:, 1:] = (log_det[:, 1:] - log_det[:, :-1]).abs()
        return cmi

    @staticmethod
    def flatten_cov(cov) -> torch.Tensor:
        """
        Vectorise covariance parameters for use as boundary-scorer features.
        Returns (B, L, feat_dim):
            full:     C*(C+1)/2  (lower-triangular vech)
            low-rank: C + C*r
        """
        if isinstance(cov, tuple):
            D, U = cov
            return torch.cat([D, U.flatten(-2, -1)], dim=-1)   # (B, L, C+C*r)
        else:
            C     = cov.shape[-1]
            dev   = cov.device
            tr, tc = torch.tril_indices(C, C, device=dev)
            return cov[:, :, tr, tc]                            # (B, L, C*(C+1)/2)

    @staticmethod
    def mvg_nll_loss(
        target: torch.Tensor,
        mu: torch.Tensor,
        cov,
        beta: float = 0.5,
    ) -> torch.Tensor:
        """
        beta-NLL loss (Seitzer et al. 2022) in multivariate form.

        Standard NLL: 0.5 * (mahal + log_det + C*log(2*pi))
        beta-NLL:     0.5 * (mahal + beta * log_det + C*log(2*pi))

        beta=1 -> standard NLL; beta=0 -> MSE-like (no variance learning).
        Recommended: beta=0.5 (Seitzer 2022 default).

        target, mu: (B, L, C)
        Returns scalar.
        """
        r       = MultivariateGaussianModel.mahalanobis_residual(target, mu, cov)
        mahal   = (r * r).sum(-1)                              # (B, L)
        ld      = MultivariateGaussianModel.log_det(cov)       # (B, L)
        C       = target.shape[-1]
        nll     = 0.5 * (mahal + beta * ld + C * math.log(2.0 * math.pi))
        # Normalize by C so lambda_nll is scale-invariant across channel counts.
        # Without this, NLL grows linearly with C, causing explosion on ECL/Traffic.
        nll     = nll / C
        return nll.clamp(max=1e4).mean()

    # ----------------------------------------------------------
    # Convenience
    # ----------------------------------------------------------

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
