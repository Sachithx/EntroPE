# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This file contains code adapted from Meta's Byte Latent Transformer
# implementation (https://github.com/facebookresearch/blt).
#
# Modifications and extensions:
#   - End-to-end continuous entropy model (no tokenisation)
#   - Learned boundary scorer + Gumbel-sigmoid straight-through patching
#   - Rich token representation (raw + MVG distribution params)
#   - Shared multivariate patch boundaries across channels
#   - Simplified pooling architecture (no APE cross-attention or FusionDecoder)
#
# Copyright (c) 2026 Sachith Abeywickrama
#
# Licensed under the same license as the original Meta code.

"""
EntroPE_v2  (simplified architecture)

Architecture: intra-patch pool → inter-patch transformer → forecast

  RevIN  →  MVG  →  BoundaryScorer  →  GumbelSigmoid
        →  RichTokenProj  →  SimplePatchPooler
        →  GlobalTransformer (standard nn.TransformerEncoder)
        →  AttentionPoolForecastHead
        →  RevIN denorm

The APE cross-attention and FusionDecoder from the original BLT design are
replaced with a simple mean+max pool (SimplePatchPooler) and a single
learned-query attention head (AttentionPoolForecastHead). This is much more
stable on small time-series datasets.

Input:  x  (B, seq_len, n_vars)   — raw float values (data-loader format)
Output: y  (B, pred_len, n_vars)
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RevIN import RevIN
from layers.GumbelSigmoidPatcher import GumbelSigmoidPatcher, boundary_to_assignment
from layers.LearnedBoundaryScorer import LearnedBoundaryScorer
from layers.SimplePatchPooler import SimplePatchPooler
from layers.AttentionPoolForecastHead import AttentionPoolForecastHead
from layers.AdaptivePatchPooler import AdaptivePatchPooler
from layers.FusionForecastHead import FusionForecastHead
from models.MultivariateGaussianModel import MultivariateGaussianModel, MVGConfig


# ============================================================
# Diagnostic 4: flatten-and-project forecast head
# ============================================================

class SimpleFlattenHead(nn.Module):
    """
    PatchTST-style flatten-and-project head.
    Pads or truncates K to k_max_estimate, flattens, then projects.
    Much more expressive than single-query attention for small datasets.
    """

    def __init__(self, d_model: int, k_max_estimate: int, pred_len: int, n_channels: int):
        super().__init__()
        self.pred_len       = pred_len
        self.n_channels     = n_channels
        self.k_max_estimate = k_max_estimate
        self.head = nn.Linear(k_max_estimate * d_model, pred_len * n_channels)

    def forward(self, patch_embeds: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        B, K, D = patch_embeds.shape
        # Zero out padded patch slots before flattening
        h = patch_embeds * patch_mask.unsqueeze(-1).float()
        if K < self.k_max_estimate:
            pad = torch.zeros(B, self.k_max_estimate - K, D, device=h.device)
            h = torch.cat([h, pad], dim=1)
        else:
            h = h[:, :self.k_max_estimate]
        return self.head(h.reshape(B, -1)).view(B, self.pred_len, self.n_channels)


# ============================================================
# Covariance feature dimension helper
# ============================================================

def _cov_feat_dim(n_channels: int, cov_rank: Optional[int]) -> int:
    if cov_rank is None:
        return n_channels * (n_channels + 1) // 2
    return n_channels + n_channels * cov_rank


# ============================================================
# EntroPE_v2 backbone
# ============================================================

class EntroPE_v2_backbone(nn.Module):
    """
    EntroPE_v2 backbone with simplified patch architecture.

    Flow:
        RevIN → MVG → scorer → patcher → assignment
        → rich token projection → SimplePatchPooler
        → GlobalTransformer → AttentionPoolForecastHead
        → RevIN denorm
    """

    def __init__(self, configs, revin=True, affine=True, subtract_last=False):
        super().__init__()
        self.configs = configs

        C        = configs.enc_in
        d_model  = configs.d_model
        seq_len  = configs.seq_len
        pred_len = configs.pred_len
        dropout  = getattr(configs, 'dropout', 0.1)
        n_heads  = configs.n_heads

        # ---- RevIN ----
        self.revin = revin
        if revin:
            self.revin_layer = RevIN(C, affine=affine, subtract_last=subtract_last)

        # ---- MVG model ----
        cov_rank = getattr(configs, 'mvg_cov_rank', None)
        self.mvg = MultivariateGaussianModel(MVGConfig(
            n_channels = C,
            block_size = seq_len,
            n_layer    = getattr(configs, 'mvg_layers', 2),
            n_head     = getattr(configs, 'mvg_heads', 4),
            n_embd     = getattr(configs, 'mvg_embd', 64),
            dropout    = getattr(configs, 'mvg_dropout', 0.1),
            cov_rank   = cov_rank,
        ))

        # ---- Boundary scorer ----
        cov_fdim = _cov_feat_dim(C, cov_rank)
        self.scorer = LearnedBoundaryScorer(
            n_channels   = C,
            cov_feat_dim = cov_fdim,
            hidden       = getattr(configs, 'scorer_hidden', 64),
        )

        # ---- Gumbel-sigmoid patcher ----
        self.patcher = GumbelSigmoidPatcher(
            tau_init        = getattr(configs, 'gumbel_tau_init', 5.0),
            tau_min         = getattr(configs, 'gumbel_tau_min', 0.5),
            tau_decay_steps = getattr(configs, 'gumbel_tau_decay', 5000),
        )

        # ---- Diagnostic / architecture flags ----
        self.no_rich_tokens = getattr(configs, 'no_rich_tokens', False)
        self.use_flat_head  = getattr(configs, 'flat_head', False)
        self.use_ape_pooler = getattr(configs, 'ape_pooler', False)
        self.use_fusion_head = getattr(configs, 'fusion_head', False)

        # ---- Rich token projection ----
        if self.no_rich_tokens:
            rich_dim = C
        else:
            rich_dim = C + C + cov_fdim + C + 1   # x + mu + vech_L + r + cmi
        self.token_proj = nn.Linear(rich_dim, d_model)

        # ---- Positional embedding for tokens ----
        self.pos_emb = nn.Embedding(seq_len, d_model)

        # ---- Local causal transformer (shared, applied before pooler AND head) ----
        n_local = getattr(configs, 'local_layers', 0)
        if n_local > 0:
            local_layer = nn.TransformerEncoderLayer(
                d_model         = d_model,
                nhead           = n_heads,
                dim_feedforward = d_model * 4,
                dropout         = dropout,
                batch_first     = True,
                norm_first      = True,
            )
            self.local_transformer = nn.TransformerEncoder(local_layer, n_local)
        else:
            self.local_transformer = None

        # ---- Intra-patch aggregation ----
        if self.use_ape_pooler:
            # APE cross-attention: patch queries attend to full token stream
            n_local = getattr(configs, 'ape_local_layers', 0)
            self.patch_pooler = AdaptivePatchPooler(
                d_model        = d_model,
                n_heads        = n_heads,
                n_local_layers = n_local,
                dropout        = dropout,
            )
        else:
            # Simple mean+max pool (fast, less expressive)
            self.patch_pooler = SimplePatchPooler(token_dim=d_model, d_model=d_model)

        # ---- Inter-patch transformer (standard encoder) ----
        n_global_layers = getattr(configs, 'global_layers',
                                  getattr(configs, 'e_layers', 3))
        global_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,
        )
        self.global_transformer = nn.TransformerEncoder(
            encoder_layer = global_layer,
            num_layers    = n_global_layers,
        )

        # ---- Forecast head ----
        if self.use_fusion_head:
            # FusionDecoder-style: token queries attend to global patches,
            # then compress-flatten-project to forecast (most expressive)
            token_dim = getattr(configs, 'fusion_token_dim', None)
            self.forecast_head = FusionForecastHead(
                d_model    = d_model,
                n_heads    = n_heads,
                seq_len    = seq_len,
                pred_len   = pred_len,
                n_channels = C,
                token_dim  = token_dim,
                dropout    = dropout,
            )
        elif self.use_flat_head:
            k_max_est = getattr(configs, 'k_max_estimate', 20)
            self.forecast_head = SimpleFlattenHead(
                d_model        = d_model,
                k_max_estimate = k_max_est,
                pred_len       = pred_len,
                n_channels     = C,
            )
        else:
            self.forecast_head = AttentionPoolForecastHead(
                d_model    = d_model,
                pred_len   = pred_len,
                n_channels = C,
                n_heads    = n_heads,
                dropout    = dropout,
            )

        # ---- Auxiliary loss hyper-params ----
        self.target_rate   = 1.0 / max(getattr(configs, 'target_avg_patch_len', 8), 1)
        self.min_patch_len = getattr(configs, 'min_patch_len', 3)
        self.lambda_count  = getattr(configs, 'lambda_count', 1.0)
        self.lambda_min    = getattr(configs, 'lambda_min', 0.5)
        self.lambda_div    = getattr(configs, 'lambda_div', 0.1)
        self.lambda_nll    = getattr(configs, 'lambda_nll', 0.1)
        self.beta_nll      = getattr(configs, 'beta_nll', 0.5)

        # Cache for auxiliary loss and monitoring
        self._last_b      = None
        self._last_logits = None
        self._last_mvg    = None   # (x_norm, mu, cov)

    # ----------------------------------------------------------
    # Load pretrained MVG checkpoint
    # ----------------------------------------------------------

    def load_mvg_checkpoint(self, path: str):
        state = torch.load(path, map_location='cpu', weights_only=True)
        self.mvg.load_state_dict(state)
        print(f"[EntroPE_v2] Loaded MVG checkpoint: {path}")

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, n_vars)  — data-loader format (time-first)
        Returns: (B, pred_len, n_vars)
        """
        B, L, C = x.shape

        # 1. RevIN normalisation (instance norm per channel)
        if self.revin:
            x = self.revin_layer(x, 'norm')     # (B, L, C)

        # 2. MVG: joint multivariate density estimation
        mu, cov = self.mvg(x)                   # (B,L,C), cov

        # 3. Distribution statistics for boundary scoring and rich tokens
        log_det = MultivariateGaussianModel.log_det(cov)                    # (B, L)
        cmi     = MultivariateGaussianModel.cmi_signal(log_det)             # (B, L)
        r       = MultivariateGaussianModel.mahalanobis_residual(x, mu, cov)  # (B, L, C)
        vech_L  = MultivariateGaussianModel.flatten_cov(cov)                # (B, L, F)

        # 4 & 5. Boundary decisions:
        #   static_patch_len > 0  → fixed-stride patches (Diagnostic 1)
        #   cmi_threshold > 0     → CMI threshold rule on MVG signal (THE FIX)
        #   else                  → learned Gumbel-sigmoid patcher
        static_p      = getattr(self.configs, 'static_patch_len', 0)
        cmi_threshold = getattr(self.configs, 'cmi_threshold', 0.0)
        if static_p > 0:
            b = torch.zeros(B, L, device=x.device)
            b[:, ::static_p] = 1.0
            b[:, 0] = 1.0
            logits = b
            tau = 0.0
        elif cmi_threshold > 0.0:
            # Top-K CMI patching: place exactly K = floor(L × cmi_threshold) boundaries
            # at the timesteps with the highest distributional shift (CMI spikes).
            #
            # cmi_threshold is interpreted as a TARGET BOUNDARY RATE (0 < rate < 1).
            # e.g., 0.125 → 1 boundary per 8 timesteps (avg patch length 8).
            #
            # This is scale-invariant: K is fixed regardless of how the absolute CMI
            # values drift during training as the MVG becomes more discriminative.
            # A fixed absolute threshold is NOT stable (tested: patches grew 30→54).
            K_target = max(1, int(L * cmi_threshold))
            with torch.no_grad():
                # cmi[:, 0] = 0 by construction; topk draws from t=1..L-1
                _, topk_idx = cmi.topk(min(K_target, L - 1), dim=1)
                b = torch.zeros(B, L, device=x.device)
                b.scatter_(1, topk_idx, 1.0)
                b[:, 0] = 1.0   # always start a patch at t=0
            logits = cmi
            tau = 0.0
        else:
            logits = self.scorer(x, mu, vech_L, cmi, r)                    # (B, L)
            b, tau = self.patcher(logits, self.training)                    # (B, L)

        # 6. Patch assignment matrix
        #    K_max: no more patches than L // min_patch_len (prevents degenerate splits)
        K_b   = b.detach().sum(dim=1).long()
        K_max = min(int(K_b.max().item()), L // max(self.min_patch_len, 1))
        K_max = max(K_max, 1)
        A, A_grad = boundary_to_assignment(b, K_max)                       # (B, L, K_max)

        # 7. Patch validity mask: True = real patch (not padding)
        K_b        = K_b.clamp(max=K_max)
        patch_mask = (
            torch.arange(K_max, device=x.device).unsqueeze(0) < K_b.unsqueeze(1)
        )                                                                    # (B, K_max)

        # 8. Rich token features (Diagnostic 2: no_rich_tokens uses only raw x)
        if self.no_rich_tokens:
            token_emb = self.token_proj(x)                                     # (B, L, d_model)
        else:
            rich      = torch.cat([x, mu, vech_L, r, cmi.unsqueeze(-1)], dim=-1)
            token_emb = self.token_proj(rich)                                  # (B, L, d_model)

        # 8b. Positional encoding
        pos_ids   = torch.arange(L, device=x.device).unsqueeze(0)             # (1, L)
        token_emb = token_emb + self.pos_emb(pos_ids)                         # (B, L, d_model)

        # 8c. Local causal transformer (enriches tokens before pooler and head)
        if self.local_transformer is not None:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                L, device=x.device, dtype=token_emb.dtype
            )
            token_emb = self.local_transformer(
                token_emb, mask=causal_mask, is_causal=True
            )                                                                  # (B, L, d_model)

        # 9. Intra-patch aggregation
        if self.use_ape_pooler:
            # APE cross-attention uses normalised soft assignment (A_grad)
            patch_emb = self.patch_pooler(token_emb, A_grad)               # (B, K_max, d_model)
        else:
            patch_emb = self.patch_pooler(token_emb, A_grad)               # (B, K_max, d_model)

        # 10. Inter-patch transformer
        patch_emb = self.global_transformer(
            patch_emb,
            src_key_padding_mask = ~patch_mask,
        )                                                                    # (B, K_max, d_model)

        # 11. Forecast head
        if self.use_fusion_head:
            # FusionForecastHead: tokens attend to global patches, then flatten
            forecast = self.forecast_head(token_emb, patch_emb, patch_mask)
        else:
            forecast = self.forecast_head(patch_emb, patch_mask)           # (B, pred_len, C)

        # 12. RevIN denormalisation
        if self.revin:
            forecast = self.revin_layer(forecast, 'denorm')

        # Cache for auxiliary loss computation and Diagnostic 3 monitoring
        self._last_b      = b
        self._last_logits = logits
        self._last_mvg    = (x, mu, cov, cmi, log_det)

        return forecast   # (B, pred_len, n_vars)

    # ----------------------------------------------------------
    # Auxiliary losses
    # ----------------------------------------------------------

    def auxiliary_losses(self) -> torch.Tensor:
        """
        Boundary and MVG regularisation:
          rate_loss — push avg boundary rate toward target_rate
          min_loss  — penalise patches shorter than min_patch_len
          div_loss  — prevent boundary collapse (maximise diversity)
          nll_loss  — keep MVG calibrated during joint training
        """
        b   = self._last_b
        L   = b.shape[1]

        # Patch-rate regulariser
        rate_loss = ((b.mean(dim=1) - self.target_rate) ** 2).mean()

        # Min-length regulariser: penalise >1 boundary in any min_patch_len window
        min_p = self.min_patch_len
        if min_p > 1 and L > min_p:
            kernel     = b.new_ones(1, 1, min_p)
            local_count = F.conv1d(b.unsqueeze(1), kernel, padding=0).squeeze(1)
            min_loss   = F.relu(local_count - 1.0).pow(2).mean()
        else:
            min_loss = b.new_tensor(0.0)

        # Diversity regulariser (maximise boundary entropy)
        p_mean   = b.mean().clamp(1e-6, 1.0 - 1e-6)
        div_loss = -(p_mean * p_mean.log() + (1 - p_mean) * (1 - p_mean).log())

        # MVG NLL calibration (causal next-step prediction)
        x, mu, cov, *_ = self._last_mvg
        if x.shape[1] > 1:
            cov_prev = (
                cov[:, :-1] if isinstance(cov, torch.Tensor)
                else (cov[0][:, :-1], cov[1][:, :-1])
            )
            nll_loss = MultivariateGaussianModel.mvg_nll_loss(
                x[:, 1:], mu[:, :-1], cov_prev, beta=self.beta_nll,
            )
        else:
            nll_loss = b.new_tensor(0.0)

        return (
              self.lambda_count * rate_loss
            + self.lambda_min   * min_loss
            - self.lambda_div   * div_loss   # negate: maximise diversity
            + self.lambda_nll   * nll_loss
        )

    # ----------------------------------------------------------
    # Monitoring
    # ----------------------------------------------------------

    def monitoring_dict(self) -> dict:
        b = self._last_b
        if b is None:
            return {}
        p   = b.mean()
        ent = -(p * p.clamp(1e-9).log() + (1 - p) * (1 - p).clamp(1e-9).log())
        d = {
            'avg_patches':      b.sum(dim=1).mean().item(),
            'boundary_rate':    p.item(),
            'boundary_entropy': ent.item(),
            'gumbel_tau':       self.patcher.current_tau(),
        }

        # Diagnostic 3: MVG variance statistics
        if self._last_mvg is not None and len(self._last_mvg) == 5:
            _, _, cov, cmi, log_det = self._last_mvg
            with torch.no_grad():
                if isinstance(cov, tuple):
                    D, _ = cov                              # low-rank: D is the diagonal
                    log_diag = D.log()
                else:
                    diag     = cov.diagonal(dim1=-2, dim2=-1)   # (B, L, C)
                    log_diag = diag.log()

                d['mvg_log_diag_min']  = log_diag.min().item()
                d['mvg_log_diag_mean'] = log_diag.mean().item()
                d['mvg_log_diag_max']  = log_diag.max().item()
                d['mvg_log_det_min']   = log_det.min().item()
                d['mvg_log_det_mean']  = log_det.mean().item()
                d['mvg_log_det_max']   = log_det.max().item()
                d['mvg_cmi_min']       = cmi.min().item()
                d['mvg_cmi_mean']      = cmi.mean().item()
                d['mvg_cmi_max']       = cmi.max().item()
        return d


# ============================================================
# Model wrapper (mirrors models/EntroPE.py interface)
# ============================================================

class Model(nn.Module):
    """
    EntroPE_v2 model wrapper.
    Exposes the same interface as models/EntroPE.py so exp_main.py
    can instantiate it via model_dict['EntroPE_v2'].

    Input/output: (B, seq_len, n_vars) → (B, pred_len, n_vars)
    """

    def __init__(self, configs, **kwargs):
        super().__init__()
        self.model = EntroPE_v2_backbone(
            configs       = configs,
            revin         = bool(getattr(configs, 'revin', 1)),
            affine        = bool(getattr(configs, 'affine', 1)),
            subtract_last = bool(getattr(configs, 'subtract_last', 0)),
        )

        # Optionally load pretrained MVG checkpoint
        mvg_ckpt = getattr(configs, 'mvg_checkpoint', None)
        if mvg_ckpt and os.path.isfile(mvg_ckpt):
            self.model.load_mvg_checkpoint(mvg_ckpt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, n_vars) → (B, pred_len, n_vars)"""
        return self.model(x)

    def auxiliary_losses(self) -> torch.Tensor:
        return self.model.auxiliary_losses()

    def monitoring_dict(self) -> dict:
        return self.model.monitoring_dict()
