# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This file contains code adapted from Meta's Byte Latent Transformer
# implementation (https://github.com/facebookresearch/blt).
#
# Modifications and extensions:
#   - Gumbel-sigmoid straight-through patch boundary estimation
#   - Annealed temperature schedule
#   - Soft patch assignment matrix construction
#
# Copyright (c) 2026 Sachith Abeywickrama
#
# Licensed under the same license as the original Meta code.

"""
GumbelSigmoidPatcher

Differentiable patch boundary decision via Gumbel-sigmoid + straight-through
estimator.  During training, boundaries are sampled with Gumbel noise and a
temperature tau that anneals from tau_init to tau_min over tau_decay_steps
gradient steps.  During inference, boundaries are the hard argmax (sigmoid > 0.5).

The output `b` carries straight-through gradients so that the boundary scorer
(LearnedBoundaryScorer) receives meaningful gradient signal from the
forecasting loss.

Helper: boundary_to_assignment(b, K_max)
  Converts the binary boundary vector b into a hard assignment matrix
  A (B, T, K_max) that maps each timestep to its patch index, together with
  a gradient-carrying soft mask A_grad (B, T, K_max) for use in the
  differentiable pooling step.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelSigmoidPatcher(nn.Module):
    """
    Gumbel-sigmoid patch boundary sampler with temperature annealing.

    Args:
        tau_init:        initial temperature (high = softer boundaries)
        tau_min:         minimum temperature (training converges here)
        tau_decay_steps: number of steps to decay from tau_init to tau_min
    """

    def __init__(
        self,
        tau_init: float = 5.0,
        tau_min: float = 0.5,
        tau_decay_steps: int = 5000,
    ):
        super().__init__()
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.tau_decay_steps = tau_decay_steps
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    def current_tau(self) -> float:
        decay = math.exp(-self.step.item() / max(self.tau_decay_steps, 1))
        return max(self.tau_min, self.tau_init * decay)

    def forward(
        self,
        logits: torch.Tensor,   # (B, L) boundary logits from scorer
        training: bool,
    ) -> tuple[torch.Tensor, float]:
        """
        Returns (b, tau).
          b: (B, L) boundary decisions.
             Hard {0,1} values in the forward pass.
             Carries soft gradient in the backward pass (straight-through).
          b[:, 0] is always forced to 1 (first timestep starts a patch).
        """
        tau = self.current_tau()

        if training:
            # Logistic noise (Gumbel-sigmoid reparameterisation)
            u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
            g = torch.log(u) - torch.log1p(-u)          # standard logistic noise
            soft = torch.sigmoid((logits + g) / tau)    # (B, L)
            hard = (soft > 0.5).float()
            # Straight-through: forward = hard, backward = soft gradient
            b = hard + (soft - soft.detach())
            self.step += 1
        else:
            b = (torch.sigmoid(logits) > 0.5).float()

        # First position always starts a patch
        b = b.clone()
        b[:, 0] = 1.0

        return b, tau


# ============================================================
# Assignment matrix helpers
# ============================================================

def boundary_to_assignment(
    b: torch.Tensor,
    K_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert binary boundary vector b to a hard patch assignment matrix.

    The patch index of timestep t = cumsum(b)[t] - 1  (0-indexed).
    Boundary positions start new patches; non-boundary positions continue
    the current patch.

    Args:
        b:     (B, T) boundary decisions (float {0,1} or soft via STE).
               Uses b.detach() > 0.5 for the hard cumsum (forward pass).
        K_max: maximum number of patches (pad to this size).

    Returns:
        A      (B, T, K_max)  hard one-hot assignment (float), no gradient.
        A_grad (B, T, K_max)  gradient-carrying version:
                              A * (1 + (b - b.detach())).unsqueeze(-1)
                              Forward: identical to A.
                              Backward: gradient flows to b via STE chain.
    """
    B, T    = b.shape
    b_hard  = (b.detach() > 0.5).float()              # (B, T), hard, no grad
    patch_idx = b_hard.cumsum(dim=1).long() - 1        # (B, T), 0-indexed patch id
    patch_idx = patch_idx.clamp(0, K_max - 1)

    A = F.one_hot(patch_idx, num_classes=K_max).float()  # (B, T, K_max), no grad

    # Gradient path: scale A by (1 + (b - b.detach())) so that d(A_grad)/d(logit)
    # is non-zero even though A itself is piecewise constant.
    # Forward: (1 + (b - b.detach())) == 1  (since b.detach() == b in fwd).
    # Backward: chain rule reaches LearnedBoundaryScorer through b.
    A_grad = A * (1.0 + (b - b.detach()).unsqueeze(-1))   # (B, T, K_max)

    return A, A_grad


def soft_pool(
    token_emb: torch.Tensor,   # (B, T, d)
    A_grad: torch.Tensor,      # (B, T, K_max)
) -> torch.Tensor:
    """
    Differentiable average-pooling of token embeddings into patch embeddings.
    Uses A_grad so gradients propagate back to boundary logits.

    Returns patch_emb (B, K_max, d).
    """
    # Weighted sum: (B, K_max, d)
    patch_emb = torch.einsum('btk,btd->bkd', A_grad, token_emb)
    # Normalise by sum of weights per patch
    weights   = A_grad.sum(dim=1).clamp(min=1e-8)         # (B, K_max)
    patch_emb = patch_emb / weights.unsqueeze(-1)          # (B, K_max, d)
    return patch_emb
