"""
SimplePatchPooler

Aggregate variable-length patches of rich tokens into fixed-size patch
embeddings using weighted mean and max pooling. Gradient flows to the
boundary scorer through A_soft (the gradient-carrying assignment matrix
from GumbelSigmoidPatcher.boundary_to_assignment).

This replaces the AdaptivePatchEncoder (APE) cross-attention from the
old EntroPE_v2. By the time we pool, each token already carries a full
per-timestep description (x, mu, vech_L, r, cmi), so we only need a
simple aggregation, not another attention mechanism.
"""

import torch
import torch.nn as nn


class SimplePatchPooler(nn.Module):
    """
    Aggregate rich tokens within each patch via weighted mean + soft-max pool.

    Args:
        token_dim: dimension of input token embeddings
        d_model:   output patch embedding dimension

    Inputs:
        tokens:  (B, T, token_dim)  — projected rich token embeddings
        A_soft:  (B, T, K_max)      — soft assignment matrix with gradient pathway
                                       (A_hard * STE scaling from boundary_to_assignment)

    Output:
        patch_embeds: (B, K_max, d_model)
    """

    def __init__(self, token_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(2 * token_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,   # (B, T, token_dim)
        A_soft: torch.Tensor,   # (B, T, K_max)
    ) -> torch.Tensor:
        # Normalize A_soft along time so each patch's weights sum to 1
        weights = A_soft / (A_soft.sum(dim=1, keepdim=True) + 1e-6)   # (B, T, K_max)

        # Weighted mean per patch: gradient flows through A_soft → b → scorer
        mean_pool = torch.einsum("btd,btk->bkd", tokens, weights)      # (B, K_max, token_dim)

        # Soft max-pool: max over T for each (patch, feature) slot
        # Use A_soft as a mask: zero-out contributions from other patches
        scores = tokens.unsqueeze(2) * A_soft.unsqueeze(-1)            # (B, T, K_max, token_dim)
        mask   = A_soft.unsqueeze(-1) < 1e-6                           # (B, T, K_max, 1)
        scores = scores.masked_fill(mask, -1e9)
        max_pool = scores.max(dim=1).values                            # (B, K_max, token_dim)
        # Replace sentinel -1e9 values in truly empty patch slots with 0
        max_pool = max_pool.masked_fill(max_pool < -1e8, 0.0)

        combined = torch.cat([mean_pool, max_pool], dim=-1)            # (B, K_max, 2*token_dim)
        return self.norm(self.proj(combined))                           # (B, K_max, d_model)
