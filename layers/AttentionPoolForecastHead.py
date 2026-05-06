"""
AttentionPoolForecastHead

Single learned-query attention over K_max patch embeddings to produce a
single context vector, then linear projection to (pred_len, n_channels).

This replaces the FusionDecoder + linear head chain from the old EntroPE_v2.
The FusionDecoder was designed for byte-level autoregressive generation (BLT);
for forecasting we only need a single-pass aggregation from patches to future.
"""

import math

import torch
import torch.nn as nn


class AttentionPoolForecastHead(nn.Module):
    """
    Reduce K_max patch embeddings to one context vector via learned-query
    attention, then project to (pred_len * n_channels).

    Args:
        d_model:    patch embedding dimension
        pred_len:   number of future steps to predict
        n_channels: number of time series variables
        n_heads:    attention heads (must divide d_model)
        dropout:    attention dropout

    Inputs:
        patch_embeds: (B, K_max, d_model)
        patch_mask:   (B, K_max) — True for real patches, False for padding

    Output:
        forecast: (B, pred_len, n_channels)
    """

    def __init__(
        self,
        d_model: int,
        pred_len: int,
        n_channels: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pred_len   = pred_len
        self.n_channels = n_channels

        # Learned query: single vector aggregates all patch information
        self.query = nn.Parameter(torch.randn(1, 1, d_model) / math.sqrt(d_model))

        self.attn = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_len * n_channels)

    def forward(
        self,
        patch_embeds: torch.Tensor,   # (B, K_max, d_model)
        patch_mask:   torch.Tensor,   # (B, K_max) bool — True = real patch
    ) -> torch.Tensor:
        B = patch_embeds.shape[0]
        Q = self.query.expand(B, -1, -1)   # (B, 1, d_model)

        # key_padding_mask: True = IGNORE that key position (PyTorch convention)
        key_padding_mask = ~patch_mask     # (B, K_max): True for padding

        out, _ = self.attn(
            query           = Q,
            key             = patch_embeds,
            value           = patch_embeds,
            key_padding_mask = key_padding_mask,
            need_weights    = False,
        )                                  # (B, 1, d_model)

        out = self.norm(out.squeeze(1))    # (B, d_model)
        forecast_flat = self.head(out)     # (B, pred_len * n_channels)
        return forecast_flat.view(B, self.pred_len, self.n_channels)
