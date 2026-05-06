"""
FusionForecastHead

Clean reimplementation of FusionDecoder + FlattenHead from the original
EntroPE backbone, stripped of BLT-specific machinery.

Core mechanism (identical to old FusionDecoder → FlattenHead):
  1. Token queries attend to global patch keys/values (cross-attention).
     This injects patch-level global context into every token position.
  2. Residual update + LayerNorm.
  3. Compress each token from d_model → token_dim (parameter-efficient).
  4. Flatten all L compressed tokens → linear projection to (pred_len, C).

The compress → flatten approach keeps the flat projection matrix small:
  L × token_dim → pred_len × C
  e.g. 96 × 16 = 1536 → 96 × 7 = 672   (~1M params)

Compared to naive d_model flatten:
  96 × 64 = 6144 → 672  (~4M params — dominated by head, hard to train)

Input:
    token_emb:   (B, L, d_model)  — pre-GlobalTransformer token embeddings
    patch_emb:   (B, K, d_model)  — GlobalTransformer output patch embeddings
    patch_mask:  (B, K) bool      — True = real patch, False = padding
Output:
    forecast:    (B, pred_len, n_channels)
"""

import torch
import torch.nn as nn


class FusionForecastHead(nn.Module):

    def __init__(
        self,
        d_model:    int,
        n_heads:    int,
        seq_len:    int,
        pred_len:   int,
        n_channels: int,
        token_dim:  int | None = None,   # compress dim; default d_model // 4
        dropout:    float      = 0.1,
    ):
        super().__init__()
        self.pred_len   = pred_len
        self.n_channels = n_channels

        token_dim = token_dim or max(d_model // 4, 8)

        # Cross-attention: token queries attend to global patch keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm = nn.LayerNorm(d_model)

        # Compress d_model → token_dim before flattening
        self.compress = nn.Linear(d_model, token_dim)

        # Flat projection: L * token_dim → pred_len * n_channels
        self.head = nn.Linear(seq_len * token_dim, pred_len * n_channels)

    def forward(
        self,
        token_emb:  torch.Tensor,   # (B, L, d_model)
        patch_emb:  torch.Tensor,   # (B, K, d_model)
        patch_mask: torch.Tensor,   # (B, K) bool — True = real
    ) -> torch.Tensor:              # (B, pred_len, n_channels)

        B = token_emb.shape[0]

        # Cross-attention: tokens attend to global patches
        key_padding_mask = ~patch_mask      # True = ignore (padding)
        delta, _ = self.cross_attn(
            query            = token_emb,
            key              = patch_emb,
            value            = patch_emb,
            key_padding_mask = key_padding_mask,
            need_weights     = False,
        )
        h = self.norm(token_emb + delta)    # (B, L, d_model)

        # Compress → flatten → project
        h = self.compress(h)                # (B, L, token_dim)
        return self.head(h.reshape(B, -1)).view(B, self.pred_len, self.n_channels)
