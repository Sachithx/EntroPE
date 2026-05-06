"""
AdaptivePatchPooler

Clean reimplementation of the APE (Adaptive Patch Encoder) cross-attention
from the original EntroPE backbone, stripped of all BLT-specific machinery
(special tokens, vocab tables, discrete embeddings, pydantic Args, etc.).

Core mechanism (identical to old APE):
  1. Initialise patch queries by mean-pooling their constituent token embeddings.
  2. Cross-attention: patch queries attend to ALL token keys/values.
  3. Residual update + LayerNorm.

Optionally prefixes the token stream with a lightweight causal local
transformer (n_local_layers > 0) before cross-attention, matching the
n_layers_local_encoder=1 path in the old backbone.

Input:
    token_emb  (B, L, d_model)   — projected rich-token embeddings
    A          (B, L, K)         — soft patch-assignment matrix
                                   A[b, t, k] = weight of token t for patch k
                                   (sums to ≈1 over L for each k)
Output:
    patch_emb  (B, K, d_model)
"""

import torch
import torch.nn as nn


class AdaptivePatchPooler(nn.Module):

    def __init__(
        self,
        d_model:        int,
        n_heads:        int,
        n_local_layers: int   = 0,    # 0 = no local transformer (pure cross-attn)
        dropout:        float = 0.1,
    ):
        super().__init__()

        # Optional local causal transformer applied to token stream first
        if n_local_layers > 0:
            local_layer = nn.TransformerEncoderLayer(
                d_model         = d_model,
                nhead           = n_heads,
                dim_feedforward = d_model * 4,
                dropout         = dropout,
                batch_first     = True,
                norm_first      = True,
            )
            self.local_tf = nn.TransformerEncoder(local_layer, n_local_layers)
        else:
            self.local_tf = None

        # Cross-attention: patch queries ← token keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        token_emb: torch.Tensor,   # (B, L, d_model)
        A:         torch.Tensor,   # (B, L, K)  soft assignment weights
    ) -> torch.Tensor:             # (B, K, d_model)

        # Optional local self-attention on token stream (causal)
        h = token_emb
        if self.local_tf is not None:
            # Causal mask so each token only sees prior context
            L = h.shape[1]
            causal = nn.Transformer.generate_square_subsequent_mask(
                L, device=h.device, dtype=h.dtype
            )
            h = self.local_tf(h, mask=causal, is_causal=True)

        # Initialise patch queries via normalised mean pool
        A_norm  = A / (A.sum(dim=1, keepdim=True) + 1e-6)   # (B, L, K)
        patch_q = torch.einsum('blk,bld->bkd', A_norm, h)   # (B, K, d_model)

        # Cross-attention: patch queries attend to full token sequence
        delta, _ = self.cross_attn(
            query       = patch_q,
            key         = h,
            value       = h,
            need_weights = False,
        )

        return self.norm(patch_q + delta)   # (B, K, d_model)
