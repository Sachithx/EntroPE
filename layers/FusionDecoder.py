"""
FusionDecoder: Fuses global patch representations with local token representations.

Decodes token-level features while incorporating global patch context via cross-attention.
"""

from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from layers.Args import LocalModelBase, LocalModelArgs
from layers.BaseTransformer import CrossAttention


class FusionDecoder(LocalModelBase):
    """
    Fusion Decoder with cross-attention over global patch embeddings.

    Token embeddings are refined through transformer layers while attending to
    patch-level latent representations for global context integration.
    """

    def __init__(self, args: LocalModelArgs):
        super().__init__(args)

        # ----------------------------------
        # Cross-Attention Configuration
        # ----------------------------------
        self.cross_attn_decoder = args.cross_attn_decoder
        self.cross_attn_all_layers_decoder = args.cross_attn_all_layers_decoder
        self.cross_attn_init_by_pooling = args.cross_attn_init_by_pooling
        self.cross_attn_nheads = args.cross_attn_nheads

        # ----------------------------------
        # Output Normalization
        # ----------------------------------
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

        # ----------------------------------
        # Cross-Attention Layers (if enabled)
        # ----------------------------------
        if self.cross_attn_decoder:
            self._init_cross_attention_layers(args)

    # ----------------------------------
    # Cross-Attention Layer Builder
    # ----------------------------------
    def _init_cross_attention_layers(self, args: LocalModelArgs):
        """Initialize cross-attention layers."""
        self.cross_attn_layers = nn.ModuleList()

        num_layers = (
            args.n_layers if self.cross_attn_all_layers_decoder else 1
        )

        for _ in range(num_layers):
            self.cross_attn_layers.append(
                CrossAttention(
                    dim=self.dim,
                    head_dim=self.dim // self.cross_attn_nheads,
                    n_heads=self.cross_attn_nheads,
                    n_kv_heads=self.cross_attn_nheads,
                    norm_eps=args.norm_eps,
                )
            )

    # ----------------------------------
    # Forward Pass
    # ----------------------------------
    def forward(
        self,
        tokens: torch.Tensor,
        embeds: Optional[torch.Tensor],
        patch_embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union[torch.Tensor, str]] = None,
        cross_mask: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        """
        Forward pass for fusion decoding.

        Args:
            tokens: Token IDs (batch_size, seq_len) — used only for shape
            embeds: Token embeddings from encoder (batch_size, seq_len, dim)
            patch_embeds: Patch-level global embeddings (batch_size, num_patches, dim)
            mask: Self-attention mask (default = causal)
            cross_mask: Cross-attention mask (token → patch)
            cache: Optional KV cache (inference)

        Returns:
            (decoded_embeddings, cache)
        """

        bs, seqlen = tokens.shape
        assert embeds is not None, "FusionDecoder requires token embeddings"

        # Default attention mask
        if mask is None:
            mask = "causal"

        h = embeds

        # ----------------------------------
        # Rotary / Positional Encoding
        # ----------------------------------
        freqs_cis = self.rope(seqlen=seqlen) if self.use_rope else None

        # ----------------------------------
        # Input Dropout
        # ----------------------------------
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ----------------------------------
        # Transformer + Cross-Attention
        # ----------------------------------
        for i, layer in enumerate(self.layers):

            # Apply cross-attention if enabled
            if self.cross_attn_decoder and (
                self.cross_attn_all_layers_decoder or i == 0
            ):
                cross_layer_idx = i if self.cross_attn_all_layers_decoder else 0

                h_cross = self.cross_attn_layers[cross_layer_idx](
                    x=h,
                    kv=patch_embeds,
                    mask=cross_mask,
                )

                h = h + h_cross

            # Apply transformer self-attention block
            h = layer(
                h,
                mask=mask,
                freq_cis=freqs_cis,
                attn_impl=self.attn_impl,
            )

        # ----------------------------------
        # Output Projection & Normalization
        # ----------------------------------
        h = self.norm(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return h.float(), cache
