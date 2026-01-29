"""
APE (Adaptive Patch Encoder) module for local encoding with cross-attention.
"""

from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

RMSNorm = nn.RMSNorm

from layers.Args import LocalModelBase, LocalModelArgs
from layers.BaseTransformer import CrossAttention
from utils.layer_utils import downsample


class PatchEncoder(LocalModelBase):
    """
    Adaptive Patch Encoder (APE)
    Performs local token encoding and optional patch-level cross-attention.
    """

    def __init__(self, args: LocalModelArgs):
        super().__init__(args)

        # ----------------------------------
        # Configuration Flags
        # ----------------------------------
        self.apply_transformer = args.use_local_encoder_transformer
        self.downsampling_by_pooling = args.downsampling_by_pooling
        self.expects_hash_embeddings = args.encoder_hash_byte_group_size is not None

        # Cross-attention config
        self.cross_attn_encoder = args.cross_attn_encoder
        self.cross_attn_all_layers_encoder = args.cross_attn_all_layers_encoder
        self.cross_attn_init_by_pooling = args.cross_attn_init_by_pooling
        self.cross_attn_nheads = args.cross_attn_nheads

        # ----------------------------------
        # Token Embedding
        # ----------------------------------
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        # ----------------------------------
        # Cross-Attention Layers (if enabled)
        # ----------------------------------
        if self.cross_attn_encoder:
            self.cross_attn_layers = nn.ModuleList()

            num_cross_layers = (
                args.n_layers if self.cross_attn_all_layers_encoder else 1
            )

            for _ in range(num_cross_layers):
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
    # Embedding
    # ----------------------------------
    def apply_embedding(self, tokens, embeds):
        return self.tok_embeddings(tokens)

    # ----------------------------------
    # Forward Pass
    # ----------------------------------
    def forward(
        self,
        tokens: torch.Tensor,
        embeds: Optional[torch.Tensor] = None,
        patch_embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union[torch.Tensor, str]] = None,
        cross_mask: Optional[torch.Tensor] = None,
        num_patches: Optional[int] = None,
        patch_ids: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        bs, seqlen = tokens.shape

        # Default to causal attention
        if mask is None:
            mask = "causal"

        # ----------------------------------
        # Token Embedding + Positional Encoding
        # ----------------------------------
        h = self.apply_embedding(tokens, embeds)

        if self.use_rope:
            freqs_cis = self.rope(seqlen=seqlen)
        else:
            device = h.device
            pos_ids = torch.arange(seqlen, device=device).unsqueeze(0)
            pos_emb = self.pos_embeddings(pos_ids)
            h = h + pos_emb
            freqs_cis = None

        h = F.dropout(h, p=self.dropout, training=self.training)

        # ----------------------------------
        # Local Transformer Layers
        # ----------------------------------
        for i, layer in enumerate(self.layers):
            h = layer(h, mask=mask, freq_cis=freqs_cis, attn_impl=self.attn_impl)

            # Apply cross-attention if enabled
            if self.cross_attn_encoder and (
                self.cross_attn_all_layers_encoder or i == len(self.layers) - 1
            ):
                patch_embeds = self.apply_cross_attention(
                    h=h,
                    patch_embeds=patch_embeds,
                    layer_idx=i,
                    bs=bs,
                    num_patches=num_patches,
                    patch_ids=patch_ids,
                    cross_mask=cross_mask,
                )

        # Residual patch output (if cross-attn enabled)
        h_residual = patch_embeds if self.cross_attn_encoder else None

        return (h, h_residual), cache

    # ----------------------------------
    # Cross-Attention Helper
    # ----------------------------------
    def apply_cross_attention(
        self,
        h,
        patch_embeds,
        layer_idx,
        bs,
        num_patches,
        patch_ids,
        cross_mask,
    ):
        # Initialize patch embeddings via pooling if needed
        if self.cross_attn_init_by_pooling and patch_embeds is None:
            patch_embeds = downsample(
                h,
                num_patches,
                patch_ids=patch_ids,
                downsampling_by_pooling=self.downsampling_by_pooling,
                patch_size=self.patch_size,
            )

            if self.patch_embedding_projection is not None:
                patch_embeds = self.patch_embedding_projection(patch_embeds)
                patch_embeds = patch_embeds.reshape(
                    bs, patch_embeds.shape[1] * self.cross_attn_k, self.dim
                )

        # Choose correct cross-attention layer
        effective_layer_idx = (
            layer_idx if self.cross_attn_all_layers_encoder else 0
        )

        patch_delta = self.cross_attn_layers[effective_layer_idx](
            x=patch_embeds,
            kv=h,
            mask=cross_mask,
        )

        # Residual update
        return patch_embeds + patch_delta
