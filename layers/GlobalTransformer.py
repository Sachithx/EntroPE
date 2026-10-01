# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This file contains code adapted from Meta's Byte Latent Transformer
# implementation (https://github.com/facebookresearch/blt).
#
# Modifications and extensions:
#   - Entropy-guided dynamic patching
#   - Adaptive patch encoder
#   - Time-series specific modeling components
#   - Percentile based thresholding for entropy computation
#
# Copyright (c) 2026 Sachith Abeywickrama
#
# Licensed under the same license as the original Meta code.


"""
GlobalTransformer: Processes global patch-level representations.

This module applies transformer layers to patch-level embeddings to capture
global dependencies across patches in the sequence.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from layers.Args import BaseTransformerArgs
from layers.BaseTransformer import BaseTransformer


class GlobalTransformer(BaseTransformer):
    """
    Global transformer for patch-level processing.
    
    Processes patch embeddings through transformer layers to capture global
    context. Optionally projects input embeddings to the model dimension.
    
    Args:
        args: Configuration arguments for the transformer
    """
    
    def __init__(self, args: BaseTransformerArgs):
        super().__init__(args)
        
        self.dropout = args.dropout
        self.eos_id = args.eos_id
        self.dim_token_emb = args.dim_token_emb
        
        # Optional projection from embedding dimension to model dimension
        self.token_embedding_projection = self._init_embedding_projection(args)
    
    def _init_embedding_projection(self, args: BaseTransformerArgs) -> Optional[nn.Linear]:
        """
        Initialize embedding projection layer if dimensions don't match.
        
        Returns:
            Linear projection layer or None if dimensions match
        """
        if args.dim_token_emb is not None and args.dim_token_emb != self.dim:
            return nn.Linear(args.dim_token_emb, args.dim, bias=False)
        return None
    
    def forward(
        self,
        tokens: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union[torch.Tensor, str]] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        """
        Forward pass through global transformer.
        
        Args:
            tokens: Token IDs for shape reference (batch_size, seq_len)
            tok_idx: Optional token indices
            embeds: Patch embeddings (batch_size, num_patches, dim)
            mask: Attention mask (optional)
            cache: Optional KV cache for inference
            
        Returns:
            Tuple of (transformed_embeddings, cache)
        """
        bs, seqlen = tokens.shape
        h = embeds
        
        # Project embeddings to model dimension if needed
        if self.token_embedding_projection is not None and h.shape[-1] != self.dim:
            h = self.token_embedding_projection(h)
        
        # Apply dropout
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Process through transformer layers
        h = super().forward(h, tok_idx=tok_idx, mask=mask, attn_impl=self.attn_impl)
        
        return h, cache
    