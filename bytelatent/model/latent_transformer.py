# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Global Transformer and Cross-Attention for ByteLatentTransformer.

This module implements:
- CrossAttention: Bidirectional attention mechanism for encoder-decoder interaction
- GlobalTransformer: Patch-level transformer for processing compressed representations

The global transformer operates on patch-level representations created by the
local encoder, enabling efficient processing of long sequences.
"""

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask

from bytelatent.base_transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    flex_attention_comp,
    repeat_kv,
)
from bytelatent.model.utils import create_causal_mask

logger = logging.getLogger(__name__)

# Use PyTorch's built-in RMSNorm
RMSNorm = nn.RMSNorm


# ============================================================================
# Cross-Attention Module
# ============================================================================

class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for attending between different sequence representations.
    
    This module enables bidirectional attention between query and key-value sequences,
    commonly used for encoder-decoder architectures or hierarchical representations.
    
    Features:
    - Multi-head attention with grouped query attention support
    - RMSNorm for query and key-value normalization
    - Support for multiple attention implementations (flex_attention, sdpa)
    - Residual connection
    
    Note: Rotary Position Embeddings (RoPE) are not supported in cross-attention.
    
    Args:
        dim: Model dimension
        head_dim: Dimension per attention head
        n_heads: Number of query heads
        n_kv_heads: Number of key-value heads (for grouped query attention)
        norm_eps: Epsilon for RMSNorm
    
    Example:
        >>> cross_attn = CrossAttention(dim=512, head_dim=64, n_heads=8, n_kv_heads=8, norm_eps=1e-5)
        >>> queries = torch.randn(2, 10, 512)  # (batch, seq_len_q, dim)
        >>> keys_values = torch.randn(2, 20, 512)  # (batch, seq_len_kv, dim)
        >>> output = cross_attn(queries, keys_values)
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        # Separate normalization for queries and key-values
        self.cross_attn_norm_q = RMSNorm(dim, eps=norm_eps)
        self.cross_attn_norm_kv = RMSNorm(dim, eps=norm_eps)

        # Query, Key, Value projections
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)

        # Output projection
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        mask: Optional[Union[BlockMask, torch.Tensor, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        """
        Forward pass of cross-attention.
        
        Args:
            x: Query tensor of shape (batch_size, seq_len_q, dim)
            kv: Key-value tensor of shape (batch_size, seq_len_kv, dim)
            mask: Optional attention mask (BlockMask, Tensor, or "causal")
            attn_impl: Attention implementation ("flex_attention" or "sdpa")
        
        Returns:
            Output tensor of shape (batch_size, seq_len_q, dim) with residual connection
        """
        bsz, seq_len, _ = x.shape
        _, slen_kv, _ = kv.shape

        # Normalize queries and key-values
        x_norm = self.cross_attn_norm_q(x)
        kv_norm = self.cross_attn_norm_kv(kv)

        # Project to queries, keys, values
        xq = self.wq(x_norm)
        xk = self.wk(kv_norm)
        xv = self.wv(kv_norm)

        # Reshape for multi-head attention
        output_shape = xq.shape
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, slen_kv, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, slen_kv, self.n_kv_heads, self.head_dim)

        # Repeat key-values for grouped query attention
        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        # Apply attention based on implementation
        if attn_impl == "flex_attention":
            assert mask is None or isinstance(mask, BlockMask), \
                "flex_attention requires BlockMask or None"
            assert flex_attention_comp is not None, \
                "flex_attention is not compiled or available"
            
            # Rearrange to (batch, heads, seq_len, head_dim)
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            output = flex_attention_comp(xq, xk, xv, block_mask=mask)
            output = output.transpose(1, 2).contiguous()

        elif attn_impl == "sdpa":
            # Rearrange to (batch, heads, seq_len, head_dim)
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            
            # Handle different mask types
            is_causal = isinstance(mask, str) and mask == "causal"
            mask_tensor = mask if isinstance(mask, torch.Tensor) else None
            
            # Ensure mask is float or bool for SDPA
            if isinstance(mask_tensor, torch.Tensor) and \
               mask_tensor.dtype not in [torch.float32, torch.bool]:
                mask_tensor = mask_tensor.to(torch.float32)
            
            output = F.scaled_dot_product_attention(
                xq, xk, xv, 
                is_causal=is_causal, 
                attn_mask=mask_tensor
            )
            output = output.transpose(1, 2).contiguous()

        else:
            raise NotImplementedError(f"Unsupported attn_impl: {attn_impl}")

        # Project output and add residual
        output = self.wo(output.reshape(output_shape))
        return x + output

    def init_weights(self, base_std: Optional[float] = None, factor: float = 1.0):
        """
        Initialize cross-attention weights.
        
        Uses truncated normal initialization with depth-aware scaling.
        
        Args:
            base_std: Base standard deviation (computed if None)
            factor: Scaling factor for depth-aware initialization
        """
        std = base_std or (self.dim ** (-0.5)) / factor

        # Initialize all projection matrices
        for proj in [self.wq, self.wk, self.wv, self.wo]:
            nn.init.trunc_normal_(
                proj.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )

        # Reset normalization parameters
        self.cross_attn_norm_q.reset_parameters()
        self.cross_attn_norm_kv.reset_parameters()


# ============================================================================
# Global Transformer
# ============================================================================

class GlobalTransformer(BaseTransformer):
    """
    Global transformer for patch-level sequence processing.
    
    The global transformer operates on compressed patch representations created
    by the local encoder, enabling efficient processing of long sequences while
    maintaining global context.
    
    Key features:
    - Operates on patch-level representations
    - Optional projection from patch embedding dimension to model dimension
    - Learned position embeddings
    - Causal or bidirectional attention
    - Inherits full transformer capabilities from BaseTransformer
    
    Args:
        args: Configuration arguments including dimensions, layers, attention settings
    
    Example:
        >>> args = BaseTransformerArgs(
        ...     dim=512,
        ...     n_layers=6,
        ...     n_heads=8,
        ...     dim_token_emb=256,
        ... )
        >>> global_transformer = GlobalTransformer(args)
        >>> patch_embeds = torch.randn(2, 20, 256)  # (batch, num_patches, patch_dim)
        >>> tokens = torch.randint(0, 1000, (2, 20))
        >>> output, _ = global_transformer(tokens=tokens, embeds=patch_embeds)
    """

    def __init__(self, args: BaseTransformerArgs):
        super().__init__(args)
        
        self.dropout = args.dropout
        self.eos_id = args.eos_id
        self.dim_token_emb = args.dim_token_emb

        # Optional projection from patch embedding dimension to model dimension
        self.token_embedding_projection = None
        if args.dim_token_emb is not None and args.dim_token_emb != self.dim:
            self.token_embedding_projection = nn.Linear(
                args.dim_token_emb,
                args.dim,
                bias=False,
            )
            logger.info(
                f"Global transformer using token embedding projection: "
                f"{args.dim_token_emb} -> {self.dim}"
            )

        # Learned position embeddings for patches
        self.pos_embeddings = nn.Embedding(self.max_seqlen, args.dim)

    def forward(
        self,
        tokens: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, torch.Tensor, str]] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """
        Forward pass of global transformer.
        
        Args:
            tokens: Token IDs of shape (batch_size, num_patches)
            tok_idx: Optional token indices
            embeds: Patch embeddings of shape (batch_size, num_patches, dim_token_emb)
            mask: Optional attention mask
            cache: Optional KV cache for generation
        
        Returns:
            Tuple of (output_embeddings, cache)
            - output_embeddings: (batch_size, num_patches, dim)
            - cache: Updated KV cache
        """
        assert embeds is not None, "Patch embeddings must be provided to global transformer"
        
        bs, seqlen = tokens.shape

        h = embeds

        # Add learned position embeddings
        pos_ids = torch.arange(seqlen, device=h.device).unsqueeze(0)
        pos_emb = self.pos_embeddings(pos_ids)
        h = h + pos_emb

        # Create causal mask if not provided
        if mask is None:
            mask = create_causal_mask(
                seqlen,
                self.attn_impl,
                self.attn_bias_type,
                tokens=tokens,
                eos_id=self.eos_id,
            )

        # Project to model dimension if needed
        if self.token_embedding_projection is not None and h.shape[-1] != self.dim:
            h = self.token_embedding_projection(h)

        # Apply dropout
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Process through transformer layers
        h = super().forward(h, tok_idx=tok_idx, mask=mask, attn_impl=self.attn_impl)
        
        return h, cache

    def init_weights(self):
        """
        Initialize global transformer weights.
        
        Initializes:
        - All transformer layers (via super().init_weights())
        - Position embeddings
        - Token embedding projection (if present)
        """
        # Initialize base transformer layers
        super().init_weights()
        
        # Initialize position embeddings
        std = self.dim ** (-0.5)
        nn.init.trunc_normal_(
            self.pos_embeddings.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )
        
        # Initialize token embedding projection if present
        if self.token_embedding_projection is not None:
            # Use dim_token_emb for projection std
            proj_std = self.dim_token_emb ** (-0.5)
            nn.init.trunc_normal_(
                self.token_embedding_projection.weight,
                mean=0.0,
                std=proj_std,
                a=-3 * proj_std,
                b=3 * proj_std,
            )
            