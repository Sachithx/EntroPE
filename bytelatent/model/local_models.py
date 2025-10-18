# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Local Models for ByteLatentTransformer.

This module implements the local encoder and decoder components used in BLT:
- LocalEncoder: Processes byte-level tokens and optionally creates patch representations
- LocalDecoder: Decodes patch representations back to byte-level predictions

Both models support optional cross-attention mechanisms for hierarchical processing.
"""

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask

from bytelatent.base_transformer import (
    BaseTransformerArgs,
    InitStdFactor,
    RotaryEmbedding,
    TransformerBlock,
)
from bytelatent.model.latent_transformer import CrossAttention
from bytelatent.model.utils import create_causal_mask, downsample
from bytelatent.tokenizers.blt_tokenizer import BOE_ID

logger = logging.getLogger(__name__)

# Use PyTorch's built-in RMSNorm
RMSNorm = nn.RMSNorm


# ============================================================================
# Configuration
# ============================================================================

class LocalModelArgs(BaseTransformerArgs):
    """
    Configuration arguments for local encoder/decoder models.
    
    Extends BaseTransformerArgs with local-specific parameters for
    byte-level processing, cross-attention, and patching.
    """
    
    # Attention configuration
    attn_impl: str | None = "xformers"
    attn_bias_type: str | None = "local_block_causal"

    # Model dimensions and architecture
    dropout: float
    vocab_size: int
    patch_size: float
    sliding_window: int | None
    use_rope: bool
    max_encoder_seq_length: int
    
    # Cross-attention configuration
    cross_attn_encoder: bool | None
    cross_attn_decoder: bool | None
    cross_attn_k: int | None
    cross_attn_init_by_pooling: bool
    cross_attn_all_layers_encoder: bool = False
    cross_attn_all_layers_decoder: bool = False
    cross_attn_nheads: int | None

    # Patching configuration
    patching_mode: str
    use_local_encoder_transformer: bool
    downsampling_by_pooling: str | None
    encoder_hash_byte_group_size: object | None = None

    # Embedding dimensions
    dim_token_emb: int
    dim_patch_emb: int | None


# ============================================================================
# Base Local Model
# ============================================================================

class LocalModelBase(nn.Module):
    """
    Base class for local encoder and decoder models.
    
    Provides shared functionality including:
    - Transformer layers
    - Position embeddings (learned or RoPE)
    - Token/patch embedding projections
    - Weight initialization
    """

    def __init__(self, args: LocalModelArgs):
        super().__init__()

        # Core configuration
        self.dim = args.dim
        self.dropout = args.dropout
        self.vocab_size = args.vocab_size
        self.patch_size = args.patch_size
        self.dim_patch_emb = args.dim_patch_emb

        # Attention configuration
        self.attn_impl = args.attn_impl
        self.attn_bias_type = args.attn_bias_type
        self.sliding_window = args.sliding_window
        self.use_rope = args.use_rope
        self.max_encoder_seq_length = args.max_encoder_seq_length
        self.init_std_factor = args.init_std_factor
        
        # Cross-attention configuration
        self.cross_attn_encoder = getattr(args, "cross_attn_encoder", None)
        self.cross_attn_decoder = getattr(args, "cross_attn_decoder", None)
        self.cross_attn_k = getattr(args, "cross_attn_k", None)
        
        # Special tokens
        self.eos_id = args.eos_id
        self.boe_id = BOE_ID

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(args) for _ in range(args.n_layers)]
        )

        # Position embeddings: RoPE or learned
        if not self.use_rope:
            self.pos_embeddings = nn.Embedding(self.max_encoder_seq_length, args.dim)
            self.rope = None
        else:
            self.rope = RotaryEmbedding(
                theta=args.rope_theta,
                head_dim=args.head_dim or args.dim // args.n_heads,
                max_seqlen=args.max_seqlen,
                rope_use_fp32_in_outer_product=args.rope_use_fp32_in_outer_product,
            )
            self.pos_embeddings = None

        # Token embedding projection (if needed)
        self.token_embedding_projection = (
            nn.Linear(args.dim_token_emb, args.dim, bias=False)
            if hasattr(args, "dim_token_emb") and args.dim_token_emb != self.dim
            else None
        )

        # Patch embedding projection (currently disabled)
        self.patch_embedding_projection = None
        
        # Cross-attention layers (initialized in subclasses)
        self.cross_attn_layers = None

    def _should_create_patch_projection(self, args: LocalModelArgs) -> bool:
        """Determine if patch embedding projection is needed."""
        dimension_mismatch = (
            getattr(args, "dim_patch_emb", None) and 
            args.dim_patch_emb != self.dim
        )

        cross_attn_conditions = (
            (args.cross_attn_encoder and args.cross_attn_init_by_pooling) or
            (args.cross_attn_decoder and args.cross_attn_init_by_pooling)
        )

        return dimension_mismatch or cross_attn_conditions

    def _create_patch_projection(self, args: LocalModelArgs) -> Optional[nn.Linear]:
        """Create patch embedding projection layer if needed."""
        if not self._should_create_patch_projection(args):
            return None

        output_dim = args.dim_token_emb * (self.cross_attn_k or 1)

        return nn.Linear(
            in_features=args.dim_patch_emb,
            out_features=output_dim,
            bias=False,
        )

    def apply_embedding(self, tokens: torch.Tensor, embeds: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply token embeddings.
        
        Args:
            tokens: Token IDs
            embeds: Pre-computed embeddings (optional)
        
        Returns:
            Token embeddings
        """
        if embeds is not None:
            return embeds
        else:
            return self.tok_embeddings(tokens)

    def init_weights(self, init_std: Optional[float] = None):
        """
        Initialize model weights.
        
        Uses truncated normal initialization with depth-aware scaling for
        transformer layers and appropriate initialization for embeddings.
        
        Args:
            init_std: Standard deviation for initialization (computed if None)
        """
        # Initialize RoPE if using it
        if self.use_rope and hasattr(self, "rope") and self.rope is not None:
            self.rope.reset_parameters()
        
        # Initialize normalization layers
        if hasattr(self, "norm"):
            self.norm.reset_parameters()

        # Compute initialization standard deviation
        init_std = init_std or (self.dim ** (-0.5))
        
        # Initialize token embeddings
        if hasattr(self, "tok_embeddings"):
            nn.init.trunc_normal_(
                self.tok_embeddings.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
        
        # Initialize position embeddings
        if self.pos_embeddings is not None:
            nn.init.trunc_normal_(
                self.pos_embeddings.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        # Initialize transformer layers with depth-aware scaling
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(None, factor)

        # Initialize output projection (if exists)
        if hasattr(self, "output"):
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        # Initialize token embedding projection
        if self.token_embedding_projection is not None:
            nn.init.trunc_normal_(
                self.token_embedding_projection.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        # Initialize patch embedding projection
        if self.patch_embedding_projection is not None:
            patch_emb_std = self.dim_patch_emb ** (-0.5)
            nn.init.trunc_normal_(
                self.patch_embedding_projection.weight,
                mean=0.0,
                std=patch_emb_std,
                a=-3 * patch_emb_std,
                b=3 * patch_emb_std,
            )

        # Initialize cross-attention layers (if they exist)
        if hasattr(self, "cross_attn_layers") and self.cross_attn_layers is not None:
            for depth, layer in enumerate(self.cross_attn_layers):
                factor = {
                    InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                    InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                    InitStdFactor.DIM_RATIO: self.dim / 4096,
                    InitStdFactor.DISABLED: 1.0,
                }[self.init_std_factor]

                layer.init_weights(None, factor)


# ============================================================================
# Local Encoder
# ============================================================================

class LocalEncoder(LocalModelBase):
    """
    Local encoder for byte-level token processing.
    
    The encoder processes byte tokens through transformer layers and optionally
    creates patch-level representations using cross-attention or pooling.
    
    Features:
    - Byte-level token encoding
    - Optional transformer processing
    - Optional cross-attention to create patch representations
    - Flexible downsampling strategies
    """

    def __init__(self, args: LocalModelArgs):
        super().__init__(args)

        # Configuration
        self.apply_transformer = args.use_local_encoder_transformer
        self.downsampling_by_pooling = args.downsampling_by_pooling
        self.expects_hash_embeddings = args.encoder_hash_byte_group_size is not None
        self.cross_attn_encoder = args.cross_attn_encoder
        self.cross_attn_all_layers_encoder = args.cross_attn_all_layers_encoder
        self.cross_attn_init_by_pooling = args.cross_attn_init_by_pooling
        self.cross_attn_nheads = args.cross_attn_nheads

        # Token embeddings
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        # Cross-attention layers (if enabled)
        if self.cross_attn_encoder:
            self.cross_attn_layers = nn.ModuleList()
            layers_to_add = args.n_layers if self.cross_attn_all_layers_encoder else 1
            for _ in range(layers_to_add):
                self.cross_attn_layers.append(
                    CrossAttention(
                        dim=self.dim,
                        head_dim=self.dim // self.cross_attn_nheads,
                        n_heads=self.cross_attn_nheads,
                        n_kv_heads=self.cross_attn_nheads,
                        norm_eps=args.norm_eps,
                    )
                )

    def apply_embedding(self, tokens: torch.Tensor, embeds: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply token embeddings (always use tok_embeddings for encoder)."""
        return self.tok_embeddings(tokens)

    def forward(
        self,
        tokens: torch.Tensor,
        embeds: Optional[torch.Tensor] = None,
        patch_embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, torch.Tensor, str]] = None,
        cross_mask: Optional[torch.Tensor] = None,
        num_patches: Optional[int] = None,
        patch_ids: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ) -> Tuple[Tuple[torch.Tensor, Optional[torch.Tensor]], Optional[List]]:
        """
        Forward pass of local encoder.
        
        Args:
            tokens: Input token IDs of shape (batch_size, seq_len)
            embeds: Optional pre-computed embeddings
            patch_embeds: Optional patch embeddings for cross-attention
            mask: Attention mask
            cross_mask: Cross-attention mask
            num_patches: Number of patches
            patch_ids: Patch ID for each token position
            cache: Optional KV cache for generation
        
        Returns:
            Tuple of ((token_embeddings, patch_embeddings), cache)
        """
        bs, seqlen = tokens.shape

        # Create causal mask if not provided
        if mask is None:
            mask = create_causal_mask(
                seqlen,
                self.attn_impl,
                self.attn_bias_type,
                sliding_window=self.sliding_window,
                tokens=tokens,
                eos_id=self.eos_id,
            )

        # Apply token embeddings
        h = self.apply_embedding(tokens, embeds)

        # Add position embeddings
        if self.use_rope:
            freqs_cis = self.rope(seqlen=seqlen)
        else:
            pos_ids = torch.arange(seqlen, device=h.device).unsqueeze(0)
            pos_emb = self.pos_embeddings(pos_ids)
            h = h + pos_emb
            freqs_cis = None

        h = F.dropout(h, p=self.dropout, training=self.training)

        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            h = layer(h, mask=mask, freq_cis=freqs_cis, attn_impl=self.attn_impl)
            
            # Apply cross-attention at appropriate layers
            if self.cross_attn_encoder and (
                i == len(self.layers) - 1 or self.cross_attn_all_layers_encoder
            ):
                patch_embeds = self.apply_cross_attention(
                    h, patch_embeds, i, bs, num_patches, patch_ids, cross_mask
                )

        h_residual = patch_embeds if self.cross_attn_encoder else None
        return (h, h_residual), cache

    def apply_cross_attention(
        self,
        h: torch.Tensor,
        patch_embeds: Optional[torch.Tensor],
        layer_idx: int,
        bs: int,
        num_patches: int,
        patch_ids: torch.Tensor,
        cross_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply cross-attention to create or refine patch embeddings.
        
        Args:
            h: Token-level hidden states
            patch_embeds: Existing patch embeddings (if any)
            layer_idx: Current layer index
            bs: Batch size
            num_patches: Number of patches
            patch_ids: Patch ID for each token
            cross_mask: Cross-attention mask
        
        Returns:
            Updated patch embeddings
        """
        # Initialize patch embeddings by pooling if needed
        if self.cross_attn_init_by_pooling and patch_embeds is None:
            patch_embeds = downsample(
                h,
                num_patches,
                patch_ids=patch_ids,
                downsampling_by_pooling=self.downsampling_by_pooling,
                patch_size=self.patch_size,
            )
            
            # Project if needed
            if self.patch_embedding_projection is not None:
                patch_embeds = self.patch_embedding_projection(patch_embeds)
                patch_embeds = patch_embeds.reshape(
                    bs, patch_embeds.shape[1] * self.cross_attn_k, self.dim
                )

        # Apply cross-attention
        layer_idx = layer_idx if self.cross_attn_all_layers_encoder else 0
        patch_embeds_cross = self.cross_attn_layers[layer_idx](
            x=patch_embeds,
            kv=h,
            mask=cross_mask,
        )
        
        return patch_embeds + patch_embeds_cross


# ============================================================================
# Local Decoder
# ============================================================================

class LocalDecoder(LocalModelBase):
    """
    Local decoder for byte-level prediction.
    
    The decoder takes token-level embeddings from the encoder and patch-level
    representations from the global transformer, and produces byte-level outputs.
    
    Features:
    - Token-level decoding
    - Optional cross-attention with patch representations
    - Causal masking for autoregressive generation
    """

    def __init__(self, args: LocalModelArgs):
        super().__init__(args)

        # Configuration
        self.cross_attn_decoder = args.cross_attn_decoder
        self.cross_attn_all_layers_decoder = args.cross_attn_all_layers_decoder
        self.cross_attn_init_by_pooling = args.cross_attn_init_by_pooling
        self.cross_attn_nheads = args.cross_attn_nheads

        # Output normalization
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        # Cross-attention layers (if enabled)
        if self.cross_attn_decoder:
            self.cross_attn_layers = nn.ModuleList()
            layers_to_add = args.n_layers if self.cross_attn_all_layers_decoder else 1
            for _ in range(layers_to_add):
                self.cross_attn_layers.append(
                    CrossAttention(
                        dim=self.dim,
                        head_dim=self.dim // self.cross_attn_nheads,
                        n_heads=self.cross_attn_nheads,
                        n_kv_heads=self.cross_attn_nheads,
                        norm_eps=args.norm_eps,
                    )
                )

    def forward(
        self,
        tokens: torch.Tensor,
        embeds: torch.Tensor,
        patch_embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, torch.Tensor, str]] = None,
        cross_mask: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """
        Forward pass of local decoder.
        
        Args:
            tokens: Input token IDs of shape (batch_size, seq_len)
            embeds: Token embeddings from encoder (required)
            patch_embeds: Patch embeddings from global transformer
            mask: Attention mask
            cross_mask: Cross-attention mask for patch embeddings
            cache: Optional KV cache for generation
        
        Returns:
            Tuple of (output_predictions, cache)
        """
        bs, seqlen = tokens.shape
        assert embeds is not None, "Embeddings must be provided to decoder"

        # Create causal mask if not provided
        if mask is None:
            mask = create_causal_mask(
                seqlen,
                self.attn_impl,
                self.attn_bias_type,
                sliding_window=self.sliding_window,
                tokens=tokens,
                eos_id=self.eos_id,
            )

        h = embeds

        # Add position embeddings
        if self.use_rope:
            freqs_cis = self.rope(seqlen=seqlen)
        else:
            pos_ids = torch.arange(seqlen, device=h.device).unsqueeze(0)
            pos_emb = self.pos_embeddings(pos_ids)
            h = h + pos_emb
            freqs_cis = None

        h = F.dropout(h, p=self.dropout, training=self.training)

        # Process through transformer layers with optional cross-attention
        for i, layer in enumerate(self.layers):
            # Apply cross-attention to incorporate patch information
            if self.cross_attn_decoder and (
                i == 0 or self.cross_attn_all_layers_decoder
            ):
                h_cross = self.cross_attn_layers[i](
                    x=h,
                    kv=patch_embeds,
                    mask=cross_mask,
                )
                h = h + h_cross

            # Self-attention and feedforward
            h = layer(h, mask=mask, freq_cis=freqs_cis, attn_impl=self.attn_impl)

        # Final normalization and output
        h_preds = self.norm(h)
        h_preds = F.dropout(h_preds, p=self.dropout, training=self.training)
        h_preds = h_preds.float()
        
        return h_preds, cache
    