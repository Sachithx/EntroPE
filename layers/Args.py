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


from pydantic import BaseModel, ConfigDict
import abc
from enum import Enum
from typing import Any, Optional

import torch
import torch.nn as nn

from layers.BaseTransformer import TransformerBlock, RotaryEmbedding
from layers.Constants import Constants


# ============================================================================
# Init Scaling Modes
# ============================================================================

class InitStdFactor(Enum):
    DISABLED = "disabled"
    GLOBAL_DEPTH = "global_depth"
    CURRENT_DEPTH = "current_depth"
    DIM_RATIO = "dim_ratio"


# ============================================================================
# Sequence Model Interface
# ============================================================================

class SequenceModelWithOutput(abc.ABC):
    @abc.abstractmethod
    def get_output_seq_len(self) -> int:
        pass


# ============================================================================
# Base Transformer Arguments
# ============================================================================

class BaseTransformerArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dim: int = 512
    n_layers: int = 8

    head_dim: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None

    ffn_dim_multiplier: float | None = None
    multiple_of: int = 256

    norm_eps: float = 1e-5

    rope_theta: float = 10000.0
    rope_use_fp32_in_outer_product: bool = False

    init_base_std: float | None = None
    init_std_factor: InitStdFactor = InitStdFactor.DISABLED

    max_seqlen: int = 96

    attn_impl: str | None = "sdpa"
    attn_bias_type: str | None = None

    eos_id: int | None = Constants.EOS_ID


# ============================================================================
# EntroPE Model Arguments
# ============================================================================

class EntroPEArgs(BaseTransformerArgs):
    # General
    seed: int = 2025
    vocab_size: int = 256
    weight_tying: bool = False
    patch_in_forward: bool = True

    # Dimensions
    dim_token: int | None = None
    dim_global: int = 64
    dim_local_decoder: int = 32
    dim_local_encoder: int = 32

    n_layers_global: int = 2
    n_layers_local_decoder: int = 2
    n_layers_local_encoder: int = 2

    # Patching
    patch_size: float | None = None
    patching_mode: str | None = None
    patching_threshold: float | None = None
    patching_threshold_add: float | None = None
    monotonicity: bool = False

    patching_batch_size: int = 1
    patching_device: str = "cuda"
    max_patch_length: int | None = None

    # Encoder / Decoder
    use_local_encoder_transformer: bool = False
    max_encoder_seq_length: int | None = None
    pad_to_max_length: bool = False
    share_encoder_decoder_emb: bool = True

    # Cross Attention
    cross_attn_encoder: bool = False
    cross_attn_decoder: bool = False

    cross_attn_window_encoder: int | None = None
    cross_attn_window_decoder: int | None = None

    cross_attn_k: int | None = None
    cross_attn_nheads: int | None = None

    cross_attn_all_layers_decoder: bool = False
    cross_attn_all_layers_encoder: bool = False

    cross_attn_use_flex_attention: bool = True
    cross_attn_init_by_pooling: bool = False

    # Encoder hash (compat only)
    encoder_hash_byte_group_size: list | None = None
    encoder_hash_byte_group_vocab: int = 30000
    encoder_hash_byte_group_nb_functions: int = 3
    encoder_enable_byte_ngrams: bool = False

    # Model behavior
    non_linearity: str = "swiglu"
    use_rope: bool = True
    recompute_attn: bool = True

    init_use_gaussian: bool = True
    init_use_depth: str = "current"

    attn_bias_type: str = "causal"
    alpha_depth: str = "disabled"
    max_length: int = 2048

    # Norm / FFN
    norm_eps: float = 1e-5
    norm_affine: bool = True
    pre_norm: bool = True
    norm_type: str = "rmsnorm"

    multiple_of: int = 128
    ffn_dim_multiplier: float = 1.0
    dropout: float = 0.0

    # Extra
    downsampling_by_pooling: str | None = None

    n_heads_global: int = 4
    n_heads_local_decoder: int = 4
    n_heads_local_encoder: int = 4

    n_kv_heads: int | None = None
    n_kv_heads_global: int | None = None

    local_attention_window_len: int | None = None

    # Logging / checkpoints
    log_patch_lengths: bool = False
    layer_ckpt: str = "all"

    # Entropy patching model
    dataset_name: str | None = None
    entropy_model_checkpoint_dir: str | None = None


# ============================================================================
# Local Model Arguments
# ============================================================================

class LocalModelArgs(BaseTransformerArgs):
    attn_impl: str | None = "xformers"
    attn_bias_type: str | None = "local_block_causal"

    dropout: float
    vocab_size: int
    patch_size: float

    sliding_window: int | None
    use_rope: bool
    max_encoder_seq_length: int

    cross_attn_encoder: bool | None
    cross_attn_decoder: bool | None
    cross_attn_k: int | None
    cross_attn_init_by_pooling: bool

    patching_mode: str
    use_local_encoder_transformer: bool

    downsampling_by_pooling: str | None

    encoder_hash_byte_group_size: Any | None = None

    cross_attn_all_layers_encoder: bool = False
    cross_attn_all_layers_decoder: bool = False
    cross_attn_nheads: int | None

    dim_token_emb: int
    dim_patch_emb: int | None


# ============================================================================
# Local Encoder Base
# ============================================================================

class LocalModelBase(nn.Module):
    def __init__(self, args: LocalModelArgs):
        super().__init__()

        # Core dimensions
        self.dim = args.dim
        self.dropout = args.dropout
        self.vocab_size = args.vocab_size
        self.patch_size = args.patch_size
        self.dim_patch_emb = args.dim_patch_emb

        # Attention settings
        self.attn_impl = args.attn_impl
        self.attn_bias_type = args.attn_bias_type
        self.sliding_window = args.sliding_window

        # Positional encoding
        self.use_rope = args.use_rope
        self.max_encoder_seq_length = args.max_encoder_seq_length

        # Init control
        self.init_std_factor = args.init_std_factor

        # Cross-attention flags
        self.cross_attn_encoder = getattr(args, "cross_attn_encoder", None)
        self.cross_attn_decoder = getattr(args, "cross_attn_decoder", None)
        self.cross_attn_k = getattr(args, "cross_attn_k", None)

        # Tokens
        self.eos_id = args.eos_id
        self.boe_id = Constants.BOE_ID

        # Transformer layers
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])

        # Positional Embeddings
        if not self.use_rope:
            self.pos_embeddings = nn.Embedding(self.max_encoder_seq_length, args.dim)
            self.rope = None
        else:
            self.pos_embeddings = None
            self.rope = RotaryEmbedding(
                theta=args.rope_theta,
                head_dim=args.head_dim or args.dim // args.n_heads,
                max_seqlen=args.max_seqlen,
                rope_use_fp32_in_outer_product=args.rope_use_fp32_in_outer_product,
            )

        # Token embedding projection (if needed)
        self.token_embedding_projection = (
            nn.Linear(args.dim_token_emb, args.dim, bias=False)
            if hasattr(args, "dim_token_emb") and args.dim_token_emb != self.dim
            else None
        )

        # Patch embedding projection (lazy build)
        self.patch_embedding_projection = None


    # ========================================================
    # Patch Projection Logic
    # ========================================================

    def _should_create_patch_projection(self, args: LocalModelArgs) -> bool:
        dim_mismatch = getattr(args, "dim_patch_emb") and args.dim_patch_emb != self.dim

        cross_attn_needed = (
            (args.cross_attn_encoder and args.cross_attn_init_by_pooling)
            or (args.cross_attn_decoder and args.cross_attn_init_by_pooling)
        )

        return dim_mismatch or cross_attn_needed


    def _create_patch_projection(self, args: LocalModelArgs):
        if not self._should_create_patch_projection(args):
            return None

        out_dim = args.dim_token_emb * (self.cross_attn_k or 1)

        return nn.Linear(
            in_features=args.dim_patch_emb,
            out_features=out_dim,
            bias=False,
        )


    # ========================================================
    # Embedding Routing
    # ========================================================

    def apply_embedding(self, tokens, embeds):
        return embeds if embeds is not None else self.tok_embeddings(tokens)


    # ========================================================
    # Weight Initialization
    # ========================================================

    def init_weights(self, init_std=None):
        # Reset rope / norm
        if self.rope is not None:
            self.rope.reset_parameters()

        if hasattr(self, "norm"):
            self.norm.reset_parameters()

        init_std = init_std or (self.dim ** -0.5)

        # Token embeddings
        if hasattr(self, "tok_embeddings"):
            nn.init.trunc_normal_(self.tok_embeddings.weight, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)

        # Positional embeddings (if not rope)
        if self.pos_embeddings is not None:
            nn.init.trunc_normal_(self.pos_embeddings.weight, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)

        # Transformer blocks
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(None, factor)

        # Output head
        if hasattr(self, "output"):
            nn.init.trunc_normal_(self.output.weight, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)

        # Token projection
        if self.token_embedding_projection is not None:
            nn.init.trunc_normal_(self.token_embedding_projection.weight, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)

        # Patch projection
        if self.patch_embedding_projection is not None:
            patch_std = self.dim_patch_emb ** -0.5
            nn.init.trunc_normal_(self.patch_embedding_projection.weight, mean=0.0, std=patch_std, a=-3*patch_std, b=3*patch_std)

        # Cross-attention layers
        if hasattr(self, "cross_attn_layers") and self.cross_attn_layers is not None:
            for depth, layer in enumerate(self.cross_attn_layers):
                factor = {
                    InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                    InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                    InitStdFactor.DIM_RATIO: self.dim / 4096,
                    InitStdFactor.DISABLED: 1.0,
                }[self.init_std_factor]

                layer.init_weights(None, factor)
