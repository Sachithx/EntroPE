# Copyright (c) Meta Platforms, Inc. and affiliates.

from enum import Enum, auto
from typing import Optional

import torch
from pydantic import model_validator
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from typing_extensions import Self

from bytelatent.base_transformer import (
    BaseTransformerArgs,
    InitStdFactor,
    SequenceModelWithOutput,
)
from layers.Patcher import Patcher, PatcherArgs
from bytelatent.model.latent_transformer import GlobalTransformer
from bytelatent.model.local_models import LocalDecoder, LocalEncoder, LocalModelArgs
from bytelatent.model.utils import downsample
from bytelatent.tokenizers.constants import BOE_ID, BOS_ID, EOS_ID, OFFSET, PAD_ID


# ============================================================================
# Utility Functions
# ============================================================================

def causal_mask(b, h, q_idx, kv_idx):
    """Causal mask for attention."""
    return q_idx >= kv_idx


def get_encoder_dim_token_emb(args):
    if args.dim_token is not None:
        return args.dim_token
    elif args.use_local_encoder_transformer:
        return args.dim_local_encoder
    else:
        return args.dim_global // args.patch_size


def get_encoder_dim_patch_emb(args):
    if args.cross_attn_encoder:
        if args.cross_attn_init_by_pooling:
            return args.dim_local_encoder
        else:
            return args.dim_global
    return None


def get_global_dim_patch_emb(args):
    dim_token_emb = get_encoder_dim_token_emb(args)
    if args.cross_attn_encoder:
        return dim_token_emb * args.cross_attn_k
    elif (
        args.downsampling_by_pooling is None
        or not args.downsampling_by_pooling
        or len(args.downsampling_by_pooling) == 0
    ):
        return dim_token_emb * args.patch_size
    else:
        return dim_token_emb * sum(
            [
                pooling in args.downsampling_by_pooling
                for pooling in ["avg", "min", "max"]
            ]
        )


def get_decoder_dim_token_emb(args):
    if args.share_encoder_decoder_emb:
        return get_encoder_dim_token_emb(args)
    elif args.dim_token is not None:
        return args.dim_token
    else:
        return args.dim_local_decoder


def fill_tokens(tokens, patch_size, fill_id):
    """Pad tokens to make sequence length divisible by patch_size."""
    batch_size, seq_len = tokens.shape
    if seq_len % patch_size == 0:
        return tokens
    else:
        remaining = patch_size - seq_len % patch_size
        final_padding = tokens.new(batch_size, remaining).fill_(fill_id)
        return torch.cat((tokens, final_padding), dim=1)


def patch_ids_from_lengths(patch_lengths, seq_len):
    """Generate patch IDs from patch lengths."""
    bs, num_patches = patch_lengths.shape
    cum_d = torch.cat(
        [
            torch.zeros(bs, 1, dtype=patch_lengths.dtype, device=patch_lengths.device),
            patch_lengths.cumsum(dim=-1),
        ],
        dim=-1,
    )
    patch_ids = (cum_d.unsqueeze(-1) <= torch.arange(seq_len, device=cum_d.device)).sum(
        dim=-2
    ) - 1
    assert not (
        torch.max(patch_ids) > patch_lengths.shape[-1] or torch.min(patch_ids) < 0
    ), f"Invalid patch_ids: max={torch.max(patch_ids)}, min={torch.min(patch_ids)}"
    return patch_ids


def create_patch_mask_from_ids(
    patch_ids, num_patches, window=None, patches_as_queries=False
):
    """Create attention mask from patch IDs."""
    bs, seq_len = patch_ids.shape
    if not patches_as_queries:
        q_ids = patch_ids.unsqueeze(-1).expand(bs, seq_len, num_patches)
        kv_ids = (
            torch.arange(num_patches, device=patch_ids.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(bs, seq_len, num_patches)
        )
    else:
        kv_ids = patch_ids.unsqueeze(1).expand(bs, num_patches, seq_len)
        q_ids = (
            torch.arange(num_patches, device=patch_ids.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(bs, num_patches, seq_len)
        )
    
    if window is None:
        mask = q_ids == kv_ids
    else:
        mask = (kv_ids <= q_ids) & (q_ids < kv_ids + window)
    return mask


def cross_attn_mask(
    patch_ids,
    patch_lengths,
    N,
    patches_as_queries=False,
    cross_attn_k=1,
    window=None,
    block_mask=True,
):
    """Create cross-attention mask."""
    bs = patch_ids.shape[0]
    with torch.no_grad():
        cross_mask = create_patch_mask_from_ids(
            patch_ids,
            patch_lengths.shape[1],
            window=window,
            patches_as_queries=patches_as_queries,
        ).repeat_interleave(cross_attn_k, dim=1 if patches_as_queries else -1)
        
        q_len = patch_lengths.shape[1] * cross_attn_k if patches_as_queries else N
        kv_len = N if patches_as_queries else patch_lengths.shape[1] * cross_attn_k
        
        assert cross_mask.shape == (bs, q_len, kv_len)

        if block_mask:
            def patch_mask(b, h, q_idx, kv_idx):
                return cross_mask[b, q_idx, kv_idx]

            return create_block_mask(
                patch_mask,
                B=bs,
                H=None,
                Q_LEN=q_len,
                KV_LEN=kv_len,
                _compile=True,
            )
        else:
            return torch.where(
                cross_mask, torch.tensor(0.0), torch.tensor(float("-inf"))
            ).unsqueeze(1)


def get_blt_input(
    tokens: torch.Tensor,
    enforce_patch_size_multiple: bool,
    nb_boe: int,
    patch_size: int,
    boe_id: int,
):
    """Prepare encoder and decoder tokens for BLT."""
    batch_size, seq_len = tokens.shape
    local_encoder_tokens = tokens
    local_decoder_tokens = tokens

    if nb_boe > 0:
        padded_patch = tokens.new(batch_size, nb_boe).fill_(boe_id)
        local_encoder_tokens = torch.cat((padded_patch, local_encoder_tokens), dim=1)

    if enforce_patch_size_multiple and local_encoder_tokens.shape[-1] % patch_size != 0:
        local_encoder_tokens = fill_tokens(local_encoder_tokens, patch_size, boe_id)

    return local_encoder_tokens, None, local_decoder_tokens


# ============================================================================
# Model Arguments
# ============================================================================

class ByteLatentTransformerArgs(BaseTransformerArgs):
    # Basic configuration
    seed: int = 2025
    vocab_size: int = 256
    weight_tying: bool = False
    patch_in_forward: bool = True

    # Architecture dimensions
    dim_token: int | None = None
    dim_global: int = 64
    dim_local_decoder: int = 32
    dim_local_encoder: int = 32
    n_layers_global: int = 2
    n_layers_local_decoder: int = 2
    n_layers_local_encoder: int = 2

    # Patching configuration
    patch_size: float | None = None
    patching_mode: str | None = None
    patching_threshold: float | None = None
    patching_threshold_add: float | None = None
    monotonicity: bool = False
    patching_batch_size: int = 1
    patching_device: str = "cuda"
    max_patch_length: int | None = None

    # Encoder/Decoder configuration
    use_local_encoder_transformer: bool = False
    max_encoder_seq_length: int | None = None
    pad_to_max_length: bool = False
    share_encoder_decoder_emb: bool = True

    # Cross attention
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

    # Encoder hash configurations (kept for compatibility but unused)
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

    # Normalization
    norm_eps: float = 1e-5
    norm_affine: bool = True
    pre_norm: bool = True
    norm_type: str = "rmsnorm"

    # FFN configuration
    multiple_of: int = 128
    ffn_dim_multiplier: float = 1.0
    dropout: float = 0

    # Additional parameters
    downsampling_by_pooling: str | None = None
    n_heads_global: int = 4
    n_heads_local_decoder: int = 4
    n_heads_local_encoder: int = 4
    n_kv_heads: int | None = None
    n_kv_heads_global: int | None = None
    local_attention_window_len: int | None = None

    # Logging and checkpointing
    log_patch_lengths: bool = False
    layer_ckpt: str = "all"

    # Patching model
    dataset_name: str | None = None
    entropy_model_checkpoint_dir: str | None = None

    @model_validator(mode="after")
    def check_hash_byte_sizes(self) -> Self:
        """Convert hash byte group size from string to list if needed."""
        if (
            self.encoder_hash_byte_group_size is not None
            and isinstance(self.encoder_hash_byte_group_size, str)
        ):
            self.encoder_hash_byte_group_size = [
                int(x)
                for x in self.encoder_hash_byte_group_size.split(",")
                if len(x) > 0
            ]
        return self


# ============================================================================
# Model Creation Functions
# ============================================================================

def create_global_transformer(args: ByteLatentTransformerArgs) -> GlobalTransformer:
    """Create global transformer with appropriate arguments."""
    global_args = args.model_copy(
        deep=True,
        update=dict(
            dim=args.dim_global,
            n_layers=args.n_layers_global,
            n_heads=args.n_heads_global,
            n_kv_heads=args.n_kv_heads_global,
            local_attention_window_len=None,
            dim_token_emb=get_global_dim_patch_emb(args),
            dim_patch_emb=None,
            cross_attn_encoder=False,
            cross_attn_decoder=False,
        ),
    )
    return GlobalTransformer(global_args)


def create_local_encoder(args: ByteLatentTransformerArgs) -> LocalEncoder:
    """Create local encoder with appropriate arguments."""
    local_encoder_args = LocalModelArgs(
        dim=args.dim_local_encoder,
        n_layers=args.n_layers_local_encoder,
        n_heads=args.n_heads_local_encoder,
        dim_token_emb=get_encoder_dim_token_emb(args),
        dim_patch_emb=get_encoder_dim_patch_emb(args),
        cross_attn_encoder=args.cross_attn_encoder,
        cross_attn_decoder=False,
        cross_attn_k=args.cross_attn_k if args.cross_attn_encoder else None,
        cross_attn_init_by_pooling=args.cross_attn_init_by_pooling,
        head_dim=args.head_dim,
        max_encoder_seq_length=args.max_encoder_seq_length,
        dropout=args.dropout,
        vocab_size=args.vocab_size,
        norm_eps=args.norm_eps,
        patch_size=args.patch_size,
        sliding_window=args.local_attention_window_len,
        use_rope=args.use_rope,
        rope_theta=args.rope_theta,
        rope_use_fp32_in_outer_product=args.rope_use_fp32_in_outer_product,
        init_base_std=args.init_base_std,
        init_std_factor=args.init_std_factor,
        n_kv_heads=args.n_kv_heads,
        attn_impl=args.attn_impl,
        attn_bias_type="causal",
        multiple_of=args.multiple_of,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
        patching_mode=args.patching_mode,
        use_local_encoder_transformer=args.use_local_encoder_transformer,
        downsampling_by_pooling=args.downsampling_by_pooling,
        cross_attn_all_layers_encoder=args.cross_attn_all_layers_encoder,
        cross_attn_all_layers_decoder=args.cross_attn_all_layers_decoder,
        cross_attn_nheads=args.cross_attn_nheads,
        eos_id=args.eos_id,
    )
    return LocalEncoder(local_encoder_args)


def create_local_decoder(args: ByteLatentTransformerArgs) -> LocalDecoder:
    """Create local decoder with appropriate arguments."""
    local_decoder_args = LocalModelArgs(
        dim=args.dim_local_decoder,
        n_layers=args.n_layers_local_decoder,
        n_heads=args.n_heads_local_decoder,
        dim_token_emb=get_decoder_dim_token_emb(args),
        dim_patch_emb=args.dim_global,
        cross_attn_encoder=False,
        cross_attn_decoder=args.cross_attn_decoder,
        cross_attn_init_by_pooling=False,
        cross_attn_k=args.cross_attn_k if args.cross_attn_decoder else None,
        head_dim=args.head_dim,
        max_encoder_seq_length=args.max_encoder_seq_length,
        max_seqlen=args.max_encoder_seq_length,
        dropout=args.dropout,
        vocab_size=args.vocab_size,
        norm_eps=args.norm_eps,
        patch_size=args.patch_size,
        sliding_window=args.local_attention_window_len,
        use_rope=args.use_rope,
        rope_theta=args.rope_theta,
        rope_use_fp32_in_outer_product=args.rope_use_fp32_in_outer_product,
        init_base_std=args.init_base_std,
        init_std_factor=args.init_std_factor,
        n_kv_heads=args.n_kv_heads,
        attn_impl=args.attn_impl,
        attn_bias_type="causal",
        multiple_of=args.multiple_of,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
        patching_mode=args.patching_mode,
        use_local_encoder_transformer=args.use_local_encoder_transformer,
        downsampling_by_pooling=args.downsampling_by_pooling,
        cross_attn_all_layers_encoder=args.cross_attn_all_layers_encoder,
        cross_attn_all_layers_decoder=args.cross_attn_all_layers_decoder,
        cross_attn_nheads=args.cross_attn_nheads,
        eos_id=args.eos_id,
    )
    return LocalDecoder(local_decoder_args)


# ============================================================================
# Main Model
# ============================================================================

class ByteLatentTransformer(nn.Module, SequenceModelWithOutput):
    """
    ByteLatentTransformer (BLT) - A byte-level language model that processes
    byte sequences by dynamically segmenting them into patches.
    
    Uses local encoders, global transformers, and local decoders for efficient
    encoding and decoding of byte sequences with patch-based processing.
    """

    def __init__(self, args: ByteLatentTransformerArgs):
        super().__init__()

        # Core configuration
        self.weight_tying = args.weight_tying
        self.patch_size = args.patch_size
        self.patching_mode = args.patching_mode
        self.boe_id, self.bos_id, self.pad_id, self.eos_id = (
            BOE_ID, BOS_ID, PAD_ID, EOS_ID
        )
        self.downsampling_by_pooling = args.downsampling_by_pooling
        self.patching_threshold = args.patching_threshold
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seqlen = args.max_seqlen

        # Cross attention configuration
        self.cross_attn_encoder = args.cross_attn_encoder
        self.cross_attn_decoder = args.cross_attn_decoder
        self.cross_attn_k = args.cross_attn_k
        self.cross_attn_window_encoder = args.cross_attn_window_encoder
        self.cross_attn_window_decoder = args.cross_attn_window_decoder
        self.cross_attn_use_flex_attention = args.cross_attn_use_flex_attention

        # Create model components
        self.local_encoder = create_local_encoder(args)
        self.global_transformer = create_global_transformer(args)
        self.local_decoder = create_local_decoder(args)

        # Patcher module
        if args.patch_in_forward:
            self.patcher = Patcher(
                PatcherArgs(
                    entropy_model_checkpoint_dir=args.entropy_model_checkpoint_dir,
                    dataset_name=args.dataset_name,
                    patching_mode=args.patching_mode,
                    threshold=args.patching_threshold,
                    threshold_add=args.patching_threshold_add,
                    monotonicity=args.monotonicity,
                    max_patch_length=args.max_patch_length,
                    patching_batch_size=args.patching_batch_size,
                )
            )

        # Initialize weights automatically
        # self.init_weights()

    def get_output_seq_len(self):
        """Return maximum sequence length."""
        return self.max_seqlen

    def forward(
        self,
        tokens: torch.Tensor,
        patch_lengths: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of ByteLatentTransformer.
        
        Args:
            tokens: Input token IDs of shape (batch_size, seq_len)
            patch_lengths: Optional precomputed patch lengths (batch_size, num_patches)
        
        Returns:
            Output logits of shape (batch_size, seq_len, vocab_size)
        """
        bs, N = tokens.shape

        # Prepare inputs
        nb_boe = int(0 if self.patching_mode != "" else self.patch_size - 1)
        local_encoder_tokens, _, local_decoder_tokens = get_blt_input(
            tokens=tokens,
            enforce_patch_size_multiple=False,
            nb_boe=nb_boe,
            patch_size=self.patch_size,
            boe_id=self.boe_id,
        )

        # Generate patches
        if patch_lengths is None:
            assert hasattr(self, "patcher"), "Patcher not defined and no patch_lengths passed"
            patch_lengths, _ = self.patcher.patch(
                local_encoder_tokens,
                include_next_token=True,
                threshold=self.patcher.threshold,
            )
            # print(patch_lengths[4])
        else:
            if nb_boe > 0:
                patch_lengths[:, 0] += nb_boe

        assert torch.min(patch_lengths) >= 0, "Patch lengths must be non-negative"

        # Generate patch IDs
        patch_ids = patch_ids_from_lengths(patch_lengths, local_encoder_tokens.shape[-1])

        # Prepare cross-attention mask for encoder if needed
        cross_attn_mask_enc = None
        if self.cross_attn_encoder:
            cross_attn_mask_enc = cross_attn_mask(
                patch_ids, patch_lengths, N,
                patches_as_queries=True,
                cross_attn_k=self.cross_attn_k,
                window=self.cross_attn_window_encoder,
                block_mask=self.cross_attn_use_flex_attention,
            )

        # Local encoder
        (h_encoder, h_cross), _ = self.local_encoder(
            tokens=local_encoder_tokens,
            embeds=None,
            patch_embeds=None,
            cross_mask=cross_attn_mask_enc,
            num_patches=patch_lengths.shape[1],
            patch_ids=patch_ids,
        )

        # Downsample to patch representations
        if not self.cross_attn_encoder:
            h = downsample(
                h_encoder, patch_lengths.shape[1], patch_lengths, patch_ids,
                downsampling_by_pooling=self.downsampling_by_pooling,
                patch_size=self.patch_size,
            )
        else:
            h = h_cross.view(bs, patch_lengths.shape[1], -1)

        # Global transformer
        global_tokens = tokens.new(h.shape[0], h.shape[1]).fill_(self.boe_id)
        rows, cols = torch.where(local_encoder_tokens == self.eos_id)
        eos_patch_ids = patch_ids[rows, cols]
        global_tokens[rows, eos_patch_ids] = self.eos_id

        h, _ = self.global_transformer(embeds=h, tokens=global_tokens)

        # Prepare decoder inputs
        dec_embeds = h_encoder[:, nb_boe : nb_boe + N, :]

        # Prepare cross-attention mask for decoder if needed
        cross_attn_mask_dec = None
        if self.cross_attn_decoder:
            cross_attn_mask_dec = cross_attn_mask(
                patch_ids, patch_lengths, N,
                patches_as_queries=False,
                cross_attn_k=self.cross_attn_k,
                window=self.cross_attn_window_decoder,
                block_mask=self.cross_attn_use_flex_attention,
            )

        # Local decoder
        output, _ = self.local_decoder(
            embeds=dec_embeds,
            patch_embeds=h,
            tokens=local_decoder_tokens,
            cross_mask=cross_attn_mask_dec,
        )

        return output

    def init_weights(self):
        """Initialize all model weights."""
        self.local_encoder.init_weights()
        self.global_transformer.init_weights()
        self.local_decoder.init_weights()
