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


__all__ = ['EntroPE_backbone']

import math
import torch
from torch import nn
import warnings
from typing import Optional

from layers.Constants import Constants
from layers.RevIN import RevIN
from layers.FlattenHead import FlattenHead
from layers.Args import SequenceModelWithOutput, InitStdFactor, EntroPEArgs, LocalModelArgs
from layers.PatchEncoder import PatchEncoder
from layers.GlobalTransformer import GlobalTransformer
from layers.FusionDecoder import FusionDecoder
from layers.Patcher import Patcher, PatcherArgs
from layers.Tokenizer import build_tokenizer
from utils.layer_utils import (
    get_decoder_dim_token_emb, 
    get_entrope_input, 
    patch_ids_from_lengths, 
    cross_attn_mask, 
    downsample,
    get_encoder_dim_token_emb, 
    get_encoder_dim_patch_emb, 
    get_global_dim_patch_emb
)

warnings.filterwarnings("ignore")


# ============================================================================
# Core EntroPE Model
# ============================================================================

class EntroPE(nn.Module, SequenceModelWithOutput):
    """
    EntroPE: Entropy-guided Patching for time series transformers.
    
    This model uses information-theoretic principles to dynamically determine
    patch boundaries rather than using fixed-length segmentation.
    """

    def __init__(self, args: EntroPEArgs):
        super().__init__()

        # Store core configuration
        self._init_core_config(args)
        self._init_cross_attention_config(args)
        
        # Create model components
        self.patch_encoder = create_patch_encoder(args)
        self.global_transformer = create_global_transformer(args)
        self.fusion_decoder = create_fusion_decoder(args)
        
        # Initialize patcher if needed
        if args.patch_in_forward:
            self.patcher = Patcher(
                PatcherArgs(
                    entropy_model_checkpoint_dir=args.entropy_model_checkpoint_dir,
                    dataset_name=args.dataset_name,
                    patching_mode=args.patching_mode,
                    quantile_threshold=args.patching_threshold,
                    monotonicity=args.monotonicity,
                    max_patch_length=args.max_patch_length,
                    patch_size=args.patch_size,
                    patching_batch_size=args.patching_batch_size,
                )
            )

    def _init_core_config(self, args: EntroPEArgs):
        """Initialize core configuration parameters."""
        self.weight_tying = args.weight_tying
        self.patch_size = args.patch_size
        self.patching_mode = args.patching_mode
        self.downsampling_by_pooling = args.downsampling_by_pooling
        self.patching_threshold = args.patching_threshold
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seqlen = args.max_seqlen
        
        # Special tokens
        self.boe_id = Constants.BOE_ID
        self.bos_id = Constants.BOS_ID
        self.pad_id = Constants.PAD_ID
        self.eos_id = Constants.EOS_ID

    def _init_cross_attention_config(self, args: EntroPEArgs):
        """Initialize cross-attention configuration."""
        self.cross_attn_encoder = args.cross_attn_encoder
        self.cross_attn_decoder = args.cross_attn_decoder
        self.cross_attn_k = args.cross_attn_k
        self.cross_attn_window_encoder = args.cross_attn_window_encoder
        self.cross_attn_window_decoder = args.cross_attn_window_decoder
        self.cross_attn_use_flex_attention = args.cross_attn_use_flex_attention

    def get_output_seq_len(self):
        """Return maximum sequence length."""
        return self.max_seqlen

    def forward(
        self,
        tokens: torch.Tensor,
        patch_lengths: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of EntroPE.
        
        Args:
            tokens: Input token IDs of shape (batch_size, seq_len)
            patch_lengths: Optional precomputed patch lengths (batch_size, num_patches)
        
        Returns:
            Output logits of shape (batch_size, seq_len, vocab_size)
        """
        bs, N = tokens.shape

        # Prepare inputs
        nb_boe = 0 if self.patching_mode != "" else self.patch_size - 1
        patch_encoder_tokens, _, fusion_decoder_tokens = get_entrope_input(
            tokens=tokens,
            enforce_patch_size_multiple=False,
            nb_boe=nb_boe,
            patch_size=self.patch_size,
            boe_id=self.boe_id,
        )

        # Generate or validate patch lengths
        patch_lengths = self._get_patch_lengths(patch_encoder_tokens, patch_lengths, nb_boe)
        patch_ids = patch_ids_from_lengths(patch_lengths, patch_encoder_tokens.shape[-1])

        # Encode patches with local encoder
        h_encoder, h_cross = self._encode_patches(
            patch_encoder_tokens, patch_ids, patch_lengths, N
        )

        # Downsample to patch-level representations
        h_patches = self._downsample_to_patches(
            h_encoder, h_cross, patch_ids, patch_lengths, bs
        )

        # Process patches with global transformer
        h_global = self._apply_global_transformer(
            h_patches, patch_encoder_tokens, patch_ids, bs
        )

        # Decode to token-level predictions
        output = self._decode_to_tokens(
            h_encoder, h_global, fusion_decoder_tokens, patch_ids, patch_lengths, N, nb_boe
        )

        return output

    def forward_patch_level(
        self,
        tokens: torch.Tensor,
        patch_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Same as forward() up through the global transformer, but returns the
        patch-level representations directly instead of running them through
        the fusion decoder. Used for head variants (e.g. a PatchTST-style
        flatten+linear head) that predict straight from patch embeddings,
        matching PatchTST's own architecture which has no token-level decode
        step at all.

        Returns: h_global of shape (batch_size, num_patches, dim_global)
        """
        bs, N = tokens.shape

        nb_boe = 0 if self.patching_mode != "" else self.patch_size - 1
        patch_encoder_tokens, _, _ = get_entrope_input(
            tokens=tokens,
            enforce_patch_size_multiple=False,
            nb_boe=nb_boe,
            patch_size=self.patch_size,
            boe_id=self.boe_id,
        )

        patch_lengths = self._get_patch_lengths(patch_encoder_tokens, patch_lengths, nb_boe)
        patch_ids = patch_ids_from_lengths(patch_lengths, patch_encoder_tokens.shape[-1])

        h_encoder, h_cross = self._encode_patches(
            patch_encoder_tokens, patch_ids, patch_lengths, N
        )

        h_patches = self._downsample_to_patches(
            h_encoder, h_cross, patch_ids, patch_lengths, bs
        )

        h_global = self._apply_global_transformer(
            h_patches, patch_encoder_tokens, patch_ids, bs
        )

        return h_global

    def _get_patch_lengths(
        self, 
        patch_encoder_tokens: torch.Tensor, 
        patch_lengths: Optional[torch.Tensor],
        nb_boe: int
    ) -> torch.Tensor:
        """Generate or validate patch lengths."""
        if patch_lengths is None:
            assert hasattr(self, "patcher"), "Patcher not defined and no patch_lengths passed"
            patch_lengths, _ = self.patcher.patch(
                patch_encoder_tokens,
                include_next_token=True
            )
        else:
            if nb_boe > 0:
                patch_lengths[:, 0] += nb_boe

        assert torch.min(patch_lengths) >= 0, "Patch lengths must be non-negative"
        return patch_lengths

    def _encode_patches(
        self,
        patch_encoder_tokens: torch.Tensor,
        patch_ids: torch.Tensor,
        patch_lengths: torch.Tensor,
        N: int
    ):
        """Encode patches with local encoder and optional cross-attention."""
        cross_attn_mask_enc = None
        if self.cross_attn_encoder:
            cross_attn_mask_enc = cross_attn_mask(
                patch_ids, patch_lengths, N,
                patches_as_queries=True,
                cross_attn_k=self.cross_attn_k,
                window=self.cross_attn_window_encoder,
                block_mask=self.cross_attn_use_flex_attention,
            )

        (h_encoder, h_cross), _ = self.patch_encoder(
            tokens=patch_encoder_tokens,
            embeds=None,
            patch_embeds=None,
            cross_mask=cross_attn_mask_enc,
            num_patches=patch_lengths.shape[1],
            patch_ids=patch_ids,
        )
        
        return h_encoder, h_cross

    def _downsample_to_patches(
        self,
        h_encoder: torch.Tensor,
        h_cross: torch.Tensor,
        patch_ids: torch.Tensor,
        patch_lengths: torch.Tensor,
        bs: int
    ) -> torch.Tensor:
        """Downsample token representations to patch representations."""
        if not self.cross_attn_encoder:
            h = downsample(
                h_encoder, patch_lengths.shape[1], patch_lengths, patch_ids,
                downsampling_by_pooling=self.downsampling_by_pooling,
                patch_size=self.patch_size,
            )
        else:
            h = h_cross.view(bs, patch_lengths.shape[1], -1)
        
        return h

    def _apply_global_transformer(
        self,
        h_patches: torch.Tensor,
        patch_encoder_tokens: torch.Tensor,
        patch_ids: torch.Tensor,
        bs: int
    ) -> torch.Tensor:
        """Apply global transformer to patch representations."""
        # Create global tokens (all BOE except EOS positions)
        global_tokens = h_patches.new_full((bs, h_patches.shape[1]), self.boe_id, dtype=torch.long)
        
        # Mark EOS positions
        rows, cols = torch.where(patch_encoder_tokens == self.eos_id)
        eos_patch_ids = patch_ids[rows, cols]
        global_tokens[rows, eos_patch_ids] = self.eos_id

        h_global, _ = self.global_transformer(embeds=h_patches, tokens=global_tokens)
        return h_global

    def _decode_to_tokens(
        self,
        h_encoder: torch.Tensor,
        h_global: torch.Tensor,
        fusion_decoder_tokens: torch.Tensor,
        patch_ids: torch.Tensor,
        patch_lengths: torch.Tensor,
        N: int,
        nb_boe: int
    ) -> torch.Tensor:
        """Decode global representations back to token-level predictions."""
        # Extract relevant encoder embeddings
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

        # Decode with fusion decoder
        output, _ = self.fusion_decoder(
            embeds=dec_embeds,
            patch_embeds=h_global,
            tokens=fusion_decoder_tokens,
            cross_mask=cross_attn_mask_dec,
        )

        return output


# ============================================================================
# EntroPE Backbone for Time Series 
# ============================================================================

class EntroPE_backbone(nn.Module):
    """
    EntroPE Backbone for time series forecasting with reversible normalization.
    
    Args:
        configs: Configuration object containing model hyperparameters
        pretrain_head: Whether to use pretraining head
        head_type: Type of prediction head ('flatten' or custom)
        individual: Whether to use individual linear layers per variable
        revin: Whether to use reversible instance normalization
        affine: Whether to use affine transformation in RevIN
        subtract_last: Whether to subtract last value in RevIN
    """
    
    def __init__(
        self, 
        configs, 
        pretrain_head=False, 
        head_type='flatten', 
        individual=False, 
        revin=True, 
        affine=True, 
        subtract_last=False, 
        **kwargs
    ):
        super().__init__()
        
        # Initialize reversible normalization
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(
                configs.enc_in, 
                affine=affine, 
                subtract_last=subtract_last
            )
        
        # Build and initialize EntroPE model
        model_args = self._build_entrope_args(configs)
        self.backbone = EntroPE(model_args)

        # Initialize tok_embeddings from entropy model's wte.
        # Both encode the same discrete vocab, so the entropy model's learned
        # token representations give the backbone a warm start.
        # Only applied when dims match (n_embd == d_model).
        if (hasattr(self.backbone, 'patcher')
                and self.backbone.patcher.patching_mode.value == 'entropy'):
            _em = self.backbone.patcher._base_entropy_model
            if _em is not None:
                bb_dim  = self.backbone.patch_encoder.tok_embeddings.weight.shape[1]
                ent_dim = _em.transformer.wte.weight.shape[1]
                if bb_dim == ent_dim:
                    self.backbone.patch_encoder.tok_embeddings.weight.data.copy_(
                        _em.transformer.wte.weight.float()
                    )
                    print(f"  [EntroPE_backbone] Initialized tok_embeddings "
                          f"(dim={bb_dim}) from entropy model")
                else:
                    print(f"  [EntroPE_backbone] Embedding dim mismatch "
                          f"(backbone={bb_dim}, entropy={ent_dim}); "
                          f"set --n_embd {bb_dim} when training entropy model to enable init.")

        # Initialize tokenizer
        self.tokenizer = build_tokenizer(configs)
        
        # Prediction head / decoder. decoder_mode selects between:
        #   'flatten' (default) -- original Linear(d_model*seq_len -> pred_len)
        #       head, optionally preceded by decoder_pool_size's AdaptiveAvgPool1d
        #       (see that flag's own comment). Param count scales with L (and
        #       with H, via the Linear's target_window dim) unless pooled.
        #   'horizon_query'    -- H fixed-sinusoidal-position forecast queries
        #                         cross-attend directly to the backbone's
        #                         point-level output X (memory length = seq_len,
        #                         varies with L, but the query/head parameters
        #                         do not).
        #   'latent_resampler' -- X is first resampled into a fixed number of
        #                         latent tokens via cross-attention, then the
        #                         same H forecast queries attend to that
        #                         fixed-size latent set instead of X directly.
        # The last two give O(d^2) decoder params independent of BOTH L and H
        # (queries are a fixed sinusoidal table sliced to pred_len, not a
        # learned H-sized matrix) -- unlike 'flatten', pooled or not, whose
        # Linear(nf, target_window) always bakes H into its weight shape.
        # IMPORTANT: the old head/pool machinery is only constructed when
        # actually used -- otherwise its (L-dependent) parameters would still
        # be counted even though forward() never calls it for the new modes.
        self.decoder_mode = getattr(configs, 'decoder_mode', 'flatten')
        self.n_vars = configs.enc_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual
        self.pred_len = configs.pred_len

        if self.decoder_mode == 'flatten':
            self.decoder_pool_size = getattr(configs, 'decoder_pool_size', None)
            if self.decoder_pool_size is not None:
                self.decoder_pool = nn.AdaptiveAvgPool1d(self.decoder_pool_size)
                self.head_nf = configs.d_model * self.decoder_pool_size
            else:
                self.decoder_pool = None
                self.head_nf = configs.d_model * configs.seq_len
            self.head = self._build_head(configs)
        elif self.decoder_mode == 'patchtst_head':
            # PatchTST-style head: flatten+linear directly on patch-level
            # global-transformer output (dim_global), skipping the fusion
            # decoder entirely -- matches PatchTST's own architecture, which
            # has no token-level decode step. Dynamic patching produces a
            # variable number of patches per batch, so --decoder_pool_size
            # (adaptive-pooled over the patch dimension) is required to give
            # the head a fixed input size; for a fixed-patch-count baseline
            # (e.g. static patching) set it to that exact patch count so the
            # two variants' heads have identical parameter shapes.
            global_dim = getattr(configs, 'global_d_model', None) or configs.d_model
            self.decoder_pool_size = getattr(configs, 'decoder_pool_size', None)
            if self.decoder_pool_size is None:
                raise ValueError(
                    "--decoder_mode patchtst_head requires --decoder_pool_size "
                    "(fixed number of patch slots for the flatten+linear head)."
                )
            self.decoder_pool = nn.AdaptiveAvgPool1d(self.decoder_pool_size)
            self.head_nf = global_dim * self.decoder_pool_size
            self.head = self._build_head(configs)
        else:
            self.decoder_pool_size = None
            self.decoder_pool = None
            self.head = None

        if self.decoder_mode in ('horizon_query', 'latent_resampler'):
            self.hq_cross_attn = nn.MultiheadAttention(
                configs.d_model, configs.n_heads, dropout=configs.dropout, batch_first=True
            )
            self.hq_output_head = nn.Linear(configs.d_model, 1)
            max_horizon = max(configs.pred_len, 768)  # headroom to slice from, built once
            self.register_buffer(
                'horizon_pos_emb', self._sinusoidal_embedding(max_horizon, configs.d_model),
                persistent=False,
            )
            if self.decoder_mode == 'latent_resampler':
                n_latent = getattr(configs, 'n_latent_tokens', 32)
                self.latent_tokens = nn.Parameter(torch.randn(1, n_latent, configs.d_model) * 0.02)
                self.resample_cross_attn = nn.MultiheadAttention(
                    configs.d_model, configs.n_heads, dropout=configs.dropout, batch_first=True
                )

    @staticmethod
    def _sinusoidal_embedding(length, dim):
        pe = torch.zeros(length, dim)
        position = torch.arange(0, length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _decode_length_independent(self, x, pred_len):
        """
        x: [bs*nvars, seq_len, d_model] -- the backbone's point-level output.
        Returns: [bs*nvars, pred_len] (one scalar per future timestep; each row
        of x is already a single channel, matching the rest of this file's
        channel-independent convention).
        """
        bnv = x.shape[0]
        queries = self.horizon_pos_emb[:pred_len].unsqueeze(0).expand(bnv, -1, -1).to(x.dtype)

        if self.decoder_mode == 'latent_resampler':
            latents = self.latent_tokens.expand(bnv, -1, -1)
            memory, _ = self.resample_cross_attn(latents, x, x, need_weights=False)
        else:
            memory = x

        out, _ = self.hq_cross_attn(queries, memory, memory, need_weights=False)  # [bnv, pred_len, d_model]
        out = self.hq_output_head(out).squeeze(-1)  # [bnv, pred_len]
        return out
    
    def _build_entrope_args(self, configs) -> EntroPEArgs:
        """Build EntroPE arguments from configuration."""
        return EntroPEArgs(
            # Core settings
            seed=configs.random_seed,
            vocab_size=configs.vocab_size,
            max_length=configs.seq_len,
            max_seqlen=configs.seq_len,
            max_encoder_seq_length=configs.seq_len,
            
            # Attention windows
            local_attention_window_len=configs.local_attention_window_len,
            encoder_self_attn_within_patch=getattr(configs, 'encoder_self_attn_within_patch', False),
            cross_attn_window_decoder=configs.cross_attn_window_decoder,
            cross_attn_window_encoder=configs.cross_attn_window_encoder,
            
            # Model dimensions
            # global_* CLI overrides let the Global Transformer be sized independently
            # from the Patch Encoder / Fusion Decoder (e.g. to match a baseline's own
            # published hyperparameters); None -> old shared-d_model behavior.
            dim_global=getattr(configs, 'global_d_model', None) or configs.d_model,
            dim_local_encoder=configs.d_model,
            dim_local_decoder=configs.d_model,

            # Layer configurations
            n_layers_global=getattr(configs, 'global_e_layers', None) or configs.e_layers,
            n_layers_local_encoder=1,
            n_layers_local_decoder=1,

            # Attention heads
            n_heads_global=getattr(configs, 'global_n_heads', None) or configs.n_heads,
            n_heads_local_encoder=configs.n_heads,
            n_heads_local_decoder=configs.n_heads,
            multiple_of_global=getattr(configs, 'global_d_ff', None),
            
            # Patching configuration
            patch_size=configs.max_patch_length,
            patch_in_forward=True,
            patching_batch_size=configs.enc_in * configs.batch_size,
            patching_device=getattr(configs, 'patching_device', 'cuda'),
            patching_mode=getattr(configs, 'boundary_method', 'entropy'),
            patching_threshold=configs.patching_threshold,
            max_patch_length=configs.max_patch_length,
            monotonicity=configs.monotonicity,
            pad_to_max_length=True,
            
            # Cross-attention settings
            cross_attn_encoder=True,
            cross_attn_decoder=True,
            cross_attn_k=configs.cross_attn_k,
            cross_attn_nheads=configs.n_heads,
            cross_attn_all_layers_encoder=True,
            cross_attn_all_layers_decoder=True,
            cross_attn_use_flex_attention=False,
            cross_attn_init_by_pooling=True,

            # Model architecture
            non_linearity=configs.activation,
            use_rope=True,
            attn_impl="sdpa",
            attn_bias_type="causal",
            multiple_of=configs.d_ff,
            dropout=configs.dropout,
            
            # Training settings
            layer_ckpt="none",
            init_use_gaussian=True,
            init_use_depth="current",
            alpha_depth="disabled",
            log_patch_lengths=True,
            
            # Dataset and checkpointing
            dataset_name=configs.model_id_name,
            entropy_model_checkpoint_dir=configs.entropy_model_checkpoint_dir,
            downsampling_by_pooling="max",
            use_local_encoder_transformer=True,
            share_encoder_decoder_emb=False
        )
    
    def _build_head(self, configs):
        """Build prediction head based on configuration."""
        if self.pretrain_head:
            return nn.Sequential(
                nn.Dropout(configs.fc_dropout),
                nn.Conv1d(self.head_nf, configs.enc_in, 1)
            )
        elif self.head_type == 'flatten':
            return FlattenHead(
                individual=self.individual,
                n_vars=self.n_vars,
                nf=self.head_nf,
                target_window=configs.pred_len,
                head_dropout=configs.head_dropout
            )
        else:
            raise ValueError(f"Unknown head_type: {self.head_type}")
    
    def forward(self, z, channel_idx=None):
        """
        Forward pass through the model.

        Args:
            z: Input tensor of shape [batch_size, n_vars, seq_len]
            channel_idx: Optional 1D LongTensor of channel indices actually present in
                z's n_vars dimension (e.g. a random subset for high-channel-count
                datasets). Only RevIN needs this -- its affine_weight/affine_bias are
                indexed per real channel identity; everything past the reshape below
                is channel-agnostic (each row of bs*n_vars is processed independently
                regardless of which channel it came from), so channel_idx is not
                threaded any further than RevIN.

        Returns:
            Output tensor of shape [batch_size, n_vars, pred_len]
        """
        bs, nvars, seq_len = z.shape

        # Apply reversible normalization
        if self.revin:
            z = z.permute(0, 2, 1)  # [bs, seq_len, nvars]
            z = self.revin_layer(z, 'norm', channel_idx=channel_idx)
            z = z.permute(0, 2, 1)  # [bs, nvars, seq_len]
        
        # Reshape for tokenization: treat each variable independently
        z = z.reshape(bs * nvars, seq_len)
        
        # Tokenize input
        z, _, _ = self.tokenizer.context_input_transform(z)
        device = next(self.backbone.parameters()).device
        z = z.to(device)
        
        # Pass through EntroPE backbone
        if self.decoder_mode == 'patchtst_head':
            z = self.backbone.forward_patch_level(z)  # [bs*nvars, num_patches, dim_global]
            z = z.permute(0, 2, 1)  # [bs*nvars, dim_global, num_patches]
            z = self.decoder_pool(z)  # [bs*nvars, dim_global, R]
            z = z.view(bs, nvars, z.shape[1], z.shape[2])  # [bs, nvars, dim_global, R]
            z = self.head(z)  # [bs, nvars, pred_len]
            if self.revin:
                z = z.permute(0, 2, 1)
                z = self.revin_layer(z, 'denorm', channel_idx=channel_idx)
                z = z.permute(0, 2, 1)
            return z

        z = self.backbone(z)  # [bs * nvars, seq_len, d_model]

        if self.decoder_mode in ('horizon_query', 'latent_resampler'):
            z = self._decode_length_independent(z, self.pred_len)  # [bs*nvars, pred_len]
            z = z.view(bs, nvars, self.pred_len)  # [bs, nvars, pred_len]
            if self.revin:
                z = z.permute(0, 2, 1)
                z = self.revin_layer(z, 'denorm', channel_idx=channel_idx)
                z = z.permute(0, 2, 1)
            return z

        # Reshape back to batch format
        z = z.view(bs, nvars, z.shape[1], z.shape[2])
        z = z.permute(0, 1, 3, 2)  # [bs, nvars, d_model, seq_len]

        # Pool seq_len -> fixed R before the head, if enabled, so the head's
        # param count doesn't depend on the (possibly much larger) input length.
        if self.decoder_pool is not None:
            d_model = z.shape[2]
            z = z.reshape(bs * nvars, d_model, seq_len)
            z = self.decoder_pool(z)  # [bs*nvars, d_model, R]
            z = z.view(bs, nvars, d_model, self.decoder_pool_size)

        # Apply prediction head
        z = self.head(z)  # [bs, nvars, pred_len]
        
        # Apply reversible denormalization
        if self.revin:
            z = z.permute(0, 2, 1)  # [bs, pred_len, nvars]
            z = self.revin_layer(z, 'denorm', channel_idx=channel_idx)
            z = z.permute(0, 2, 1)  # [bs, nvars, pred_len]
        
        return z


# ============================================================================
# Model Component Creation Functions
# ============================================================================

def create_global_transformer(args: EntroPEArgs) -> GlobalTransformer:
    """Create global transformer with appropriate configuration."""
    global_args = args.model_copy(
        deep=True,
        update=dict(
            dim=args.dim_global,
            n_layers=args.n_layers_global,
            n_heads=args.n_heads_global,
            n_kv_heads=args.n_kv_heads_global,
            local_attention_window_len=args.local_attention_window_len,
            dim_token_emb=get_global_dim_patch_emb(args),
            dim_patch_emb=None,
            cross_attn_encoder=False,
            cross_attn_decoder=False,
            multiple_of=args.multiple_of_global if args.multiple_of_global is not None else args.multiple_of,
        ),
    )
    return GlobalTransformer(global_args)


def create_patch_encoder(args: EntroPEArgs) -> PatchEncoder:
    """Create patch encoder with appropriate configuration."""
    patch_encoder_args = LocalModelArgs(
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
        max_seqlen=args.max_encoder_seq_length,
        dropout=args.dropout,
        vocab_size=args.vocab_size,
        norm_eps=args.norm_eps,
        patch_size=args.patch_size,
        sliding_window=args.local_attention_window_len,
        encoder_self_attn_within_patch=args.encoder_self_attn_within_patch,
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
    return PatchEncoder(patch_encoder_args)


def create_fusion_decoder(args: EntroPEArgs) -> FusionDecoder:
    """Create fusion decoder with appropriate configuration."""
    fusion_decoder_args = LocalModelArgs(
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
        max_seqlen=args.max_encoder_seq_length,  # RoPE covers full seq length
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
    return FusionDecoder(fusion_decoder_args)
