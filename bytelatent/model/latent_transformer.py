# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask
# from xformers.ops import AttentionBias

from bytelatent.base_transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    flex_attention_comp,
    repeat_kv,
)
from bytelatent.model.utils import create_causal_mask

logger = logging.getLogger()
# try:
#     from apex.normalization.fused_layer_norm import FusedRMSNorm

#     RMSNorm = FusedRMSNorm
# except (ImportError, ModuleNotFoundError):
#     logging.debug("Apex not found. Using nn.RMSNorm")
RMSNorm = nn.RMSNorm


class CrossAttention(nn.Module):
    """
    CrossAttention block to attend to the encoder states from the decoder.
    Rope is not supported.
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

        self.cross_attn_norm_q = nn.RMSNorm(dim, eps=norm_eps)
        self.cross_attn_norm_kv = RMSNorm(dim, eps=norm_eps)

        self.wq = nn.Linear(
            dim,
            n_heads * head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            n_heads * head_dim,
            dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        mask: Optional[Union[BlockMask, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        # B S D
        bsz, seq_len, _ = x.shape
        _, slen_kv, _ = kv.shape
        x_norm = self.cross_attn_norm_q(x)
        kv = self.cross_attn_norm_kv(kv)

        xq = self.wq(x_norm)
        xk = self.wk(kv)
        xv = self.wv(kv)

        output_shape = xq.shape
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, slen_kv, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, slen_kv, self.n_kv_heads, self.head_dim)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        if attn_impl == "flex_attention":
            assert mask is None or isinstance(mask, BlockMask)
            assert flex_attention_comp is not None, "flex_attention is not compiled or available"
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            output = flex_attention_comp(xq, xk, xv, block_mask=mask)
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

        # elif attn_impl == "xformers":
        #     from xformers.ops.fmha import memory_efficient_attention

        #     assert mask is None or isinstance(mask)
        #     xq, xk, xv = [t.reshape(1, -1, *t.shape[2:]) for t in (xq, xk, xv)]
        #     output = memory_efficient_attention(xq, xk, xv, attn_bias=mask)
        #     output = output.view(output_shape)

        elif attn_impl == "sdpa":
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            is_causal = isinstance(mask, str) and mask == "causal"
            mask_tensor = mask if isinstance(mask, torch.Tensor) else None
            if isinstance(mask_tensor, torch.Tensor) and mask_tensor.dtype not in [torch.float32, torch.bool]:
                mask_tensor = mask_tensor.to(torch.float32)
            output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=is_causal, attn_mask=mask_tensor)
            output = output.transpose(1, 2).contiguous()

        else:
            raise NotImplementedError(f"Unsupported attn_impl: {attn_impl}")

        output = self.wo(output.reshape(output_shape))
        return x + output


    def init_weights(self, base_std: float, factor: float = 1.0):
        std = base_std or (self.dim ** (-0.5)) / factor

        nn.init.trunc_normal_(
            self.wq.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

        nn.init.trunc_normal_(
            self.wk.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

        nn.init.trunc_normal_(
            self.wv.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )
        self.cross_attn_norm_q.reset_parameters()
        self.cross_attn_norm_kv.reset_parameters()


class GlobalTransformer(BaseTransformer):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__(args)
        self.dropout = args.dropout
        self.eos_id = args.eos_id
        self.dim_token_emb = args.dim_token_emb

        self.token_embedding_projection = None
        if args.dim_token_emb is not None and args.dim_token_emb != self.dim:
            self.token_embedding_projection = nn.Linear(
                args.dim_token_emb,
                args.dim,
                bias=False,
            )
        self.pos_embeddings = nn.Embedding(self.max_seqlen, args.dim)
        print(f"[Global] Token embedding projection: {self.token_embedding_projection}")

    def forward(
        self,
        tokens: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, torch.Tensor, str]] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        """
        Similar to BaseTransformer.forward, but with an additional embeds argument
        and projection to the token space.
        """
        bs, seqlen = tokens.shape

        h = embeds
        pos_ids = torch.arange(seqlen, device=h.device).unsqueeze(0)  # [1, seq_len]
        pos_emb = self.pos_embeddings(pos_ids)                       # [1, seq_len, dim]

        h = h + pos_emb

        # mask = (
        #     mask
        #     if mask is not None
        #     else create_causal_mask(
        #         seqlen,
        #         self.attn_impl,
        #         self.attn_bias_type,
        #         tokens=tokens,
        #         eos_id=self.eos_id,
        #     )
        # )

        if self.token_embedding_projection is not None and h.shape[-1] != self.dim:
            h = self.token_embedding_projection(h)

        h = F.dropout(h, p=self.dropout, training=self.training)

        h = super().forward(h, tok_idx=tok_idx, mask=mask, attn_impl=self.attn_impl)
        return h, cache

    def init_weights(self):
        super().init_weights()
        std = self.dim_token_emb ** (-0.5)
        if self.token_embedding_projection is not None:
            nn.init.trunc_normal_(
                self.token_embedding_projection.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
