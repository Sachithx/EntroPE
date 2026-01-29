import abc
import logging
import os
from enum import Enum
from typing import Optional, Tuple, Union

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    flex_attention,
)

from layers.Constants import Constants

logger = logging.getLogger()

RMSNorm = nn.RMSNorm

# ============================================================
# Flex Attention Compilation Control
# ============================================================

if int(os.environ.get("BLT_ALLOW_MISSING_FLEX_ATTENTION", False)) == 0:
    flex_attention_comp = torch.compile(flex_attention)
else:
    flex_attention_comp = None


# ============================================================
# Initialization Scaling Modes
# ============================================================

class InitStdFactor(Enum):
    DISABLED = "disabled"
    GLOBAL_DEPTH = "global_depth"
    CURRENT_DEPTH = "current_depth"
    DIM_RATIO = "dim_ratio"


# ============================================================
# Transformer Config
# ============================================================

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


# ============================================================
# Loss Utility
# ============================================================

def cross_entropy(pred, target, **kwargs):
    return F.nll_loss(
        F.log_softmax(pred.flatten(end_dim=-2).float(), -1),
        target.flatten(end_dim=-1),
        **kwargs,
    )


# ============================================================
# KV Head Replication
# ============================================================

def repeat_kv(x: torch.Tensor, n_rep: int, dim: int) -> torch.Tensor:
    """Repeat KV heads to match query heads."""
    assert dim == 2, "Only dim=2 supported"
    bs, slen, n_kv_heads, head_dim = x.shape

    if n_rep == 1:
        return x

    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# ============================================================
# Rotary Positional Embeddings (RoPE)
# ============================================================

def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    rope_use_fp32_in_outer_product: bool = False,
):
    """Precompute cos/sin rotation matrices."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(end, device=freqs.device)

    if rope_use_fp32_in_outer_product:
        t = t.float()

    freqs = torch.outer(t, freqs).float()
    cos, sin = freqs.cos(), freqs.sin()

    return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int):
    ndim = x.ndim
    assert 0 <= seq_dim < ndim
    assert freqs_cis.shape == (x.shape[seq_dim], x.shape[-3], 2, 2)

    shape = [
        d if i == seq_dim or i == ndim - 3 else 1
        for i, d in enumerate(x.shape[:-2])
    ] + [2, 2]

    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, seq_dim, freqs_cis):
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_, seq_dim).float()

    xq_out = (xq_ * freqs_cis).sum(5).flatten(3)
    xk_out = (xk_ * freqs_cis).sum(5).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# ============================================================
# Sequence Mask Helpers
# ============================================================

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def lengths_to_start_ids(lengths):
    doc_start = lengths.cumsum(0).roll(1)
    doc_start[0] = 0
    return doc_start


def lengths_to_local_ids(lengths):
    nb_seqs = lengths.size(0)
    total_len = lengths.sum()

    doc_id = torch.repeat_interleave(lengths)
    doc_start = lengths_to_start_ids(lengths)
    doc_start = doc_start[doc_id]

    tok_id = torch.arange(total_len, device=lengths.device) - doc_start
    return doc_id, tok_id


def generate_doc_mask_mod(mask_mod, lengths, kv_lengths=None):
    """Generate document-level attention mask modifier."""
    kv_lengths = kv_lengths if kv_lengths is not None else lengths

    q_doc, q_tok = lengths_to_local_ids(lengths)
    kv_doc, kv_tok = lengths_to_local_ids(kv_lengths)

    q_max = lengths.sum() - 1
    kv_max = kv_lengths.sum() - 1

    def doc_mask_mod(b, h, q_idx, kv_idx):
        q_cap = torch.minimum(q_max, q_idx)
        kv_cap = torch.minimum(kv_max, kv_idx)

        valid = (q_idx <= q_max) & (kv_idx <= kv_max)
        same_doc = q_doc[q_cap] == kv_doc[kv_cap]

        inner = mask_mod(b, h, q_tok[q_cap], kv_tok[kv_cap])

        return same_doc & inner & valid

    return doc_mask_mod


# ============================================================
# RotaryEmbedding Module
# ============================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, theta, head_dim, max_seqlen=1024, rope_use_fp32_in_outer_product=False):
        super().__init__()

        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen
        self.rope_use_fp32_in_outer_product = rope_use_fp32_in_outer_product

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                dim=head_dim,
                end=max_seqlen,
                theta=theta,
                rope_use_fp32_in_outer_product=self.rope_use_fp32_in_outer_product,
            ),
            persistent=False,
        )

    def reset_parameters(self):
        self.freqs_cis[...] = precompute_freqs_cis(
            dim=self.head_dim,
            end=self.max_seqlen,
            theta=self.theta,
            rope_use_fp32_in_outer_product=self.rope_use_fp32_in_outer_product,
        )

    def forward(self, seqlen=None, tok_idx=None):
        assert (seqlen is not None) or (tok_idx is not None)

        if tok_idx is not None:
            return self.freqs_cis[tok_idx]

        return self.freqs_cis[:seqlen]


# ============================================================
# Attention Layer
# ============================================================

class Attention(nn.Module):
    def __init__(self, dim, head_dim, n_heads, n_kv_heads, rope_theta):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)

        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x, freq_cis, tok_idx=None, mask=None, attn_impl="sdpa"):
        bsz, seq_len, _ = x.shape

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        out_shape = xq.shape

        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[:seq_len])

        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        if attn_impl == "sdpa":
            xq, xk, xv = map(lambda t: t.transpose(1, 2), (xq, xk, xv))

            is_causal = isinstance(mask, str) and mask == "causal"
            mask_tensor = mask if isinstance(mask, torch.Tensor) else None

            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                is_causal=is_causal,
                attn_mask=mask_tensor,
            )

            output = output.transpose(1, 2).contiguous()

        else:
            raise NotImplementedError(attn_impl)

        return self.wo(output.reshape(out_shape))

    def reset_parameters(self, init_std=None, factor=1.0):
        std = init_std or (self.dim ** -0.5) / factor

        for w in [self.wq, self.wk, self.wv, self.wo]:
            nn.init.trunc_normal_(w.weight, mean=0.0, std=std, a=-3*std, b=3*std)


# ============================================================
# Feedforward
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier=None, mp_size=1):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)

        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % mp_size == 0

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def reset_parameters(self, init_std=None, factor=1.0):
        in_std = init_std or (self.dim ** -0.5) / factor
        out_std = init_std or (self.hidden_dim ** -0.5) / factor

        nn.init.trunc_normal_(self.w1.weight, mean=0, std=in_std, a=-3*in_std, b=3*in_std)
        nn.init.trunc_normal_(self.w3.weight, mean=0, std=in_std, a=-3*in_std, b=3*in_std)
        nn.init.trunc_normal_(self.w2.weight, mean=0, std=out_std, a=-3*out_std, b=3*out_std)


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()

        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // self.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        assert self.n_heads % self.n_kv_heads == 0
        assert args.dim % self.n_heads == 0

        self.attention = Attention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
        )

        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freq_cis, tok_idx=None, mask=None, attn_impl="sdpa"):
        attn_out = self.attention(self.attention_norm(x), freq_cis, tok_idx, mask, attn_impl)
        h = x + attn_out
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.feed_forward.reset_parameters(init_std, factor)

        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()


# ============================================================
# Base Transformer Model
# ============================================================

class SequenceModelWithOutput(abc.ABC):
    @abc.abstractmethod
    def get_output_seq_len(self) -> int:
        pass


class BaseTransformer(nn.Module, SequenceModelWithOutput):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()

        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = args.init_std_factor

        self.attn_impl = args.attn_impl
        self.attn_bias_type = args.attn_bias_type
        self.max_seqlen = args.max_seqlen
        self.eos_id = args.eos_id

        self.rope_embeddings = RotaryEmbedding(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.max_seqlen,
            rope_use_fp32_in_outer_product=args.rope_use_fp32_in_outer_product,
        )

        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])

    def get_output_seq_len(self):
        return self.max_seqlen

    def forward(self, h, tok_idx=None, mask=None, attn_impl="sdpa"):
        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)

        for layer in self.layers:
            h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)

        return h

    def init_weights(self):
        self.rope_embeddings.reset_parameters()

        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(self.init_base_std, factor)


# ============================================================
# Cross-Attention Block
# ============================================================

class CrossAttention(nn.Module):
    """Decoder cross-attention (no RoPE)."""

    def __init__(self, dim, head_dim, n_heads, n_kv_heads, norm_eps):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = n_heads // n_kv_heads

        self.cross_attn_norm_q = RMSNorm(dim, eps=norm_eps)
        self.cross_attn_norm_kv = RMSNorm(dim, eps=norm_eps)

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)

        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x, kv, mask=None, attn_impl="sdpa"):
        bsz, seq_len, _ = x.shape
        _, slen_kv, _ = kv.shape

        x = self.cross_attn_norm_q(x)
        kv = self.cross_attn_norm_kv(kv)

        xq = self.wq(x)
        xk = self.wk(kv)
        xv = self.wv(kv)

        out_shape = xq.shape

        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, slen_kv, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, slen_kv, self.n_kv_heads, self.head_dim)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        if attn_impl == "sdpa":
            xq, xk, xv = map(lambda t: t.transpose(1, 2), (xq, xk, xv))

            is_causal = isinstance(mask, str) and mask == "causal"
            mask_tensor = mask if isinstance(mask, torch.Tensor) else None

            if isinstance(mask_tensor, torch.Tensor) and mask_tensor.dtype not in (torch.float32, torch.bool):
                mask_tensor = mask_tensor.float()

            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                is_causal=is_causal,
                attn_mask=mask_tensor,
            )

            output = output.transpose(1, 2).contiguous()

        else:
            raise NotImplementedError(attn_impl)

        return x + self.wo(output.reshape(out_shape))

    def init_weights(self, base_std: float, factor: float = 1.0):
        std = base_std or (self.dim ** -0.5) / factor

        for w in [self.wq, self.wk, self.wv, self.wo]:
            nn.init.trunc_normal_(w.weight, mean=0, std=std, a=-3*std, b=3*std)

        self.cross_attn_norm_q.reset_parameters()
        self.cross_attn_norm_kv.reset_parameters()
