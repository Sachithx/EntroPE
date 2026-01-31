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
Patcher (Entropy-Guided Dynamic Patching with Pre-trained GPT Entropy Model)
"""

import math
import os
import time
import json
import logging
from collections import defaultdict
from contextlib import nullcontext
from enum import Enum
from functools import lru_cache

import torch
from pydantic import BaseModel
from torch.nn import functional as F

from models.GPT2EntropyModel import GPTConfig, GPT

logger = logging.getLogger()

# ============================================================
# Enums & Config
# ============================================================

class PatchingModeEnum(str, Enum):
    entropy = "entropy"
    static = "static"


class PatcherArgs(BaseModel):
    patching_mode: PatchingModeEnum = PatchingModeEnum.entropy
    dataset_name: str | None = None
    patching_device: str = "cuda"
    entropy_model_checkpoint_dir: str | None = None
    realtime_patching: bool = True

    quantile_threshold: float | None = None
    max_patch_length: int | None = None
    patch_size: float = 4.5
    patching_batch_size: int = 512
    device: str = "cuda"
    monotonicity: bool = False
    log_time: bool = False

    def build(self) -> "Patcher":
        return Patcher(self)


# ============================================================
# Entropy Model Loader
# ============================================================

def load_entropy_model(checkpoint_dir, state_path, device="cpu"):
    """Load pretrained GPT entropy model."""
    with open(os.path.join(checkpoint_dir, "params.json")) as fr:
        params = json.loads(fr.read())["entropy_model"]
    torch.set_default_dtype(torch.bfloat16)
    entropy_args = GPTConfig(
        n_layer=params["n_layer"],
        n_head=params["n_head"],
        n_embd=params["n_embd"],
        dropout=params["dropout"],
        bias=params["bias"],
        vocab_size=params["vocab_size"],
        block_size=params["block_size"],
    )

    model = GPT(entropy_args)

    model.load_state_dict(
        torch.load(state_path, map_location=device, weights_only=True)["model_state_dict"],
        strict=True
    )

    model = model.to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    return model, entropy_args


# ============================================================
# Entropy Computation
# ============================================================

def entropy(scores: torch.Tensor) -> torch.Tensor:
    """Compute token entropy from logits [bs, seq, vocab]."""
    log_probs = F.log_softmax(scores, dim=-1)
    probs = log_probs.exp()
    return -(log_probs * probs).sum(dim=-1)


def calculate_entropies(
    tokens: torch.Tensor,
    entropy_model,
    patching_batch_size: int,
    device: str | None = None,
    enable_grad: bool = False,
):
    """Compute entropy + predictions in batches."""
    grad_ctx = nullcontext() if enable_grad else torch.no_grad()

    with grad_ctx:
        entropies, preds = [], []

        max_len = getattr(entropy_model, "max_length", 96)
        batch_numel = max_len * patching_batch_size
        splits = torch.split(tokens.flatten(), batch_numel)

        for split in splits:
            pad_len = (max_len - (split.numel() % max_len)) % max_len
            if pad_len:
                pad = torch.zeros(pad_len, dtype=split.dtype, device=split.device)
                split = torch.cat([split, pad])

            split = split.view(-1, max_len)

            if device:
                split = split.to(device)

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred, _ = entropy_model(split)

            pred = pred.view(-1, pred.shape[-1])[: split.numel() - pad_len]
            preds.append(pred)
            entropies.append(entropy(pred))

        entropies = torch.cat(entropies).view(tokens.shape)
        preds = torch.cat(preds).view(tokens.shape[0], -1)

    return entropies, preds


# ============================================================
# Patch Mask Logic
# ============================================================

def patch_start_mask_from_entropy_with_monotonicity(entropies, threshold):
    """Start new patch when entropy increases > threshold."""
    bs, seq_len = entropies.shape
    mask = torch.zeros_like(entropies, dtype=torch.bool)
    mask[:, 0] = True

    if seq_len > 1:
        diffs = entropies[:, 1:] - entropies[:, :-1]
        mask[:, 1:] = diffs > threshold

    return mask


def patch_start_mask_from_entropy_with_monotonicity_adaptive(entropies, threshold):
    """Same as above, but threshold is per-sample."""
    bs, seq_len = entropies.shape
    mask = torch.zeros_like(entropies, dtype=torch.bool)
    mask[:, 0] = True

    if seq_len > 1:
        diffs = entropies[:, 1:] - entropies[:, :-1]
        mask[:, 1:] = diffs > threshold.unsqueeze(1)

    return mask


# ============================================================
# Patch ID & Length Conversion
# ============================================================

def patch_start_ids_from_patch_start_mask(mask):
    """Convert boolean mask → patch start index tensor."""
    bs, seq_len = mask.shape
    max_patches = mask.sum(dim=1).max()

    if max_patches == 0:
        return torch.full((bs, seq_len), seq_len, dtype=torch.long, device=mask.device)

    positions = torch.arange(seq_len, device=mask.device).repeat(bs, 1)
    pad = torch.full((bs, seq_len), seq_len, dtype=torch.long, device=mask.device)

    positions = torch.cat([positions, pad], dim=1)
    mask = torch.cat([mask, ~mask], dim=1)

    return positions[mask].view(bs, seq_len)[:, :max_patches]


def patch_lengths_from_start_ids(start_ids, seq_len):
    """Convert patch start indices → patch lengths."""
    last = torch.full_like(start_ids[:, :1], seq_len - 1)
    ends = torch.cat([start_ids[:, 1:] - 1, last], dim=1)

    lengths = ends - start_ids + 1

    assert torch.all(lengths >= 0)
    return lengths


# ============================================================
# Entropy Patch Finder
# ============================================================

def find_entropy_patch_start_ids(
    entropies,
    patch_size=None,
    monotonicity=False,
    include_next_token=True,
    quantile_threshold=None,
):
    """Compute patch start positions from entropy."""
    bs, seq_len = entropies.shape

    first = torch.zeros((bs, 1), dtype=torch.long, device=entropies.device)

    # Static top-k patching
    if quantile_threshold is None:
        num_patches = seq_len // patch_size
        starts = entropies.topk(int(num_patches) - 2, dim=1).indices
        starts = starts.sort(dim=1).values

    # Quantile-based adaptive patching
    else:
        ent = entropies.float()

        if monotonicity:
            diffs = ent[:, 1:] - ent[:, :-1]
            threshold = torch.quantile(diffs, quantile_threshold, dim=1)
            mask = patch_start_mask_from_entropy_with_monotonicity_adaptive(entropies, threshold)
        else:
            threshold = torch.quantile(ent, quantile_threshold, dim=1)
            mask = entropies > threshold.unsqueeze(1)

        if not include_next_token:
            mask = mask[:, :-1]

        starts = patch_start_ids_from_patch_start_mask(mask)

    return torch.cat([first, starts + 1], dim=1)


# ============================================================
# Utility Helpers
# ============================================================

def split_large_numbers(lst, m):
    out = []
    for v in lst:
        while v > m:
            out.append(m)
            v -= m
        out.append(v)
    return out


def rightpad(seq, pad_id, max_len):
    return seq + [pad_id] * (max_len - len(seq))


# ============================================================
# Main Patcher Class
# ============================================================

class Patcher:
    """Dynamic patcher supporting static and entropy-based segmentation."""

    def __init__(self, args: PatcherArgs):
        self.args = args
        self.patching_mode = args.patching_mode
        self.monotonicity = args.monotonicity
        self.quantile_threshold = args.quantile_threshold
        self.max_patch_length = args.max_patch_length
        self.patch_size = args.patch_size
        self.batch_size = args.patching_batch_size
        self.device = args.device
        self.log_time = args.log_time

        if self.log_time:
            self.log = defaultdict(float)

        self.state_path = os.path.join(
            args.entropy_model_checkpoint_dir,
            f"{args.dataset_name}.pt"
        )

        self._entropy_models = {}

        self._base_entropy_model, _ = load_entropy_model(
            args.entropy_model_checkpoint_dir,
            self.state_path,
        )

    # ---------------------------------------
    # Entropy model device cache
    # ---------------------------------------

    def _get_entropy_model_for_device(self, device):
        key = str(device)

        if key not in self._entropy_models:
            import copy
            model = copy.deepcopy(self._base_entropy_model).to(device).eval()
            for p in model.parameters():
                p.requires_grad = False
            self._entropy_models[key] = model

        return self._entropy_models[key]

    # ---------------------------------------
    # Main patching entry point
    # ---------------------------------------

    def patch(self, tokens, include_next_token=False, preds=None, entropies=None):
        bs, seq_len = tokens.shape
        seq_len_eff = seq_len + int(include_next_token)

        # ---------- STATIC ----------
        if self.patching_mode == PatchingModeEnum.static:
            num_patches = math.ceil(seq_len_eff / self.patch_size)
            patch_lengths = torch.full((bs, num_patches), self.patch_size, device=tokens.device)

            if seq_len_eff % self.patch_size:
                patch_lengths[:, -1] = seq_len_eff % self.patch_size

            return patch_lengths, None

        # ---------- ENTROPY ----------
        t0 = time.time() if self.log_time else None

        if entropies is not None:
            scores = entropies.float()
        elif preds is not None:
            scores = entropy(preds)
        else:
            scores, _ = calculate_entropies(
                tokens,
                self._get_entropy_model_for_device(tokens.device),
                self.batch_size,
                tokens.device,
            )

        patch_start_ids = find_entropy_patch_start_ids(
            scores,
            self.patch_size,
            include_next_token=include_next_token,
            monotonicity=self.monotonicity,
            quantile_threshold=self.quantile_threshold,
        )

        patch_lengths = patch_lengths_from_start_ids(patch_start_ids, seq_len_eff)

        # ---------- POSTPROCESS ----------
        if self.max_patch_length is not None:
            patch_lengths = [
                split_large_numbers(pl, self.max_patch_length)
                for pl in patch_lengths.tolist()
            ]
            max_len = max(len(pl) for pl in patch_lengths)
            patch_lengths = torch.tensor(
                [rightpad(pl, 0, max_len) for pl in patch_lengths],
                device=tokens.device
            )

        expected = tokens.numel() + include_next_token * bs
        assert patch_lengths.sum().item() == expected

        return patch_lengths, scores


# ============================================================
# Distributed Helpers
# ============================================================

@lru_cache()
def get_is_torch_run():
    return os.environ.get("LOCAL_RANK") is not None


@lru_cache()
def get_local_rank():
    return int(os.environ["LOCAL_RANK"]) if get_is_torch_run() else 0
