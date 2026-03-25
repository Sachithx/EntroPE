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
#   - Alternative boundary detection methods (LocalDiff, VarianceCP, CUSUM, Random)
#
# Copyright (c) 2026 Sachith Abeywickrama
#
# Licensed under the same license as the original Meta code.


"""
Patcher (Entropy-Guided Dynamic Patching with Pre-trained GPT Entropy Model)
Supports multiple boundary detection strategies for ablation studies.
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
    entropy          = "entropy"           # conditional entropy from frozen GPT
    static           = "static"            # uniform fixed-length patches
    local_diff       = "local_diff"        # |x_t - x_{t-1}| peaks
    variance_cp      = "variance_cp"       # sliding window variance ratio
    cusum            = "cusum"             # cumulative sum change-point
    random           = "random"            # random boundaries (same avg count)
    empirical_entropy = "empirical_entropy" # EAPformer-style: Shannon entropy of value distribution
    frequency_based  = "frequency_based"   # spectral energy shift (short-time FFT)


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
    variance_window: int = 4      # half-window for variance_cp method
    freq_window: int = 16         # window length for frequency_based FFT
    empirical_bins: int = 16      # histogram bins for empirical_entropy

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
# Entropy-Based Patch Mask Logic
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
# Alternative Boundary Detection Methods
# ============================================================

def patch_start_mask_from_local_diff(tokens: torch.Tensor,
                                     quantile_threshold: float) -> torch.Tensor:
    """
    LocalDiff: boundaries at positions where |token[t] - token[t-1]|
    exceeds the quantile_threshold-th quantile of all differences.
    Returns boolean mask (bs, seq_len).
    """
    bs, seq_len = tokens.shape
    tokens_f = tokens.float()
    diffs = torch.abs(tokens_f[:, 1:] - tokens_f[:, :-1])   # (bs, seq_len-1)

    # Clamp to [0,1] to avoid quantile issues with uniform sequences
    if diffs.numel() > 0 and diffs.max() > 0:
        threshold = torch.quantile(diffs, quantile_threshold, dim=1, keepdim=True)
    else:
        threshold = torch.zeros(bs, 1, device=tokens.device)

    mask = torch.zeros(bs, seq_len, dtype=torch.bool, device=tokens.device)
    mask[:, 0] = True
    mask[:, 1:] = diffs > threshold
    return mask


def patch_start_mask_from_variance_cp(tokens: torch.Tensor,
                                       window: int,
                                       quantile_threshold: float) -> torch.Tensor:
    """
    VarianceCP: sliding window variance change-point detection.
    For each position t, compare variance of left vs right windows.
    Boundaries where the ratio is high (high variance contrast).
    Returns boolean mask (bs, seq_len).
    """
    bs, seq_len = tokens.shape
    tokens_f = tokens.float()
    eps = 1e-6
    window = max(2, min(window, seq_len // 4))  # safety clamp

    scores = torch.zeros(bs, seq_len, dtype=torch.float32, device=tokens.device)

    # Use unfold for efficiency
    # left window: tokens_f[:, t-window:t], right: tokens_f[:, t:t+window]
    if seq_len > 2 * window:
        # Unfold into windows: shape (bs, num_positions, window)
        left_windows = tokens_f.unfold(1, window, 1)   # (bs, seq_len-window+1, window)
        right_windows = tokens_f.unfold(1, window, 1)

        left_var  = left_windows.var(dim=2)   # (bs, seq_len-window+1)
        right_var = right_windows.var(dim=2)

        # Align: for position t, left covers [t-window, t), right covers [t, t+window)
        # left_var[t] = var of tokens[t:t+window], right_var[t+window] = var of tokens[t+window:t+2*window]
        # So boundary score at position t+window = max/min ratio
        n_pos = left_windows.shape[1] - window
        if n_pos > 0:
            lv = left_var[:, :n_pos]
            rv = right_var[:, window:window + n_pos]
            ratio = (torch.max(lv, rv) / (torch.min(lv, rv) + eps)).float()
            scores[:, window: window + n_pos] = ratio

    if scores.max() > 0:
        threshold = torch.quantile(scores, quantile_threshold, dim=1, keepdim=True)
    else:
        threshold = torch.zeros(bs, 1, device=tokens.device)

    mask = torch.zeros(bs, seq_len, dtype=torch.bool, device=tokens.device)
    mask[:, 0] = True
    mask[:, 1:] = scores[:, 1:] > threshold
    return mask


def patch_start_mask_from_cusum(tokens: torch.Tensor,
                                 quantile_threshold: float) -> torch.Tensor:
    """
    CUSUM: cumulative sum of (token - mean).
    Large absolute CUSUM values indicate regime change points.
    Returns boolean mask (bs, seq_len).
    """
    tokens_f = tokens.float()
    mean = tokens_f.mean(dim=1, keepdim=True)
    deviations = tokens_f - mean
    cusum = torch.cumsum(deviations, dim=1)
    scores = torch.abs(cusum)

    if scores.max() > 0:
        threshold = torch.quantile(scores, quantile_threshold, dim=1, keepdim=True)
    else:
        threshold = torch.zeros(tokens.shape[0], 1, device=tokens.device)

    mask = scores > threshold
    mask[:, 0] = True
    return mask


def patch_start_mask_from_empirical_entropy(tokens: torch.Tensor,
                                              window: int,
                                              quantile_threshold: float,
                                              n_bins: int = 16) -> torch.Tensor:
    """
    EAPformer-style empirical Shannon entropy boundary detection.
    For each position t, compute the Shannon entropy of the empirical value
    distribution within a sliding window of length `window` centred at t.
    Boundaries are placed where this entropy is HIGH (value distribution is
    more spread out / more uncertain), which is EAPformer's Eq. 4 concept.

    Key difference from EntroPE: this measures *within-window distributional spread*
    (statistical entropy), whereas EntroPE measures *predictive uncertainty* of the
    next token given causal history (conditional entropy from a learned model).

    Returns boolean mask (bs, seq_len).
    """
    bs, seq_len = tokens.shape
    tokens_f = tokens.float()
    half_w = max(1, window // 2)
    eps = 1e-8

    # Compute empirical Shannon entropy for each position using sliding window
    scores = torch.zeros(bs, seq_len, dtype=torch.float32, device=tokens.device)

    # Use torch.unfold to create overlapping windows
    # Pad to keep output length == seq_len
    padded = torch.nn.functional.pad(tokens_f, (half_w, half_w), mode='replicate')
    # Shape: (bs, seq_len, window)
    windows = padded.unfold(1, window, 1)[:, :seq_len, :]

    # Compute histogram-based entropy for each window
    # Normalise token values to [0, n_bins)
    tok_min = tokens_f.min(dim=1, keepdim=True)[0]
    tok_max = tokens_f.max(dim=1, keepdim=True)[0]
    tok_range = (tok_max - tok_min).clamp(min=eps)

    for b in range(bs):
        win_b = windows[b]  # (seq_len, window)
        # Normalise windows to [0, n_bins)
        normed = ((win_b - tok_min[b]) / tok_range[b] * (n_bins - 1)).clamp(0, n_bins - 1)
        normed_int = normed.long()  # (seq_len, window)

        for t in range(seq_len):
            counts = torch.bincount(normed_int[t], minlength=n_bins).float()
            p = counts / counts.sum()
            p = p[p > 0]
            scores[b, t] = -(p * p.log()).sum()

    if scores.max() > 0:
        threshold = torch.quantile(scores, quantile_threshold, dim=1, keepdim=True)
    else:
        threshold = torch.zeros(bs, 1, device=tokens.device)

    mask = scores > threshold
    mask[:, 0] = True
    return mask


def patch_start_mask_from_frequency(tokens: torch.Tensor,
                                     window: int,
                                     quantile_threshold: float) -> torch.Tensor:
    """
    Frequency-based boundary detection: boundaries at positions where the
    dominant spectral content shifts between adjacent short-time windows.
    Uses the L2 distance between consecutive window FFT magnitude spectra.

    Returns boolean mask (bs, seq_len).
    """
    bs, seq_len = tokens.shape
    tokens_f = tokens.float()
    half_w = max(1, window // 2)

    # Pad and create overlapping windows (bs, seq_len, window)
    padded = torch.nn.functional.pad(tokens_f, (half_w, half_w), mode='replicate')
    windows = padded.unfold(1, window, 1)[:, :seq_len, :]  # (bs, seq_len, window)

    # Apply Hann window to reduce spectral leakage
    hann = torch.hann_window(window, device=tokens.device)
    windows = windows * hann.unsqueeze(0).unsqueeze(0)

    # Compute FFT magnitude spectra: (bs, seq_len, window//2+1)
    spectra = torch.fft.rfft(windows, dim=-1).abs()
    # Normalise each spectrum
    spec_norm = spectra / (spectra.sum(dim=-1, keepdim=True) + 1e-8)

    # Spectral shift score = L2 distance between adjacent spectra
    scores = torch.zeros(bs, seq_len, dtype=torch.float32, device=tokens.device)
    if seq_len > 1:
        diff = (spec_norm[:, 1:, :] - spec_norm[:, :-1, :]).pow(2).sum(dim=-1).sqrt()
        scores[:, 1:] = diff

    if scores.max() > 0:
        threshold = torch.quantile(scores, quantile_threshold, dim=1, keepdim=True)
    else:
        threshold = torch.zeros(bs, 1, device=tokens.device)

    mask = scores > threshold
    mask[:, 0] = True
    return mask


def patch_start_mask_random(tokens: torch.Tensor,
                             avg_patch_size: float) -> torch.Tensor:
    """
    Random: random boundary placement with same average boundary count
    as entropy-based method (expected 1 boundary per avg_patch_size tokens).
    Returns boolean mask (bs, seq_len).
    """
    bs, seq_len = tokens.shape
    p_boundary = 1.0 / max(avg_patch_size, 1.0)
    rand = torch.rand(bs, seq_len, device=tokens.device)
    mask = rand < p_boundary
    mask[:, 0] = True  # first position is always a patch start
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
    """Dynamic patcher supporting multiple boundary detection strategies."""

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
        self.variance_window = args.variance_window
        self.freq_window = args.freq_window
        self.empirical_bins = args.empirical_bins

        if self.log_time:
            self.log = defaultdict(float)

        # Only load the entropy model when needed
        self._entropy_models = {}
        self._base_entropy_model = None

        if self.patching_mode == PatchingModeEnum.entropy:
            self.state_path = os.path.join(
                args.entropy_model_checkpoint_dir,
                f"{args.dataset_name}.pt"
            )
            self._base_entropy_model, _ = load_entropy_model(
                args.entropy_model_checkpoint_dir,
                self.state_path,
            )
        else:
            self.state_path = None

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
    # Postprocess patch lengths (shared logic)
    # ---------------------------------------

    def _postprocess_patch_lengths(self, patch_lengths, tokens, include_next_token):
        """Apply max_patch_length splitting and assert correctness."""
        bs = tokens.shape[0]

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
        assert patch_lengths.sum().item() == expected, (
            f"patch_lengths sum {patch_lengths.sum().item()} != expected {expected}"
        )
        return patch_lengths

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

        # ---------- LOCAL DIFF ----------
        elif self.patching_mode == PatchingModeEnum.local_diff:
            qt = self.quantile_threshold if self.quantile_threshold is not None else 0.75
            mask = patch_start_mask_from_local_diff(tokens, qt)
            patch_start_ids = patch_start_ids_from_patch_start_mask(mask)
            patch_lengths = patch_lengths_from_start_ids(patch_start_ids, seq_len_eff)
            return self._postprocess_patch_lengths(patch_lengths, tokens, include_next_token), None

        # ---------- VARIANCE CP ----------
        elif self.patching_mode == PatchingModeEnum.variance_cp:
            qt = self.quantile_threshold if self.quantile_threshold is not None else 0.75
            mask = patch_start_mask_from_variance_cp(tokens, self.variance_window, qt)
            patch_start_ids = patch_start_ids_from_patch_start_mask(mask)
            patch_lengths = patch_lengths_from_start_ids(patch_start_ids, seq_len_eff)
            return self._postprocess_patch_lengths(patch_lengths, tokens, include_next_token), None

        # ---------- CUSUM ----------
        elif self.patching_mode == PatchingModeEnum.cusum:
            qt = self.quantile_threshold if self.quantile_threshold is not None else 0.75
            mask = patch_start_mask_from_cusum(tokens, qt)
            patch_start_ids = patch_start_ids_from_patch_start_mask(mask)
            patch_lengths = patch_lengths_from_start_ids(patch_start_ids, seq_len_eff)
            return self._postprocess_patch_lengths(patch_lengths, tokens, include_next_token), None

        # ---------- RANDOM ----------
        elif self.patching_mode == PatchingModeEnum.random:
            mask = patch_start_mask_random(tokens, self.patch_size)
            patch_start_ids = patch_start_ids_from_patch_start_mask(mask)
            patch_lengths = patch_lengths_from_start_ids(patch_start_ids, seq_len_eff)
            return self._postprocess_patch_lengths(patch_lengths, tokens, include_next_token), None

        # ---------- EMPIRICAL ENTROPY (EAPformer-style) ----------
        elif self.patching_mode == PatchingModeEnum.empirical_entropy:
            qt = self.quantile_threshold if self.quantile_threshold is not None else 0.75
            mask = patch_start_mask_from_empirical_entropy(
                tokens, self.freq_window, qt, self.empirical_bins
            )
            patch_start_ids = patch_start_ids_from_patch_start_mask(mask)
            patch_lengths = patch_lengths_from_start_ids(patch_start_ids, seq_len_eff)
            return self._postprocess_patch_lengths(patch_lengths, tokens, include_next_token), None

        # ---------- FREQUENCY-BASED ----------
        elif self.patching_mode == PatchingModeEnum.frequency_based:
            qt = self.quantile_threshold if self.quantile_threshold is not None else 0.75
            mask = patch_start_mask_from_frequency(tokens, self.freq_window, qt)
            patch_start_ids = patch_start_ids_from_patch_start_mask(mask)
            patch_lengths = patch_lengths_from_start_ids(patch_start_ids, seq_len_eff)
            return self._postprocess_patch_lengths(patch_lengths, tokens, include_next_token), None

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

        return self._postprocess_patch_lengths(patch_lengths, tokens, include_next_token), scores


# ============================================================
# Distributed Helpers
# ============================================================

@lru_cache()
def get_is_torch_run():
    return os.environ.get("LOCAL_RANK") is not None


@lru_cache()
def get_local_rank():
    return int(os.environ["LOCAL_RANK"]) if get_is_torch_run() else 0
