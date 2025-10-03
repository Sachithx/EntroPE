"""
Patcher module for dynamic patching of token sequences based on entropy.
Copyright (c) Meta Platforms, Inc. and affiliates.
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

class PatchingModeEnum(str, Enum):
    """Enumeration of available patching modes"""
    entropy = "entropy"
    static = "static"


class PatcherArgs(BaseModel):
    """Configuration arguments for the Patcher"""
    patching_mode: PatchingModeEnum = PatchingModeEnum.entropy
    dataset_name: str | None = None
    patching_device: str = "cuda"
    entropy_model_checkpoint_dir: str | None = None
    realtime_patching: bool = True
    threshold: float = 1.335442066192627
    threshold_add: float | None = None
    max_patch_length: int | None = None
    patch_size: float = 4.5
    patching_batch_size: int = 512
    device: str = "cuda"
    monotonicity: bool = False
    log_time: bool = False

    def build(self) -> "Patcher":
        """Build and return a Patcher instance"""
        return Patcher(self)


def load_entropy_model(
        entropy_model_checkpoint_dir, state_dict_path, device="cpu"):
    with open(os.path.join(entropy_model_checkpoint_dir, "params.json")) as fr:
        reloaded = json.loads(fr.read())
    # print(reloaded)
    torch.set_default_dtype(torch.bfloat16)
    model_params = reloaded["entropy_model"]
    logger.warning(
        "Update checkpoint to load attn and sliding window args from checkpoint"
    )
    # print("Loading entropy model with params:", model_params)
    entropy_model_args = GPTConfig(
        n_layer=model_params["n_layer"],
        n_head=model_params["n_head"],
        n_embd=model_params["n_embd"],
        dropout=model_params["dropout"],
        bias=model_params["bias"],
        vocab_size=model_params["vocab_size"],
        block_size=model_params["block_size"]
    )
    entropy_model = GPT(entropy_model_args)

    entropy_model.load_state_dict(torch.load(
        state_dict_path, 
        map_location=device, 
        weights_only=True
        )["model_state_dict"], strict=True)
    
    entropy_model.to(device)
    entropy_model = entropy_model.eval()
    # no grads for the model:
    for param in entropy_model.parameters():
        param.requires_grad = False
    return entropy_model, entropy_model_args


def entropy(scores):
    """
    Compute entropy for each token in the batch.
    
    Args:
        scores: Tensor of shape [bs, seq_len, vocab]
        
    Returns:
        Tensor of shape [bs, seq_len] containing entropy values
        
    Note: Uses natural logarithm
    """
    log_probs = F.log_softmax(scores, dim=-1)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    return -p_log_p.sum(dim=-1)


def calculate_entropies(
    tokens: torch.Tensor,
    entropy_model,
    patching_batch_size: int,
    device: str | None = None,
    enable_grad: bool = False,
):
    """
    Calculate entropies for all tokens in batched manner.
    
    Args:
        tokens: 2D tensor of shape [batch_size, seq_len]
        entropy_model: Model to compute entropy predictions
        patching_batch_size: Batch size for processing
        device: Device to run model on ('cuda' or 'cpu')
        enable_grad: Whether to enable gradients
        
    Returns:
        Tuple of (entropies, predictions) both reshaped to match input tokens
    """
    grad_context = nullcontext() if enable_grad else torch.no_grad()
    
    with grad_context:
        entropies = []
        preds = []
        max_length = getattr(entropy_model, "max_length", 96)
        batch_numel = max_length * patching_batch_size
        
        # Split tokens into manageable chunks
        splits = torch.split(tokens.flatten(), batch_numel)
        
        for split in splits:
            # Pad to max_length
            pad_size = (max_length - (split.numel() % max_length)) % max_length
            if pad_size > 0:
                pad = torch.zeros(
                    pad_size, 
                    dtype=split.dtype, 
                    device=split.device, 
                    requires_grad=False
                )
                split = torch.cat((split, pad), dim=0)
            
            split = split.reshape(-1, max_length)
            
            if device is not None:
                split = split.to(device)
            
            # Get predictions from entropy model
            pred, _ = entropy_model(split)
            pred = pred.reshape(-1, pred.shape[-1])[:split.numel() - pad_size, :]
            preds.append(pred)
            
            # Calculate entropy
            pred_entropies = entropy(pred)
            entropies.append(pred_entropies)
        
        # Concatenate and reshape to original token shape
        concat_entropies = torch.cat(entropies, dim=0).reshape(tokens.shape)
        concat_preds = torch.cat(preds, dim=0).reshape(tokens.shape[0], -1)
    
    return concat_entropies, concat_preds


def patch_start_mask_from_entropy_with_monotonicity(entropies, threshold):
    """
    Create patch start mask using monotonicity constraint.
    
    A new patch starts when the difference in entropy exceeds the threshold.
    
    Args:
        entropies: Tensor of shape [bs, seq_len]
        threshold: Threshold value
        
    Returns:
        Boolean mask of shape [bs, seq_len] indicating patch starts
    """
    bs, seq_len = entropies.shape
    
    if seq_len == 0:
        return entropies > threshold
    
    mask = torch.zeros_like(entropies, dtype=torch.bool)
    mask[:, 0] = True  # First token always starts a patch
    
    # Calculate differences between consecutive elements
    differences = entropies[:, 1:] - entropies[:, :-1]
    
    # New patch when difference exceeds threshold
    condition = differences > threshold
    mask[:, 1:] = condition
    
    return mask


def patch_start_mask_global_and_monotonicity(entropies, threshold, threshold_add=0):
    """
    Create patch start mask using both global and monotonicity constraints.
    
    Args:
        entropies: Tensor of shape [bs, seq_len]
        threshold: Global threshold for entropy value
        threshold_add: Additional threshold for entropy difference
        
    Returns:
        Boolean mask of shape [bs, seq_len] indicating patch starts
    """
    bs, seq_len = entropies.shape
    
    if seq_len == 0:
        return entropies > threshold
    
    mask = torch.zeros_like(entropies, dtype=torch.bool)
    mask[:, 0] = True
    
    # Calculate differences
    differences = entropies[:, 1:] - entropies[:, :-1]
    
    # New patch when: difference > threshold_add AND entropy > threshold AND previous is not start
    condition = (differences > threshold_add) & (entropies[:, 1:] > threshold) & (~mask[:, :-1])
    mask[:, 1:] = condition
    
    return mask


def patch_start_ids_from_patch_start_mask(patch_start_mask):
    """
    Convert boolean patch start mask to patch start IDs.
    
    Args:
        patch_start_mask: Boolean tensor of shape [bs, seq_len]
        
    Returns:
        Tensor of shape [bs, max_patches] with patch start positions
    """
    bs, trunc_seq_len = patch_start_mask.shape
    max_patches = patch_start_mask.sum(dim=1).max()
    
    if max_patches == 0:
        return torch.full(
            (bs, trunc_seq_len),
            trunc_seq_len,
            dtype=torch.long,
            device=patch_start_mask.device,
        )
    
    # Create position indices
    patch_ids = torch.arange(trunc_seq_len, device=patch_start_mask.device).unsqueeze(0).repeat(bs, 1)
    
    # Padding for extraction
    extra_patch_ids = torch.full(
        (bs, trunc_seq_len),
        trunc_seq_len,
        dtype=torch.long,
        device=patch_start_mask.device,
    )
    
    all_patch_ids = torch.cat((patch_ids, extra_patch_ids), dim=1)
    patch_start_mask_padded = torch.cat((patch_start_mask, ~patch_start_mask), dim=1)
    
    patch_start_ids = all_patch_ids[patch_start_mask_padded].reshape(bs, trunc_seq_len)[:, :max_patches]
    
    return patch_start_ids


def check_non_zero_after_zero(tensor):
    """Check if there are non-zero values after zero values in each row"""
    zero_mask = tensor == 0
    shifted_mask = torch.cat([
        torch.zeros(tensor.shape[0], 1, dtype=torch.bool, device=tensor.device),
        zero_mask[:, :-1],
    ], dim=1)
    non_zero_after_zero = (tensor != 0) & shifted_mask
    return non_zero_after_zero.any()


def patch_lengths_from_start_ids(patch_start_ids, seq_len):
    """
    Calculate patch lengths from start IDs.
    
    Args:
        patch_start_ids: Tensor with patch start positions, e.g., [0, 1, 7, 7, 7, 7, 7]
        seq_len: Length of the sequence
        
    Returns:
        Patch lengths, e.g., [1, 6] for the example above
    """
    last_ids = torch.full_like(patch_start_ids[:, :1], seq_len - 1)
    patch_end_ids = torch.cat((patch_start_ids[:, 1:] - 1, last_ids), dim=1)
    patch_lengths = patch_end_ids - patch_start_ids + 1
    
    assert torch.all(patch_lengths >= 0), f"Negative patch lengths: {patch_lengths}"
    assert not check_non_zero_after_zero(patch_lengths), f"Invalid patch lengths: {patch_lengths}"
    
    return patch_lengths


def find_entropy_patch_start_ids(
    entropies,
    patch_size=None,
    threshold=None,
    threshold_add=None,
    monotonicity=False,
    include_next_token=True,
):
    """
    Find patch start IDs using entropy values.
    
    Args:
        entropies: Entropy values for each token
        patch_size: Static patch size (if threshold is None)
        threshold: Entropy threshold for dynamic patching
        threshold_add: Additional threshold for monotonicity mode
        monotonicity: Whether to use monotonicity constraint
        include_next_token: Whether to include next token in calculations
        
    Returns:
        Tensor of patch start IDs
    """
    bs, seq_len = entropies.shape[:2]
    
    # First token always starts a patch
    first_ids = torch.tensor([0], dtype=torch.long, device=entropies.device).unsqueeze(0).repeat(bs, 1)
    preds_truncation_len = first_ids.shape[1]
    
    if threshold is None:
        # Static patching: use top-k entropies
        num_patches = seq_len // patch_size
        patch_start_ids = entropies.topk(int(num_patches) - 2, dim=1).indices
        patch_start_ids = patch_start_ids.sort(dim=1).values
    else:
        # Dynamic patching based on threshold
        if monotonicity:
            patch_start_mask = patch_start_mask_from_entropy_with_monotonicity(entropies, threshold)
        elif threshold_add is not None and threshold is not None:
            patch_start_mask = patch_start_mask_global_and_monotonicity(entropies, threshold, threshold_add)
        else:
            patch_start_mask = entropies > threshold
        
        if not include_next_token:
            patch_start_mask = patch_start_mask[:, :-1]
        
        patch_start_ids = patch_start_ids_from_patch_start_mask(patch_start_mask)
    
    patch_start_ids = torch.cat((first_ids, patch_start_ids + preds_truncation_len), dim=1)
    return patch_start_ids


def rightpad(seq, pad_id, max_len):
    """Right-pad a sequence to max_len"""
    return seq + [pad_id] * (max_len - len(seq))


def split_large_numbers(lst, m):
    """Split numbers larger than m into multiple chunks of size m"""
    new_lst = []
    for i in lst:
        if i > m:
            while i > m:
                new_lst.append(m)
                i -= m
            new_lst.append(i)
        else:
            new_lst.append(i)
    
    assert sum(new_lst) == sum(lst), f"Sum mismatch: {sum(new_lst)} != {sum(lst)}"
    return new_lst


class Patcher:
    """
    Patcher for dynamically segmenting token sequences into patches.
    
    Supports both static and entropy-based dynamic patching strategies.
    """
    
    def __init__(self, patcher_args: PatcherArgs):
        self.patcher_args = patcher_args
        self.patching_mode = patcher_args.patching_mode
        self.realtime_patching = patcher_args.realtime_patching
        
        # Entropy model configuration
        self.entropy_model_checkpoint_dir = patcher_args.entropy_model_checkpoint_dir
        self.dataset_name = patcher_args.dataset_name
        self.state_path = os.path.join(
            patcher_args.entropy_model_checkpoint_dir, 
            f"{patcher_args.dataset_name}.pt"
        )
        
        # Device-specific entropy model cache
        self._entropy_models = {}
        
        # Load base entropy model once
        self._base_entropy_model, _ = load_entropy_model(
            self.entropy_model_checkpoint_dir,
            self.state_path,
        )
        
        # Patching parameters
        self.threshold = patcher_args.threshold
        self.threshold_add = patcher_args.threshold_add
        self.max_patch_length = patcher_args.max_patch_length
        self.patch_size = patcher_args.patch_size
        self.patching_batch_size = patcher_args.patching_batch_size
        self.device = patcher_args.device
        self.monotonicity = patcher_args.monotonicity
        
        # Time logging
        self.log_time = patcher_args.log_time
        if self.log_time:
            self.log = defaultdict(float)
    
    def _get_entropy_model_for_device(self, device):
        """
        Get or create entropy model for specific device.
        
        Uses caching to avoid recreating models for the same device.
        """
        device_str = str(device)
        
        if device_str not in self._entropy_models:
            try:
                import copy
                entropy_model = copy.deepcopy(self._base_entropy_model)
                entropy_model = entropy_model.to(device)
                entropy_model.eval()
                
                # Disable gradients for inference
                for param in entropy_model.parameters():
                    param.requires_grad = False
                
                self._entropy_models[device_str] = entropy_model
            except Exception as e:
                print(f"Warning: Could not create entropy model for device {device}: {e}")
                # Fallback to base model
                return self._base_entropy_model
        
        return self._entropy_models[device_str]
    
    def patch(
        self,
        tokens: torch.Tensor,
        include_next_token: bool = False,
        preds: torch.Tensor | None = None,
        entropies: torch.Tensor | None = None,
        threshold: float = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Patch a sequence of tokens.
        
        Args:
            tokens: 2D tensor of shape [batch_size, seq_len]
            include_next_token: Whether to include next token in patch calculations
            preds: Pre-computed predictions (optional)
            entropies: Pre-computed entropies (optional)
            threshold: Override threshold value (optional)
            
        Returns:
            Tuple of (patch_lengths, scores) where:
                - patch_lengths: Tensor of shape [batch_size, max_num_patches]
                - scores: Entropy or other scores (or None for static patching)
        """
        bs, seq_len = tokens.shape
        seq_len_next_tok = seq_len + 1 if include_next_token else seq_len
        scores = None
        
        # STATIC PATCHING
        if self.patching_mode == PatchingModeEnum.static:
            num_patches = math.ceil(seq_len_next_tok / self.patch_size)
            patch_lengths = torch.zeros(
                (bs, num_patches),
                dtype=tokens.dtype,
                device=tokens.device,
            ).fill_(self.patch_size)
            
            # Adjust last patch if sequence doesn't divide evenly
            if seq_len_next_tok % self.patch_size != 0:
                patch_lengths[:, -1] = seq_len_next_tok % self.patch_size
        
        # ENTROPY-BASED DYNAMIC PATCHING
        elif self.patching_mode == PatchingModeEnum.entropy:
            if self.log_time:
                start_time = time.time()
            
            # Get or compute entropies
            if entropies is not None:
                scores = entropies.to(dtype=torch.float32)
            elif preds is not None:
                scores = entropy(preds)
            else:
                scores, _ = calculate_entropies(
                    tokens,
                    self._get_entropy_model_for_device(tokens.device),
                    self.patching_batch_size,
                    tokens.device,
                )
            
            if self.log_time:
                self.log["calculate_entropies"] += time.time() - start_time
                start_time = time.time()
            
            # Find patch start positions
            patch_start_ids = find_entropy_patch_start_ids(
                scores,
                self.patch_size,
                include_next_token=include_next_token,
                threshold=threshold if threshold is not None else self.threshold,
                threshold_add=self.threshold_add,
                monotonicity=self.monotonicity,
            )
            
            if self.log_time:
                self.log["find_entropy_patch_start_ids"] += time.time() - start_time
                start_time = time.time()
            
            # Convert start IDs to lengths
            patch_lengths = patch_lengths_from_start_ids(patch_start_ids, seq_len_next_tok)
            
            if self.log_time:
                self.log["patch_lengths_from_start_ids"] += time.time() - start_time
                start_time = time.time()
        
        # Post-processing: split patches that exceed max length
        if self.max_patch_length is not None:
            patch_lengths = [
                split_large_numbers(pl, self.max_patch_length)
                for pl in patch_lengths.tolist()
            ]
            max_len = max(len(pl) for pl in patch_lengths)
            patch_lengths = [rightpad(pl, 0, max_len=max_len) for pl in patch_lengths]
            patch_lengths = torch.tensor(
                patch_lengths, 
                dtype=tokens.dtype, 
                device=tokens.device
            )
        
        # Validate patch lengths
        assert not check_non_zero_after_zero(patch_lengths), "Invalid patch lengths structure"
        
        # Trim trailing zeros
        last_non_zero_col = (patch_lengths != 0).flip(dims=[1]).int().argmax(dim=1).min()
        patch_lengths = patch_lengths[:, :patch_lengths.shape[1] - last_non_zero_col]
        
        # Verify total length
        expected_total = tokens.numel() + include_next_token * tokens.shape[0]
        actual_total = torch.sum(patch_lengths).item()
        assert actual_total == expected_total, \
            f"Patch length sum mismatch: {actual_total} != {expected_total}"
        
        if self.log_time:
            self.log["postprocessing_patch_lengths"] += time.time() - start_time
            self.log["tokens"] += patch_lengths.sum().item()
        
        return patch_lengths, scores


@lru_cache()
def get_is_torch_run() -> bool:
    """Check if running in distributed torch environment"""
    return os.environ.get("LOCAL_RANK") is not None


@lru_cache()
def get_local_rank() -> int:
    """Get local rank for distributed training"""
    if get_is_torch_run():
        return int(os.environ["LOCAL_RANK"])
    return 0
