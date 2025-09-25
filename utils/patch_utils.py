import numpy as np
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
import os
import argparse

from chronos import MeanScaleUniformBins, ChronosConfig
from pathlib import Path

def build_tokenizer(quant_range, vocab_size, context_length, prediction_length):
    """
    Build a tokenizer that maps byte values to tokens.
    """
    low_limit = -1 * quant_range
    high_limit = quant_range

    tokenizer_config = ChronosConfig(
        tokenizer_class='MeanScaleUniformBins',
        tokenizer_kwargs={'low_limit': low_limit, 'high_limit': high_limit},
        context_length=context_length,
        prediction_length=prediction_length,   
        n_tokens=vocab_size,
        n_special_tokens=4,
        pad_token_id=-1,
        eos_token_id=0,
        use_eos_token=False,
        model_type='causal',
        num_samples=1,
        temperature=1.0,
        top_k=50,
        top_p=1.0
    )

    tokenizer = MeanScaleUniformBins(low_limit, high_limit, tokenizer_config)
    
    # Store original tensors on CPU for multi-GPU compatibility
    cpu_boundaries = tokenizer.boundaries.cpu()
    cpu_centers = tokenizer.centers.cpu()
    
    def patched_input_transform(context, scale=None):
        context = context.to(dtype=torch.float32)
        attention_mask = ~torch.isnan(context)
        
        # Create device-specific tensors for this call (don't modify original)
        device = context.device
        boundaries = cpu_boundaries.to(device)
        centers = cpu_centers.to(device)
        
        # Continue with original logic
        if scale is None:
            scale = torch.nansum(
                torch.abs(context) * attention_mask, dim=-1
            ) / torch.nansum(attention_mask, dim=-1)
            scale[~(scale > 0)] = 1.0

        scaled_context = context / scale.unsqueeze(dim=-1)
        token_ids = (
            torch.bucketize(
                input=scaled_context,
                boundaries=boundaries,  # Use local boundaries
                right=True,
            )
            + tokenizer.config.n_special_tokens
        )

        token_ids.clamp_(0, tokenizer.config.n_tokens - 1)
        token_ids[~attention_mask] = tokenizer.config.pad_token_id

        return token_ids, attention_mask, scale
    
    def patched_output_transform(samples, scale):
        device = samples.device
        centers = cpu_centers.to(device)
        
        scale_unsqueezed = scale.unsqueeze(-1).unsqueeze(-1)
        indices = torch.clamp(
            samples - tokenizer.config.n_special_tokens - 1,
            min=0,
            max=len(centers) - 1,
        )
        return centers[indices] * scale_unsqueezed
    
    # Replace the methods
    tokenizer._input_transform = patched_input_transform
    tokenizer.output_transform = patched_output_transform
    
    return tokenizer