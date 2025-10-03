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
    # Create a new config with prediction_length=1
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

    # Create a new tokenizer with the updated config
    tokenizer = MeanScaleUniformBins(low_limit, high_limit, tokenizer_config)
    return tokenizer