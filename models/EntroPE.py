"""EntroPE Model for Time Series Forecasting"""

__all__ = ['Model']

from typing import Optional
import torch
from torch import nn
from torch import Tensor

from layers.EntroPE_backbone import EntroPE_backbone


class Model(nn.Module):
    """
    EntroPE model wrapper that handles input/output transformations
    and optional decomposition.
    
    Args:
        configs: Configuration object with model hyperparameters
        pretrain_head: Whether to use pretraining head
        head_type: Type of prediction head ('flatten' or custom)
        verbose: Whether to print model information
        **kwargs: Additional keyword arguments passed to backbone
    """
    
    def __init__(self, configs, max_seq_len=1024, d_k=None, d_v=None, 
                 norm='BatchNorm', attn_dropout=0., act="gelu", 
                 key_padding_mask='auto', padding_var=None, attn_mask=None, 
                 res_attention=True, pre_norm=False, store_attn=False, 
                 pe='zeros', learn_pe=True, pretrain_head=False, 
                 head_type='flatten', verbose=False, **kwargs):
        super().__init__()
        
        # Extract configuration parameters
        self.decomposition = configs.decomposition
        
        # Build EntroPE backbone
        self.model = EntroPE_backbone(
            configs=configs,
            pretrain_head=pretrain_head,
            head_type=head_type,
            individual=configs.individual,
            revin=configs.revin,
            affine=configs.affine,
            subtract_last=configs.subtract_last,
            **kwargs
        )
        
        if verbose:
            self._print_model_info(configs)
    
    def _print_model_info(self, configs):
        """Print model configuration information"""
        print(f"EntroPE Model Configuration:")
        print(f"  Input channels: {configs.enc_in}")
        print(f"  Context window: {configs.seq_len}")
        print(f"  Prediction window: {configs.pred_len}")
        print(f"  Decomposition: {self.decomposition}")
        print(f"  RevIN: {configs.revin}")
        print(f"  Individual: {configs.individual}")
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_vars]
            
        Returns:
            Output tensor of shape [batch_size, pred_len, n_vars]
        """
        # Transform: [batch_size, seq_len, n_vars] -> [batch_size, n_vars, seq_len]
        x = x.permute(0, 2, 1)
        
        # Pass through backbone
        x = self.model(x)
        
        # Transform back: [batch_size, n_vars, pred_len] -> [batch_size, pred_len, n_vars]
        x = x.permute(0, 2, 1)
        
        return x
    