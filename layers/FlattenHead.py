import torch
import torch.nn as nn


class Flatten_Head(nn.Module):
    """
    Flattening head for converting patch representations to predictions.
    
    Args:
        individual: Whether to use separate linear layers per variable
        n_vars: Number of variables
        nf: Number of input features (flattened dimension)
        target_window: Prediction length
        head_dropout: Dropout probability
    """
    
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.flattens = nn.ModuleList([nn.Flatten(start_dim=-2) for _ in range(n_vars)])
            self.linears = nn.ModuleList([nn.Linear(nf, target_window) for _ in range(n_vars)])
            self.dropouts = nn.ModuleList([nn.Dropout(head_dropout) for _ in range(n_vars)])
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
    
    def forward(self, x):
        """
        Forward pass through the head.
        
        Args:
            x: Input tensor of shape [bs, nvars, d_model, patch_num]
            
        Returns:
            Output tensor of shape [bs, nvars, target_window]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])     # [bs, d_model * patch_num]
                z = self.linears[i](z)                  # [bs, target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)               # [bs, nvars, target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        
        return x