"""
GlobalTransformer module for processing global patch-level representations.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from bytelatent.model.latent_transformer import BaseTransformer, BaseTransformerArgs
from bytelatent.model.utils import create_causal_mask


class GlobalTransformer(BaseTransformer):
    """
    Global transformer that processes patch-level embeddings.
    
    Extends BaseTransformer with optional token embedding projection and
    positional embeddings for global-level sequence processing.
    """
    
    def __init__(self, args: BaseTransformerArgs):
        super().__init__(args)
        
        self.dropout = args.dropout
        self.eos_id = args.eos_id
        self.dim_token_emb = args.dim_token_emb
        
        # Optional projection from token embedding dimension to model dimension
        self.token_embedding_projection = None
        if args.dim_token_emb is not None and args.dim_token_emb != self.dim:
            self.token_embedding_projection = nn.Linear(
                args.dim_token_emb,
                args.dim,
                bias=False,
            )
        
        # Positional embeddings
        self.pos_embeddings = nn.Embedding(self.max_seqlen, args.dim)
        
        print(f"[Global] Token embedding projection: {self.token_embedding_projection}")
    
    def forward(
        self,
        tokens: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union[torch.Tensor, str]] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """
        Forward pass through the global transformer.
        
        Args:
            tokens: Token IDs of shape [batch_size, seq_len]
            tok_idx: Optional token position indices
            embeds: Pre-computed embeddings of shape [batch_size, seq_len, dim_token_emb]
            mask: Attention mask
            cache: KV cache for generation
            
        Returns:
            Tuple of (hidden_states, cache)
        """
        bs, seqlen = tokens.shape
        
        # Use provided embeddings
        h = embeds
        
        # Add positional embeddings
        pos_ids = torch.arange(seqlen, device=h.device).unsqueeze(0)  # [1, seq_len]
        pos_emb = self.pos_embeddings(pos_ids)                        # [1, seq_len, dim]
        h = h + pos_emb
        
        # Create causal mask if not provided
        if mask is None:
            mask = create_causal_mask(
                seqlen,
                self.attn_impl,
                self.attn_bias_type,
                tokens=tokens,
                eos_id=self.eos_id,
            )
        
        # Project to model dimension if needed
        if self.token_embedding_projection is not None and h.shape[-1] != self.dim:
            h = self.token_embedding_projection(h)
        
        # Apply dropout
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Process through transformer layers
        h = super().forward(h, tok_idx=tok_idx, mask=mask, attn_impl=self.attn_impl)
        
        return h, cache
    
    def init_weights(self):
        """
        Initialize model weights.
        
        Uses truncated normal initialization for the token embedding projection
        with std based on the token embedding dimension.
        """
        super().init_weights()
        
        if self.token_embedding_projection is not None:
            std = self.dim_token_emb ** (-0.5)
            nn.init.trunc_normal_(
                self.token_embedding_projection.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
