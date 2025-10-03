"""
FusionDecoder module for fusing global and local representations.
"""

from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from bytelatent.model.local_models import LocalModelBase, LocalModelArgs
from bytelatent.model.latent_transformer import CrossAttention
from bytelatent.model.utils import create_causal_mask

RMSNorm = nn.RMSNorm


class FusionDecoder(LocalModelBase):
    """
    Fusion decoder that combines local token representations with global patch embeddings.
    
    Uses cross-attention to inject global context into local representations,
    enabling the model to leverage both local and global information.
    """
    
    def __init__(self, args: LocalModelArgs):
        super().__init__(args)
        
        # Configuration flags
        self.cross_attn_decoder = args.cross_attn_decoder
        self.cross_attn_all_layers_decoder = args.cross_attn_all_layers_decoder
        self.cross_attn_init_by_pooling = args.cross_attn_init_by_pooling
        self.cross_attn_nheads = args.cross_attn_nheads
        
        # Output normalization
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        # Cross-attention layers
        if self.cross_attn_decoder:
            self.cross_attn_layers = nn.ModuleList()
            num_cross_attn_layers = args.n_layers if self.cross_attn_all_layers_decoder else 1
            
            for _ in range(num_cross_attn_layers):
                self.cross_attn_layers.append(
                    CrossAttention(
                        dim=self.dim,
                        head_dim=self.dim // self.cross_attn_nheads,
                        n_heads=self.cross_attn_nheads,
                        n_kv_heads=self.cross_attn_nheads,
                        norm_eps=args.norm_eps,
                    )
                )
    
    def forward(
        self,
        tokens: torch.Tensor,
        embeds: Optional[torch.Tensor],
        patch_embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union[torch.Tensor, str]] = None,
        cross_mask: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """
        Forward pass through the fusion decoder.
        
        Args:
            tokens: Token IDs of shape [batch_size, seq_len]
            embeds: Local token embeddings of shape [batch_size, seq_len, dim]
            patch_embeds: Global patch embeddings for cross-attention
            mask: Attention mask for self-attention
            cross_mask: Attention mask for cross-attention
            cache: KV cache for generation
            
        Returns:
            Tuple of (output_predictions, cache)
        """
        bs, seqlen = tokens.shape
        
        # Embeddings are required
        assert embeds is not None, "Embeddings must be provided to FusionDecoder"
        
        # Create causal mask if not provided
        if mask is None:
            mask = create_causal_mask(
                seqlen,
                self.attn_impl,
                self.attn_bias_type,
                sliding_window=self.sliding_window,
                tokens=tokens,
                eos_id=self.eos_id,
            )
        
        # Start with provided embeddings
        h = embeds
        
        # Add positional embeddings
        device = h.device
        pos_ids = torch.arange(seqlen, device=device).unsqueeze(0)  # [1, seq_len]
        pos_emb = self.pos_embeddings(pos_ids)                      # [1, seq_len, dim]
        h = h + pos_emb
        
        # Apply dropout
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Process through layers with cross-attention
        for i, layer in enumerate(self.layers):
            # Apply cross-attention at first layer or all layers based on config
            if self.cross_attn_decoder and (
                i == 0 or self.cross_attn_all_layers_decoder
            ):
                # Cross-attention: h attends to patch_embeds
                cross_attn_layer_idx = i if self.cross_attn_all_layers_decoder else 0
                h_cross = self.cross_attn_layers[cross_attn_layer_idx](
                    x=h,
                    kv=patch_embeds,
                    mask=cross_mask,
                )
                h = h + h_cross
            
            # Self-attention within decoder
            h = layer(h, mask=mask, freq_cis=None, attn_impl=self.attn_impl)
        
        # Final normalization
        h_preds = self.norm(h)
        h_preds = F.dropout(h_preds, p=self.dropout, training=self.training)
        h_preds = h_preds.float()
        
        return h_preds, cache