"""
APE (Adaptive Patch Encoder) module for local encoding with cross-attention.
"""

from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from bytelatent.model.local_models import LocalModelBase, LocalModelArgs
from bytelatent.model.latent_transformer import CrossAttention
from bytelatent.model.utils import create_causal_mask, downsample


class APE(LocalModelBase):
    """
    Adaptive Patch Encoder with token embeddings and optional cross-attention.
    
    Processes token sequences with transformer layers and optional cross-attention
    to global patch embeddings.
    """
    
    def __init__(self, args: LocalModelArgs):
        super().__init__(args)
        
        # Configuration flags
        self.apply_transformer = args.use_local_encoder_transformer
        self.downsampling_by_pooling = args.downsampling_by_pooling
        self.expects_hash_embeddings = args.encoder_hash_byte_group_size is not None
        self.cross_attn_encoder = args.cross_attn_encoder
        self.cross_attn_all_layers_encoder = args.cross_attn_all_layers_encoder
        self.cross_attn_init_by_pooling = args.cross_attn_init_by_pooling
        self.cross_attn_nheads = args.cross_attn_nheads
        
        # Token embeddings
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        
        # Cross-attention layers
        if self.cross_attn_encoder:
            self.cross_attn_layers = nn.ModuleList()
            num_cross_attn_layers = args.n_layers if self.cross_attn_all_layers_encoder else 1
            
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
    
    def apply_embedding(self, tokens: torch.Tensor, embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply token embeddings.
        
        Args:
            tokens: Token IDs of shape [batch_size, seq_len]
            embeds: Optional pre-computed embeddings (unused)
            
        Returns:
            Token embeddings of shape [batch_size, seq_len, dim]
        """
        return self.tok_embeddings(tokens)
    
    def forward(
        self,
        tokens: torch.Tensor,
        embeds: Optional[torch.Tensor] = None,
        patch_embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union[torch.Tensor, str]] = None,
        cross_mask: Optional[torch.Tensor] = None,
        num_patches: Optional[int] = None,
        patch_ids: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ) -> Tuple[Tuple[torch.Tensor, Optional[torch.Tensor]], Optional[List]]:
        """
        Forward pass through the APE encoder.
        
        Args:
            tokens: Input tokens of shape [batch_size, seq_len]
            embeds: Optional pre-computed embeddings
            patch_embeds: Optional global patch embeddings
            mask: Attention mask
            cross_mask: Cross-attention mask
            num_patches: Number of patches for downsampling
            patch_ids: Patch boundary indices
            cache: KV cache for generation
            
        Returns:
            Tuple of ((hidden_states, residual_patch_embeds), cache)
        """
        bs, seqlen = tokens.shape
        
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
        
        # Token embeddings
        h = self.apply_embedding(tokens, embeds)
        
        # Add positional information
        if self.use_rope:
            freqs_cis = self.rope(seqlen=seqlen)
        else:
            # Learned positional embeddings
            device = h.device
            pos_ids = torch.arange(seqlen, device=device).unsqueeze(0)  # [1, seq_len]
            pos_emb = self.pos_embeddings(pos_ids)                      # [1, seq_len, dim]
            h = h + pos_emb
            freqs_cis = None
        
        # Apply dropout
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            h = layer(h, mask=mask, freq_cis=freqs_cis, attn_impl=self.attn_impl)
            
            # Apply cross-attention at specified layers
            if self.cross_attn_encoder and (
                i == len(self.layers) - 1 or self.cross_attn_all_layers_encoder
            ):
                patch_embeds = self.apply_cross_attention(
                    h=h,
                    patch_embeds=patch_embeds,
                    layer_idx=i,
                    bs=bs,
                    num_patches=num_patches,
                    patch_ids=patch_ids,
                    cross_mask=cross_mask,
                )
        
        # Return hidden states and residual patch embeddings
        h_residual = patch_embeds if self.cross_attn_encoder else None
        return (h, h_residual), cache
    
    def apply_cross_attention(
        self,
        h: torch.Tensor,
        patch_embeds: Optional[torch.Tensor],
        layer_idx: int,
        bs: int,
        num_patches: Optional[int],
        patch_ids: Optional[torch.Tensor],
        cross_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply cross-attention between token representations and patch embeddings.
        
        Args:
            h: Token hidden states of shape [batch_size, seq_len, dim]
            patch_embeds: Global patch embeddings (or None to initialize)
            layer_idx: Current layer index
            bs: Batch size
            num_patches: Number of patches
            patch_ids: Patch boundary indices
            cross_mask: Cross-attention mask
            
        Returns:
            Updated patch embeddings
        """
        # Initialize patch embeddings by pooling if needed
        if self.cross_attn_init_by_pooling and patch_embeds is None:
            patch_embeds = downsample(
                h,
                num_patches,
                patch_ids=patch_ids,
                downsampling_by_pooling=self.downsampling_by_pooling,
                patch_size=self.patch_size,
            )
            
            # Apply optional projection
            if self.patch_embedding_projection is not None:
                patch_embeds = self.patch_embedding_projection(patch_embeds)
                patch_embeds = patch_embeds.reshape(
                    bs, patch_embeds.shape[1] * self.cross_attn_k, self.dim
                )
        
        # Select appropriate cross-attention layer
        cross_attn_layer_idx = layer_idx if self.cross_attn_all_layers_encoder else 0
        
        # Apply cross-attention: patch_embeds attend to token representations
        patch_embeds_cross = self.cross_attn_layers[cross_attn_layer_idx](
            x=patch_embeds,
            kv=h,
            mask=cross_mask,
        )
        
        # Residual connection
        return patch_embeds + patch_embeds_cross
    