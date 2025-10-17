__all__ = ['EntroPE_backbone']

import torch
from torch import nn
import warnings

from layers.RevIN import RevIN
from layers.FlattenHead import Flatten_Head
from bytelatent.model.blt import ByteLatentTransformerArgs, ByteLatentTransformer
from utils.patch_utils import build_tokenizer

warnings.filterwarnings("ignore")


class EntroPE_backbone(nn.Module):
    """
    EntroPE Backbone model combining ByteLatentTransformer with reversible normalization.
    
    Args:
        configs: Configuration object containing model hyperparameters
        pretrain_head: Whether to use pretraining head
        head_type: Type of head ('flatten' or custom)
        individual: Whether to use individual linear layers per variable
        revin: Whether to use reversible instance normalization
        affine: Whether to use affine transformation in RevIN
        subtract_last: Whether to subtract last value in RevIN
    """
    
    def __init__(self, configs, pretrain_head=False, head_type='flatten', 
                 individual=False, revin=True, affine=True, subtract_last=False, **kwargs):
        super().__init__()
        
        # Reversible Instance Normalization
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(configs.enc_in, affine=affine, subtract_last=subtract_last)
        
        # ByteLatentTransformer Configuration
        model_args = self._build_transformer_args(configs)
        self.backbone = ByteLatentTransformer(model_args)
        
        # Tokenizer
        self.tokenizer = build_tokenizer(
            quant_range=configs.quant_range,
            vocab_size=configs.vocab_size,
            context_length=configs.seq_len,
            prediction_length=configs.seq_len
        )
        
        # Prediction Head
        self.head_nf = configs.dim_local_decoder * configs.seq_len
        self.n_vars = configs.enc_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual
        
        self.head = self._build_head(configs)
    
    def _build_transformer_args(self, configs):
        """Build ByteLatentTransformer arguments from configs"""
        return ByteLatentTransformerArgs(
            # Core settings
            seed=configs.random_seed,
            vocab_size=configs.vocab_size,
            max_length=configs.seq_len,
            max_seqlen=configs.seq_len,
            max_encoder_seq_length=configs.seq_len,
            local_attention_window_len=configs.seq_len,
            
            # Model dimensions
            dim_global=configs.dim_global,
            dim_local_encoder=configs.dim_local_encoder,
            dim_local_decoder=configs.dim_local_decoder,
            
            # Layer configurations
            n_layers_global=configs.n_layers_global,
            n_layers_local_encoder=configs.n_layers_local_encoder,
            n_layers_local_decoder=configs.n_layers_local_decoder,
            
            # Attention heads
            n_heads_global=configs.n_heads_global,
            n_heads_local_encoder=configs.n_heads_local_encoder,
            n_heads_local_decoder=configs.n_heads_local_decoder,
            
            # Patching configuration
            patch_size=configs.max_patch_length,
            patch_in_forward=True,
            patching_batch_size=configs.patching_batch_size,
            patching_device="cuda",
            patching_mode="entropy",
            patching_threshold=configs.patching_threshold,
            patching_threshold_add=configs.patching_threshold_add,
            max_patch_length=configs.max_patch_length,
            monotonicity=configs.monotonicity,
            pad_to_max_length=True,
            
            # Cross-attention settings
            cross_attn_encoder=True,
            cross_attn_decoder=True,
            cross_attn_k=configs.cross_attn_k,
            cross_attn_nheads=configs.cross_attn_nheads,
            cross_attn_all_layers_encoder=True,
            cross_attn_all_layers_decoder=True,
            cross_attn_use_flex_attention=False,
            cross_attn_init_by_pooling=True,
            
            # Encoder hash settings
            encoder_hash_byte_group_size=[10],
            encoder_hash_byte_group_vocab=2**4,
            encoder_hash_byte_group_nb_functions=2,
            encoder_enable_byte_ngrams=False,
            
            # Model architecture
            non_linearity="gelu",
            use_rope=True,
            attn_impl="sdpa",
            attn_bias_type="causal",
            multiple_of=configs.multiple_of,
            dropout=configs.dropout,
            
            # Training settings
            layer_ckpt="none",
            init_use_gaussian=True,
            init_use_depth="current",
            alpha_depth="disabled",
            log_patch_lengths=True,
            
            # Dataset and checkpointing
            dataset_name=configs.model_id_name,
            entropy_model_checkpoint_dir=configs.entropy_model_checkpoint_dir,
            downsampling_by_pooling="max",
            use_local_encoder_transformer=True,
            share_encoder_decoder_emb=False
        )
    
    def _build_head(self, configs):
        """Build prediction head based on configuration"""
        if self.pretrain_head:
            return self._create_pretrain_head(self.head_nf, configs.enc_in, configs.fc_dropout)
        elif self.head_type == 'flatten':
            return Flatten_Head(
                individual=self.individual,
                n_vars=self.n_vars,
                nf=self.head_nf,
                target_window=configs.pred_len,
                head_dropout=configs.head_dropout
            )
        else:
            raise ValueError(f"Unknown head_type: {self.head_type}")
    
    def _create_pretrain_head(self, head_nf, n_vars, dropout):
        """Create pretraining head"""
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(head_nf, n_vars, 1)
        )
    
    def forward(self, z):
        """
        Forward pass through the model.
        
        Args:
            z: Input tensor of shape [batch_size, n_vars, seq_len]
            
        Returns:
            Output tensor of shape [batch_size, n_vars, pred_len]
        """
        bs, nvars, seq_len = z.shape
        
        # Apply reversible normalization
        if self.revin:  
            z = z.permute(0, 2, 1)      # [bs, seq_len, nvars]
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)      # [bs, nvars, seq_len]
        
        # Reshape for tokenization
        z = z.reshape(bs * nvars, seq_len)
        
        # Tokenize input
        z, _, _ = self.tokenizer.context_input_transform(z)
        z = z.cuda()
        
        # Pass through backbone
        z = self.backbone(z)            # [bs * nvars, patch_num, d_model]
        
        # Reshape back to batch format
        z = z.view(bs, nvars, z.shape[1], z.shape[2])
        z = z.permute(0, 1, 3, 2)       # [bs, nvars, d_model, patch_num]
        
        # Apply prediction head
        z = self.head(z)                # [bs, nvars, pred_len]
        
        # Apply reversible denormalization
        if self.revin:
            z = z.permute(0, 2, 1)      # [bs, pred_len, nvars]
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)      # [bs, nvars, pred_len]
        
        return z
    