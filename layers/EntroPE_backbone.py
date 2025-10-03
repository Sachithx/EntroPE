__all__ = ['EntroPE_backbone']

# Cell
# from typing import Callable, Optional
import torch
from torch import nn
# from torch import Tensor
# import torch.nn.functional as F
# import numpy as np
from layers.RevIN import RevIN
# from patcher_backbone import CAPE_TST_backbone
import warnings
warnings.filterwarnings("ignore")
from bytelatent.model.blt import ByteLatentTransformerArgs, ByteLatentTransformer
from utils.patch_utils import build_tokenizer

# Cell
class EntroPE_backbone(nn.Module):
    def __init__(self, configs, pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, 
                 subtract_last = False, **kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(configs.enc_in, affine=affine, subtract_last=subtract_last)
        # -----------------------------------------------
        model_args = ByteLatentTransformerArgs(
            seed=configs.random_seed,
            vocab_size=configs.vocab_size,                       # Small byte-level vocab
            max_length=configs.seq_len,                        # Max full sequence length
            max_seqlen=configs.seq_len,
            max_encoder_seq_length=configs.seq_len,
            local_attention_window_len=configs.seq_len,        # Local window, 128 is sufficient for small models

            dim_global=configs.dim_global,                        # Lower than default 512
            dim_local_encoder=configs.dim_local_encoder,
            dim_local_decoder=configs.dim_local_decoder,

            n_layers_global=configs.n_layers_global,
            n_layers_local_encoder=configs.n_layers_local_encoder,
            n_layers_local_decoder=configs.n_layers_local_decoder,

            n_heads_global=configs.n_heads_global,                      # Reduce heads
            n_heads_local_encoder=configs.n_heads_local_encoder,
            n_heads_local_decoder=configs.n_heads_local_decoder,

            patch_size=configs.max_patch_length,
            patch_in_forward=True,                # Patch in forward pass
            patching_batch_size=configs.patching_batch_size,
            patching_device="cuda",               # Use CPU for patching in small model
            patching_mode="entropy",
            patching_threshold=configs.patching_threshold,
            patching_threshold_add=configs.patching_threshold_add,           # No additional threshold
            max_patch_length=configs.max_patch_length,
            monotonicity=configs.monotonicity,            # Monotonic patching
            pad_to_max_length=True,

            cross_attn_encoder=True,
            cross_attn_decoder=True,
            cross_attn_k=configs.cross_attn_k,
            cross_attn_nheads=configs.cross_attn_nheads,
            cross_attn_all_layers_encoder=True,
            cross_attn_all_layers_decoder=True,
            cross_attn_use_flex_attention=False,
            cross_attn_init_by_pooling=True,

            encoder_hash_byte_group_size=[10],   # Fewer hash sizes
            encoder_hash_byte_group_vocab=2**4,
            encoder_hash_byte_group_nb_functions=2,
            encoder_enable_byte_ngrams=False,

            non_linearity="gelu",
            use_rope=False,
            attn_impl="sdpa",                      # Efficient PyTorch attention
            attn_bias_type="causal",

            multiple_of=configs.multiple_of,                     # Multiple of 128 for efficient attention
            dropout=configs.dropout,                # Lower dropout
            layer_ckpt="none",                     # No checkpointing in small model
            init_use_gaussian=True,
            init_use_depth="current",
            alpha_depth="disabled",
            log_patch_lengths=True,

            dataset_name=configs.model_id_name,  # Dataset name for patching
            entropy_model_checkpoint_dir=configs.entropy_model_checkpoint_dir,  # Directory for entropy model checkpoint
            downsampling_by_pooling="max",         # Efficient downsampling
            use_local_encoder_transformer=True,
            share_encoder_decoder_emb=False         # Save memory if possible
        )

        self.backbone = ByteLatentTransformer(model_args)
        self.tokenizer = build_tokenizer(
            quant_range=configs.quant_range,
            vocab_size=configs.vocab_size,
            context_length=configs.seq_len,
            prediction_length=configs.seq_len
        )
        # self.backbone, self.tokenizer = CAPE_TST_backbone(configs).place()
        # -------------- model replace end --------------

        # Head
        self.head_nf = configs.dim_local_decoder * configs.seq_len  # number of input components for head (patches)
        self.n_vars = configs.enc_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, configs.enc_in, configs.fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, configs.pred_len, head_dropout=configs.head_dropout)
        
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        bs, nvars, seq_len = z.shape
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)                                                            # z: [bs x nvars x seq_len]

        # -----------------------------------------------
        z = z.reshape(bs * nvars, seq_len)
        z, _, _ = self.tokenizer.context_input_transform(z)
        z = z.cuda()
        z = self.backbone(z)                                                                    # z: [bs * nvars x patch_num x d_model]

        z = z.view(bs, nvars, z.shape[1], z.shape[2]).permute(0, 1, 3, 2)                       # z: [bs x nvars x d_model x patch_num]
        # -------------- model replace end --------------

        z = self.head(z)                                                                        # z: [bs x nvars x target_window] 
    
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
