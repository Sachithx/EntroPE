import warnings
warnings.filterwarnings("ignore")
from bytelatent.model.blt import ByteLatentTransformerArgs, ByteLatentTransformer
from utils.patch_utils import build_tokenizer

## Set backbone
class CAPE_TST_backbone():
    def __init__(self, configs):
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
    
    def place(self):
        return self.backbone, self.tokenizer
