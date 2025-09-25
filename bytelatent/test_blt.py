# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from bytelatent.constants import BLT_DATA
from bytelatent.data.data_types import Batch
from bytelatent.data.ngram_processor import NgramProcessor
from bytelatent.model.blt import (
    ByteLatentTransformer,
    ByteLatentTransformerArgs,
    EmbeddingType,
    compute_hash_embeddings,
    create_global_transformer,
    create_local_decoder,
    create_local_encoder,
    cross_attn_mask,
    decoder_patch_ids_from_lengths,
    get_blt_input,
    init_embeddings,
    patch_ids_from_lengths,
)
from bytelatent.model.latent_transformer import CrossAttention
from bytelatent.model.utils import create_causal_mask
from bytelatent.optim import OptimArgs, build_optimizer
from bytelatent.tokenizers.constants import EOS_ID
from bytelatent.train import compute_loss


def batch_to_tensors_and_gpu(batch):
    x = torch.from_numpy(batch.x)
    y = torch.from_numpy(batch.y)
    mask = None if batch.mask is None else torch.from_numpy(batch.mask)
    patch_lengths = (
        None if batch.patch_lengths is None else torch.from_numpy(batch.patch_lengths)
    )
    ngram_ids = None if batch.ngram_ids is None else torch.from_numpy(batch.ngram_ids)

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        if mask is not None:
            mask = mask.cuda()
        if patch_lengths is not None:
            patch_lengths = patch_lengths.cuda()
        if ngram_ids is not None:
            ngram_ids = ngram_ids.cuda()
    return x, y, mask, patch_lengths, ngram_ids


def fake_batch():
    batch_dict = torch.load(os.path.join(BLT_DATA, "test_batch.pt"), weights_only=False)
    del batch_dict["x2"]
    del batch_dict["y2"]
    del batch_dict["src_names"]
    return Batch(**batch_dict)


def create_args(cross_attention=False):
    transformer_args = ByteLatentTransformerArgs(
        # Base args provided
        n_heads=8,
        dim=512,
        vocab_size=260,
        # Additional args from command line
        dim_token=256,
        patch_size=6,
        patching_mode="space",
        tie_local_encoder_decoder_logits=False,
        patch_in_forward=False,
        max_encoder_seq_length=12288,
        pad_to_max_length=True,
        encoder_lm_loss=False,
        patching_threshold=3.1439168453216553,
        encoder_hash_byte_group_size=[4],
        encoder_hash_byte_group_vocab=50002,
        encoder_hash_byte_group_nb_functions=3,
        cross_attn_encoder=cross_attention,  # True,
        cross_attn_decoder=cross_attention,  # True,
        cross_attn_window_encoder=512,
        cross_attn_window_decoder=512,
        dim_local_encoder=256,
        dim_local_decoder=256,
        cross_attn_k=8,
        cross_attn_nheads=4,
        cross_attn_all_layers_decoder=True,
        cross_attn_all_layers_encoder=True,
        cross_attn_use_flex_attention=True,
        cross_attn_init_by_pooling=True,
        log_patch_lengths=True,
        non_linearity="swiglu",
        use_rope=True,
        recompute_fc1_out=False,
        recompute_fc3_out=False,
        recompute_attn=False,
        custom_bwd=False,
        layer_ckpt="none",
        use_local_encoder_transformer=True,
        init_use_gaussian=True,
        init_use_depth="current",
        attn_bias_type="block_causal",
        attn_impl="xformers",
        alpha_depth="disabled",
        max_length=256,
        local_attention_window_len=512,
        max_seqlen=12288,
        downsampling_by_pooling="max",
        eos_id=EOS_ID,
    )
    return transformer_args


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
class TestByteLatentTransformer:
    def test_local_encoder(self):
        args = create_args()
        device = torch.device("cuda")
        local_encoder = create_local_encoder(args).to(device)

        batch = fake_batch()
        tokens, _, _, patch_lengths, _ = batch_to_tensors_and_gpu(batch)

        local_encoder_tokens, _, _ = get_blt_input(
            tokens=tokens,
            enforce_patch_size_multiple=False,
            nb_boe=0,
            patch_size=local_encoder.patch_size,
            boe_id=local_encoder.boe_id,
        )

        patch_ids = patch_ids_from_lengths(
            patch_lengths, local_encoder_tokens.shape[-1]
        )

        encoder_hash_tok_embedding = init_embeddings(
            args,
            EmbeddingType.HASH_TOK,
            local_encoder_dim=local_encoder.dim,
            encoder_hash_byte_group_size=args.encoder_hash_byte_group_size,
        ).to(device)

        local_encoder_embeds = compute_hash_embeddings(
            local_encoder_tokens=local_encoder_tokens,
            local_encoder=local_encoder,
            encoder_hash_tok_embedding=encoder_hash_tok_embedding,
            encoder_hash_byte_group_nb_functions=args.encoder_hash_byte_group_nb_functions,
            encoder_hash_byte_group_size=args.encoder_hash_byte_group_size,
            encoder_hash_byte_group_vocab=args.encoder_hash_byte_group_vocab,
        )

        reference_path = os.path.join(BLT_DATA, "local_encoder_tokens.pt")
        reference_tokens = torch.load(reference_path).to(device)
        torch.testing.assert_close(
            local_encoder_tokens,
            reference_tokens,
            msg="Generated tokens don't match reference tokens",
        )

        (h_encoder, h_cross), cache_encoder = local_encoder(
            tokens=local_encoder_tokens,
            embeds=local_encoder_embeds,
            patch_embeds=None,
            cross_mask=None,
            num_patches=patch_lengths.shape[1],
            patch_ids=patch_ids,
        )

        assert h_encoder is not None
        assert h_cross is None
        assert cache_encoder is None

        expected_shape = (
            local_encoder_tokens.shape[0],
            local_encoder_tokens.shape[1],
            local_encoder.dim,
        )
        assert h_encoder.shape == expected_shape

    def test_local_encoder_cross_attention(self):
        args = create_args(cross_attention=True)
        device = torch.device("cuda")
        local_encoder = create_local_encoder(args).to(device)

        batch = fake_batch()
        tokens, _, _, patch_lengths, _ = batch_to_tensors_and_gpu(batch)

        local_encoder_tokens, _, _ = get_blt_input(
            tokens=tokens,
            enforce_patch_size_multiple=False,
            nb_boe=0,
            patch_size=local_encoder.patch_size,
            boe_id=local_encoder.boe_id,
        )

        patch_ids = patch_ids_from_lengths(
            patch_lengths, local_encoder_tokens.shape[-1]
        )

        encoder_hash_tok_embedding = init_embeddings(
            args,
            EmbeddingType.HASH_TOK,
            local_encoder_dim=local_encoder.dim,
            encoder_hash_byte_group_size=args.encoder_hash_byte_group_size,
        ).to(device)

        cross_attn_mask_enc = cross_attn_mask(
            patch_ids,
            patch_lengths,
            local_encoder_tokens.shape[-1],
            patches_as_queries=True,
            cross_attn_k=args.cross_attn_k,
            window=args.cross_attn_window_encoder,
            block_mask=True,
        )

        local_encoder_embeds = compute_hash_embeddings(
            local_encoder_tokens=local_encoder_tokens,
            local_encoder=local_encoder,
            encoder_hash_tok_embedding=encoder_hash_tok_embedding,
            encoder_hash_byte_group_nb_functions=args.encoder_hash_byte_group_nb_functions,
            encoder_hash_byte_group_size=args.encoder_hash_byte_group_size,
            encoder_hash_byte_group_vocab=args.encoder_hash_byte_group_vocab,
        )
        (h_encoder, h_cross), cache_encoder = local_encoder(
            tokens=local_encoder_tokens,
            embeds=local_encoder_embeds,
            patch_embeds=None,
            cross_mask=cross_attn_mask_enc,
            num_patches=patch_lengths.shape[1],
            patch_ids=patch_ids,
        )
        assert h_encoder is not None
        assert h_cross is not None
        assert cache_encoder is None
        expected_shape = (
            local_encoder_tokens.shape[0],
            local_encoder_tokens.shape[1],
            local_encoder.dim,
        )
        assert h_encoder.shape == expected_shape
        assert h_cross.shape == (2, 2048, local_encoder.dim)

    def test_local_decoder_cross_attention(self):
        args = create_args(cross_attention=True)
        device = torch.device("cuda")
        local_decoder = create_local_decoder(args).to(device)

        test_files = {
            "dec_embeds": "dec_embeds.pt",
            "decoder_tokens": "local_decoder_tokens.pt",
            "patch_embeds": "decoder_patch_cross_embeds.pt",
        }
        batch = fake_batch()
        _, _, _, patch_lengths, _ = batch_to_tensors_and_gpu(batch)

        tensors = {
            name: torch.load(os.path.join(BLT_DATA, filename)).float().to(device)
            for name, filename in test_files.items()
        }
        decoder_patch_ids = decoder_patch_ids_from_lengths(
            patch_lengths, 0, tensors["decoder_tokens"].shape[-1]
        )
        cross_attn_mask_dec = cross_attn_mask(
            decoder_patch_ids,
            patch_lengths,
            tensors["decoder_tokens"].shape[-1],
            patches_as_queries=False,
            cross_attn_k=args.cross_attn_k,
            window=args.cross_attn_window_decoder,
            block_mask=True,
        )
        output, _ = local_decoder(
            embeds=tensors["dec_embeds"],
            patch_embeds=tensors["patch_embeds"],
            tokens=tensors["decoder_tokens"],
            cross_mask=cross_attn_mask_dec,
            cache=None,
        )
        assert output is not None
        assert output.shape == (2, tensors["decoder_tokens"].shape[1], args.vocab_size)

    def test_local_decoder(self):
        args = create_args()
        device = torch.device("cuda")

        local_decoder = create_local_decoder(args).to(device)

        test_files = {
            "dec_embeds": "dec_embeds.pt",
            "decoder_tokens": "local_decoder_tokens.pt",
            "patch_embeds": "decoder_patch_embeds.pt",
        }

        tensors = {
            name: torch.load(os.path.join(BLT_DATA, filename)).float().to(device)
            for name, filename in test_files.items()
        }

        output, cache_decoder = local_decoder(
            embeds=tensors["dec_embeds"],
            patch_embeds=tensors["patch_embeds"],
            tokens=tensors["decoder_tokens"],
            cross_mask=None,
            cache=None,
        )
        assert output is not None
        expected_shape = (
            tensors["decoder_tokens"].shape[0],
            tensors["decoder_tokens"].shape[1],
            args.vocab_size,
        )
        assert output.shape == expected_shape
        assert cache_decoder is None

    def test_global_transformer(self):
        args = create_args()
        device = torch.device("cuda")
        global_transformer = create_global_transformer(args).to(device)

        test_files = {
            "global_embeds": "global_embeds.pt",
            "global_tokens": "global_tokens.pt",
        }
        tensors = {
            name: torch.load(os.path.join(BLT_DATA, filename)).float().to(device)
            for name, filename in test_files.items()
        }
        h, cache = global_transformer(
            embeds=tensors["global_embeds"], tokens=tensors["global_tokens"]
        )
        h is not None
        assert h.shape == (2, 256, 512)
        assert cache is None

    def test_blt_transformer_init(self):
        args = create_args()
        model = ByteLatentTransformer(args)
        assert model is not None

    @pytest.mark.parametrize("attn_impl", ["sdpa", "xformers"])
    def test_blt_transformer_forward(self, attn_impl):
        args = create_args()
        if attn_impl == "sdpa":
            os.environ["BLT_SUPPRESS_ATTN_ERROR"] = "1"
        else:
            os.environ["BLT_SUPPRESS_ATTN_ERROR"] = "0"

        args = args.model_copy(update=dict(attn_impl=attn_impl))
        model = ByteLatentTransformer(args)
        model = model.cuda()
        batch = fake_batch()
        x, _, _, patch_lengths, ngram_ids = batch_to_tensors_and_gpu(batch)

        output = model(
            tokens=x,
            patch_lengths=patch_lengths,
            ngram_ids=ngram_ids,
        )
        assert output is not None
        expected_shape = (
            x.shape[0],
            x.shape[1],
            args.vocab_size,
        )
        assert output.shape == expected_shape

    def test_blt_transformer_cross_attn_forward(self):
        args = create_args(cross_attention=True)
        model = ByteLatentTransformer(args)
        model = model.cuda()
        batch = fake_batch()
        x, y, mask, patch_lengths, ngram_ids = batch_to_tensors_and_gpu(batch)

        output = model(
            tokens=x,
            patch_lengths=patch_lengths,
            ngram_ids=ngram_ids,
        )
        assert output is not None
        expected_shape = (
            x.shape[0],
            x.shape[1],
            args.vocab_size,
        )
        assert output.shape == expected_shape

    def test_cross_attention_rand(self):
        x = torch.randn(2, 256, 512, device="cuda")
        kv = torch.randn(2, 256, 512, device="cuda")
        cross_attention = CrossAttention(
            dim=512,
            head_dim=64,
            n_heads=8,
            n_kv_heads=4,
            norm_eps=1e-6,
        ).to("cuda")
        mask = create_causal_mask(
            x.shape[1], "flex_attention", None, sliding_window=None
        )
        output = cross_attention(x, kv, mask)
        assert output is not None
        assert output.shape == (2, 256, 512)

    def test_ngram_embeddings(self):
        ngram_to_size = {
            2: 38396,
            3: 50000,
            4: 50000,
            5: 50000,
            6: 50000,
            7: 50000,
            8: 50000,
        }
        batch = fake_batch()
        ngram_processor = NgramProcessor(BLT_DATA, ngram_to_size)
        ngram_ids = ngram_processor.encode_token_ngrams(batch.x)
        ngram_ids = np.stack(ngram_ids, axis=0)
        batch = replace(batch, ngram_ids=ngram_ids)
        args = create_args(cross_attention=True)
        args = args.model_copy(
            update=dict(
                encoder_ngram_to_size_str="2:38396,3:50000,4:50000,5:50000,6:50000,7:50000,8:50000",
                encoder_enable_byte_ngrams=True,
                ngram_vocab_sizes=ngram_processor.ngram_vocab_sizes,
            )
        )
        model = ByteLatentTransformer(args)
        model = model.cuda()
        x, _, _, patch_lengths, ngram_ids = batch_to_tensors_and_gpu(batch)

        output = model(
            tokens=x,
            patch_lengths=patch_lengths,
            ngram_ids=ngram_ids,
        )
        assert output is not None
        expected_shape = (
            x.shape[0],
            x.shape[1],
            args.vocab_size,
        )
        assert output.shape == expected_shape

    def test_loss_backward(self):
        args = create_args()
        args = args.model_copy(update=dict(attn_impl="xformers"))
        batch = fake_batch()
        model = ByteLatentTransformer(args)
        steps = 10
        optimizer, scheduler = build_optimizer(model, OptimArgs(lr=4e-04), steps)
        model = model.cuda()
        x, y, mask, patch_lengths, ngram_ids = batch_to_tensors_and_gpu(batch)

        initial_loss = None
        final_loss = None
        for step in range(steps):
            output = model(
                tokens=x,
                patch_lengths=patch_lengths,
                ngram_ids=ngram_ids,
            )
            loss, _ = compute_loss(output, y, mask, 1.0)
            if step == 0:
                initial_loss = loss.item()
            if step == steps - 1:
                final_loss = loss.item()
            prev_loss = loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        assert (
            final_loss < initial_loss
        ), f"Training did not reduce loss: initial {initial_loss}, final {final_loss}"
