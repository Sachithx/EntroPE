# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

import torch

from bytelatent.constants import BLT_DATA
from bytelatent.data.iterators.arrow_iterator import ArrowFileIteratorState
from bytelatent.data.iterators.preprocess_iterator import PreprocessIterator
from bytelatent.data.patcher import PatcherArgs, PatchingModeEnum, entropy
from bytelatent.entropy_model import load_entropy_model
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs

ENTROPY_MODEL = "transformer_100m"
ARROW_TEST_DATA = str(BLT_DATA / "stackexchange.chunk.00.jsonl.shard_00.arrow")


def test_entropy_model():
    initial_state = ArrowFileIteratorState(
        file_path=None,
        num_workers=1,
        worker_id=0,
        preprocess_dir=None,
        entropy_model_name=ENTROPY_MODEL,
        dataset_files=[ARROW_TEST_DATA],
        row_num=0,
        arrow_batch_size=100,
        s3_profile=None,
        file_format="arrow",
    )
    arrow_file = initial_state.build()
    tokenizer_args = TokenizerArgs(
        name="blt",
        init_kwargs={
            "bpe_tokenizer_path": BLT_DATA / "tokenizer_final_32k.minus_inf_ws.model"
        },
    )
    entropy_model = load_entropy_model(
        BLT_DATA / "checkpoint_0100000_consolidated",
        os.path.join(
            BLT_DATA,
            "entropy_model.pth",
        ),
    ).cuda()
    preprocess_iter = PreprocessIterator(
        arrow_file,
        tokenizer_args=tokenizer_args,
        patcher_args=PatcherArgs(patching_mode=PatchingModeEnum.entropy),
        add_patches=False,
    )
    for example in preprocess_iter.create_iter():
        tokens = torch.tensor(example.tokens).unsqueeze(0)
        expected_entropies = torch.tensor(example.entropies).unsqueeze(0)
        preds = entropy_model(tokens.cuda())
        pred_entropies = entropy(preds)
        assert pred_entropies.shape == expected_entropies.shape
        assert torch.allclose(
            pred_entropies.cpu(), expected_entropies, rtol=1.0, atol=3.5
        )
        break
