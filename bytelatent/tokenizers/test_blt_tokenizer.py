# Copyright (c) Meta Platforms, Inc. and affiliates.
import json

from bytelatent.constants import BLT_DATA
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs


def test_tokenizer_bytes():
    with open("fixtures/tokenizer_data.json") as f:
        data = json.load(f)

    examples: list[str] = data["texts"]
    examples_tokens: list[list[int]] = data["tokens"]

    tokenizer = BltTokenizer(bpe_delim=False)
    for i in range(len(examples)):
        assert tokenizer.encode(examples[i]) == examples_tokens[i]


def test_tokenizer_bpe():
    with open("fixtures/tokenizer_data_bpe_delim.json") as f:
        data = json.load(f)

    examples: list[str] = data["texts"]
    examples_tokens: list[list[int]] = data["tokens"]

    tokenizer = BltTokenizer(bpe_delim=True)
    for i in range(len(examples)):
        assert tokenizer.encode(examples[i]) == examples_tokens[i]


def test_build_tokenizer_from_args():
    tokenizer_args = TokenizerArgs(
        name="blt",
        init_kwargs={
            "bpe_tokenizer_path": BLT_DATA / "tokenizer_final_32k.minus_inf_ws.model"
        },
    )
    tokenizer = tokenizer_args.build()
    assert tokenizer.encode("test text") is not None
