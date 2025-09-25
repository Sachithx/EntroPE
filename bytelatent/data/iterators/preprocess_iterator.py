# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Any, Generator

import torch
from pydantic import BaseModel, ConfigDict

from bytelatent.data.data_types import BltExample
from bytelatent.data.iterators.abstract_iterator import (
    PydanticIteratorState,
    StatefulIterator,
)
from bytelatent.data.iterators.arrow_iterator import (
    ArrowFileIterator,
    ArrowFileIteratorState,
)
from bytelatent.data.iterators.limit_iterator import LimitIterator, LimitIteratorState
from bytelatent.data.iterators.looping_iterator import (
    LoopingIterator,
    LoopingIteratorState,
)
from bytelatent.data.patcher import Patcher, PatcherArgs, PatchingModeEnum
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs


class PreprocessIteratorState(PydanticIteratorState):
    model_config = ConfigDict(extra="forbid")
    arrow_file_iterator_state: (
        ArrowFileIteratorState | LoopingIteratorState | LimitIteratorState
    )
    add_tokens: bool
    add_patches: bool
    tokenizer_args: TokenizerArgs
    patcher_args: PatcherArgs

    def build(self):
        arrow_iterator = self.arrow_file_iterator_state.build()
        return PreprocessIterator(
            arrow_iterator,
            patcher_args=self.patcher_args,
            tokenizer_args=self.tokenizer_args,
            add_tokens=self.add_tokens,
            add_patches=self.add_patches,
        )


class PreprocessIterator(StatefulIterator):
    """
    Take BltExamples with fields filled in only from ArrowFileIterator, and fill in fields that require
    preprocessing like tokenization and patching
    """

    def __init__(
        self,
        arrow_iterator: ArrowFileIterator | LoopingIterator | LimitIterator,
        *,
        patcher_args: PatcherArgs,
        tokenizer_args: TokenizerArgs,
        add_tokens: bool = True,
        add_patches: bool = True,
    ):
        self.arrow_iterator = arrow_iterator
        self.tokenizer_args = tokenizer_args
        self.patcher_args = patcher_args
        self.add_tokens = add_tokens
        self.add_patches = add_patches
        self.tokenizer: BltTokenizer | None = None
        self.patcher: Patcher | None = None

    def get_state(self) -> PreprocessIteratorState:
        """
        The only state to maintain here is from arrow, there
        isn't any internal state on this iterator.
        """
        return PreprocessIteratorState(
            arrow_file_iterator_state=self.arrow_iterator.get_state(),
            tokenizer_args=self.tokenizer_args,
            patcher_args=self.patcher_args,
            add_tokens=self.add_tokens,
            add_patches=self.add_patches,
        )

    def create_iter(self) -> Generator[BltExample, Any, None]:
        if self.tokenizer is None and self.add_tokens:
            self.tokenizer = self.tokenizer_args.build()
        if self.patcher is None and self.add_patches:
            self.patcher = self.patcher_args.build()

        example_iter = self.arrow_iterator.create_iter()
        for example in example_iter:
            if self.add_tokens:
                tokens = self.tokenizer.encode(example.text)
            else:
                tokens = example.tokens
            if (
                self.patcher is not None
                and self.patcher.patching_mode == PatchingModeEnum.entropy
            ):
                assert (
                    example.entropies is not None
                ), "For patching, entropies cannot be None"
                entropies = torch.tensor(example.entropies).unsqueeze(0)
            else:
                entropies = None
            if self.patcher is None:
                patch_lengths = None
            else:
                patch_lengths = self.patcher.patch(
                    torch.tensor(tokens).unsqueeze(0),
                    include_next_token=False,
                    entropies=entropies,
                )[0][0].tolist()
            yield BltExample(
                sample_id=example.sample_id,
                text=example.text,
                tokens=tokens,
                mask=[True] * len(tokens),
                patch_lengths=patch_lengths,
                entropies=example.entropies,
            )
