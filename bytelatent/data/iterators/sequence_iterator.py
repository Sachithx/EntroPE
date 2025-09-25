# Copyright (c) Meta Platforms, Inc. and affiliates.
from logging import getLogger
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from bytelatent.data.data_types import BltSequence
from bytelatent.data.iterators.abstract_iterator import (
    PydanticIteratorState,
    StatefulIterator,
)
from bytelatent.data.iterators.arrow_iterator import ArrowFileIterator
from bytelatent.data.iterators.limit_iterator import LimitIterator
from bytelatent.data.iterators.looping_iterator import LoopingIterator
from bytelatent.data.iterators.preprocess_iterator import (
    PreprocessIterator,
    PreprocessIteratorState,
)

logger = getLogger()


class SequencePackingArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_seq_len: int
    buffer_size: int


class SequenceIteratorState(PydanticIteratorState):
    model_config = ConfigDict(extra="forbid")
    sequence_packing_args: SequencePackingArgs
    preprocess_iterator_state: PreprocessIteratorState
    # If None, rng is disabled.
    rng_state: dict[str, Any] | None

    def build(self):
        preprocess_iterator = self.preprocess_iterator_state.build()
        return SequenceIterator(
            preprocess_iterator,
            sequence_packing_args=self.sequence_packing_args,
            rng_state=self.rng_state,
        )


def get_datafile(
    iterator: PreprocessIterator | ArrowFileIterator | LoopingIterator | LimitIterator,
):
    if isinstance(iterator, ArrowFileIterator):
        return f"file={iterator.file_path} n_shards={len(iterator.dataset_files) if iterator.dataset_files is not None else None}"
    elif isinstance(iterator, PreprocessIterator):
        return get_datafile(iterator.arrow_iterator)
    elif isinstance(iterator, LoopingIterator):
        return get_datafile(iterator.file_iterator)
    elif isinstance(iterator, LimitIterator):
        return get_datafile(iterator.base_iterator)
    else:
        raise NotImplementedError()


class SequenceIterator(StatefulIterator):
    def __init__(
        self,
        preprocess_iterator: PreprocessIterator,
        *,
        rng_state: dict[str, Any] | None,
        sequence_packing_args: SequencePackingArgs,
    ):
        self.preprocess_iterator = preprocess_iterator
        self.sequence_packing_args = sequence_packing_args
        self.output_seq_len = sequence_packing_args.output_seq_len
        self.buffer_size = sequence_packing_args.buffer_size
        if rng_state is None:
            self.rng = None
        else:
            self.rng = np.random.default_rng()
            self.rng.bit_generator.state = rng_state

    def get_state(self):
        # TODO: need to also perist the current shuffle buffer
        return SequenceIteratorState(
            sequence_packing_args=self.sequence_packing_args,
            preprocess_iterator_state=self.preprocess_iterator.get_state(),
            rng_state=None if self.rng is None else self.rng.bit_generator.state,
        )

    def create_iter(self):
        example_iter = self.preprocess_iterator.create_iter()
        n_buffer_patches = self.buffer_size * self.output_seq_len

        patch_lengths: list[int] = []
        tokens: list[int] = []
        mask: list[bool] = []
        first = True
        logger.info(
            "Starting first buffer for: %s",
            get_datafile(self.preprocess_iterator),
        )
        for example in example_iter:
            assert example.tokens is not None
            assert example.mask is not None
            if self.preprocess_iterator.add_patches:
                assert example.patch_lengths is not None
                assert len(example.tokens) == sum(example.patch_lengths)
            else:
                assert example.patch_lengths is None
            assert len(example.tokens) != 0
            assert len(example.mask) != 0
            assert len(example.tokens) == len(example.mask)

            tokens.extend(example.tokens)
            mask.extend(example.mask)
            if self.preprocess_iterator.add_patches:
                patch_lengths.extend(example.patch_lengths)
            else:
                # This lets the rest of the code work as expected and just yield byte seqs
                patch_lengths.extend([1] * len(example.tokens))

            while len(patch_lengths) >= n_buffer_patches:
                if first:
                    first = False
                    logger.info(
                        "First buffer complete for: %s",
                        get_datafile(self.preprocess_iterator),
                    )

                x_patches = np.array(patch_lengths[:n_buffer_patches]).reshape(
                    self.buffer_size, self.output_seq_len
                )
                seq_tokens = []
                seq_mask = []
                start_id = 0
                # We fix the number of patches and therefore global steps per batch
                # so we have a variable number of tokens we need to account for
                for num_tokens in x_patches.sum(axis=-1):
                    seq_tokens.append(tokens[start_id : start_id + num_tokens])
                    seq_mask.append(mask[start_id : start_id + num_tokens])
                    start_id += num_tokens

                assert start_id == x_patches.sum()

                # Remove what we just added from the buffer
                patch_lengths = patch_lengths[n_buffer_patches:]
                tokens = tokens[x_patches.sum() :]
                mask = mask[x_patches.sum() :]

                seq_patch_lengths: list[list[int]] = x_patches.tolist()
                assert len(seq_patch_lengths) == self.buffer_size
                if self.rng is None:
                    permutations = list(range(len(seq_patch_lengths)))
                else:
                    permutations = self.rng.permutation(len(seq_patch_lengths))

                for idx in permutations:
                    assert len(seq_patch_lengths[idx]) == self.output_seq_len
                    assert (
                        sum(seq_patch_lengths[idx])
                        == len(seq_tokens[idx])
                        == len(seq_mask[idx])
                    ), f"{sum(seq_patch_lengths[idx])}, {len(seq_tokens[idx])} {len(seq_mask[idx])}, idx={idx}"
                    assert seq_patch_lengths[idx][0] > 0, f"{seq_patch_lengths[idx]}"
                    if self.preprocess_iterator.add_patches:
                        yield BltSequence(
                            tokens=seq_tokens[idx],
                            mask=seq_mask[idx],
                            patch_lengths=seq_patch_lengths[idx],
                        )
                    else:
                        yield BltSequence(
                            tokens=seq_tokens[idx],
                            mask=seq_mask[idx],
                            patch_lengths=None,
                        )
