# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Any

import numpy as np
from pydantic import ConfigDict

from bytelatent.data.iterators.abstract_iterator import (
    PydanticIteratorState,
    StatefulIterator,
)
from bytelatent.data.iterators.sequence_iterator import SequenceIteratorState


class SamplingIteratorState(PydanticIteratorState):
    model_config = ConfigDict(extra="forbid")
    rng_state: dict[str, Any]
    source_to_weight: dict[str, float]
    source_to_iterator_state: dict[str, SequenceIteratorState]

    def build(self) -> "SamplingIterator":
        return SamplingIterator(
            rng_state=self.rng_state,
            source_to_weight=self.source_to_weight,
            source_to_iterator={
                source: state.build()
                for source, state in self.source_to_iterator_state.items()
            },
        )


class SamplingIterator(StatefulIterator):
    def __init__(
        self,
        *,
        rng_state: dict[str, Any],
        source_to_weight: dict[str, float],
        source_to_iterator: dict[str, StatefulIterator],
    ):
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = rng_state
        self.source_to_weight = source_to_weight
        self.source_to_iterator = source_to_iterator

    def get_state(self) -> SamplingIteratorState:
        return SamplingIteratorState(
            rng_state=self.rng.bit_generator.state,
            source_to_weight=self.source_to_weight,
            source_to_iterator_state={
                source: iterator.get_state()
                for source, iterator in self.source_to_iterator.items()
            },
        )

    def create_iter(self):
        n_sources = len(self.source_to_weight)
        possible_sources = []
        weights = []
        for source, w in self.source_to_weight.items():
            possible_sources.append(source)
            weights.append(w)

        source_to_python_iter = {
            source: self.source_to_iterator[source].create_iter()
            for source in possible_sources
        }
        while True:
            norm_weights = np.array(weights) / np.array(weights).sum()
            source_choice = possible_sources[self.rng.choice(n_sources, p=norm_weights)]
            yield next(source_to_python_iter[source_choice])
