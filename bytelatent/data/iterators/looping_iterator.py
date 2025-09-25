# Copyright (c) Meta Platforms, Inc. and affiliates.

from bytelatent.data.iterators.abstract_iterator import (
    PydanticIteratorState,
    StatefulIterator,
)
from bytelatent.data.iterators.arrow_iterator import (
    ArrowFileIterator,
    ArrowFileIteratorState,
)


class LoopingIteratorState(PydanticIteratorState):
    file_iterator_state: ArrowFileIteratorState
    epoch: int

    def build(self) -> "LoopingIterator":
        return LoopingIterator(
            file_iterator=self.file_iterator_state.build(),
            epoch=self.epoch,
        )


class LoopingIterator(StatefulIterator):
    def __init__(self, file_iterator: ArrowFileIterator, epoch: int = -1):
        self.file_iterator = file_iterator
        self.epoch = epoch

    def get_state(self):
        return LoopingIteratorState(
            file_iterator_state=self.file_iterator.get_state(), epoch=self.epoch
        )

    def create_iter(self):
        while True:
            self.epoch += 1
            iterator = self.file_iterator.create_iter()
            yield from iterator
