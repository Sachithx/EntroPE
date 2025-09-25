from pydantic import ConfigDict

from bytelatent.data.iterators.abstract_iterator import (
    PydanticIteratorState,
    StatefulIterator,
)
from bytelatent.data.iterators.arrow_iterator import ArrowFileIteratorState
from bytelatent.data.iterators.dev_iterators import BltTestIteratorState


class LimitIteratorState(PydanticIteratorState):
    model_config = ConfigDict(extra="forbid")
    base_iterator_state: (
        BltTestIteratorState | ArrowFileIteratorState | PydanticIteratorState
    )
    n_yielded: int
    limit: int

    def build(self) -> "LimitIterator":
        return LimitIterator(
            base_iterator=self.base_iterator_state.build(),
            n_yielded=self.n_yielded,
            limit=self.limit,
        )


class LimitIterator(StatefulIterator):
    def __init__(self, base_iterator: StatefulIterator, limit: int, n_yielded: int = 0):
        self.base_iterator = base_iterator
        self.n_yielded = n_yielded
        self.limit = limit

    def get_state(self):
        return LimitIteratorState(
            base_iterator_state=self.base_iterator.get_state(),
            n_yielded=self.n_yielded,
            limit=self.limit,
        )

    def create_iter(self):
        iterator = self.base_iterator.create_iter()
        try:
            while self.n_yielded < self.limit or self.limit < 0:
                yield next(iterator)
                self.n_yielded += 1
        except StopIteration:
            pass
