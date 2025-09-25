import pandas as pd
from pydantic import ConfigDict

from bytelatent.data.data_types import BltExample
from bytelatent.data.iterators.abstract_iterator import (
    PydanticIteratorState,
    StatefulIterator,
)


class BltTestIteratorState(PydanticIteratorState):
    model_config = ConfigDict(extra="forbid")
    position: int
    total: int

    def build(self):
        blt_iter = BltTestIteratorState(total=self.total)
        blt_iter.position = self.position
        return blt_iter


class BltTestIterator(StatefulIterator):
    def __init__(self, total: int):
        self.position = 0
        self.total = total

    def get_state(self):
        return BltTestIteratorState(position=self.position, total=self.total)

    def create_iter(self):
        for i in range(self.total):
            self.position += 1
            yield BltExample(
                sample_id=f"test_{i}",
                text=f"This is some test {i} text.",
                tokens=None,
                mask=None,
                entropies=None,
                patch_lengths=None,
            )


class BltTestWithEntropiesIteratorState(PydanticIteratorState):
    model_config = ConfigDict(extra="forbid")
    position: int
    total: int

    def build(self):
        blt_iter = BltTestWithEntropiesIteratorState(total=self.total)
        blt_iter.position = self.position
        return blt_iter


class BltTestWithEntropiesIterator(StatefulIterator):
    def __init__(self, total: int):
        self.position = 0
        self.total = total

    def get_state(self):
        return BltTestIteratorState(position=self.position, total=self.total)

    def create_iter(self):
        text = "Daenerys Targaryen is in Game of Thrones, a fantasy epic by George R.R. Martin."
        df = pd.read_json("fixtures/tokens_with_entropies.json")
        tokens = df["token_ids"].tolist()
        entropies = df["entropies"].tolist()
        # BOS and EOS
        assert len(tokens) == len(text) + 2
        for i in range(self.total):
            self.position += 1
            yield BltExample(
                sample_id=f"test_{i}",
                text=text,
                tokens=tokens,
                mask=[True] * len(tokens),
                entropies=entropies,
                patch_lengths=None,
            )
