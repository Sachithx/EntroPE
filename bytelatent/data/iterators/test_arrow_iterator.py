# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import pyarrow as pa

# pyarrow needs the initialization from this import
import pyarrow.dataset  # pyright: ignore

from bytelatent.constants import BLT_DATA
from bytelatent.data.iterators.arrow_iterator import (
    ArrowFileIterator,
    ArrowFileIteratorState,
)

ENTROPY_MODEL = "transformer_100m"
ARROW_TEST_DATA_1 = str(BLT_DATA / "stackexchange.chunk.00.jsonl.shard_00.arrow")
ARROW_TEST_DATA_2 = str(BLT_DATA / "stackexchange.chunk.00.jsonl.shard_01.arrow")


def test_basic_arrow_file():
    dataset = pa.dataset.dataset(ARROW_TEST_DATA_1, format="arrow")
    n_head = 1000
    head_df = dataset.head(n_head).to_pandas()

    initial_state = ArrowFileIteratorState(
        file_path=None,
        num_workers=1,
        worker_id=0,
        preprocess_dir=None,
        entropy_model_name=ENTROPY_MODEL,
        dataset_files=[ARROW_TEST_DATA_1],
        row_num=0,
        arrow_batch_size=100,
        s3_profile=None,
        file_format="arrow",
    )
    arrow_file = initial_state.build()
    start_state = arrow_file.get_state()
    assert start_state.row_num == initial_state.row_num

    sample_id = None
    for example in arrow_file.create_iter():
        sample_id = example.sample_id
        assert head_df.iloc[0]["sample_id"] == sample_id
        break

    assert arrow_file.get_state().row_num == 1
    arrow_file = initial_state.build()
    for example in arrow_file.create_iter():
        assert example.sample_id == sample_id
        assert head_df.iloc[0]["sample_id"] == sample_id
        break

    # Test resume far enough in to be past the batch size of 100
    resumed_state = ArrowFileIteratorState(
        file_path=None,
        num_workers=1,
        worker_id=0,
        preprocess_dir=None,
        entropy_model_name=ENTROPY_MODEL,
        dataset_files=[ARROW_TEST_DATA_1],
        row_num=251,
        arrow_batch_size=100,
        s3_profile=None,
        file_format="arrow",
    )
    arrow_file = resumed_state.build()
    for example in arrow_file.create_iter():
        assert example.sample_id == head_df.iloc[251]["sample_id"]
        assert arrow_file.get_state().row_num == 252
        break

    world_rank = 1
    world_size = 4
    # Test World Size and Rank
    rank_state = ArrowFileIteratorState(
        file_path=None,
        num_workers=world_size,
        worker_id=world_rank,
        preprocess_dir=None,
        entropy_model_name=ENTROPY_MODEL,
        dataset_files=[ARROW_TEST_DATA_1],
        row_num=0,
        arrow_batch_size=100,
        s3_profile=None,
        file_format="arrow",
    )
    arrow_file = rank_state.build()
    expected_ids = []
    for i in range(n_head):
        if i % world_size == world_rank:
            expected_ids.append(head_df.iloc[i]["sample_id"])
    print(len(expected_ids))
    i = 0
    for example in arrow_file.create_iter():
        assert example.sample_id == expected_ids[i]
        i += 1
        if i >= len(expected_ids):
            break


def test_read_jsonl_from_arrow():
    arrow_iterator = ArrowFileIterator(
        file_path="fixtures/test_docs.jsonl",
        num_workers=1,
        worker_id=0,
        preprocess_dir=None,
        entropy_model_name=None,
        file_format="json",
        arrow_batch_size=100,
    )
    iterator = arrow_iterator.create_iter()
    for i, example in enumerate(iterator):
        assert example.sample_id == str(i)
        assert example.text == f"test_{i}"
