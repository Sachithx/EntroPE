# Copyright (c) Meta Platforms, Inc. and affiliates.
import pickle
from pathlib import Path

import numpy as np

from bytelatent import ByteLatentError

LOOKUP_OFFSET = 4


def apply_lookup_table_wrapper(ngram_to_idx: dict[tuple, int], lookup_offset=1):
    """
    Wrapper function for applying the lookup table to each n-gram.

    :param ngram: Array of numbers representing an n-gram.
    :param lookup_table: Dictionary where keys are tuples (n-grams) and values are the desired outputs.
    :param lookup_offset: Offset to add to the lookup result.
    :return: The value associated with the n-gram tuple in the dictionary, or None if not found.
    """

    def apply_lookup_table(ngram):
        """
        Function to apply to each n-gram: converts it to a tuple and looks it up in a dictionary.

        :param ngram: Array of numbers representing an n-gram.
        :return: The value associated with the n-gram tuple in the dictionary, or None if not found.
        """
        # Convert the n-gram to a tuple
        ngram_tuple = tuple(ngram)

        if ngram_tuple not in ngram_to_idx:
            return 0
        else:
            return ngram_to_idx[ngram_tuple] + lookup_offset

    return apply_lookup_table


def get_byte_ngrams_ids(
    byte_array: np.ndarray, n: int, ngram_to_idx: dict[tuple, int], pad_value=0
):
    """
    Generate n-grams from a 2D numpy array.

    :param n: The length of each n-gram.
    :param pad_value: The value used for padding of the byte values to maintain the same dimensions for the n-grams.
    :return: A 2D numpy array where each element is the ID of an n-gram offset by LOOKUP_OFFSET.
    """
    num_rows, num_cols = byte_array.shape

    # Create an array to hold the padded version of the original array
    padded_array = np.pad(
        byte_array, ((0, 0), (n - 1, 0)), mode="constant", constant_values=pad_value
    )

    # Use stride tricks to avoid explicit looping
    strided = np.lib.stride_tricks.as_strided
    shape = (num_rows, num_cols, n)
    strides = padded_array.strides[:2] + (padded_array.strides[1],)
    ngrams = strided(padded_array, shape=shape, strides=strides)

    ngram_ids = np.apply_along_axis(
        apply_lookup_table_wrapper(ngram_to_idx, lookup_offset=LOOKUP_OFFSET), 2, ngrams
    )
    assert ngram_ids.shape == byte_array.shape
    return ngram_ids


def reload_tables(
    ngram_table_dir: str, ngram_to_size: dict[int, int], offset: int = LOOKUP_OFFSET
) -> tuple[dict[int, list], dict[tuple, int], dict[int, int]]:
    """
    Reload lookup tables from a directory. Reload only the ngrams in the dictionary and per ngram,
    only load up to the max specified size. Return the actual number of ngrams taken per ngram size.
    """
    idx_to_ngram_tables = {}
    ngram_to_idx_tables = {}
    vocab_sizes = {}
    for ngram, size in ngram_to_size.items():
        with open(Path(ngram_table_dir) / f"ngram-{ngram}.pickle", "rb") as f:
            # These are already sorted by count
            # Value: tuple of: count, ngram, dataset
            ngram_data: list[tuple[tuple, tuple[int, int, str]]] = pickle.load(f)[
                "counts"
            ]
            table = [ngram for ngram, _ in ngram_data][:size]
            if len(table) != size:
                raise ValueError(
                    f"Ngram table for {ngram}-gram is not large enough to get {size} ngrams, max size is {len(ngram_data)}"
                )
            ngram_to_idx = {ngram: idx for idx, ngram in enumerate(table)}
            actual_size = len(table)
            idx_to_ngram_tables[ngram] = table
            ngram_to_idx_tables[ngram] = ngram_to_idx
            vocab_sizes[ngram] = actual_size + offset
    return ngram_to_idx_tables, ngram_to_idx_tables, vocab_sizes


def parse_ngram_to_size(ngram_to_size_str: str | None) -> dict[int, int]:
    if ngram_to_size_str is None:
        return None
    ngram_to_size = {}
    for entry in ngram_to_size_str.split(","):
        ngram, size = entry.split(":")
        ngram = int(ngram)
        size = int(size)
        ngram_to_size[ngram] = size
    return ngram_to_size


class NgramProcessor:
    def __init__(
        self,
        ngram_table_dir: str | None = None,
        ngram_to_size: dict[int, int] | None = None,
    ):
        if ngram_table_dir is None or ngram_to_size is None:
            raise ByteLatentError(
                "ngram_table_dir and ngram_to_size cannot be none if enable_byte_ngrams is True"
            )
        (
            self.ngram_to_idx_tables,
            self.idx_to_ngram_tables,
            self.ngram_vocab_sizes,
        ) = reload_tables(ngram_table_dir, ngram_to_size)
        # Lowest to highest ngram
        self.ngram_sizes = sorted(list(self.ngram_to_idx_tables.keys()))
        # Although the model might not use all the ngrams, we need the tokenizer
        # to produce ngram_ids such that index zero is the 2-gram, later on in
        # src.model.megabyte.Megabyte.forward
        assert self.ngram_sizes[0] == 2

    def encode_single_ngram_table(self, data: np.ndarray, n: int):
        """
        Return the n-grams of the input data for a given n
        numpy array with ids of shape data.shape
        """
        return get_byte_ngrams_ids(data, n, self.ngram_to_idx_tables[n], pad_value=0)

    def encode_token_ngrams(self, data: np.ndarray):
        """
        Return the n-grams of the input data.
        output shape: [ids with data.shape for n in self.ngram_sizes]
        """
        return [self.encode_single_ngram_table(data, n) for n in self.ngram_sizes]
