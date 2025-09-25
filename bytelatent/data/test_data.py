import os
import pickle

import pytest
from omegaconf import OmegaConf

from bytelatent.args import TrainArgs
from bytelatent.constants import BLT_DATA


def get_test_config():
    if "BLT_INTERNAL" in os.environ:
        internal_dir = os.environ["BLT_INTERNAL"]
    else:
        internal_dir = "../internal-blt/configs"
    test_config = os.path.join(internal_dir, "tests.yaml")
    return test_config


@pytest.mark.skipif(
    not os.path.exists(get_test_config()),
    reason="Skipping since internal config is missing",
)
def test_first_batch_matches():
    test_config_path = get_test_config()
    default_cfg = OmegaConf.create(TrainArgs().model_dump())
    file_cfg = OmegaConf.load(test_config_path)
    merged_cfg = OmegaConf.merge(default_cfg, file_cfg)
    merged_cfg = OmegaConf.to_container(merged_cfg, resolve=True, throw_on_missing=True)
    train_args = TrainArgs.model_validate(merged_cfg)
    # MP doesn't work with async very well, but it doesn't change logic
    train_args.data.load_async = False

    # Test data created by pickling first batch in train loop then exiting
    with open(os.path.join(BLT_DATA, "fixtures", "first_batch_0.pickle"), "rb") as f:
        first_batch = pickle.load(f)

    # Emulate 1 node, 8 gpu training
    data_loader = train_args.data.build_from_rank(0, 8)
    batch_iterator = data_loader.create_iter()
    print("Getting first batch")
    batch = next(batch_iterator)
    assert (batch.x == first_batch.x).all()
    assert (batch.y == first_batch.y).all()
    assert (batch.mask == first_batch.mask).all()
    assert (batch.patch_lengths == first_batch.patch_lengths).all()
    assert batch.ngram_ids is None and first_batch.ngram_ids is None
    assert batch.is_final == False and batch.is_final == False
