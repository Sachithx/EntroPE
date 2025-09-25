# Copyright (c) Meta Platforms, Inc. and affiliates.
import subprocess
from pathlib import Path

import luigi

# CHANGEME: Change this to point to your data
BASE_DIR = Path("datasets")
DATASETS = ["dclm"]
TARGET_DIR = Path("entropy_preprocess")

SHARD_SCRIPT = """split -C 2500m -d {source} {destination}.shard_"""


def list_dataset_shards(dataset: str):
    dataset_dir = BASE_DIR / dataset
    return list(dataset_dir.glob("*.chunk.*.jsonl"))


class ChunkFile(luigi.ExternalTask):
    file = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.file)


class ShardDatasetChunk(luigi.Task):
    dataset_name = luigi.Parameter()
    chunk_file = luigi.Parameter()

    def _chunk_filename(self):
        return Path(self.chunk_file).name

    def requires(self):
        return ChunkFile(self.chunk_file)

    def run(self):
        destination_dir = TARGET_DIR / str(self.dataset_name)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / self._chunk_filename()
        subprocess.check_output(
            SHARD_SCRIPT.format(source=str(self.chunk_file), destination=destination),
            shell=True,
        )
        (
            Path(TARGET_DIR)
            / str(self.dataset_name)
            / f"{self._chunk_filename()}.shard.COMPLETE"
        ).touch()

    def output(self):
        return luigi.LocalTarget(
            TARGET_DIR
            / str(self.dataset_name)
            / f"{self._chunk_filename()}.shard.COMPLETE"
        )


class ShardDataset(luigi.WrapperTask):
    dataset_name = luigi.Parameter()

    def requires(self):
        for f in list_dataset_shards(self.dataset_name):
            yield ShardDatasetChunk(dataset_name=self.dataset_name, chunk_file=str(f))


class ShardAllDatasets(luigi.WrapperTask):
    def requires(self):
        for d in DATASETS:
            yield ShardDataset(dataset_name=d)


if __name__ == "__main__":
    luigi.build([ShardAllDatasets()], local_scheduler=True, workers=128)
