# Copyright (c) Meta Platforms, Inc. and affiliates.
import subprocess
from pathlib import Path

import submitit
import typer


class PreprocessEntropiesJob(submitit.helpers.Checkpointable):
    def __init__(self) -> None:
        pass

    def __call__(self, shard_file: str, output_filename: str):
        subprocess.run(
            [
                "python",
                "-u",
                "-m",
                "bytelatent.preprocess.preprocess_entropies",
                str(shard_file),
                str(output_filename),
            ],
            check=True,
        )
        return True


def chunk(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main(
    job_folder: str,
    input_dir: str,
    output_dir: str,
    qos: str = "explore",
    slurm_batch_size: int = 1000,
    check_only: bool = False,
    wait: bool = False,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    shard_files = [
        p for p in input_dir.glob("*.jsonl.shard*") if "COMPLETE" not in p.name
    ]
    if check_only:
        exist = []
        missing = []
        for shard_file in shard_files:
            shard_file = Path(shard_file)
            complete_file = output_dir / f"{shard_file.name}.arrow.complete"
            if complete_file.exists():
                exist.append(complete_file)
            else:
                missing.append(complete_file)
        print("Checked for output files for input_dir=", input_dir)
        print("Exist:", len(exist))
        print("Missing:", len(missing))
        print(missing)
        return
    print("Running parallel job over N files=", len(shard_files))
    print("Input Directory:", input_dir)
    print("Output Directory:", output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    executor = submitit.SlurmExecutor(job_folder)
    executor.update_parameters(
        # 12 hours in minutes
        time=60 * 12,
        qos=qos,
        exclusive="user",
        cpus_per_task=4,
        num_gpus=1,
        mem_per_gpu="80G",
        array_parallelism=slurm_batch_size,
    )

    jobs = []
    n_batches = 0
    n_skipped = 0
    n_launched = 0
    for file_batch in chunk(shard_files, slurm_batch_size):
        with executor.batch():
            for shard_file in file_batch:
                output_filename = Path(output_dir) / f"{shard_file.name}.arrow"
                complete_output_filename = (
                    Path(output_dir) / f"{shard_file.name}.arrow.complete"
                )
                if complete_output_filename.exists():
                    n_skipped += 1
                else:
                    job = executor.submit(
                        PreprocessEntropiesJob(), str(shard_file), str(output_filename)
                    )
                    n_launched += 1
                    jobs.append(job)
        n_batches += 1
    print("launched array jobs n=", n_launched)
    print("skipped (completed) array jobs n=", n_skipped)
    print("number of slurm batches=", n_batches)
    if wait:
        output = [job.result() for job in jobs]
        assert all(output)


if __name__ == "__main__":
    typer.run(main)
