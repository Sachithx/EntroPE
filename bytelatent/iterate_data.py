import json

import pyarrow
import typer
from rich.progress import track

from bytelatent.data.iterators.multiprocess_iterator import MultiprocessIteratorState
from bytelatent.logger import init_logger


def main(
    state_file: str,
    steps: int = 3_000,
    io_thread_count: int = 2,
    cpu_count: int = 2,
    log_freq: int = 100,
):
    init_logger()
    pyarrow.set_io_thread_count(io_thread_count)
    pyarrow.set_cpu_count(cpu_count)
    with open(state_file) as f:
        train_state = json.load(f)
        dl_state = MultiprocessIteratorState(**train_state["data_loader_state"])
        packing_iterator_state = dl_state.base_iterator_state
        print("building")
        packing_iterator = packing_iterator_state.build()
        print("iter")
        batch_iter = packing_iterator.create_iter()
        print("looping")
        for i in track(range(steps)):
            _ = next(batch_iter)
            if i % log_freq == 0:
                print(pyarrow.default_memory_pool())
        print(i)
        print(pyarrow.default_memory_pool())


if __name__ == "__main__":
    typer.run(main)
