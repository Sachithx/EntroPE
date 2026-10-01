import numpy as np


def get_lr(iteration, max_iters, warmup_iters, learning_rate, min_lr, decay_lr=True):
    """Cosine learning-rate schedule with linear warmup.

    Args:
        iteration: current iteration.
        max_iters: total number of iterations.
        warmup_iters: number of linear-warmup iterations.
        learning_rate: peak learning rate.
        min_lr: minimum learning rate at the end of decay.
        decay_lr: whether to cosine-decay after warmup.
    """
    # Linear warmup
    if warmup_iters > 0 and iteration < warmup_iters:
        return learning_rate * (iteration / warmup_iters)

    if not decay_lr:
        return learning_rate

    # Cosine decay from learning_rate down to min_lr
    decay_ratio = (iteration - warmup_iters) / max(1, (max_iters - warmup_iters))
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)
