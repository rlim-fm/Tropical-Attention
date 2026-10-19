"""Independent, recorded seeds for each source of randomness in a run.

The seeds are spawned from ``(global_seed, seed_index)`` and **nothing else**.
They do not depend on the cell. So two cells with the same seed index share
their data draw and their initial weights, and differ only in their factors.
That pairing is what makes the seed-paired tests in ``expkit.stats`` valid. Keep
it: never fold a factor level into a seed.
"""
import os
import random
from typing import Sequence

import numpy as np

DEFAULT_STREAMS = ('init', 'data', 'sampler')


def derive_seeds(global_seed: int, seed_index: int,
                 streams: Sequence[str] = DEFAULT_STREAMS) -> dict:
    """``{stream: uint32 seed}``, one independent child per named stream.

    The children come in the order the streams are named, so appending a stream
    leaves the earlier seeds unchanged. Renaming or reordering one does change
    them.
    """
    children = np.random.SeedSequence([int(global_seed), int(seed_index)]).spawn(len(streams))
    return {name: int(c.generate_state(1, dtype=np.uint32)[0])
            for name, c in zip(streams, children)}


def seed_everything(seed: int) -> None:
    """Seed the global python, numpy and (if installed) torch generators."""
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_determinism() -> None:
    """Ask torch for deterministic kernels where they exist (a no-op without torch).

    These stay non-deterministic, and that is tolerated:

    * ``warn_only=True``: a few CUDA kernels (scatter/index_add backward among
      them) have no deterministic implementation, so torch warns instead of
      failing the run.
    * A result is reproducible on the same device type and library version, not
      across them. CPU and CUDA runs of one config differ in the last bits, and
      so do runs under two cuBLAS versions. Each record stores its device and
      torch version for that reason.
    * Draw batch orders from a CPU ``torch.Generator`` (or numpy). Then the
      batch sequence is the same on every device, even where the arithmetic
      is not.
    """
    try:
        import torch
    except ImportError:
        return
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
