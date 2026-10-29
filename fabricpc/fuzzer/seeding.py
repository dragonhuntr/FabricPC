"""Seed handling so every graph can be regenerated on its own.

Each graph in a batch gets its own seed derived from the master seed and the
graph's index with :class:`numpy.random.SeedSequence`. Graph ``i`` therefore
depends only on ``(seed, i)``: asking for 5 graphs or 500 yields the same first
five, and a single graph from a large batch can be regenerated without
generating the ones before it.

All randomness in the fuzzer must flow through the ``numpy.random.Generator``
returned here. Never use the global ``random`` / ``np.random`` state.
"""

from __future__ import annotations

import numpy as np


def graph_seed(master_seed: int, index: int) -> int:
    """Return the 32-bit seed for graph ``index`` of a run seeded ``master_seed``."""
    if master_seed < 0 or index < 0:
        raise ValueError("master_seed and index must be non-negative")
    seq = np.random.SeedSequence(master_seed, spawn_key=(index,))
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def graph_rng(master_seed: int, index: int, attempt: int = 0) -> np.random.Generator:
    """Random generator for one generation attempt of graph ``index``.

    ``attempt`` lets the pipeline retry a graph that failed validation with
    fresh randomness while staying reproducible.
    """
    seq = np.random.SeedSequence(master_seed, spawn_key=(index, attempt))
    return np.random.default_rng(seq)
