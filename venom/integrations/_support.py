"""Deterministic fixtures shared by the sibling adapters.

These are *glue*, not analysis: small NumPy stacks and score functions so
that the sibling implementations have something well-formed to run on
when the caller does not supply its own.  Every number a sibling reports
is computed by the sibling.
"""
from __future__ import annotations

from typing import Any, Callable, List


def default_blocks(dim: int = 4, count: int = 4, seed: int = 0) -> List[Callable]:
    """A deterministic stream-additive block stack (NumPy).

    The sibling repos (TRACE-PATH, SKIP-RUN-, NOISE-STORM) expect blocks
    with the semantics ``state = state + block(state)``.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    matrices = [rng.normal(scale=0.4, size=(dim, dim)) for _ in range(count)]

    def make(m):
        def block(state):
            return np.tanh(m @ np.asarray(state, dtype=float))
        return block

    return [make(m) for m in matrices]


def default_score(dim: int = 4) -> Callable:
    """A deterministic scalar score of a state vector."""
    import numpy as np

    weights = np.linspace(0.5, 1.5, dim)

    def score(state):
        return float(np.dot(weights, np.asarray(state, dtype=float)))
    return score


def default_writes(count: int = 4, tokens: int = 3, dim: int = 5, seed: int = 5):
    """Deterministic per-block write tensors of shape ``(T, D)``."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return [rng.normal(size=(tokens, dim)) * (0.2 * (i + 1))
            for i in range(count)]


def default_weights(count: int = 3, rows: int = 6, cols: int = 5, seed: int = 0):
    """Deterministic 2-D weight matrices for the weight-space siblings."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, cols)) for _ in range(count)]


def to_numpy(value: Any):
    """Convert a torch tensor / array-like into a NumPy array."""
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)
