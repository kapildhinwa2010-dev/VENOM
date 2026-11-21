"""Trace integrations: TRACE-PATH, BUS-TRACE, GEOMETRY-SCAN.

* TRACE-PATH transplants the write of one block from a clean run into a
  corrupted run and measures how much clean behaviour recovers.
* BUS-TRACE summarises how much each block writes into a residual stream.
* GEOMETRY-SCAN measures the shape of the hidden-state space per layer.

The sibling implementations work on *stream-additive* blocks
(`state = state + block(state)`) and on plain NumPy arrays.  When the
caller does not supply its own blocks, the adapters build a small
deterministic NumPy stack so that the sibling's real code path runs.  That
stack is glue, not a re-implementation: all the analysis is the sibling's.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from venom.integrations._base import guard, ok, require

SOURCE_TRACE_PATH = "TRACE-PATH"
SOURCE_BUS_TRACE = "BUS-TRACE"
SOURCE_GEOMETRY_SCAN = "GEOMETRY-SCAN"


from venom.integrations._support import default_blocks as _default_blocks
from venom.integrations._support import default_score as _default_score
from venom.integrations._support import default_writes as _default_writes


# --------------------------------------------------------------------------- #
# TRACE-PATH
# --------------------------------------------------------------------------- #

def trace_path_available() -> bool:
    """Is the TRACE-PATH sibling importable?"""
    from venom.integrations._loader import load
    return load("trace_path") is not None


def trace_path_run(blocks: Optional[List[Callable]] = None,
                   x_clean: Optional[Any] = None,
                   x_corrupt: Optional[Any] = None,
                   score_fn: Optional[Callable] = None,
                   threshold: float = 0.5) -> Dict[str, Any]:
    """Localise which block carries a fact, by activation patching.

    What it does: calls TRACE-PATH's ``trace_blocks`` - for every block it
    replaces that block's write in the corrupted run with the clean run's
    write and reports the normalised recovery, plus the first block whose
    transplant reaches the threshold.

    What it does NOT prove: that the block *stores* the fact, only that
    restoring its write restores the measured score in this synthetic
    stream.  The result depends entirely on the ``score_fn`` and on the
    blocks handed in; with the adapter's default toy stack it says nothing
    about any real model.
    """
    def _run():
        module, failure = require("trace_path", SOURCE_TRACE_PATH)
        if failure is not None:
            return failure
        import numpy as np

        stack = list(blocks) if blocks else _default_blocks()
        clean = np.ones(4) if x_clean is None else np.asarray(x_clean, dtype=float)
        corrupt = -0.5 * np.ones(4) if x_corrupt is None else np.asarray(x_corrupt, dtype=float)
        score = score_fn if score_fn is not None else _default_score()
        report = module.trace_blocks(stack, clean, corrupt, score,
                                     threshold=float(threshold))
        return ok(
            SOURCE_TRACE_PATH,
            threshold=float(threshold),
            num_blocks=len(stack),
            report=report,
            note=(
                "Activation patching on a synthetic stream. 'birth_index' "
                "says which block's write, restored, brought the score "
                "back - it is not a claim that the block stores the fact."
            ),
        )
    return guard(SOURCE_TRACE_PATH, _run)


# --------------------------------------------------------------------------- #
# BUS-TRACE
# --------------------------------------------------------------------------- #

def bus_trace_available() -> bool:
    """Is the BUS-TRACE sibling importable?"""
    from venom.integrations._loader import load
    return load("bus_trace") is not None


def bus_trace_run(writes: Optional[Any] = None,
                  taps: Optional[Dict[str, Any]] = None,
                  k: int = 3) -> Dict[str, Any]:
    """Summarise how much each block writes into the residual stream.

    What it does: calls BUS-TRACE's ``write_report`` on per-block write
    tensors (shape ``(T, D)`` or ``(B, T, D)``) and ``top_writers`` to rank
    them.  When VENOM's own bus taps are supplied they are converted to
    NumPy and used as the writes.

    What it does NOT prove: that a block with a large write is important,
    or that a "silent" block is useless - a block can write little because
    the prompt did not need it.  The taps are also read-only snapshots;
    nothing here modifies the host.
    """
    def _run():
        module, failure = require("bus_trace", SOURCE_BUS_TRACE)
        if failure is not None:
            return failure
        import numpy as np

        data = writes
        if data is None and taps:
            data = []
            for name in sorted(taps, key=lambda n: str(n)):
                value = taps[name]
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                data.append(np.asarray(value, dtype=float))
        if data is None:
            data = _default_writes()
        report = module.write_report(data)
        writers = module.top_writers(report, k=int(k))
        return ok(
            SOURCE_BUS_TRACE,
            num_blocks=int(report["num_blocks"]),
            report=report,
            top_writers=writers,
            note=(
                "Write magnitude, not importance: a large write is not "
                "evidence that the block matters, and a silent block may "
                "simply not have been needed for this prompt."
            ),
        )
    return guard(SOURCE_BUS_TRACE, _run)


# --------------------------------------------------------------------------- #
# GEOMETRY-SCAN
# --------------------------------------------------------------------------- #

def geometry_scan_available() -> bool:
    """Is the GEOMETRY-SCAN sibling importable?"""
    from venom.integrations._loader import load
    return load("geometry_scan") is not None


def geometry_scan_run(states_by_layer: Optional[Any] = None,
                      names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Measure the geometry of the hidden-state space, layer by layer.

    What it does: calls GEOMETRY-SCAN's ``geometry_report`` on per-layer
    hidden states (samples x dim) and reports intrinsic dimension, the
    principal-direction variance share and the anisotropy (mean pairwise
    cosine) per layer.

    What it does NOT prove: that a low intrinsic dimension means the layer
    stores less, or that anisotropy is a defect.  These are shape
    statistics of the samples they were given - with few samples they are
    noisy, and they say nothing about the model's understanding.
    """
    def _run():
        module, failure = require("geometry_scan", SOURCE_GEOMETRY_SCAN)
        if failure is not None:
            return failure
        if states_by_layer is None:
            from venom.integrations._harness import tiny_states
            states_by_layer = tiny_states(layers=4, samples=10, dim=8, seed=2)
        report = module.geometry_report(states_by_layer, names)
        return ok(
            SOURCE_GEOMETRY_SCAN,
            report=report,
            note=(
                "Shape statistics of the samples supplied. They do not say "
                "how much the layer 'knows', and they are only as stable as "
                "the number of samples given."
            ),
        )
    return guard(SOURCE_GEOMETRY_SCAN, _run)
