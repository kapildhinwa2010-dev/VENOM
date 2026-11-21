"""Verification integrations: Draft_ponit_harness, NOISE-STORM.

* Draft_ponit_harness (``drift-point``) measures the first generated
  token at which a rank-k delta replay diverges from the original model.
* NOISE-STORM pressure-tests every block of a stream-additive stack with
  calibrated Gaussian noise and ranks the blocks by how fragile they are.

Both are *experimental* instruments.  Neither one says anything about
VENOM's own tools, and a drift point or a fragility ranking on a toy model
is not a result about a trained network.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from venom.integrations._base import guard, ok, require

SOURCE_DRIFT_POINT = "Draft_ponit_harness"
SOURCE_NOISE_STORM = "NOISE-STORM"


# --------------------------------------------------------------------------- #
# Draft_ponit_harness
# --------------------------------------------------------------------------- #

def drift_point_available() -> bool:
    """Is the Draft_ponit_harness sibling importable?"""
    from venom.integrations._loader import load
    return load("drift_point") is not None


def drift_point_run(prompt: str = "Tokyo is the capital of",
                   rank: int = 2,
                   steps: int = 4,
                   cal_prompts: Optional[List[str]] = None,
                   model: Any = None,
                   tok: Any = None) -> Dict[str, Any]:
    """Find the first token where a compressed replay diverges.

    What it does: drives the sibling's own ``collect_deltas``,
    ``build_bases``, ``Replay`` and ``measure_drift`` - it generates with
    the replay off and on and reports the index of the first
    disagreement.  With no model supplied, VENOM builds a tiny
    randomly-initialised GPT-2 locally (no download).

    What it does NOT prove: anything about a trained model.  On the
    default random-weight model the drift point is a property of that toy
    model; it is a wiring check that the adapter runs the sibling's real
    code.  It also says nothing about VENOM's own tools or interventions.
    """
    def _run():
        module, failure = require("drift_point", SOURCE_DRIFT_POINT)
        if failure is not None:
            return failure
        if model is None or tok is None:
            from venom.integrations._harness import tiny_model
            used_model, used_tok = tiny_model()
        else:
            used_model, used_tok = model, tok

        calibration = list(cal_prompts) if cal_prompts else [
            "The sun rises in the",
            "Water boils at",
            prompt,
        ]
        store = module.collect_deltas(used_model, used_tok, calibration)
        bases = module.build_bases(store, int(rank))
        replay = module.Replay(used_model, bases)
        try:
            result = module.measure_drift(used_model, used_tok, prompt, replay,
                                          steps=int(steps))
        finally:
            replay.remove()
        return ok(
            SOURCE_DRIFT_POINT,
            rank=int(rank),
            steps=int(steps),
            report=result,
            note=(
                "Measured on a tiny RANDOM-WEIGHT GPT-2 built locally "
                "unless a model is supplied. The drift point is a property "
                "of that model, not a general claim about compression."
            ),
        )
    return guard(SOURCE_DRIFT_POINT, _run)


# --------------------------------------------------------------------------- #
# NOISE-STORM
# --------------------------------------------------------------------------- #

def noise_storm_available() -> bool:
    """Is the NOISE-STORM sibling importable?"""
    from venom.integrations._loader import load
    return load("noise_storm") is not None


def noise_storm_run(blocks: Optional[List[Callable]] = None,
                    x: Optional[Any] = None,
                    score_fn: Optional[Callable] = None,
                    sigma_fracs: Optional[List[float]] = None,
                    trials: int = 2,
                    seed: int = 0) -> Dict[str, Any]:
    """Rank the blocks of a stack by how fragile they are under noise.

    What it does: calls NOISE-STORM's ``storm_report`` - for each block
    it adds Gaussian noise calibrated to that block's own write norm and
    measures the score degradation, then ranks the blocks.

    What it does NOT prove: that the most fragile block is the most
    important one, or that the fragility transfers to real inputs.  The
    measurement depends on the score function and on the synthetic noise;
    it is a property of the stack it was given.
    """
    def _run():
        module, failure = require("noise_storm", SOURCE_NOISE_STORM)
        if failure is not None:
            return failure
        import numpy as np

        if blocks is None:
            from venom.integrations._support import default_blocks
            stack = default_blocks()
        else:
            stack = list(blocks)
        start = np.ones(4) if x is None else np.asarray(x, dtype=float)
        if score_fn is None:
            from venom.integrations._support import default_score
            score = default_score()
        else:
            score = score_fn
        fracs = tuple(float(s) for s in (sigma_fracs if sigma_fracs else [0.25, 0.5, 1.0]))
        report = module.storm_report(stack, start, score, fracs,
                                     trials=int(trials), seed=int(seed))
        return ok(
            SOURCE_NOISE_STORM,
            sigma_fracs=list(fracs),
            trials=int(trials),
            report=report,
            note=(
                "Fragility under synthetic noise on the stack that was "
                "supplied. The most fragile block is not necessarily the "
                "most important one, and nothing was changed."
            ),
        )
    return guard(SOURCE_NOISE_STORM, _run)
