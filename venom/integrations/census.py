"""Census integrations: LAYER-CENSUS-, DEAD-CELL, TWIN-TOWER, SKIP-RUN-, POWERLAW-PROBE.

All five describe the *structure* of a model rather than its behaviour:

* LAYER-CENSUS-  - an atlas of the weight matrices (shape, Frobenius norm,
  effective rank, spectrum decay).
* DEAD-CELL      - zero rows/columns and never-active neurons.
* TWIN-TOWER     - duplicated / redundant layers by cross-layer cosine.
* SKIP-RUN-      - how much of the depth is real, by removing every k-th
  block and watching the score.
* POWERLAW-PROBE - the power-law exponent of a layer's activations versus
  its weights.

Every one of them is **report only**: nothing here edits a weight matrix
or removes a block.  A layer that looks dead may simply not have been
needed for the prompt that was run.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from venom.integrations._base import guard, ok, require
from venom.integrations._support import default_weights as _default_weights

SOURCE_LAYER_CENSUS = "LAYER-CENSUS-"
SOURCE_DEAD_CELL = "DEAD-CELL"
SOURCE_TWIN_TOWER = "TWIN-TOWER"
SOURCE_SKIP_RUN = "SKIP-RUN-"
SOURCE_POWERLAW = "POWERLAW-PROBE"


# --------------------------------------------------------------------------- #
# LAYER-CENSUS-
# --------------------------------------------------------------------------- #

def layer_census_available() -> bool:
    """Is the LAYER-CENSUS- sibling importable?"""
    from venom.integrations._loader import load
    return load("layer_census") is not None


def layer_census_run(weights: Optional[Any] = None,
                     names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Build the sibling's per-layer atlas of weight matrices.

    What it does: calls LAYER-CENSUS-'s ``census`` over a list of 2-D
    weight matrices and reports shape, parameter count, Frobenius norm,
    effective rank and the top-1 / top-10 variance shares per layer.

    What it does NOT prove: that a low effective rank means the layer is
    redundant or prunable.  Effective rank is a spectrum statistic of the
    matrix; whether the layer can be compressed without changing the
    model's behaviour needs a behavioural test (and a drift proof).
    """
    def _run():
        module, failure = require("layer_census", SOURCE_LAYER_CENSUS)
        if failure is not None:
            return failure
        matrices = list(weights) if weights else _default_weights()
        report = module.census(matrices, names)
        return ok(
            SOURCE_LAYER_CENSUS,
            report=report,
            note=(
                "Report only: a low effective rank is a spectrum statistic, "
                "not a licence to prune. Nothing was modified."
            ),
        )
    return guard(SOURCE_LAYER_CENSUS, _run)


# --------------------------------------------------------------------------- #
# DEAD-CELL
# --------------------------------------------------------------------------- #

def dead_cell_available() -> bool:
    """Is the DEAD-CELL sibling importable?"""
    from venom.integrations._loader import load
    return load("dead_cell") is not None


def dead_cell_run(weights: Optional[Any] = None,
                  activations: Optional[Any] = None,
                  act_threshold: float = 1e-6) -> Dict[str, Any]:
    """Map dead weight structure and never-active neurons.

    What it does: calls DEAD-CELL's ``dead_cells`` over a list of weight
    matrices (optionally paired with activation snapshots of shape
    ``(samples, neurons)``) and aggregates the result with
    ``total_waste``.

    What it does NOT prove: that a zero row or a never-firing neuron is
    dead weight.  A neuron can be silent because the prompts never
    needed it.  Nothing is pruned and no weight is touched.
    """
    def _run():
        module, failure = require("dead_cell", SOURCE_DEAD_CELL)
        if failure is not None:
            return failure
        import numpy as np

        matrices = list(weights) if weights else _default_weights(seed=4)
        if activations is None:
            # give the default fixture two obviously dead rows to find
            matrices = [m.copy() for m in matrices]
            matrices[0][0, :] = 0.0
            matrices[0][1, :] = 0.0
            acts = None
        else:
            acts = list(activations)
        entries = module.dead_cells(matrices, acts,
                                    act_threshold=float(act_threshold))
        total = module.total_waste(entries)
        return ok(
            SOURCE_DEAD_CELL,
            act_threshold=float(act_threshold),
            entries=entries,
            total=total,
            note=(
                "Waste is measured against the samples supplied. A neuron "
                "that never fired may simply not have been needed. Report "
                "only - nothing was pruned."
            ),
        )
    return guard(SOURCE_DEAD_CELL, _run)


# --------------------------------------------------------------------------- #
# TWIN-TOWER
# --------------------------------------------------------------------------- #

def twin_tower_available() -> bool:
    """Is the TWIN-TOWER sibling importable?"""
    from venom.integrations._loader import load
    return load("twin_tower") is not None


def twin_tower_run(weights: Optional[Any] = None,
                   cos_tol: float = 0.9) -> Dict[str, Any]:
    """Cluster layers that are near-duplicates of each other.

    What it does: calls TWIN-TOWER's ``duplication_report`` over a stack
    of same-shape weight matrices and reports the clusters, the number of
    duplicate layers and the redundancy rate.

    What it does NOT prove: that two similar layers can be merged, or that
    the model would behave the same afterwards.  Cosine similarity between
    weight matrices is not a behavioural equivalence test.
    """
    def _run():
        module, failure = require("twin_tower", SOURCE_TWIN_TOWER)
        if failure is not None:
            return failure
        import numpy as np

        matrices = list(weights) if weights else _default_weights(seed=6)
        if len(matrices) >= 2:
            # make the second layer a near-duplicate of the first
            matrices[1] = matrices[0] * 0.99 + 0.01 * matrices[1]
        report = module.duplication_report(matrices, cos_tol=float(cos_tol))
        return ok(
            SOURCE_TWIN_TOWER,
            cos_tol=float(cos_tol),
            report=report,
            note=(
                "Weight-space similarity only. It does not show that two "
                "layers can be merged, and no layer was merged."
            ),
        )
    return guard(SOURCE_TWIN_TOWER, _run)


# --------------------------------------------------------------------------- #
# SKIP-RUN-
# --------------------------------------------------------------------------- #

def skip_run_available() -> bool:
    """Is the SKIP-RUN- sibling importable?"""
    from venom.integrations._loader import load
    return load("skip_run") is not None


def skip_run_run(blocks: Optional[List[Callable]] = None,
                 x: Optional[Any] = None,
                 score_fn: Optional[Callable] = None,
                 k_values: Optional[List[int]] = None,
                 tolerance: float = 0.05) -> Dict[str, Any]:
    """Measure how much of a block stack's depth is real.

    What it does: calls SKIP-RUN-'s ``skip_sweep`` - the stack is run
    again with every k-th block removed - and ``real_depth`` for the
    largest k that stays within the tolerance.

    What it does NOT prove: that the removed blocks are useless in general.
    They are useless *for this input and this score function*; a block that
    does nothing for one prompt can matter for another.
    """
    def _run():
        module, failure = require("skip_run", SOURCE_SKIP_RUN)
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
        ks = [int(k) for k in (k_values if k_values else [1, 2, 3, 4])]
        report = module.skip_sweep(stack, start, score, tuple(ks))
        depth = module.real_depth(report, tolerance=float(tolerance))
        return ok(
            SOURCE_SKIP_RUN,
            tolerance=float(tolerance),
            report=report,
            real_depth=depth,
            note=(
                "Depth that does not matter *for this input and score*. It "
                "does not mean those blocks are useless for other prompts, "
                "and no block was removed from any model."
            ),
        )
    return guard(SOURCE_SKIP_RUN, _run)


# --------------------------------------------------------------------------- #
# POWERLAW-PROBE
# --------------------------------------------------------------------------- #

def powerlaw_probe_available() -> bool:
    """Is the POWERLAW-PROBE sibling importable?"""
    from venom.integrations._loader import load
    return load("powerlaw_probe") is not None


def powerlaw_probe_run(prompt: str = "Tokyo is the capital of Japan.",
                       layer: int = -1,
                       model: Any = None,
                       tok: Any = None) -> Dict[str, Any]:
    """Compare the power-law exponent of a layer's activations and weights.

    What it does: drives the sibling's own ``probe`` - it collects the
    layer's activations with a forward hook, takes the singular-value
    spectrum of activations and weights and fits a power-law exponent to
    each.  With no model supplied, VENOM builds a tiny randomly-initialised
    GPT-2 locally so no download is needed.

    What it does NOT prove: anything about a trained model.  On the default
    (random-weight) model the exponents describe that toy model only; the
    "alpha gap" is a fingerprint of a particular matrix pair, not a
    statement about a network's behaviour.
    """
    def _run():
        module, failure = require("powerlaw_probe", SOURCE_POWERLAW)
        if failure is not None:
            return failure
        if model is None or tok is None:
            from venom.integrations._harness import tiny_model
            used_model, used_tok = tiny_model()
        else:
            used_model, used_tok = model, tok

        n_layers = len(module.get_layers(used_model))
        idx = module.resolve_layer_index(int(layer), n_layers)
        report = module.probe(used_model, used_tok, prompt, idx)
        return ok(
            SOURCE_POWERLAW,
            layer=idx,
            n_layers=n_layers,
            report=report,
            note=(
                "By default this runs on a tiny RANDOM-WEIGHT GPT-2 built "
                "locally, so the exponents describe that toy model. The "
                "measurement is a wiring check, not a result about a "
                "trained network."
            ),
        )
    return guard(SOURCE_POWERLAW, _run)
