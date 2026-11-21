"""Sensor integrations: INNER-VOTE-, LOGIT-STREAM, STATE-MIRROR, VARIANCE-LIES.

These siblings read *internal* signals out of a forward pass.  They are
adapters, not new sensors: VENOM's own `venom/sensors/*` remain the
built-in fallback and the default path.

The honesty rule applies to all four (see LIMITATIONS.md §4c):

* a cosine between hidden states is a similarity of representations, not a
  measurement of knowledge, understanding or introspection;
* a "birth layer" says where a logit *rose*, not that the layer caused
  the token;
* "variance captured" is a compression statistic, and VARIEANCE-LIES exists
  precisely because it does not imply "behaviour preserved".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from venom.integrations._base import guard, ok, require

SOURCE_INNER_VOTE = "INNER-VOTE-"
SOURCE_LOGIT_STREAM = "LOGIT-STREAM"
SOURCE_STATE_MIRROR = "STATE-MIRROR"
SOURCE_VARIANCE_LIES = "VARIEANCE-LIES"


# --------------------------------------------------------------------------- #
# INNER-VOTE-
# --------------------------------------------------------------------------- #

def inner_vote_available() -> bool:
    """Is the INNER-VOTE- sibling importable?"""
    from venom.integrations._loader import load
    return load("inner_vote") is not None


def inner_vote_run(states_by_layer: Optional[Any] = None,
                   low: float = 0.7) -> Dict[str, Any]:
    """Measure how much paraphrases of one prompt agree, layer by layer.

    What it does: calls INNER-VOTE's ``consensus_report`` on per-layer
    hidden states of N paraphrases and reports the mean/minimum pairwise
    cosine per layer plus the first layer where the paraphrases disagree.

    What it does NOT prove: that the model is self-consistent about
    *knowledge*.  The input is hidden states, not answers, so this is a
    representational similarity - a prompt can be answered consistently
    while the states drift, and vice versa.
    """
    def _run():
        module, failure = require("inner_vote", SOURCE_INNER_VOTE)
        if failure is not None:
            return failure
        if states_by_layer is None:
            from venom.integrations._harness import tiny_states
            data = tiny_states(layers=4, samples=3, dim=6, seed=1)
        else:
            data = states_by_layer
        report = module.consensus_report(data, low=float(low))
        return ok(
            SOURCE_INNER_VOTE,
            low=float(low),
            report=report,
            note=(
                "Pairwise cosine between hidden states of paraphrases. It is "
                "a representational agreement measure, not evidence about "
                "what the model knows."
            ),
        )
    return guard(SOURCE_INNER_VOTE, _run)


# --------------------------------------------------------------------------- #
# LOGIT-STREAM
# --------------------------------------------------------------------------- #

def logit_stream_available() -> bool:
    """Is the LOGIT-STREAM sibling importable?"""
    from venom.integrations._loader import load
    return load("logit_stream") is not None


def logit_stream_run(prefix_logits: Optional[Any] = None,
                     target: Optional[int] = None) -> Dict[str, Any]:
    """Decompose a logit-lens prefix stream into per-layer increments.

    What it does: calls LOGIT-STREAM's ``birth_certificate`` on the
    logits observed after each layer prefix and reports the per-layer
    increments for the target token plus the first layer whose argmax
    already equals the final answer.

    What it does NOT prove: causality.  A birth layer shows where the
    logit *rose*; it does not show that the layer caused the token.  A
    causal claim needs an intervention (ablate the layer, re-measure).
    """
    def _run():
        module, failure = require("logit_stream", SOURCE_LOGIT_STREAM)
        if failure is not None:
            return failure
        import numpy as np

        if prefix_logits is None:
            vocab, layers = 8, 5
            rows = []
            for i in range(layers):
                row = np.full(vocab, -2.0, dtype=float)
                row[:] += 0.05 * i
                row[3] += 0.4 * (i + 1)
                rows.append(row)
            data = rows
        else:
            data = prefix_logits
        report = module.birth_certificate(data, target)
        return ok(
            SOURCE_LOGIT_STREAM,
            report=report,
            note=(
                "Logit-lens increments, not a causal proof: the birth layer "
                "is where the logit rose, not where the decision was "
                "caused."
            ),
        )
    return guard(SOURCE_LOGIT_STREAM, _run)


# --------------------------------------------------------------------------- #
# STATE-MIRROR
# --------------------------------------------------------------------------- #

def state_mirror_available() -> bool:
    """Is the STATE-MIRROR sibling importable?"""
    from venom.integrations._loader import load
    return load("state_mirror") is not None


def state_mirror_run(states_a: Optional[Any] = None,
                     states_b: Optional[Any] = None,
                     threshold: float = 0.9) -> Dict[str, Any]:
    """Compare two twin prompts layer by layer.

    What it does: calls STATE-MIRROR's ``mirror_report`` on the per-layer
    hidden states of two prompts, reporting the per-layer cosine, the
    per-layer L2 distance, the first divergent layer and the layer with
    the largest change.

    What it does NOT prove: that the divergence was caused by the token
    that differs between the twins, nor that an early divergence means an
    early decision.  The report is a description of the two forward
    passes, nothing more.
    """
    def _run():
        module, failure = require("state_mirror", SOURCE_STATE_MIRROR)
        if failure is not None:
            return failure
        import numpy as np

        if states_a is None or states_b is None:
            rng = np.random.default_rng(3)
            a, b = [], []
            for layer in range(4):
                sa = rng.normal(size=(3, 5))
                sb = sa.copy()
                if layer >= 2:
                    sb = sb + rng.normal(scale=1.2, size=(3, 5))
                a.append(sa)
                b.append(sb)
            left = a if states_a is None else states_a
            right = b if states_b is None else states_b
        else:
            left, right = states_a, states_b
        report = module.mirror_report(left, right, threshold=float(threshold))
        return ok(
            SOURCE_STATE_MIRROR,
            threshold=float(threshold),
            report=report,
            note=(
                "A description of two forward passes. It does not attribute "
                "the divergence to the differing token and it does not make "
                "the divergent layer the cause of the output."
            ),
        )
    return guard(SOURCE_STATE_MIRROR, _run)


# --------------------------------------------------------------------------- #
# VARIANCE-LIES
# --------------------------------------------------------------------------- #

def variance_lies_available() -> bool:
    """Is the VARIANCE-LIES sibling importable?"""
    from venom.integrations._loader import load
    return load("variance_lies") is not None


def variance_lies_run(prompts: Optional[List[str]] = None,
                      ranks: Optional[List[int]] = None,
                      steps: int = 3) -> Dict[str, Any]:
    """Compare variance captured by a rank-k replay with behaviour preserved.

    What it does: drives the sibling's own ``collect_deltas`` / ``spectrum``
    / ``probe`` over a *tiny randomly-initialised* GPT-2 that VENOM builds
    locally (no download), so the sibling's code really runs.

    What it does NOT prove: anything about a trained model.  The model is
    random-weight, so the numbers describe that toy model only; they are a
    plumbing check that the adapter wires the sibling up correctly and
    never a result about language models.
    """
    def _run():
        module, failure = require("variance_lies", SOURCE_VARIANCE_LIES)
        if failure is not None:
            return failure
        from venom.integrations._harness import tiny_model

        text = list(prompts) if prompts else [
            "Tokyo is the capital of",
            "Water boils at",
        ]
        ranks_list = [int(r) for r in (ranks if ranks else [2, 4])]
        model, tok = tiny_model()
        store = module.collect_deltas(model, tok, text)
        curve = [module.probe(model, tok, store, text, r, int(steps))
                 for r in ranks_list]
        return ok(
            SOURCE_VARIANCE_LIES,
            ranks=ranks_list,
            steps=int(steps),
            curve=curve,
            note=(
                "Measured on a tiny RANDOM-WEIGHT GPT-2 built locally. The "
                "sibling's point - variance captured is not behaviour "
                "preserved - is structural; these particular numbers "
                "describe a toy model and are not a result about language "
                "models."
            ),
        )
    return guard(SOURCE_VARIANCE_LIES, _run)
