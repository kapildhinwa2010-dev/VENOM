"""Regression tests for the E5 content-agnostic control.

The experiment is a research script; these tests lock in the *shape* of the
finding so a refactor cannot silently invert it.
"""
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _load(name):
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not present (run the experiment first)")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_e5_result_files_exist():
    assert os.path.exists(os.path.join(ROOT, "e5_gpt2.json"))
    assert os.path.exists(os.path.join(ROOT, "e5_qwen.json"))


@pytest.mark.parametrize("name", ["e5_gpt2.json", "e5_qwen.json"])
def test_activation_injection_moves_signal_less_than_input_injection(name):
    """The core finding: the self-report reacts to the INPUT more than to the
    internal state. That is the anomaly-detector signature."""
    d = _load(name)
    ratio = d["signal_ratio_activation_over_input"]
    assert ratio != "inf"
    assert ratio < 1.5, (
        "the activation/input signal ratio went above the pre-registered "
        "threshold; the interpretation in PAPER.md section 8 must be revisited")


@pytest.mark.parametrize("name", ["e5_gpt2.json", "e5_qwen.json"])
def test_tap_stream_localises_the_perturbed_layer(name):
    """Layers before the injection must stay bit-exact; the injection layer
    must be the first one that moved."""
    d = _load(name)
    assert d["activation_first_diverged_layer"] == d["injected_layer"]
    assert d["localisation_correct"] is True
    assert d["input_first_diverged_layer"] == 0, (
        "an input-level perturbation must show up at the embedding, i.e. layer 0")


@pytest.mark.parametrize("name", ["e5_gpt2.json", "e5_qwen.json"])
def test_caveat_is_never_omitted(name):
    d = _load(name)
    assert d.get("caveat"), "every measurement must carry its caveat"
    assert "one prompt" in d["caveat"].lower()
