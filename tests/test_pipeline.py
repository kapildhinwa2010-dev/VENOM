"""Tests for the automated surgery pipeline and its decision ladder."""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from venom.surgeon.pipeline import (FLAG_COMPRESS, FLAG_LAYER, KEEP,  # noqa: E402
                                    LADDER_RULES, REMOVE, ZERO, decide,
                                    powerlaw_exponent, render_ladder,
                                    twin_tower_clusters)


PATH = "transformer.h.5.mlp.c_fc"
PREFIX = "transformer.h.5."


def _units(**rates):
    """Build the minimal census-shaped input the ladder consumes."""
    return {PATH: {"fired_rate": list(rates.values())}}


def test_ladder_rules_are_ordered_and_unique():
    orders = [r.order for r in LADDER_RULES]
    assert orders == sorted(orders) == list(range(1, len(LADDER_RULES) + 1))
    assert len({r.action for r in LADDER_RULES}) == len(LADDER_RULES)
    assert LADDER_RULES[-1].action == KEEP, "the last rule must be the default"


def test_first_matching_rule_wins():
    """A unit that is BOTH exactly-dead and in a twin layer must be REMOVE,
    because the ladder is an ordered chain, not a set of independent votes."""
    v = decide(_units(a=0.0), {PREFIX}, {PREFIX}, rare_threshold=0.001)
    assert v[0].action == REMOVE
    assert v[0].rule == "exactly_dead"


def test_ladder_covers_every_branch():
    v = decide(_units(dead=0.0, rare=0.0005, alive=1.0),
               {PREFIX}, {PREFIX}, rare_threshold=0.001)
    by_index = {x.index: x for x in v}
    assert by_index[0].action == REMOVE
    assert by_index[1].action == ZERO
    assert by_index[2].action == FLAG_LAYER      # alive + twin layer


def test_compressible_rule_only_fires_without_a_higher_rule():
    v = decide(_units(a=1.0), set(), {PREFIX}, rare_threshold=0.001)
    assert v[0].action == FLAG_COMPRESS


def test_keep_is_the_default():
    v = decide(_units(a=1.0), set(), set(), rare_threshold=0.001)
    assert v[0].action == KEEP
    assert v[0].rule == "default"


def test_ladder_uses_prefix_matching_not_exact_match():
    """The layer sets are path PREFIXES ('transformer.h.5.'). An exact
    comparison would silently never match."""
    v = decide(_units(a=1.0), {PREFIX}, set(), rare_threshold=0.001)
    assert v[0].action == FLAG_LAYER


def test_render_ladder_shows_every_rule_and_its_count():
    v = decide(_units(a=0.0, b=1.0), set(), set(), rare_threshold=0.001)
    text = render_ladder(v)
    assert "REMOVE" in text and "KEEP" in text
    assert "1 units" in text or "1 unit" in text
    for rule in LADDER_RULES:
        assert rule.name in text or rule.description in text


def test_powerlaw_exponent_of_a_structured_matrix_is_steeper():
    torch.manual_seed(0)
    noise = torch.randn(64, 64)
    # a rank-1 heavy-tailed direction makes the spectrum much steeper
    heavy = noise.clone()
    heavy[0] *= 30.0
    e_noise = powerlaw_exponent(noise)
    e_heavy = powerlaw_exponent(heavy)
    assert e_noise is not None and e_heavy is not None
    assert e_heavy < e_noise, "a heavier head must have a steeper slope"


def test_powerlaw_exponent_returns_none_for_tiny_input():
    assert powerlaw_exponent(torch.randn(2, 2)) is None


def test_twin_tower_clusters_finds_an_obvious_duplicate():
    taps = {
        "h_0": torch.randn(1, 4, 8),
        "h_1": torch.randn(1, 4, 8),
        "h_2": torch.randn(1, 4, 8),
    }
    taps["h_1"] = taps["h_0"] * 1.0001        # near-duplicate of layer 0
    out = twin_tower_clusters(taps, threshold=0.99)
    assert out["layers_compared"] == 3
    assert out["clusters"], "an obvious duplicate must be found"


def test_auto_surgery_dry_run_on_a_tiny_real_model(tmp_path):
    from transformers import GPT2Config, GPT2LMHeadModel

    from venom.surgeon.pipeline import auto_surgery

    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=64,
                     n_positions=32, n_inner=64)
    model = GPT2LMHeadModel(cfg).eval()
    out = auto_surgery(model, _Tok(), plan_prompts=["a b c", "d e"],
                       heldout_prompts=["f g"], apply=False)
    assert out["ok"] is True
    assert "census" in out and "ladder" in out
    assert out["ladder"]["tally"]
    assert out["applied"]["note"].startswith("dry run")


class _Tok:
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, text, return_tensors="pt"):
        ids = [(abs(hash(t)) % 30) + 1 for t in text.split()][:6] or [1]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}
