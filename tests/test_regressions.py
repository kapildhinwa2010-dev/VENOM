"""Regression tests for the v0.1.2 bug-fix batch.

Every test here corresponds to a confirmed bug found in the adversarial
self-review. Each test names the bug it locks down, so a future regression
points straight at what broke.
"""
import re

import pytest

from venom.bench.correction_bench import MockHost, run_bench
from venom.core.hostmap import fingerprint, get_blocks
from venom.core.session import Session
from venom.memory.store import MemoryStore, norm
from venom.reflex.budget import Budget
from venom.reflex.injector import Injector
from venom.reflex.triggers import Reflex
from venom.sensors.uncertainty import token_uncertainty
from venom.server import handle, make_server
from venom.surgeon.report import compressibility_report, dead_neuron_report
from venom.tools.handlers import ToolHandlers
from venom.verify.drift import drift_report


# ---------------------------------------------------------------- B1: hostmap
def _fake_model(model_type, blocks_attr="model.layers", norm_attr="model.norm", layers=4):
    """Synthetic HF-shaped model. Paths are set exactly as the real classes use."""
    class Block:
        pass

    Cfg = type("Cfg", (), {
        "hidden_size": 768,
        "vocab_size": 32000,
        "num_attention_heads": 12,
        "model_type": model_type,
    })

    m = type("M", (), {})()
    m.config = Cfg

    blocks_holder = type("Holder", (), {})()
    blocks_holder.layers = [Block() for _ in range(layers)]
    norm_holder = type("Holder", (), {})()
    setattr(norm_holder, norm_attr.split(".")[-1], object())

    if blocks_attr == "transformer.h":
        m.transformer = blocks_holder
        m.transformer.h = blocks_holder.layers
        setattr(m.transformer, norm_attr.split(".")[-1], object())
    else:
        m.model = blocks_holder
        setattr(m.model, norm_attr.split(".")[-1], object())

    m.lm_head = object()
    return m


@pytest.mark.parametrize("model_type,expected", [
    ("gpt2", "gpt2"), ("llama", "llama"), ("qwen2", "qwen2"),
    ("mistral", "mistral"), ("gemma", "gemma"), ("phi", "phi"),
])
def test_b1_hostmap_respects_model_type(model_type, expected):
    """B1: every non-GPT-2 model used to be labelled 'llama'."""
    if model_type == "gpt2":
        m = _fake_model("gpt2", blocks_attr="transformer.h", norm_attr="transformer.ln_f")
    elif model_type == "phi":
        m = _fake_model("phi", norm_attr="model.final_layernorm")
    else:
        m = _fake_model(model_type)
    fp = fingerprint(m, model_type)
    assert fp["family"] == expected, f"{model_type} detected as {fp['family']}"


def test_b1_hostmap_phi_norm_path_exists():
    """B1: Phi was given 'model.norm', which does not exist on the model."""
    m = _fake_model("phi", norm_attr="model.final_layernorm")
    fp = fingerprint(m, "phi")
    assert fp["norm_path"] == "model.final_layernorm"
    assert hasattr(m.model, "final_layernorm")


def test_b1_get_blocks_raises_clear_error():
    """B1: get_blocks() raised KeyError('blocks_path') with no context."""
    fp = {"family": "unknown"}
    with pytest.raises(KeyError) as exc:
        get_blocks(object(), fp)
    assert "blocks_path" in str(exc.value)


# ------------------------------------------------- B2: punctuation normalisation
def test_b2_norm_does_not_join_words():
    """B2: norm() deleted punctuation, merging 'hello,world' -> 'helloworld'."""
    assert norm("hello,world") == "hello world"
    assert norm("a.b") == "a b"


def test_b2_distinct_keys_do_not_collide():
    """B2: 'don't'/'dont' used to collapse into one slot, losing a fact."""
    s = MemoryStore()
    s.remember("don't stop", "value A")
    s.remember("dont stop", "value B")
    assert s.stats()["slots"] == 2
    assert s.recall("don't stop").value == "value A"
    assert s.recall("dont stop").value == "value B"


def test_b2_comma_key_is_recallable():
    s = MemoryStore()
    s.remember("hello world", "hi")
    assert s.recall("hello,world").value == "hi"


# --------------------------------------------- B3: digest survives persistence
def test_b3_tampered_session_file_abstains():
    """B3: a tampered session file loaded as a HIT - the digest guard died
    at the persistence boundary."""
    sess = Session(session_id="s")
    sess.store.remember("london", "uk")
    blob = sess.to_dict()
    blob["memory"][0]["value"] = "TAMPERED"

    restored = Session.from_dict(blob)
    r = restored.store.recall("london")
    assert r.status == "abstain"
    assert r.reason == "digest_mismatch"


def test_b3_untampered_session_file_still_hits():
    sess = Session()
    sess.store.remember("london", "uk")
    restored = Session.from_dict(sess.to_dict())
    assert restored.store.recall("london").value == "uk"


# ------------------------------------------ B4: session keeps its own bounds
def test_b4_memory_capacity_survives_roundtrip():
    """B4: restoring a session silently reset the memory cap to 1024."""
    sess = Session()
    sess.store.capacity = 7
    restored = Session.from_dict(sess.to_dict())
    assert restored.store.capacity == 7


def test_b4_budget_usage_survives_roundtrip():
    """B4: restoring a session re-armed the full tool-call budget."""
    sess = Session()
    sess.budget.take()
    sess.budget.take()
    restored = Session.from_dict(sess.to_dict())
    assert restored.budget.used == 2
    assert restored.budget.state()["left"] == 1


def test_b4_budget_max_survives_roundtrip():
    sess = Session()
    sess.budget = Budget(max_calls=9)
    restored = Session.from_dict(sess.to_dict())
    assert restored.budget.max_calls == 9


# -------------------------------------------------- B5: multi-hit injection
def test_b5_matching_keys_returns_every_hit():
    s = MemoryStore()
    s.remember("capital of france", "Paris")
    s.remember("capital of italy", "Rome")
    keys = list(Reflex(s).matching_keys("capital of france and capital of italy"))
    assert len(keys) == 2
    assert "capital of france" in keys and "capital of italy" in keys


def test_b5_multi_hit_injection_is_not_dead_code():
    s = MemoryStore()
    s.remember("capital of france", "Paris")
    s.remember("capital of italy", "Rome")
    inj = Injector(s)
    out = inj.maybe_inject("tell me about capital of france and capital of italy")
    assert out["changed"]
    assert len(out["recalls"]) == 2
    assert "Paris" in out["injected"] and "Rome" in out["injected"]


def test_b5_multi_hit_format_is_parseable_by_mockhost():
    """B5: the multi-line CONTEXT format was not parseable by MockHost."""
    s = MemoryStore()
    s.remember("capital of france", "Paris")
    s.remember("capital of italy", "Rome")
    inj = Injector(s)
    prompt = inj.as_prompt_fn()("capital of france and capital of italy")
    assert MockHost({})(prompt).strip() == "Paris ; Rome"


# ------------------------------------------ B6: abstention must not hide harm
def test_b6_correct_to_abstain_counts_as_harm():
    """B6: replacing a correct answer with 'abstain' was invisible in
    `harmed` and inflated `net_gain`."""
    r = drift_report(["paris is in france"], ["abstain"], ["france"])
    assert r["harmed"] == 1
    assert r["net_gain"] == -1
    assert r["abstained"] == 0


def test_b6_wrong_to_abstain_is_an_abstention_not_a_harm():
    r = drift_report(["paris is in germany"], ["abstain"], ["france"])
    assert r["abstained"] == 1
    assert r["harmed"] == 0


def test_b6_identical_abstentions_are_unchanged():
    r = drift_report(["abstain"], ["abstain"], ["france"])
    assert r["unchanged"] == 1
    assert r["abstained"] == 0


def test_b6_categories_are_exhaustive():
    r = drift_report(["a", "b", "c", "d"],
                     ["a", "abstain", "x", "y"], ["a", "b", "c", "d"])
    assert r["unchanged"] + r["changed"] == r["n"]
    assert r["changed"] == r["fixed"] + r["harmed"] + r["shifted"] + r["abstained"]


# ------------------------------------------------------ B7/B8: JSON-RPC
def test_b7_notification_without_id_gets_no_reply():
    """B7: a notification has no id; answering it is a JSON-RPC violation."""
    assert handle({"jsonrpc": "2.0", "method": "exit"}, make_server()) is None


def test_b7_request_with_id_gets_a_reply():
    """B7: a request that happened to be named 'notifications/...' was dropped."""
    resp = handle({"jsonrpc": "2.0", "id": 5, "method": "notifications/initialized"},
                  make_server())
    assert resp is not None
    assert resp["id"] == 5


def test_b8_malformed_arguments_do_not_lose_the_id():
    """B8: a malformed tools/call answered with id=null and code -32700."""
    resp = handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                   "params": {"name": "venom.recall", "arguments": [1, 2]}}, make_server())
    assert resp["id"] == 9
    assert resp["result"]["isError"] is True


def test_b8_malformed_params_do_not_lose_the_id():
    resp = handle({"jsonrpc": "2.0", "id": 10, "method": "tools/call",
                   "params": "not-a-dict"}, make_server())
    assert resp["id"] == 10


def test_b8_tool_exception_becomes_an_error_result_not_a_crash():
    class Boom(ToolHandlers):
        def call(self, name, args):
            raise RuntimeError("boom")

    resp = handle({"jsonrpc": "2.0", "id": 11, "method": "tools/call",
                   "params": {"name": "venom.recall", "arguments": {}}},
                  Boom(MemoryStore()))
    assert resp["id"] == 11
    assert resp["result"]["isError"] is True


# --------------------------------------------------- B9: empty logit row
def test_b9_empty_logit_row_does_not_raise():
    """B9: token_uncertainty([]) raised ValueError and killed the sweep."""
    u = token_uncertainty([])
    assert u["is_uncertain"] is True
    assert u["confidence"] == 0.0


# ------------------------------------------------- B10: gold must be a word
def test_b10_gold_match_is_word_bounded():
    """B10: substring matching made 'the cat sat' a match for 'C++'."""
    from venom.bench.correction_bench import _match
    assert _match("the cat sat", "C++") is False
    assert _match("ukraine", "uk") is False


def test_b10_real_gold_still_matches():
    from venom.bench.correction_bench import _match
    assert _match("the capital is paris", "Paris") is True
    assert _match("Paris", "paris") is True


# ------------------------------------------- B12: injector -> bench adapter
def test_b12_injector_feeds_run_bench_end_to_end():
    """B12: maybe_inject returns a dict; run_bench needs a prompt function."""
    s = MemoryStore()
    s.remember("capital of france", "Paris")
    inj = Injector(s)
    host = MockHost({"capital of france": "Berlin"})
    facts = [{"q": "capital of france", "gold": "Paris"}]
    out = run_bench(host, facts, inj.as_prompt_fn())
    assert out["fixed"] == 1
    assert out["harmed"] == 0


def test_b12_run_bench_accepts_repeats_and_seed():
    class Seeded:
        def __init__(self):
            self.n = 0

        def set_seed(self, seed):
            self.n = seed

        def __call__(self, prompt):
            return "Paris"

    host = Seeded()
    facts = [{"q": "capital of france", "gold": "Paris"}]
    out = run_bench(host, facts, None, repeats=3, seed=7)
    assert out["baseline_acc"] == 1.0


# ------------------------------------------------- B18: corrupt_for_test
def test_b18_corrupt_for_test_raises_clear_keyerror():
    s = MemoryStore()
    with pytest.raises(KeyError) as exc:
        s.corrupt_for_test("atlantis", "x")
    assert "atlantis" in str(exc.value)


# ------------------------------------------------- B19: surgeon schema
def test_b19_zero_spectrum_keeps_the_same_schema():
    r = compressibility_report({"z": [0.0, 0.0]})
    assert set(r["details"]["z"]) == {"eff_rank", "full_rank", "ratio", "compressible"}


# ------------------------------------------------- B20: low_threshold
def test_b20_low_bucket_survives_a_high_threshold():
    counts = {"a": 0, "b": 50, "c": 900}
    r = dead_neuron_report(counts, 1000, threshold=0.01, low_threshold=0.1)
    assert r["dead"] == 1
    assert r["low"] == 1
    assert r["active"] == 1


# ------------------------------------------------- B16: budget per answer
def test_b16_initialize_rearms_the_budget():
    """B16: the cap was per process, so the host went mute after 3 calls."""
    h = make_server()
    handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "venom.time_now", "arguments": {}}}, h)
    handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "venom.time_now", "arguments": {}}}, h)
    assert h.budget.used == 2
    handle({"jsonrpc": "2.0", "id": 3, "method": "initialize"}, h)
    assert h.budget.used == 0
