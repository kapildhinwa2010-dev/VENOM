"""Tests for the v0.3.0 tool surface: self-read, intervention and state tools.

The self-read tools are the interesting ones: they must work on a real
forward pass, and they must degrade honestly when the host gives them nothing.
"""
import json

import pytest

from venom.core.runtime import VenomConfig, VenomRuntime
from venom.memory.store import MemoryStore
from venom.reflex.budget import Budget
from venom.tools.handlers import KNOWN_TOOLS, ToolHandlers
from venom.tools.schema import ALL_TOOLS, TIER2_TOOLS


def _handlers(**kw):
    return ToolHandlers(MemoryStore(), Budget(max_calls=50), **kw)


# ------------------------------------------------------------------- surface
def test_the_full_tool_surface_exists():
    names = [t["name"] for t in ALL_TOOLS]
    assert len(names) == 14
    for expected in ("venom.recall", "venom.remember", "venom.time_now",
                     "venom.uncertainty", "venom.check_myself", "venom.why",
                     "venom.where", "venom.dial", "venom.audit",
                     "venom.verify", "venom.forget", "venom.snapshot",
                     "venom.restore", "venom.report"):
        assert expected in names


def test_tier2_excludes_the_tools_that_need_internals():
    names = [t["name"] for t in TIER2_TOOLS]
    assert "venom.recall" in names and "venom.verify" in names
    assert "venom.uncertainty" not in names
    assert "venom.check_myself" not in names
    assert "venom.dial" not in names


def test_every_tool_has_a_schema():
    for t in ALL_TOOLS:
        assert t["name"].startswith("venom.")
        assert t["description"]
        assert t["inputSchema"]["type"] == "object"
        assert isinstance(t["inputSchema"]["properties"], dict)


# ------------------------------------------------------------- memory tools
def test_forget_removes_a_fact():
    h = _handlers()
    h.call("venom.remember", {"key": "capital of france", "value": "Paris"})
    out = h.call("venom.forget", {"key": "capital of france"})
    assert out["ok"] and out["forgotten"]
    assert h.call("venom.recall", {"query": "capital of france"})["status"] == "abstain"


def test_forget_of_an_unknown_key_says_so():
    out = _handlers().call("venom.forget", {"key": "atlantis"})
    assert out["ok"] is False and out["error"] == "not_found"


def test_verify_confirms_a_correct_answer():
    h = _handlers()
    h.call("venom.remember", {"key": "capital of france", "value": "Paris"})
    out = h.call("venom.verify", {"answer": "the capital is Paris",
                                   "key": "capital of france"})
    assert out["verdict"] == "verified"


def test_verify_flags_a_contradiction():
    h = _handlers()
    h.call("venom.remember", {"key": "capital of france", "value": "Paris"})
    out = h.call("venom.verify", {"answer": "the capital is Berlin",
                                   "key": "capital of france"})
    assert out["verdict"] == "contradicted"
    assert out["stored_value"] == "Paris"


def test_verify_abstains_when_nothing_is_stored():
    out = _handlers().call("venom.verify", {"answer": "Paris"})
    assert out["verdict"] == "unknown"


def test_verify_never_guesses():
    """A key with no stored value must not be 'verified' by luck."""
    h = _handlers()
    out = h.call("venom.verify", {"answer": "Paris", "key": "unknown key"})
    assert out["verdict"] == "unknown"


# -------------------------------------------------------------- state tools
def test_snapshot_and_restore_roundtrip():
    h = _handlers()
    sid = h.call("venom.snapshot", {"tag": "v1"})["snapshot_id"]
    assert sid
    out = h.call("venom.restore", {"snapshot_id": sid})
    assert out["ok"]


def test_restore_of_an_unknown_snapshot_says_so():
    out = _handlers().call("venom.restore", {"snapshot_id": "nope"})
    assert out["ok"] is False and out["error"] == "snapshot_not_found"


def test_report_returns_readable_text():
    out = _handlers().call("venom.report", {})
    assert out["ok"] and "VENOM REPORT" in out["report"]


# ----------------------------------------------------------- degraded modes
def test_self_read_tools_degrade_honestly_without_a_probe():
    h = _handlers()
    out = h.call("venom.uncertainty", {})
    assert out["ok"] is False and out["error"] == "no_probe"

    out = h.call("venom.check_myself", {})
    assert out["ok"] is False and "error" in out

    out = h.call("venom.why", {"token": " Paris"})
    assert out["ok"] is False and "error" in out


def test_self_read_tools_degrade_honestly_without_taps():
    h = _handlers()
    out = h.call("venom.check_myself", {})
    assert "error" in out
    out = h.call("venom.audit", {})
    assert "error" in out


def test_dial_without_a_manager_says_so():
    out = _handlers().call("venom.dial", {"layer": 0, "delta": 0.1})
    assert out["ok"] is False and out["error"] == "no_dial_manager"


def test_unknown_tool_does_not_consume_the_budget():
    h = _handlers()
    h.call("venom.definitely_not_real", {})
    assert h.budget.used == 0


def test_a_raising_tool_does_not_kill_the_handler():
    """A tool that throws must come back as an error, not crash the server."""

    class ExplodingStore(MemoryStore):
        def recall(self, query):
            raise RuntimeError("boom")

    h = ToolHandlers(ExplodingStore(), Budget(max_calls=5))
    out = h.call("venom.recall", {"query": "x"})
    assert out["ok"] is False
    assert out["error"] == "tool_exception"
    assert "boom" in out["message"]


# --------------------------------------------------------------- with torch
torch = pytest.importorskip("torch")


def _tiny_model():
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, d=16):
            super().__init__()
            self.self_attn = nn.Linear(d, d)
            self.mlp = nn.Linear(d, d)

        def forward(self, x):
            return x + self.mlp(self.self_attn(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Block() for _ in range(3)])
            self.model.norm = nn.LayerNorm(16)
            self.lm_head = nn.Linear(16, 32)

        def forward(self, x):
            for blk in self.model.layers:
                x = blk(x)
            return self.lm_head(self.model.norm(x))

    return Model().eval()


class _FakeTokenizer:
    def encode(self, text):
        return [abs(hash(text)) % 32]

    def decode(self, ids):
        return f"tok_{ids[0]}" if ids else ""


def test_uncertainty_works_on_a_real_forward_pass():
    from venom.core.probe import LogitProbe

    model = _tiny_model()
    probe = LogitProbe(model)
    probe.attach()
    try:
        with torch.no_grad():
            model(torch.zeros(1, 4, 16))
        u = probe.last_uncertainty()
        assert u is not None
        assert "entropy" in u and "is_uncertain" in u
    finally:
        probe.release()


def test_probe_reports_no_logits_before_a_forward_pass():
    from venom.core.probe import LogitProbe
    probe = LogitProbe(_tiny_model())
    assert probe.last_uncertainty() is None
    assert probe.last_logits() is None


def test_check_myself_works_with_real_taps():
    from venom.core.bus import Bus
    from venom.sensors.consensus import consensus

    model = _tiny_model()
    bus = Bus(model, list(model.model.layers))
    with torch.no_grad():
        with bus:
            model(torch.zeros(1, 4, 16))
    out = consensus(bus.taps, model.model.norm, model.lm_head, _FakeTokenizer())
    assert out["layers_total"] == 3
    assert 0.0 <= out["agreement"] <= 1.0
    assert out["caveat"]
    assert "arXiv 2611.15267" in out["caveat"]


def test_audit_reports_on_real_taps():
    from venom.core.bus import Bus
    from venom.sensors.audit import audit

    model = _tiny_model()
    bus = Bus(model, list(model.model.layers))
    with torch.no_grad():
        with bus:
            model(torch.zeros(1, 4, 16))
    out = audit(bus.taps)
    assert "dead_neurons" in out
    assert "redundancy" in out
    assert "compressibility" in out
    assert "REPORT ONLY" in out["warning"]


def test_why_and_where_work_on_real_taps():
    from venom.core.bus import Bus
    from venom.sensors.attribution import where, why

    model = _tiny_model()
    bus = Bus(model, list(model.model.layers))
    with torch.no_grad():
        with bus:
            model(torch.zeros(1, 4, 16))

    out = why(" Paris", bus.taps, model.model.norm, model.lm_head, _FakeTokenizer())
    assert out["ok"] if "ok" in out else True
    assert "layers" in out and len(out["layers"]) == 3
    assert out["caveat"]

    out2 = where(" France", bus.taps, model.model.norm, model.lm_head, _FakeTokenizer())
    assert "layers" in out2 and len(out2["layers"]) == 3
    assert isinstance(out2["best_layer"], int)


# --------------------------------------------------------------- the runtime
def test_runtime_call_tool_uses_the_live_state():
    rt = VenomRuntime(VenomConfig(max_calls=10))
    rt.remember("capital of france", "Paris")
    out = rt.call_tool("venom.recall", {"query": "capital of france"})
    assert out["status"] == "hit" and out["value"] == "Paris"


def test_runtime_handlers_share_the_budget_with_the_runtime():
    rt = VenomRuntime(VenomConfig(max_calls=2))
    rt.call_tool("venom.time_now", {})
    rt.call_tool("venom.time_now", {})
    out = rt.call_tool("venom.time_now", {})
    assert out["error"] == "budget_exhausted"


def test_runtime_bind_host_wires_the_probe():
    from venom.core.probe import LogitProbe

    model = _tiny_model()
    probe = LogitProbe(model)
    rt = VenomRuntime(VenomConfig(max_calls=10))
    rt.bind_host(probe=probe)
    with torch.no_grad():
        model(torch.zeros(1, 4, 16))
    assert rt.call_tool("venom.uncertainty", {})["ok"] is True
