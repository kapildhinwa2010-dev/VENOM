"""P0 unit tests - no torch, no network, no model download."""
from venom.bench.correction_bench import MockHost, run_bench
from venom.core.hostmap import fingerprint
from venom.memory.store import MemoryStore
from venom.reflex.budget import Budget
from venom.reflex.injector import Injector
from venom.reflex.triggers import Reflex
from venom.report.compiler import compile_all, compile_drift, compile_uncertainty
from venom.server import handle, make_server
from venom.sensors.uncertainty import sequence_uncertainty, token_uncertainty
from venom.surgeon.report import compressibility_report, dead_neuron_report, redundancy_report
from venom.tools.handlers import ToolHandlers
from venom.verify.drift import drift_report
from venom.verify.rollback import RollbackManager


def test_exact_recall():
    s = MemoryStore(); s.remember("Paris", "France")
    assert s.recall("  paris ").value == "France"


def test_abstain_unknown():
    r = MemoryStore().recall("atlantis")
    assert r.status == "abstain" and r.reason == "not_found"


def test_abstain_corruption():
    s = MemoryStore(); s.remember("paris", "France"); s.corrupt_for_test("paris", "Germany")
    r = s.recall("paris")
    assert r.status == "abstain" and r.reason == "digest_mismatch"


def test_bounded_lru():
    s = MemoryStore(capacity=2)
    for k in ["a", "b", "c"]:
        s.remember(k, k)
    assert s.stats()["slots"] == 2 and s.evictions == 1 and s.recall("a").status == "abstain"


def test_reflex_memory_hit():
    s = MemoryStore(); s.remember("capital of france", "Paris")
    t = Reflex(s).decide("what is the capital of france?")
    assert t.fire and t.reason == "memory_hit"


def test_reflex_low_margin():
    t = Reflex(MemoryStore()).decide("xyz", top1=0.30, top2=0.28)
    assert t.fire and t.reason == "low_margin"


def test_reflex_no_trigger():
    assert not Reflex(MemoryStore()).decide("hello world").fire


def test_budget_cap():
    h = ToolHandlers(MemoryStore(), Budget(max_calls=2))
    h.call("venom.time_now", {}); h.call("venom.time_now", {})
    assert h.call("venom.time_now", {})["error"] == "budget_exhausted"


def test_server_tools_list():
    resp = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, make_server())
    names = [t["name"] for t in resp["result"]["tools"]]
    # the full Tier-1 surface (v0.3.0)
    for expected in ("venom.recall", "venom.remember", "venom.time_now",
                     "venom.uncertainty", "venom.check_myself", "venom.why",
                     "venom.where", "venom.dial", "venom.audit",
                     "venom.verify", "venom.forget", "venom.snapshot",
                     "venom.restore", "venom.report"):
        assert expected in names, f"{expected} missing from tools/list"


def test_server_can_serve_the_tier2_surface_only():
    """An API-only host has no access to internals -> no self-read tools."""
    from venom.tools.schema import TIER2_TOOLS
    h = make_server(tools=TIER2_TOOLS)
    resp = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, h)
    names = [t["name"] for t in resp["result"]["tools"]]
    assert "venom.recall" in names
    assert "venom.uncertainty" not in names
    assert "venom.dial" not in names


def test_bench_plumbing():
    host = MockHost({"capital of france": "Berlin"})
    out = run_bench(host, [{"q": "capital of france", "gold": "Paris"}],
                    lambda q: f"CONTEXT: Paris | {q}")
    assert out["fixed"] == 1 and out["harmed"] == 0


def test_uncertainty_certain():
    u = token_uncertainty([10.0, 0.1, 0.1, 0.1])
    assert u["confidence"] > 0.9 and not u["is_uncertain"]


def test_uncertainty_uncertain():
    u = token_uncertainty([1.0, 1.0, 1.0, 1.0])
    assert u["is_uncertain"] and u["margin"] < 0.01


def test_sequence_uncertainty():
    r = sequence_uncertainty([[10, 0, 0], [1, 1, 1]])
    assert len(r) == 2 and not r[0]["is_uncertain"] and r[1]["is_uncertain"]


def test_drift_unchanged():
    r = drift_report(["hello", "world"], ["hello", "world"])
    assert r["unchanged"] == 2 and r["changed"] == 0


def test_drift_fixed_harmed():
    r = drift_report(["paris is in germany", "london is in uk"],
                     ["paris is in france", "london is in usa"], ["france", "uk"])
    assert r["fixed"] == 1 and r["harmed"] == 1 and r["net_gain"] == 0


def test_drift_abstained():
    assert drift_report(["wrong"], ["abstain"])["abstained"] == 1


def test_hostmap_unknown():
    class F:
        pass
    assert fingerprint(F(), "x")["family"] == "unknown"


def test_hostmap_gpt2():
    class B:
        pass

    class T:
        h = [B() for _ in range(12)]
        ln_f = None

    class M:
        transformer = T
        lm_head = None
        config = type("C", (), {"hidden_size": 768, "vocab_size": 50257,
                                "num_attention_heads": 12, "model_type": "gpt2"})()

    fp = fingerprint(M(), "gpt2")
    assert fp["family"] == "gpt2" and fp["num_layers"] == 12


def test_compile_certain():
    r = compile_uncertainty({"is_uncertain": False, "confidence": 0.95, "margin": 0.8, "entropy": 0.3})
    assert "CONFIDENT" in r


def test_compile_uncertain():
    r = compile_uncertainty({"is_uncertain": True, "confidence": 0.3, "margin": 0.02, "entropy": 3.1})
    assert "VERY UNCERTAIN" in r


def test_compile_drift_pos():
    r = compile_drift({"fixed": 3, "harmed": 0, "abstained": 1, "unchanged": 10,
                        "net_gain": 3, "error": None})
    assert "NET POSITIVE" in r


def test_compile_drift_neg():
    r = compile_drift({"fixed": 1, "harmed": 5, "abstained": 0, "unchanged": 8,
                        "net_gain": -4, "error": None})
    assert "NET NEGATIVE" in r


def test_compile_all():
    r = compile_all({"uncertainty": {"is_uncertain": False, "confidence": 0.9,
                                     "margin": 0.7, "entropy": 0.5},
                     "budget": {"used": 1, "max": 3, "left": 2}})
    assert "VENOM REPORT" in r and "CONFIDENT" in r


def test_dead_neurons():
    counts = {f"l0.r{i}": (0 if i < 5 else 900) for i in range(100)}
    r = dead_neuron_report(counts, 1000)
    assert r["dead"] == 5 and r["dead_pct"] == 5.0


def test_redundancy():
    r = redundancy_report([{"i": 0, "j": 5, "cosine_sim": 0.998},
                           {"i": 1, "j": 2, "cosine_sim": 0.96}])
    assert r["near_dup_99"] == 1 and r["similar_95_99"] == 1


def test_compressibility():
    r = compressibility_report({"l0": [10.0, 9.0] + [0.1] * 8})
    assert r["compressible"] == 1 and r["details"]["l0"]["compressible"]


def test_injector_no_trigger():
    r = Injector(MemoryStore()).maybe_inject("hello world")
    assert not r["changed"]


def test_injector_memory_hit():
    s = MemoryStore(); s.remember("capital of france", "Paris")
    r = Injector(s).maybe_inject("what is the capital of france?")
    assert r["changed"] and "Paris" in r["injected"]


def test_injector_no_hit():
    r = Injector(MemoryStore()).maybe_inject("what is atlantis?")
    assert not r["changed"]


def test_injector_stats():
    s = MemoryStore(); s.remember("project deadline", "2026-01-30"); inj = Injector(s)
    inj.maybe_inject("tell me about project deadline")
    assert inj.stats()["injections"] >= 1


def test_injector_rejects_generic_key():
    """v0.1.2: a one-character key must not trigger an injection.

    Guards against memory-induced tool drift (arXiv 2605.24941): irrelevant
    memories must not shift the host's behaviour.
    """
    s = MemoryStore(); s.remember("x", "y")
    inj = Injector(s)
    r = inj.maybe_inject("tell me about x")
    assert not r["changed"] and r["reason"] == "key_too_generic"
    assert inj.stats()["rejected_generic"] == 1


def test_injector_allows_long_single_word():
    s = MemoryStore(); s.remember("photosynthesis", "process in plants")
    r = Injector(s).maybe_inject("explain photosynthesis")
    assert r["changed"]


def test_rollback_basic():
    rm = RollbackManager(); sid = rm.snapshot({"a": 1, "b": [1, 2]}, "v1")
    assert rm.restore(sid)["a"] == 1


def test_rollback_isolation():
    rm = RollbackManager(); state = {"a": 1}; sid = rm.snapshot(state)
    state["a"] = 999
    assert rm.restore(sid)["a"] == 1


def test_rollback_latest():
    rm = RollbackManager(); rm.snapshot({"v": 1}); rm.snapshot({"v": 2})
    assert rm.restore()["v"] == 2


def test_rollback_eviction():
    rm = RollbackManager(max_snapshots=2)
    rm.snapshot({"v": 1}); rm.snapshot({"v": 2}); rm.snapshot({"v": 3})
    assert len(rm.list_snapshots()) == 2


def test_rollback_not_found():
    assert RollbackManager().restore("nope") is None


def test_punctuation_insensitive_key():
    """v0.1.1: keys are normalised once, in the store, not in two places."""
    s = MemoryStore(); s.remember("capital of france.", "Paris")
    assert s.recall("capital of france") .value == "Paris"
    assert s.recall("Capital Of France?") .value == "Paris"


def test_session_roundtrip():
    from venom.core.session import Session
    sess = Session(session_id="s1")
    sess.store.remember("paris", "France")
    sess.record("fix_rate", 0.4)
    blob = sess.to_dict()
    assert blob["session_id"] == "s1" and len(blob["memory"]) == 1
    again = Session.from_dict(blob)
    assert again.store.recall("paris").value == "France"
    assert again.telemetry["fix_rate"] == 0.4


def test_server_unknown_method():
    resp = handle({"jsonrpc": "2.0", "id": 9, "method": "nope"}, make_server())
    assert resp["error"]["code"] == -32601


def test_server_notification_returns_none():
    assert handle({"jsonrpc": "2.0", "method": "notifications/initialized"}, make_server()) is None


def test_drift_length_mismatch():
    assert drift_report(["a"], ["a", "b"])["error"] == "length mismatch"


def test_unknown_tool_rejected():
    h = ToolHandlers(MemoryStore())
    out = h.call("venom.definitely_not_a_tool", {})
    assert out["error"] == "unknown_tool"
    assert "available" in out


def test_known_tool_without_host_capability_reports_clearly():
    """A known tool that needs host internals must say so, not pretend."""
    h = ToolHandlers(MemoryStore())
    out = h.call("venom.dial", {"layer": 0, "delta": 0.1})
    assert out["error"] == "no_dial_manager"


def test_reflex_budget_interaction():
    """Reflex fires but the store returns abstain -> no injection."""
    s = MemoryStore()
    r = Injector(s).maybe_inject("capital of france is where")
    assert not r["changed"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for f in fns:
        try:
            f()
            print("PASS", f.__name__)
            passed += 1
        except Exception as exc:
            print("FAIL", f.__name__, exc)
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    raise SystemExit(1 if failed else 0)
