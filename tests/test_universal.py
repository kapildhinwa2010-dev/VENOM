"""Tests for the v0.2 universal plug-in layer.

These cover the three surfaces that make the layer host-agnostic:

  * `venom.core.discovery`  - find the blocks/head/norm of ANY architecture
  * `venom.core.runtime`    - the framework-agnostic core
  * `venom.shell.attach`    - attach/detach, always restoring the host
  * `venom.shell.proxy`     - put VENOM in front of any OpenAI-compatible host
"""
import json
import threading
import urllib.request

import pytest

from venom.core.discovery import (
    discover_blocks,
    discover_final_norm,
    discover_output_head,
    fingerprint_universal,
)
from venom.core.runtime import VenomConfig, VenomRuntime
from venom.shell.attach import attach, fingerprint
from venom.shell.proxy import VenomProxy, payload_path


# ------------------------------------------------------- discovery (torch-free stubs)
class _Block:
    """Mimics a transformer block: an attention child and an MLP child."""

    class _Attn(torch.nn.Module if False else object):
        pass


def _make_tiny_blocks_model():
    import torch.nn as nn

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(4, 4)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.Linear(4, 4)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()
            self.mlp = MLP()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Block() for _ in range(5)])
            self.model.norm = nn.LayerNorm(4)
            self.lm_head = nn.Linear(4, 10)

    return Model()


def test_discovery_finds_blocks_without_any_hardcoded_path():
    model = _make_tiny_blocks_model()
    found = discover_blocks(model)
    assert found is not None
    path, blocks = found
    assert len(blocks) == 5
    assert path.endswith("layers")


def test_discovery_finds_head_and_norm():
    model = _make_tiny_blocks_model()
    assert discover_output_head(model) is not None
    assert discover_final_norm(model) is not None


def test_fingerprint_universal_reports_confidence():
    fp = fingerprint_universal(_make_tiny_blocks_model(), "tiny")
    assert fp["num_layers"] == 5
    assert fp["confidence"] >= 0.75
    assert "blocks_path" in fp


def test_fingerprint_universal_handles_a_model_with_no_blocks():
    import torch.nn as nn

    class Empty(nn.Module):
        pass

    fp = fingerprint_universal(Empty(), "empty")
    assert fp["confidence"] == 0.0
    assert "error" in fp


def test_attach_refuses_a_host_it_cannot_understand():
    import torch.nn as nn

    class Empty(nn.Module):
        pass

    with pytest.raises(RuntimeError) as exc:
        attach(Empty(), min_confidence=0.5)
    assert "confidence" in str(exc.value)


def test_attach_can_be_forced_on_an_unknown_host():
    import torch.nn as nn

    class Empty(nn.Module):
        pass

    # min_confidence=0.0 must not raise
    attach(Empty(), min_confidence=0.0)


# ------------------------------------------------------------------- runtime
def test_runtime_is_framework_agnostic():
    rt = VenomRuntime(VenomConfig(session_id="t"))
    assert rt.store.session_id == "t"
    assert rt.budget.max_calls == 3


def test_runtime_prompt_for_injects_memory():
    rt = VenomRuntime()
    rt.remember("capital of france", "Paris")
    out = rt.prompt_for("what is the capital of france?")
    assert "Paris" in out
    assert rt.telemetry["injections"] == 1


def test_runtime_prompt_for_leaves_unknown_prompts_alone():
    rt = VenomRuntime()
    out = rt.prompt_for("hello world")
    assert out == "hello world"


def test_runtime_injection_can_be_disabled():
    rt = VenomRuntime(VenomConfig(inject_memory=False))
    rt.remember("capital of france", "Paris")
    assert rt.prompt_for("capital of france?") == "capital of france?"


def test_runtime_observe_does_not_modify_anything():
    rt = VenomRuntime()
    rt.remember("capital of france", "Paris")
    obs = rt.observe("what is the capital of france?")
    assert obs["fire"] is True
    assert obs["reason"] == "memory_hit"
    assert obs["budget_left"] == 3


def test_runtime_tool_call_consumes_budget():
    rt = VenomRuntime(VenomConfig(max_calls=2))
    rt.call_tool("venom.time_now", {})
    rt.call_tool("venom.time_now", {})
    out = rt.call_tool("venom.time_now", {})
    assert out["error"] == "budget_exhausted"
    assert rt.telemetry["budget_exhausted"] == 1


def test_runtime_note_answer_rearms_the_budget():
    rt = VenomRuntime()
    rt.call_tool("venom.time_now", {})
    assert rt.budget.used == 1
    rt.note_answer()
    assert rt.budget.used == 0


def test_runtime_tool_defs_are_mcp_shaped():
    tools = VenomRuntime.tool_defs()
    assert len(tools) == 3
    for t in tools:
        assert t["name"].startswith("venom.")
        assert "inputSchema" in t


def test_runtime_stats_and_report():
    rt = VenomRuntime()
    rt.remember("x y", "z")
    stats = rt.stats()
    assert stats["memory"]["slots"] == 1
    assert "VENOM REPORT" in rt.report()


# -------------------------------------------------------------------- proxy
def test_payload_path_strips_the_v1_prefix():
    assert payload_path("/v1/chat/completions") == "chat/completions"
    assert payload_path("/v1/models") == "models"
    assert payload_path("v1/chat/completions?x=1") == "chat/completions"


def test_proxy_injects_memory_into_the_last_user_message():
    rt = VenomRuntime()
    rt.remember("capital of france", "Paris")
    proxy = VenomProxy(rt, upstream_base_url="http://127.0.0.1:1/v1")
    payload = {
        "model": "whatever",
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "what is the capital of france?"},
        ],
    }
    assert proxy._inject(payload) is True
    assert "Paris" in payload["messages"][-1]["content"]
    # the system message must be untouched
    assert payload["messages"][0]["content"] == "be terse"


def test_proxy_leaves_irrelevant_requests_alone():
    rt = VenomRuntime()
    rt.remember("capital of france", "Paris")
    proxy = VenomProxy(rt, upstream_base_url="http://127.0.0.1:1/v1")
    payload = {"model": "x", "messages": [{"role": "user", "content": "hello there"}]}
    assert proxy._inject(payload) is False
    assert payload["messages"][0]["content"] == "hello there"


def test_proxy_end_to_end_against_a_fake_upstream():
    """The proxy must forward AND inject, and the upstream must see the context."""
    import http.server
    import socketserver

    received = {}

    class Fake(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            received["body"] = json.loads(self.rfile.read(length).decode())
            out = json.dumps({"choices": [{"message": {"content": "Paris"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    server = socketserver.TCPServer(("127.0.0.1", 0), Fake)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    rt = VenomRuntime()
    rt.remember("capital of france", "Paris")
    proxy = VenomProxy(rt, upstream_base_url=f"http://127.0.0.1:{port}/v1")
    proxy.start()
    try:
        req = urllib.request.Request(
            proxy.base_url + "/chat/completions",
            data=json.dumps({
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "capital of france?"}],
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode())
        assert body["choices"][0]["message"]["content"] == "Paris"
        assert "Paris" in received["body"]["messages"][0]["content"]
        assert proxy.injected == 1
    finally:
        proxy.stop()
        server.shutdown()
        server.server_close()
