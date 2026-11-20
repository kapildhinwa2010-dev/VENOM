"""Attach VENOM to ANY AI, including closed API models.

The local `attach()` covers open-weight models. This module covers everything
else - GPT, Claude, Gemini, Llama-API, vLLM, Ollama, LM Studio - by putting
an OpenAI-compatible proxy in front of the host.

    upstream:  client -> VENOM proxy -> real provider

The proxy is transparent: it forwards the request unchanged unless VENOM has
something to add (memory context) or something to enforce (the call budget).
It never rewrites the model's answer.

    from venom.shell.proxy import VenomProxy

    proxy = VenomProxy(runtime, upstream_base_url="https://api.openai.com/v1",
                       upstream_api_key=..., upstream_model="gpt-4o-mini")
    proxy.start()
    # point your client at http://127.0.0.1:<port>/v1

Zero dependencies: `http.server` + `urllib` only.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from venom.core.runtime import VenomRuntime
from venom.tools.schema import ALL_TOOLS, TIER2_TOOLS


class VenomProxy:
    """OpenAI-compatible proxy that injects VENOM into any upstream host."""

    def __init__(self, runtime: VenomRuntime, upstream_base_url: str,
                 upstream_api_key: Optional[str] = None,
                 upstream_model: Optional[str] = None,
                 host: str = "127.0.0.1", port: int = 8765,
                 inject_memory: bool = True,
                 timeout: int = 120,
                 tier: int = 2):
        self.runtime = runtime
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.upstream_api_key = upstream_api_key
        self.upstream_model = upstream_model
        self.host = host
        self.port = port
        self.inject_memory = inject_memory
        self.timeout = timeout
        # An API host has no access to its own internals, so by default it is
        # offered the Tier-2 surface only. tier=1 offers everything, for a
        # local proxy in front of a model you control.
        self.tier = tier
        self.tool_defs = TIER2_TOOLS if tier == 2 else ALL_TOOLS

        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.requests = 0
        self.injected = 0
        self.errors = 0

    # ------------------------------------------------------------- lifecycle
    def start(self) -> "VenomProxy":
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # silence stderr logging
                pass

            def _respond(self, status: int, body: Any) -> None:
                data = json.dumps(body, default=str).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):  # noqa: N802 - http.server API
                proxy._handle(self)

            def do_GET(self):  # noqa: N802 - http.server API
                path = self.path.rstrip("/")
                if path.endswith("/venom/tools"):
                    self._respond(200, {"tools": proxy.tool_defs})
                elif path.endswith("/venom/stats"):
                    self._respond(200, proxy.stats())
                else:
                    self._respond(404, {"error": "not found"})

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self) -> "VenomProxy":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    # -------------------------------------------------------------- request
    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length") or 0)
        raw = handler.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001
            handler._respond(400, {"error": "invalid JSON"})
            return

        # a direct tool-call endpoint: POST /venom/tools/call
        if handler.path.rstrip("/").endswith("/venom/tools/call"):
            name = payload.get("name") or ""
            args = payload.get("arguments") or {}
            out = self.runtime.call_tool(name, args)
            handler._respond(200, out)
            return

        self.requests += 1
        injected = self._inject(payload)
        if injected:
            self.injected += 1

        # the upstream path mirrors the incoming path (v1/chat/completions -> ...)
        path = payload_path(handler.path)
        try:
            body, status = self._forward(payload, path)
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            handler._respond(502, {"error": {"message": str(exc), "type": "upstream_error"}})
            return

        handler._respond(status, body)

    def _inject(self, payload: Dict[str, Any]) -> bool:
        """Add memory context to the last user message, if relevant."""
        if not self.inject_memory:
            return False
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            return False

        last_user = None
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user = msg
                break
        if last_user is None:
            return False

        text = last_user.get("content")
        if not isinstance(text, str) or not text.strip():
            return False

        result = self.runtime.injector.maybe_inject(text)
        if not result["changed"]:
            return False
        last_user["content"] = result["injected"]
        self.runtime.telemetry["injections"] += 1
        return True

    def _forward(self, payload: Dict[str, Any], path: str) -> tuple:
        if self.upstream_model:
            payload = dict(payload)
            payload["model"] = self.upstream_model

        url = self.upstream_base_url + "/" + path
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.upstream_api_key:
            headers["Authorization"] = f"Bearer {self.upstream_api_key}"

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8")), resp.status
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                return json.loads(raw), exc.code
            except Exception:  # noqa: BLE001
                return {"error": {"message": raw, "type": "upstream_http_error"}}, exc.code

    # the path of the incoming request, relative to /v1
    def stats(self) -> dict:
        return {
            "requests": self.requests,
            "injected": self.injected,
            "errors": self.errors,
            "upstream": self.upstream_base_url,
            "runtime": self.runtime.stats(),
        }


def payload_path(path: str) -> str:
    """Map an incoming proxy path onto the upstream path."""
    p = path.split("?", 1)[0].strip("/")
    if p.startswith("v1/"):
        p = p[3:]
    return p or "chat/completions"
