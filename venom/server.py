"""Minimal MCP-style JSON-RPC 2.0 server over stdio. Zero dependencies."""
from __future__ import annotations

import json
import sys

from venom.memory.store import MemoryStore
from venom.reflex.budget import Budget
from venom.tools.handlers import ToolHandlers
from venom.tools.schema import ALL_TOOLS, P0_TOOLS

PROTOCOL = "2024-11-05"

# v0.3.0: the server serves the FULL tool surface. `tools` can be narrowed for
# a Tier-2 (API-only) host, which has no access to the host's internals and
# therefore must not be offered the self-read or intervention tools.
DEFAULT_TOOLS = ALL_TOOLS


def make_server(store: MemoryStore | None = None, budget: Budget | None = None,
                runtime=None, tools=None) -> ToolHandlers:
    """Build the handler set.

    If a `runtime` is given, its handlers are used so the tools share the
    runtime's store, budget and host wiring (probe, taps, dials).
    """
    if runtime is not None:
        h = runtime.handlers()
        h.tool_defs = list(tools or DEFAULT_TOOLS)
        return h
    h = ToolHandlers(store or MemoryStore(), budget or Budget())
    h.tool_defs = list(tools or DEFAULT_TOOLS)
    return h


def handle(req: dict, h: ToolHandlers):
    method = req.get("method")
    rid = req.get("id")

    # v0.1.2: a JSON-RPC notification is defined by the ABSENCE of "id", not
    # by the method name. The old check (`method.startswith("notifications/")`)
    # answered a notification that had no id (forbidden), and silently dropped
    # a request that had an id (the client waited forever).
    is_notification = "id" not in req

    if method == "initialize":
        # a new conversation turn re-arms the per-answer call cap
        h.budget.reset()
        result = {
            "protocolVersion": PROTOCOL,
            "serverInfo": {"name": "venom", "version": "0.3.0"},
            "capabilities": {"tools": {}},
        }
    elif method == "tools/list":
        result = {"tools": getattr(h, "tool_defs", DEFAULT_TOOLS)}
    elif method == "tools/call":
        p = req.get("params") or {}
        if not isinstance(p, dict):
            p = {}
        args = p.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        try:
            out = h.call(p.get("name", ""), args)
        except Exception as exc:  # noqa: BLE001 - a failing tool must not kill the RPC
            out = {"ok": False, "error": "tool_exception", "message": str(exc)}
        result = {
            "content": [{"type": "text", "text": json.dumps(out)}],
            "isError": not out.get("ok", False),
        }
    else:
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "method not found"}}

    if is_notification:
        return None
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def main() -> None:
    h = make_server()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:  # noqa: BLE001 - malformed JSON is a parse error
            resp = {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": str(exc)}}
        else:
            try:
                resp = handle(req, h)
            except Exception as exc:  # noqa: BLE001 - server must not die on bad input
                resp = {
                    "jsonrpc": "2.0",
                    # keep the request id so the client can correlate the reply
                    "id": req.get("id") if isinstance(req, dict) else None,
                    "error": {"code": -32603, "message": str(exc)},
                }
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
