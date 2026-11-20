"""Minimal MCP-style JSON-RPC 2.0 server over stdio. Zero dependencies."""
from __future__ import annotations

import json
import sys

from venom.memory.store import MemoryStore
from venom.reflex.budget import Budget
from venom.tools.handlers import ToolHandlers
from venom.tools.schema import P0_TOOLS

PROTOCOL = "2024-11-05"


def make_server(store: MemoryStore | None = None) -> ToolHandlers:
    return ToolHandlers(store or MemoryStore(), Budget())


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
            "serverInfo": {"name": "venom", "version": "0.1.2"},
            "capabilities": {"tools": {}},
        }
    elif method == "tools/list":
        result = {"tools": P0_TOOLS}
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
