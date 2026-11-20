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

    if method == "initialize":
        result = {
            "protocolVersion": PROTOCOL,
            "serverInfo": {"name": "venom", "version": "0.1.1"},
            "capabilities": {"tools": {}},
        }
    elif method == "tools/list":
        result = {"tools": P0_TOOLS}
    elif method == "tools/call":
        p = req.get("params", {})
        out = h.call(p.get("name", ""), p.get("arguments", {}))
        result = {
            "content": [{"type": "text", "text": json.dumps(out)}],
            "isError": not out.get("ok", False),
        }
    elif method and method.startswith("notifications/"):
        return None
    else:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "method not found"}}

    return {"jsonrpc": "2.0", "id": rid, "result": result}


def main() -> None:
    h = make_server()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            resp = handle(json.loads(line), h)
        except Exception as exc:  # noqa: BLE001 - server must not die on bad input
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
