"""MCP tool schemas.

Tier 1 = works with open-weight models (everything).
Tier 2 = API-only hosts get memory + verify only.
"""
from __future__ import annotations

P0_TOOLS = [
    {
        "name": "venom.recall",
        "description": "Exact lookup of a stored fact. Abstains instead of guessing.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "venom.remember",
        "description": "Store a key/value fact in the session memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
                "persistent": {"type": "boolean", "default": False},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "venom.time_now",
        "description": "Current UTC time (host models have no clock).",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

LATER_TOOLS = [
    "venom.uncertainty", "venom.check_myself", "venom.verify",
    "venom.why", "venom.where", "venom.dial", "venom.audit",
]


def tool_names() -> list:
    return [t["name"] for t in P0_TOOLS]
