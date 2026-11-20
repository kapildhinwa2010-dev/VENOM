"""MCP tool schemas.

Tier 1 = works with open-weight hosts (everything).
Tier 2 = API-only hosts get memory + verify only.

v0.3.0 - the full P0-P5 tool set from the plan's TABLE 4, plus the state
control tools the plan assumed but never specified (forget / snapshot /
restore / report).

    Tier 1 tools  venom.recall, venom.remember, venom.time_now,
                  venom.uncertainty, venom.check_myself, venom.verify,
                  venom.why, venom.where, venom.dial, venom.audit,
                  venom.forget, venom.snapshot, venom.restore, venom.report
    Tier 2 tools  venom.recall, venom.remember, venom.time_now,
                  venom.verify, venom.forget
"""
from __future__ import annotations

def _tool(name: str, description: str, properties: dict, required=None) -> dict:
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    }


P0_TOOLS = [
    _tool("venom.recall",
          "Exact lookup of a stored fact. Abstains instead of guessing.",
          {"query": {"type": "string"}}, ["query"]),
    _tool("venom.remember",
          "Store a key/value fact in the session memory.",
          {"key": {"type": "string"}, "value": {"type": "string"},
           "persistent": {"type": "boolean", "default": False}}, ["key", "value"]),
    _tool("venom.time_now",
          "Current UTC time (host models have no clock).", {}),
]

# --- Tier 1: self-read tools (need access to the host's internals) --------
SELF_READ_TOOLS = [
    _tool("venom.uncertainty",
          "How uncertain is the host about its last token? Returns entropy, margin, "
          "confidence and an is_uncertain flag.",
          {"k": {"type": "integer", "default": 5}}),
    _tool("venom.check_myself",
          "Layer-wise consensus: how many layers' logit lens agree with the final "
          "top-1 token. High agreement = the decision was broadly supported.",
          {"k": {"type": "integer", "default": 5}}),
    _tool("venom.why",
          "Why did the host pick this token? Per-layer logit-lens trajectory of a "
          "target token, and the layer where it overtook the competitor.",
          {"token": {"type": "string"}}, ["token"]),
    _tool("venom.where",
          "Where does the host carry this information? Per-layer logit-lens ranking "
          "of a concept across the network.",
          {"concept": {"type": "string"}}, ["concept"]),
]

# --- Tier 1: intervention tools -------------------------------------------
INTERVENTION_TOOLS = [
    _tool("venom.dial",
          "Adjust a bounded self-edit dial (layer scaling). Every dial is "
          "snapshot-backed and auto-rolls-back if drift exceeds the threshold.",
          {"layer": {"type": "integer"},
           "delta": {"type": "number"},
           "apply": {"type": "boolean", "default": True}},
          ["layer", "delta"]),
    _tool("venom.audit",
          "Report-only health check of the host: dead neurons, redundant layers, "
          "compressible layers. Never modifies weights.",
          {"threshold": {"type": "number", "default": 0.001}}),
]

# --- Tier 1 + Tier 2: verification and memory state -----------------------
STATE_TOOLS = [
    _tool("venom.verify",
          "Check an answer against stored memory. Returns verified / unverified / "
          "abstain and never guesses.",
          {"answer": {"type": "string"}, "key": {"type": "string"}}, ["answer"]),
    _tool("venom.forget",
          "Delete a stored fact (the memory is not append-only).",
          {"key": {"type": "string"}}, ["key"]),
    _tool("venom.snapshot",
          "Take a rollback snapshot of the host state. Returns a snapshot id.",
          {"tag": {"type": "string", "default": ""}}),
    _tool("venom.restore",
          "Restore a previous snapshot by id, or the latest one if no id is given.",
          {"snapshot_id": {"type": "string"}}),
    _tool("venom.report",
          "A one-line human-readable report of the layer's current state.",
          {}),
]

# The full Tier-1 surface.
ALL_TOOLS = P0_TOOLS + SELF_READ_TOOLS + INTERVENTION_TOOLS + STATE_TOOLS

# Tier-2 surface (API-only hosts: no access to the host's internals).
TIER2_TOOLS = [
    t for t in ALL_TOOLS
    if t["name"] in ("venom.recall", "venom.remember", "venom.time_now",
                     "venom.verify", "venom.forget", "venom.snapshot",
                     "venom.restore", "venom.report")
]

LATER_TOOLS = [
    "venom.uncertainty", "venom.check_myself", "venom.verify",
    "venom.why", "venom.where", "venom.dial", "venom.audit",
]


def tool_names(tools=None) -> list:
    return [t["name"] for t in (tools or P0_TOOLS)]
