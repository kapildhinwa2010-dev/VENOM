"""Tool dispatch. VENOM has no knowledge of its own - it only serves."""
from __future__ import annotations

from datetime import datetime, timezone

from venom.memory.store import MemoryStore
from venom.reflex.budget import Budget

P0_NAMES = ("venom.recall", "venom.remember", "venom.time_now")


class ToolHandlers:
    def __init__(self, store: MemoryStore, budget: Budget | None = None):
        self.store = store
        self.budget = budget or Budget()

    def call(self, name: str, args: dict) -> dict:
        if name not in P0_NAMES:
            return {"ok": False, "error": "unknown_tool", "tool": name}
        if not self.budget.take():
            return {"ok": False, "error": "budget_exhausted", "budget": self.budget.state()}

        if name == "venom.recall":
            r = self.store.recall(args.get("query", ""))
            return {"ok": r.status == "hit", **r.as_dict()}

        if name == "venom.remember":
            return self.store.remember(args.get("key", ""), args.get("value", ""))

        return {
            "ok": True,
            "utc": datetime.now(timezone.utc).isoformat(),
            "note": "host has no clock; VENOM supplies the time",
        }
