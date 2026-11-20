"""Session state for one host conversation: memory, budget and telemetry.

Everything in a session is JSON-serialisable so a session can be saved and
restored without touching the host model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from venom.memory.store import MemoryStore
from venom.reflex.budget import Budget


@dataclass
class Session:
    session_id: str = "default"
    store: MemoryStore = field(default_factory=MemoryStore)
    budget: Budget = field(default_factory=Budget)
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "memory": [
                {"key": k, "value": v["value"], "digest": v["digest"]}
                for k, v in self.store._data.items()
            ],
            "budget": self.budget.state(),
            "telemetry": self.telemetry,
        }

    def to_dict_safe(self) -> Dict[str, Any]:
        return self.to_dict()

    @classmethod
    def from_dict(cls, blob: Dict[str, Any]) -> "Session":
        store = MemoryStore(session_id=blob.get("session_id", "default"))
        for entry in blob.get("memory", []):
            store.remember(entry["key"], entry["value"])
        budget = Budget(max_calls=blob.get("budget", {}).get("max", 3))
        return cls(
            session_id=blob.get("session_id", "default"),
            store=store,
            budget=budget,
            telemetry=blob.get("telemetry", {}),
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "Session":
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        store = MemoryStore(session_id=blob.get("session_id", "default"))
        for entry in blob.get("memory", []):
            store.remember(entry["key"], entry["value"])
        budget = Budget(max_calls=blob.get("budget", {}).get("max", 3))
        return cls(
            session_id=blob.get("session_id", "default"),
            store=store,
            budget=budget,
            telemetry=blob.get("telemetry", {}),
        )

    def record(self, key: str, value: Any) -> None:
        self.telemetry[key] = value
