"""Exact-recall memory with a digest guard. Never guesses (NEVER_LIE principle).

Design rules:
  * exact match only - no fuzzy retrieval, no partial credit;
  * every stored value carries a digest so tampering is detectable;
  * bounded LRU so a long session cannot grow without limit;
  * unknown or corrupted -> explicit `abstain`, never a plausible guess.
"""
from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Optional

_PUNCT = re.compile(r"[^\w\s]")


def norm(key: str) -> str:
    """Canonical key form: lowercase, collapsed whitespace, punctuation -> space.

    v0.1.1 change: punctuation is stripped here (not only in the reflex layer) so
    that `remember("Paris.")` and `recall("paris")` refer to the same slot.

    v0.1.2 change: punctuation becomes a SPACE, not nothing. Deleting it merged
    distinct keys ("hello,world" -> "helloworld") and silently collapsed
    "don't"/"dont", "U.S.A."/"usa" into one slot, losing a stored fact.
    """
    return " ".join(_PUNCT.sub(" ", key.strip().lower()).split())


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


@dataclass
class Recall:
    status: str  # "hit" | "abstain"
    value: Optional[str] = None
    reason: str = ""
    key: Optional[str] = None

    def as_dict(self):
        return asdict(self)


class MemoryStore:
    """Bounded, LRU, digest-guarded exact memory. No fuzzy match, no guessing."""

    def __init__(self, capacity: int = 1024, session_id: str = "default"):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.session_id = session_id
        self._data: "OrderedDict[str, dict]" = OrderedDict()
        self.evictions = 0
        self.corruptions = 0

    def remember(self, key: str, value: str) -> dict:
        k = norm(key)
        if not k:
            return {"ok": False, "reason": "empty_key"}
        if k in self._data:
            self._data.move_to_end(k)
        self._data[k] = {"value": value, "digest": digest(value)}
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)
            self.evictions += 1
        return {"ok": True, "key": k, "digest": digest(value)}

    def has(self, key: str) -> bool:
        return norm(key) in self._data

    def recall(self, query: str) -> Recall:
        k = norm(query)
        if not k:
            return Recall("abstain", reason="empty_key")
        rec = self._data.get(k)
        if rec is None:
            return Recall("abstain", reason="not_found", key=k)
        if digest(rec["value"]) != rec["digest"]:
            self.corruptions += 1
            return Recall("abstain", reason="digest_mismatch", key=k)
        self._data.move_to_end(k)
        return Recall("hit", value=rec["value"], reason="exact", key=k)

    def corrupt_for_test(self, key: str, new_value: str) -> None:
        """Deliberately break the digest. Used by the abstain benchmark."""
        k = norm(key)
        if k not in self._data:
            raise KeyError(f"no such key in store: {key!r}")
        self._data[k]["value"] = new_value

    def restore(self, key: str, value: str, dig: Optional[str] = None) -> dict:
        """Re-insert a persisted entry WITH its persisted digest.

        v0.1.2: the digest is deliberately NOT recomputed. `remember()` would
        recompute it from the payload, which means a tampered session file
        would load as a valid hit - destroying the whole point of the digest
        guard at the persistence boundary.
        """
        k = norm(key)
        if not k:
            return {"ok": False, "reason": "empty_key"}
        self._data[k] = {"value": value, "digest": dig if dig else digest(value)}
        self._data.move_to_end(k)
        return {"ok": True, "key": k}

    def stats(self) -> dict:
        return {
            "session": self.session_id,
            "slots": len(self._data),
            "capacity": self.capacity,
            "dead_slots": self.capacity - len(self._data),
            "evictions": self.evictions,
            "corruptions": self.corruptions,
        }
