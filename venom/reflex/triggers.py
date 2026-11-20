"""Rule-based reflex. No intelligence, no model call, fires without host permission.

Two triggers:
  * `memory_hit`  - the question text contains a stored key (n-gram match);
  * `low_margin`  - the host is about to emit a low-confidence token.

Both are cheap string/int comparisons, so the reflex path can run per token.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from venom.memory.store import MemoryStore, norm


@dataclass
class Trigger:
    fire: bool
    reason: str
    key: Optional[str] = None


class Reflex:
    def __init__(self, store: MemoryStore, margin_threshold: float = 0.15, max_ngram: int = 6):
        self.store = store
        self.margin_threshold = margin_threshold
        self.max_ngram = max_ngram

    def _candidate_keys(self, text: str):
        words = norm(text).split()
        for n in range(min(self.max_ngram, len(words)), 0, -1):
            for i in range(len(words) - n + 1):
                yield " ".join(words[i:i + n])

    def decide(self, text: str, top1: Optional[float] = None, top2: Optional[float] = None) -> Trigger:
        seen = set()
        for key in self._candidate_keys(text):
            if not key or key in seen:
                continue
            seen.add(key)
            if self.store.has(key):
                return Trigger(True, "memory_hit", key)
        if top1 is not None and top2 is not None and (top1 - top2) < self.margin_threshold:
            return Trigger(True, "low_margin", None)
        return Trigger(False, "no_trigger", None)
