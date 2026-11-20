"""Memory-context injection for hosts that cannot call tools themselves.

GPT-2 and similar models have no function-calling ability, so the only way to
give them memory is to rewrite the prompt. This module does that, and only
when the reflex actually fires.
"""
from __future__ import annotations

from venom.memory.store import MemoryStore
from venom.reflex.triggers import Reflex


class Injector:
    def __init__(self, store: MemoryStore, reflex: Reflex | None = None, max_entries: int = 3):
        self.store = store
        self.reflex = reflex or Reflex(store)
        self.max_entries = max_entries
        self.injections = 0
        self.abstains = 0

    def maybe_inject(self, prompt: str, auto_recall: bool = True) -> dict:
        trigger = self.reflex.decide(prompt)
        if not trigger.fire:
            return {
                "original": prompt, "injected": prompt, "changed": False,
                "recalls": [], "reason": "no_trigger",
            }

        recalls = []
        if trigger.key and auto_recall:
            recalls.append(self.store.recall(trigger.key))
        hits = [r for r in recalls if r.status == "hit"]
        self.abstains += sum(1 for r in recalls if r.status == "abstain")

        if not hits:
            return {
                "original": prompt, "injected": prompt, "changed": False,
                "recalls": [r.as_dict() for r in recalls],
                "reason": f"trigger_{trigger.reason}_no_hit",
            }

        if len(hits) == 1:
            prefix = f"CONTEXT: {hits[0].value} | "
        else:
            entries = "\n".join(f"- {h.key}: {h.value}" for h in hits[: self.max_entries])
            prefix = f"CONTEXT:\n{entries}\n| "

        self.injections += 1
        return {
            "original": prompt, "injected": prefix + prompt, "changed": True,
            "recalls": [r.as_dict() for r in recalls],
            "reason": f"injected_{len(hits)}_via_{trigger.reason}",
        }

    def stats(self) -> dict:
        return {"injections": self.injections, "abstains": self.abstains, "store": self.store.stats()}
