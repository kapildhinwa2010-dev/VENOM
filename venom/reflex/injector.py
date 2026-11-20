"""Memory-context injection for hosts that cannot call tools themselves.

GPT-2 and similar models have no function-calling ability, so the only way to
give them memory is to rewrite the prompt. This module does that, and only
when the reflex actually fires.
"""
from __future__ import annotations

from venom.memory.store import MemoryStore
from venom.reflex.triggers import Reflex


class Injector:
    """Memory -> prompt rewrite.

    Guard rail (v0.1.2): a stored key that is too short or too generic is not
    allowed to trigger an injection. Memory-Induced Tool-Drift (arXiv 2605.24941)
    shows that irrelevant memories shift tool-call parameters (degradation
    0.38-0.83 vs 0.05-0.12 for neutral memories), so "the key appears somewhere
    in the prompt" is not sufficient evidence of relevance.
    """

    def __init__(self, store: MemoryStore, reflex: Reflex | None = None, max_entries: int = 3,
                 min_key_words: int = 2, min_key_chars: int = 6):
        self.store = store
        self.reflex = reflex or Reflex(store)
        self.max_entries = max_entries
        self.min_key_words = min_key_words
        self.min_key_chars = min_key_chars
        self.injections = 0
        self.abstains = 0
        self.rejected_generic = 0

    def _is_specific_enough(self, key: str) -> bool:
        """Reject keys too generic to be evidence of relevance."""
        words = key.split()
        if len(words) >= self.min_key_words:
            return True
        return len(key) >= self.min_key_chars

    def maybe_inject(self, prompt: str, auto_recall: bool = True) -> dict:
        trigger = self.reflex.decide(prompt)
        if not trigger.fire:
            return {
                "original": prompt, "injected": prompt, "changed": False,
                "recalls": [], "reason": "no_trigger",
            }

        recalls = []
        if auto_recall:
            # v0.1.2: collect EVERY matching key, not just the first one.
            # `decide()` stops at the first hit, so the multi-hit branch and
            # `max_entries` were unreachable dead code.
            for key in self.reflex.matching_keys(prompt):
                if not self._is_specific_enough(key):
                    self.rejected_generic += 1
                    continue
                recalls.append(self.store.recall(key))
        hits = [r for r in recalls if r.status == "hit"]
        self.abstains += sum(1 for r in recalls if r.status == "abstain")

        if not hits:
            reason = "no_trigger" if not trigger.fire else f"trigger_{trigger.reason}_no_hit"
            if trigger.fire and self.rejected_generic and not recalls:
                reason = "key_too_generic"
            return {
                "original": prompt, "injected": prompt, "changed": False,
                "recalls": [r.as_dict() for r in recalls],
                "reason": reason,
            }

        # The single-line format is the contract shared with
        # `venom/bench/correction_bench.py` (and with `MockHost`): everything
        # between "CONTEXT:" and the first "|" is the context.
        if len(hits) == 1:
            prefix = f"CONTEXT: {hits[0].value} | "
        else:
            entries = " ; ".join(h.value for h in hits[: self.max_entries])
            prefix = f"CONTEXT: {entries} | "

        self.injections += 1
        return {
            "original": prompt, "injected": prefix + prompt, "changed": True,
            "recalls": [r.as_dict() for r in recalls],
            "reason": f"injected_{len(hits)}_via_{trigger.reason}",
        }

    def as_prompt_fn(self):
        """Adapter for `venom.bench.correction_bench.run_bench`.

        v0.1.2: `run_bench` expects `venom_fn(prompt) -> prompt`, but
        `maybe_inject` returns a dict. Without this adapter the end-to-end
        pipeline could not run at all - the bench test used a hand-written
        lambda instead of the real injection format.
        """
        def inject(prompt: str) -> str:
            return self.maybe_inject(prompt)["injected"]
        return inject

    def stats(self) -> dict:
        return {
            "injections": self.injections,
            "abstains": self.abstains,
            "rejected_generic": self.rejected_generic,
            "store": self.store.stats(),
        }
