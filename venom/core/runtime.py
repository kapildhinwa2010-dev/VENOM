"""The runtime: everything VENOM does, independent of the host framework.

This is the object that makes the layer *plug-in*. It owns:

  * the memory store        (exact recall, digest-guarded, abstain)
  * the call budget         (hard cap per answer)
  * the reflex              (rule-based auto-trigger)
  * the telemetry           (what happened, in numbers)
  * the optional collaborators that expose the host's internals:
      probe    - live logits        (venom.core.probe.LogitProbe)
      taps     - per-layer hidden   (venom.core.bus.Bus.taps)
      norm     - final norm module
      lm_head  - output projection
      dials    - bounded self-edit  (venom.core.dials.DialManager)
      rollback - state snapshots    (venom.verify.rollback.RollbackManager)

It does NOT own the model. A host can be:

  * a local HuggingFace causal LM   -> `venom.shell.attach.attach()`
  * an OpenAI-compatible API model  -> `venom.shell.proxy.VenomProxy`
  * anything with a `generate()`     -> `VenomRuntime.prompt_for()`

The runtime only ever produces a *prompt* and reads *telemetry*. It never
generates the answer, and it never modifies the host's weights.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from venom.memory.store import MemoryStore
from venom.reflex.budget import Budget
from venom.reflex.injector import Injector
from venom.reflex.triggers import Reflex
from venom.tools.schema import ALL_TOOLS


@dataclass
class VenomConfig:
    """Everything tunable, in one place. Defaults are the tested values."""

    memory_capacity: int = 1024
    max_calls: int = 3
    margin_threshold: float = 0.15
    max_ngram: int = 6
    max_entries: int = 3
    min_key_words: int = 2
    min_key_chars: int = 6
    session_id: str = "default"
    # Optional on-disk memory. When set, `venom.remember` with
    # persistent=true writes here and the facts survive a restart.
    persist_path: Optional[str] = None
    # The reflex may rewrite the prompt (memory injection). Set False for hosts
    # that call tools themselves - there the reflex only *advises*.
    inject_memory: bool = True
    # Bounded self-edit is OFF unless a host explicitly wires a DialManager.
    enable_dials: bool = False
    max_delta: float = 0.1
    max_active_dials: int = 4
    drift_threshold: float = 0.02


class VenomRuntime:
    """Framework-agnostic core. Attach it to any host."""

    def __init__(self, config: Optional[VenomConfig] = None, store: Optional[MemoryStore] = None):
        self.config = config or VenomConfig()
        self.store = store or MemoryStore(
            capacity=self.config.memory_capacity,
            session_id=self.config.session_id,
            persist_path=self.config.persist_path,
        )
        self.budget = Budget(max_calls=self.config.max_calls)
        self.reflex = Reflex(
            self.store,
            margin_threshold=self.config.margin_threshold,
            max_ngram=self.config.max_ngram,
        )
        self.injector = Injector(
            self.store,
            reflex=self.reflex,
            max_entries=self.config.max_entries,
            min_key_words=self.config.min_key_words,
            min_key_chars=self.config.min_key_chars,
        )
        self.telemetry: Dict[str, Any] = {
            "answers": 0,
            "injections": 0,
            "tool_calls": 0,
            "budget_exhausted": 0,
        }
        self._t0 = time.perf_counter()

        # --- host internals (all optional) ---
        self.probe = None
        self.taps_provider: Optional[Callable[[], Dict[str, Any]]] = None
        self.norm = None
        self.lm_head = None
        self.tokenizer = None
        self.dials = None
        self.rollback = None
        self.pruner = None
        self.verifier = None

    # -------------------------------------------------------- host wiring
    def bind_host(self, *, probe=None, taps_provider=None, norm=None,
                  lm_head=None, tokenizer=None, dials=None, rollback=None,
                  pruner=None, verifier=None) -> "VenomRuntime":
        """Give the runtime access to a host's internals.

        A probe that is not attached yet is attached here - the host owns the
        lifecycle, but a caller should not have to remember a second step.
        """
        self.probe = probe
        if probe is not None and not getattr(probe, "attached", False):
            try:
                probe.attach()
            except Exception:  # noqa: BLE001 - a host without a probe is fine
                pass
        self.taps_provider = taps_provider
        self.norm = norm
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.dials = dials
        self.rollback = rollback
        self.pruner = pruner
        self.verifier = verifier
        return self
        return self

    def taps(self) -> Dict[str, Any]:
        """Current per-layer hidden states (empty if the host exposes none)."""
        if self.taps_provider is None:
            return {}
        return self.taps_provider() or {}

    # ------------------------------------------------------------- memory
    def remember(self, key: str, value: str) -> dict:
        return self.store.remember(key, value)

    def recall(self, query: str) -> dict:
        return self.store.recall(query).as_dict()

    # -------------------------------------------------------------- tools
    @staticmethod
    def tool_defs() -> List[Dict[str, Any]]:
        """MCP tool schemas this runtime serves."""
        return [dict(t) for t in ALL_TOOLS]

    def handlers(self):
        """A ToolHandlers wired to this runtime and its host."""
        from venom.tools.handlers import ToolHandlers
        from venom.verify.rollback import RollbackManager

        if self.rollback is None:
            self.rollback = RollbackManager()

        return ToolHandlers(
            self.store,
            self.budget,
            probe=self.probe,
            taps=self.taps(),
            norm=self.norm,
            lm_head=self.lm_head,
            tokenizer=self.tokenizer,
            dials=self.dials,
            rollback=self.rollback,
            report_fn=self.report,
            pruner=self.pruner,
            verifier=self.verifier,
        )

    def call_tool(self, name: str, args: Dict[str, Any]) -> dict:
        """Execute a tool against a fresh handler bound to the live state."""
        handlers = self.handlers()
        handlers.taps = self.taps()
        out = handlers.call(name, args)
        if out.get("ok"):
            self.telemetry["tool_calls"] += 1
        elif out.get("error") == "budget_exhausted":
            self.telemetry["budget_exhausted"] += 1
        return out

    # ------------------------------------------------------------- reflex
    def observe(self, text: str, top1: Optional[float] = None,
               top2: Optional[float] = None) -> dict:
        """Look at an incoming question WITHOUT changing it.

        Returns the reflex decision and the telemetry for it. An API-side host
        uses this to decide whether to call a tool; a local host uses
        `prompt_for()` instead.
        """
        trigger = self.reflex.decide(text, top1, top2)
        return {
            "fire": trigger.fire,
            "reason": trigger.reason,
            "key": trigger.key,
            "budget_left": self.budget.state()["left"],
            "memory": self.store.stats(),
        }

    def prompt_for(self, text: str) -> str:
        """The prompt the host should actually see.

        For hosts that cannot call tools (GPT-2 and friends), the memory
        context is prepended. For hosts that CAN call tools, the caller may
        prefer `observe()` and let the model decide.
        """
        if not self.config.inject_memory:
            return text
        result = self.injector.maybe_inject(text)
        if result["changed"]:
            self.telemetry["injections"] += 1
        return result["injected"]

    def note_answer(self) -> None:
        """Called when a host finishes an answer."""
        self.telemetry["answers"] += 1
        self.budget.reset()

    # --------------------------------------------------------- telemetry
    def stats(self) -> dict:
        elapsed = time.perf_counter() - self._t0
        return {
            "uptime_s": round(elapsed, 2),
            "answers": self.telemetry["answers"],
            "injections": self.telemetry["injections"],
            "tool_calls": self.telemetry["tool_calls"],
            "budget_exhausted": self.telemetry["budget_exhausted"],
            "memory": self.store.stats(),
            "injector": self.injector.stats(),
            "budget": self.budget.state(),
        }

    def report(self) -> str:
        """One-line human-readable summary of the runtime state."""
        from venom.report.compiler import compile_all

        return compile_all({
            "memory_stats": self.store.stats(),
            "budget": self.budget.state(),
        })
