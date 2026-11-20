"""Tool dispatch. VENOM has no knowledge of its own - it only serves.

v0.3.0: the full tool surface (P0 memory tools, P2 self-read tools,
P4 intervention tools, plus state control). Every self-read tool degrades
gracefully: if the host has no probe or no bus attached, it returns a clear
error instead of pretending to know something.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from venom.memory.store import MemoryStore
from venom.reflex.budget import Budget
from venom.tools.schema import ALL_TOOLS

MEMORY_TOOLS = (
    "venom.recall", "venom.remember", "venom.time_now",
    "venom.forget", "venom.verify",
)
STATE_TOOLS = ("venom.snapshot", "venom.restore", "venom.report")
SELF_READ_TOOLS = (
    "venom.uncertainty", "venom.check_myself", "venom.why", "venom.where",
)
INTERVENTION_TOOLS = ("venom.dial", "venom.audit")

KNOWN_TOOLS = tuple(t["name"] for t in ALL_TOOLS)


class ToolHandlers:
    """Executes tool calls.

    Optional collaborators let a host expose its internals:

        probe   venom.core.probe.LogitProbe  - live logits
        taps    dict                          - per-layer hidden states (Bus.taps)
        norm    torch.nn.Module               - final norm
        lm_head torch.nn.Module               - output projection
        dials   venom.core.dials.DialManager  - bounded self-edit
        rollback venom.verify.rollback.RollbackManager
    """

    def __init__(self, store: MemoryStore, budget: Budget | None = None,
                 probe=None, taps: Optional[Dict[str, Any]] = None,
                 norm=None, lm_head=None, tokenizer=None, dials=None,
                 rollback=None, report_fn=None):
        self.store = store
        self.budget = budget or Budget()
        self.probe = probe
        self.taps = taps if taps is not None else {}
        self.norm = norm
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.dials = dials
        self.rollback = rollback
        self.report_fn = report_fn

    # -------------------------------------------------------------- dispatch
    def call(self, name: str, args: Dict[str, Any]) -> dict:
        args = args or {}

        # An unknown tool is a client mistake, not a budget event: it must not
        # consume the call budget, or a typo would silently mute the host.
        if name not in KNOWN_TOOLS:
            return {"ok": False, "error": "unknown_tool", "tool": name,
                    "available": list(KNOWN_TOOLS)}

        if not self.budget.take():
            return {"ok": False, "error": "budget_exhausted",
                    "budget": self.budget.state()}

        handler = getattr(self, "_t_" + name.replace("venom.", ""), None)
        if handler is None:
            return {"ok": False, "error": "tool_not_implemented", "tool": name}
        try:
            return handler(args)
        except Exception as exc:  # noqa: BLE001 - a tool must never kill the server
            return {"ok": False, "error": "tool_exception", "tool": name,
                    "message": str(exc)}

    # ----------------------------------------------------------- memory tools
    def _t_recall(self, args):
        r = self.store.recall(args.get("query", ""))
        return {"ok": r.status == "hit", **r.as_dict()}

    def _t_remember(self, args):
        return self.store.remember(args.get("key", ""), args.get("value", ""))

    def _t_time_now(self, args):
        return {"ok": True, "utc": datetime.now(timezone.utc).isoformat(),
                "note": "host has no clock; VENOM supplies the time"}

    def _t_forget(self, args):
        """The memory is not append-only - facts can be retracted."""
        key = args.get("key", "")
        from venom.memory.store import norm
        k = norm(key)
        if k not in self.store._data:
            return {"ok": False, "error": "not_found", "key": k}
        del self.store._data[k]
        return {"ok": True, "key": k, "forgotten": True}

    def _t_verify(self, args):
        """Check an answer against stored memory.

        The verdict is one of:
          verified   - the stored fact is stated in the answer
          contradicted - the answer states a different value for the same key
          unknown    - nothing stored for that key (abstain, never guess)
        """
        answer = (args.get("answer") or "").strip()
        key = args.get("key")
        if not answer:
            return {"ok": False, "error": "empty_answer"}

        if not key:
            # look for any stored key mentioned in the answer
            from venom.memory.store import norm
            mentioned = [k for k in self.store._data if k in norm(answer)]
            if not mentioned:
                return {"ok": True, "verdict": "unknown",
                        "reason": "no stored fact is mentioned in the answer"}
            key = mentioned[0]

        rec = self.store.recall(key)
        if rec.status != "hit":
            return {"ok": True, "verdict": "unknown", "reason": rec.reason, "key": rec.key}

        from venom.memory.store import norm
        if norm(rec.value) and norm(rec.value) in norm(answer):
            return {"ok": True, "verdict": "verified", "key": rec.key,
                    "value": rec.value}
        return {"ok": True, "verdict": "contradicted", "key": rec.key,
                "stored_value": rec.value,
                "note": "the answer does not state the stored value"}

    # ----------------------------------------------------------- state tools
    def _t_snapshot(self, args):
        if self.rollback is None:
            from venom.verify.rollback import RollbackManager
            self.rollback = RollbackManager()
        state = {"dials": self.dials.state() if self.dials is not None else None,
                 "memory": {k: v["digest"] for k, v in self.store._data.items()}}
        sid = self.rollback.snapshot(state, args.get("tag", ""))
        return {"ok": True, "snapshot_id": sid,
                "snapshots": self.rollback.list_snapshots()}

    def _t_restore(self, args):
        if self.rollback is None:
            from venom.verify.rollback import RollbackManager
            self.rollback = RollbackManager()
        state = self.rollback.restore(args.get("snapshot_id"))
        if state is None:
            return {"ok": False, "error": "snapshot_not_found"}
        if self.dials is not None and state.get("dials"):
            self.dials.restore_state(state["dials"])
        return {"ok": True, "restored": state.get("dials") is not None,
                "snapshot_id": self.rollback.latest_id()}

    def _t_report(self, args):
        if self.report_fn is not None:
            return {"ok": True, "report": self.report_fn()}
        from venom.report.compiler import compile_all
        return {"ok": True, "report": compile_all({
            "memory_stats": self.store.stats(),
            "budget": self.budget.state(),
        })}

    # --------------------------------------------------------- self-read tools
    def _t_uncertainty(self, args):
        if self.probe is None:
            return {"ok": False, "error": "no_probe",
                    "note": "the host has no logits probe attached"}
        u = self.probe.last_uncertainty(k=int(args.get("k", 5)))
        if u is None:
            return {"ok": False, "error": "no_logits_captured",
                    "note": "run the host once before asking about uncertainty"}
        return {"ok": True, **u}

    def _t_check_myself(self, args):
        from venom.sensors.consensus import consensus
        out = consensus(self.taps, self.norm, self.lm_head, self.tokenizer,
                        top_k=int(args.get("k", 5)))
        return {"ok": "error" not in out, **out}

    def _t_why(self, args):
        from venom.sensors.attribution import why
        out = why(args.get("token", ""), self.taps, self.norm, self.lm_head,
                  self.tokenizer)
        return {"ok": "error" not in out, **out}

    def _t_where(self, args):
        from venom.sensors.attribution import where
        out = where(args.get("concept", ""), self.taps, self.norm, self.lm_head,
                    self.tokenizer)
        return {"ok": "error" not in out, **out}

    # ---------------------------------------------------- intervention tools
    def _t_dial(self, args):
        """Turn a dial on, or off.

        `apply: false` REMOVES the dial on that layer (the natural reading of
        "don't apply it"), which restores the host exactly.
        """
        if self.dials is None:
            return {"ok": False, "error": "no_dial_manager",
                    "note": "self-edit is not enabled for this host"}
        layer = int(args.get("layer", -1))
        if not bool(args.get("apply", True)):
            return self.dials.revert(layer)
        return self.dials.apply(layer, float(args.get("delta", 0.0)), True)

    def _t_audit(self, args):
        from venom.sensors.audit import audit
        out = audit(self.taps, float(args.get("threshold", 0.001)))
        return {"ok": "error" not in out, **out}
