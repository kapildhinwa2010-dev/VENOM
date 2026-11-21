"""Gated pruning — the plan's phase P5, and the most dangerous thing here.

The plan's exit gate for P5 is exact: *"10k prompts on, drift = 0, warna prune
radd"* — prune only if the drift is zero, otherwise cancel the prune.

So this module never prunes on its own. It:

  1. builds a **plan** (which units look dead),
  2. snapshots the affected weights,
  3. applies the prune,
  4. asks a **verifier** to measure the drift on a HELD-OUT set,
  5. **restores the weights** if the drift is not within the threshold.

The held-out set must not be the set the plan was built from. A prune that is
"verified" on the same prompts that justified it proves nothing — the plan
that chose the dead units and the prompts that confirm it are the same
evidence.

This module DOES modify weights. That is why it is separate from
`venom/surgeon/report.py`, which is report-only, and why every path through
it ends in either a verified prune or a restored model.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class PrunePlan:
    """Which units to zero. Nothing is applied until `apply()` is called."""

    target_path: str
    layer_name: str
    unit_indices: List[int] = field(default_factory=list)
    total_units: int = 0
    fraction: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_path": self.target_path,
            "layer_name": self.layer_name,
            "units": len(self.unit_indices),
            "total_units": self.total_units,
            "fraction": round(self.fraction, 6),
            "reason": self.reason,
            "unit_indices_preview": self.unit_indices[:20],
        }


def _prunable_axes(module: nn.Module) -> Optional[tuple]:
    """(weight tensor, axis of the OUTPUT units) for a prunable module.

    `nn.Linear` has weight [out_features, in_features] -> the output unit is
    a ROW. HuggingFace's `Conv1D` (which GPT-2 uses for every projection)
    has weight [in_features, out_features] -> the output unit is a COLUMN.
    Getting this wrong would zero the wrong axis, so it is resolved here
    once, from the weight shape, and never guessed at the call site.
    """
    weight = getattr(module, "weight", None)
    if weight is None or weight.dim() != 2:
        return None
    name = type(module).__name__
    if name == "Conv1D":
        return weight, 1          # [in, out] -> columns are output units
    if isinstance(module, nn.Linear):
        return weight, 0          # [out, in] -> rows are output units
    return None


def _zero_units(module: nn.Module, indices: List[int]) -> int:
    """Zero the given OUTPUT units of a module. Returns how many were zeroed."""
    axes = _prunable_axes(module)
    if axes is None:
        return 0
    weight, axis = axes
    n = weight.shape[axis]
    valid = [i for i in indices if 0 <= i < n]
    if not valid:
        return 0
    with torch.no_grad():
        if axis == 0:
            weight[valid] = 0.0
        else:
            weight[:, valid] = 0.0
    return len(valid)


def build_plan(model: nn.Module, activation_counts: Dict[str, Any],
               total_samples: int, threshold: float = 0.001,
               max_units: int = 64,
               candidates: Optional[List[str]] = None) -> Dict[str, Any]:
    """Propose a prune from observed activation counts.

    `activation_counts` maps a module path to the number of samples in which
    the module's output was non-zero (per output unit, as a list) or a single
    int for the whole module.
    """
    if total_samples <= 0:
        return {"error": "total_samples must be > 0"}

    plans: List[PrunePlan] = []
    named = dict(model.named_modules())

    for path, counts in activation_counts.items():
        module = named.get(path)
        if module is None:
            continue
        axes = _prunable_axes(module)
        if axes is None:
            continue
        out_features = axes[0].shape[axes[1]]

        if isinstance(counts, (int, float)):
            # module-level count: if the whole module never fired, all of its
            # output units are candidates
            if counts > 0:
                continue
            indices = list(range(out_features))
        else:
            indices = [i for i, c in enumerate(counts) if (c / total_samples) < threshold]
            indices = [i for i in indices if i < out_features]

        if not indices:
            continue
        if len(indices) > max_units:
            indices = indices[:max_units]

        plans.append(PrunePlan(
            target_path=path,
            layer_name=type(module).__name__,
            unit_indices=indices,
            total_units=out_features,
            fraction=len(indices) / out_features if out_features else 0.0,
            reason=f"activation rate < {threshold} over {total_samples} samples",
        ))

    return {
        "plans": [p.to_dict() for p in plans],
        "plan_count": len(plans),
        "total_units_to_prune": sum(len(p.unit_indices) for p in plans),
        "warning": (
            "This is a PLAN. Nothing has been modified. Applying it requires "
            "the gated path (apply -> verify on a held-out set -> restore if "
            "the drift is not within the threshold)."
        ),
    }


class PruneManager:
    """Applies a prune, but only keeps it if it can be proven harmless."""

    def __init__(self, model: nn.Module, *, drift_threshold: float = 0.0,
                 max_units: int = 64):
        self.model = model
        self.drift_threshold = drift_threshold
        self.max_units = max_units
        self._snapshots: Dict[str, torch.Tensor] = {}
        self.applied: List[PrunePlan] = []
        self.rolled_back = False

    # ------------------------------------------------------------- planning
    def plan(self, activation_counts: Dict[str, Any], total_samples: int,
             threshold: float = 0.001) -> Dict[str, Any]:
        out = build_plan(self.model, activation_counts, total_samples,
                         threshold, self.max_units)
        out["_objects"] = [p for p in out.get("plans", [])]
        return out

    # ------------------------------------------------------------- applying
    def apply(self, plan: PrunePlan) -> Dict[str, Any]:
        """Zero the planned units, after snapshotting the weights."""
        named = dict(self.model.named_modules())
        module = named.get(plan.target_path)
        if module is None or _prunable_axes(module) is None:
            return {"ok": False, "error": "target_not_found",
                    "target_path": plan.target_path}

        with torch.no_grad():
            if plan.target_path not in self._snapshots:
                self._snapshots[plan.target_path] = module.weight.detach().clone()
            zeroed = _zero_units(module, plan.unit_indices)

        self.applied.append(plan)
        return {
            "ok": True,
            "target_path": plan.target_path,
            "pruned_units": zeroed,
            "total_units": plan.total_units,
            "snapshot_taken": True,
            "warning": "NOT YET VERIFIED. Call verify() and restore() if the drift is too large.",
        }

    # ------------------------------------------------------------ verifying
    def verify(self, verifier: Optional[Callable[[], Dict[str, Any]]] = None,
               drift_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Measure the drift with the verifier and restore if it is too large.

        The verifier is a zero-arg callable returning `{"drift": float}` and
        MUST evaluate on a held-out set that was not used to build the plan.
        """
        threshold = self.drift_threshold if drift_threshold is None else drift_threshold

        if verifier is None:
            # A prune that cannot be verified must not stay applied. Restore
            # before returning - otherwise "never accepted without a
            # verifier" would leave the host modified anyway.
            self.restore()
            return {"ok": False, "error": "no_verifier",
                    "rolled_back": True,
                    "note": ("a prune is never accepted without a verifier; "
                             "the weights were restored")}

        try:
            report = verifier()
        except Exception as exc:  # noqa: BLE001
            self.restore()
            return {"ok": False, "error": "verifier_raised", "message": str(exc),
                    "rolled_back": True}

        if not isinstance(report, dict) or "drift" not in report:
            self.restore()
            return {"ok": False, "error": "verifier_must_report_drift",
                    "rolled_back": True}

        drift = float(report["drift"])
        if drift > threshold:
            self.restore()
            return {
                "ok": False,
                "error": "drift_exceeded",
                "drift": drift,
                "drift_threshold": threshold,
                "rolled_back": True,
                "verdict": "the prune was cancelled and the weights restored",
            }

        return {
            "ok": True,
            "drift": drift,
            "drift_threshold": threshold,
            "rolled_back": False,
            "verdict": "the prune is within the threshold on the held-out set",
            "note": ("A drift within the threshold on a held-out set is "
                     "evidence, not proof of losslessness."),
        }

    # ------------------------------------------------------------ restoring
    def restore(self) -> Dict[str, Any]:
        """Put every modified weight back exactly as it was."""
        named = dict(self.model.named_modules())
        restored = 0
        with torch.no_grad():
            for path, snapshot in self._snapshots.items():
                module = named.get(path)
                if module is None or _prunable_axes(module) is None:
                    continue
                module.weight.copy_(snapshot)
                restored += 1
        self._snapshots.clear()
        self.applied = []
        self.rolled_back = True
        return {"ok": True, "restored_modules": restored}

    # ------------------------------------------------------- the whole gate
    def prune_with_gate(self, plan: PrunePlan,
                        verifier: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        """Apply -> verify -> keep or restore. The only path that keeps a prune."""
        applied = self.apply(plan)
        if not applied.get("ok"):
            return applied
        verdict = self.verify(verifier)
        verdict["apply_result"] = applied
        if not verdict.get("ok"):
            self.rolled_back = True
        return verdict

    def is_pristine(self) -> bool:
        """True when no weight differs from the snapshot."""
        named = dict(self.model.named_modules())
        for path, snapshot in self._snapshots.items():
            module = named.get(path)
            if module is None or _prunable_axes(module) is None:
                continue
            if not torch.equal(module.weight.detach(), snapshot):
                return False
        return True
