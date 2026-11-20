"""Bounded, verified, revertible self-edits of the host's forward pass.

A "dial" scales the output of one transformer block by `(1 + delta)`. It is the
smallest self-edit this project offers:

  * it changes the residual stream at exactly one layer,
  * it changes no weights and no parameters,
  * it is undone by removing the hook, which restores the original path
    bit-for-bit (PyTorch keeps the module output untouched when a hook
    returns `None`).

What this module does NOT claim: that a dial improves the model's output.
`drift_threshold` is a *guard rail against large behavioural change*, not a
quality metric, and `logit_drift` measures how far the logits moved, not
whether the new logits are better. No quality, benefit or "self-awareness"
claim is made here - see `LIMITATIONS.md`.

`apply()` is bounded in three ways: the per-dial magnitude (`max_delta`), the
number of simultaneously active dials (`max_active_dials`) and - only when a
`verifier` is supplied - the measured post-hoc drift of the host.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from venom.verify.rollback import RollbackManager

__all__ = ["DialManager", "logit_drift"]

_EPS = 1e-12


def _flat_list(value: Any) -> List[Any]:
    """Flatten a nested list into a flat list of numbers."""
    out: List[Any] = []
    stack: List[Any] = [value]
    while stack:
        item = stack.pop(0)
        if isinstance(item, (list, tuple)):
            stack = list(item) + stack
        else:
            out.append(item)
    return out


def _shape_of(value: Any) -> Optional[tuple]:
    """Best-effort shape of a tensor or nested sequence."""
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return tuple(int(d) for d in shape)
        except TypeError:
            return None
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        subs = [_shape_of(v) for v in value]
        if any(s is None for s in subs):
            return None
        first = subs[0]
        if all(s == first for s in subs):
            return (len(value),) + first
        return None
    return None


def logit_drift(before: Any, after: Any) -> float:
    """Relative L2 change between two logit tensors.

    `||after - before|| / (||before|| + eps)`, as a float. 0.0 means the two
    are identical. A shape mismatch returns `float("inf")` - the two cannot be
    compared, and "cannot be compared" must not read as "no change".

    Works with torch tensors (no numpy required) and with plain nested
    lists. The torch path is used only when both arguments expose a torch
    tensor interface.
    """
    if before is None or after is None:
        return float("inf")

    shape_b = _shape_of(before)
    shape_a = _shape_of(after)
    if shape_b is None or shape_a is None or shape_b != shape_a:
        return float("inf")

    try:
        import torch  # optional dependency; the module works without it
    except Exception:
        torch = None

    if torch is not None and isinstance(before, torch.Tensor) and isinstance(after, torch.Tensor):
        b = before.detach().to(torch.float64)
        a = after.detach().to(torch.float64)
        if b.shape != a.shape:
            return float("inf")
        num = float(torch.linalg.vector_norm((a - b).reshape(-1)))
        den = float(torch.linalg.vector_norm(b.reshape(-1))) + _EPS
        return num / den

    # Pure-python fallback (lists, tuples, numbers).
    fb = _flat_list(before)
    fa = _flat_list(after)
    if len(fb) != len(fa):
        return float("inf")
    num_sq = 0.0
    den_sq = 0.0
    for x, y in zip(fb, fa):
        diff = float(y) - float(x)
        num_sq += diff * diff
        den_sq += float(x) * float(x)
    return (num_sq ** 0.5) / ((den_sq ** 0.5) + _EPS)


class DialManager:
    """Install and remove bounded per-block output scalings on a host model.

    The manager owns one forward hook per dialled block. Applying a dial to a
    layer that already has one REPLACES it. Removing a dial removes the hook,
    which restores the block's original output exactly.
    """

    def __init__(
        self,
        model,
        blocks,
        *,
        max_delta: float = 0.1,
        max_active_dials: int = 4,
        drift_threshold: float = 0.02,
        verifier: Optional[Callable[[], Dict[str, Any]]] = None,
    ):
        self.model = model
        self.blocks = list(blocks)
        self.max_delta = float(max_delta)
        self.max_active_dials = int(max_active_dials)
        self.drift_threshold = float(drift_threshold)
        self.verifier = verifier

        # layer -> delta
        self._dials: Dict[int, float] = {}
        # layer -> hook handle
        self._handles: Dict[int, Any] = {}
        self._rollback = RollbackManager()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _active_layers(self) -> List[int]:
        return sorted(self._dials)

    def _snapshot_state(self) -> Dict[str, Any]:
        """Current dial state, in a form the rollback manager can hash."""
        return {"dials": {int(k): float(v) for k, v in self._dials.items()}}

    def _make_hook(self, delta: float):
        scale = 1.0 + float(delta)

        def fn(module, inputs, output):
            # The block output is replaced, not mutated in place: the block's
            # parameters and the module's own storage are untouched.
            if isinstance(output, tuple):
                if not output:
                    return output
                scaled = output[0] * scale
                return (scaled,) + tuple(output[1:])
            if isinstance(output, list):
                if not output:
                    return output
                scaled = output[0] * scale
                return [scaled] + list(output[1:])
            try:
                return output * scale
            except Exception:
                # Unknown output type: pass it through unchanged.
                return None

        return fn

    def _install(self, layer: int, delta: float) -> None:
        self._uninstall(layer)
        handle = self.blocks[layer].register_forward_hook(self._make_hook(delta))
        self._handles[layer] = handle
        self._dials[layer] = float(delta)

    def _uninstall(self, layer: int) -> None:
        handle = self._handles.pop(layer, None)
        if handle is not None:
            handle.remove()
        self._dials.pop(layer, None)

    def _clear_hooks(self) -> None:
        for layer in list(self._handles):
            handle = self._handles.pop(layer)
            handle.remove()
        self._dials.clear()

    def _rebuild(self, dials: Dict[int, float]) -> None:
        """Remove every hook, then install exactly the given dials."""
        self._clear_hooks()
        for layer in sorted(int(k) for k in dials):
            self._install(layer, float(dials[layer]))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def apply(self, layer: int, delta: float, apply: bool = True) -> Dict[str, Any]:
        """Install (or replace) the dial on `layer`.

        Returns `{"ok": True, "layer", "delta", "active_dials", "snapshot_id"}`
        on success, plus `"verified"` and `"drift"` when a verifier ran.

        Failure returns are `{"ok": False, "error": ...}` and NOTHING is
        changed on the host:

          * `"delta_out_of_range"` - `abs(delta) > max_delta`
          * `"too_many_dials"`      - the cap would be exceeded
          * `"invalid_layer"`       - no such block
          * `"drift_exceeded"`      - the verifier measured too much drift
            (or raised); the dial was rolled back automatically
        """
        try:
            layer = int(layer)
        except (TypeError, ValueError):
            return {"ok": False, "error": "invalid_layer", "layer": layer}

        if layer < 0 or layer >= len(self.blocks):
            return {"ok": False, "error": "invalid_layer", "layer": layer}

        try:
            delta = float(delta)
        except (TypeError, ValueError):
            return {"ok": False, "error": "delta_out_of_range", "layer": layer, "delta": delta}

        if abs(delta) > self.max_delta:
            return {
                "ok": False,
                "error": "delta_out_of_range",
                "layer": layer,
                "delta": delta,
                "max_delta": self.max_delta,
            }

        # Re-applying to a layer that already has a dial replaces it and does
        # not count against the cap.
        if layer not in self._dials and len(self._dials) >= self.max_active_dials:
            return {
                "ok": False,
                "error": "too_many_dials",
                "layer": layer,
                "active_dials": self._active_layers(),
                "max_active_dials": self.max_active_dials,
            }

        if not apply:
            # Validation-only: report what would happen, change nothing.
            return {
                "ok": True,
                "applied": False,
                "layer": layer,
                "delta": delta,
                "active_dials": self._active_layers(),
            }

        # Snapshot the CURRENT state before touching the host, so any failure
        # below can be undone exactly.
        snapshot_id = self._rollback.snapshot(self._snapshot_state(), tag=f"before_apply_{layer}")

        self._install(layer, delta)

        if self.verifier is None:
            return {
                "ok": True,
                "layer": layer,
                "delta": delta,
                "active_dials": self._active_layers(),
                "snapshot_id": snapshot_id,
                "verified": False,
            }

        drift: float
        verifier_error: Optional[str] = None
        try:
            report = self.verifier()
            if not isinstance(report, dict) or "drift" not in report:
                raise ValueError("verifier must return a dict with a numeric 'drift' key")
            drift = float(report["drift"])
        except Exception as exc:  # a raising verifier is treated as drift
            drift = float("inf")
            verifier_error = f"{type(exc).__name__}: {exc}"

        if drift > self.drift_threshold:
            # Roll back to the pre-apply state: the layer goes back to its
            # previous dial, or to no dial at all.
            self._restore_snapshot(snapshot_id)
            result = {
                "ok": False,
                "error": "drift_exceeded",
                "layer": layer,
                "delta": delta,
                "drift": drift,
                "drift_threshold": self.drift_threshold,
                "rolled_back": True,
                "active_dials": self._active_layers(),
                "snapshot_id": snapshot_id,
            }
            if verifier_error is not None:
                result["verifier_error"] = verifier_error
            return result

        result = {
            "ok": True,
            "layer": layer,
            "delta": delta,
            "active_dials": self._active_layers(),
            "snapshot_id": snapshot_id,
            "verified": True,
            "drift": drift,
        }
        if verifier_error is not None:
            result["verifier_error"] = verifier_error
        return result

    def revert(self, layer: int) -> Dict[str, Any]:
        """Remove the dial on `layer`, restoring that block exactly."""
        try:
            layer = int(layer)
        except (TypeError, ValueError):
            return {"ok": False, "error": "invalid_layer", "layer": layer}

        if layer not in self._dials:
            return {
                "ok": False,
                "error": "no_dial",
                "layer": layer,
                "active_dials": self._active_layers(),
            }

        snapshot_id = self._rollback.snapshot(self._snapshot_state(), tag=f"before_revert_{layer}")
        self._uninstall(layer)
        return {
            "ok": True,
            "layer": layer,
            "active_dials": self._active_layers(),
            "snapshot_id": snapshot_id,
        }

    def revert_all(self) -> Dict[str, Any]:
        """Remove every dial. The host returns to its original behaviour."""
        snapshot_id = self._rollback.snapshot(self._snapshot_state(), tag="before_revert_all")
        self._clear_hooks()
        return {
            "ok": True,
            "active_dials": [],
            "snapshot_id": snapshot_id,
        }

    def state(self) -> Dict[str, Any]:
        """Current dial state."""
        return {
            "active_dials": {int(k): float(v) for k, v in sorted(self._dials.items())},
            "max_delta": self.max_delta,
            "max_active_dials": self.max_active_dials,
            "attached_layers": sorted(self._handles),
        }

    def restore(self, snapshot_id: str) -> Dict[str, Any]:
        """Restore a previously snapshotted dial state.

        Returns `{"ok": True, "active_dials": {...}, "snapshot_id": ...}` or
        `{"ok": False, "error": "unknown_snapshot"}`.
        """
        snap = self._rollback.restore(snapshot_id)
        if snap is None:
            return {"ok": False, "error": "unknown_snapshot", "snapshot_id": snapshot_id}
        dials = snap.get("dials", {}) if isinstance(snap, dict) else {}
        self._rebuild({int(k): float(v) for k, v in dials.items()})
        return {
            "ok": True,
            "active_dials": {int(k): float(v) for k, v in sorted(self._dials.items())},
            "snapshot_id": snapshot_id,
        }

    def restore_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Restore a dial state taken from `state()` (a plain dict).

        Convenience for callers that snapshot the manager state themselves
        rather than using the internal rollback ids. The dict may be the
        output of `state()` (the `"active_dials"` mapping is used) or a full
        `state()` dict.
        """
        if not isinstance(state, dict):
            return {"ok": False, "error": "invalid_state"}
        dials = state.get("active_dials", state)
        if not isinstance(dials, dict):
            return {"ok": False, "error": "invalid_state"}
        cleaned: Dict[int, float] = {}
        for key, value in dials.items():
            try:
                cleaned[int(key)] = float(value)
            except (TypeError, ValueError):
                return {"ok": False, "error": "invalid_state"}
        for layer in cleaned:
            if layer < 0 or layer >= len(self.blocks):
                return {"ok": False, "error": "invalid_layer", "layer": layer}
        self._rebuild(cleaned)
        return {
            "ok": True,
            "active_dials": {int(k): float(v) for k, v in sorted(self._dials.items())},
        }

    def is_clean(self) -> bool:
        """True when no dial is active - the host is back to original."""
        return not self._dials

    # ------------------------------------------------------------------
    # internals used by apply()/restore()
    # ------------------------------------------------------------------
    def _restore_snapshot(self, snapshot_id: str) -> None:
        snap = self._rollback.restore(snapshot_id)
        if snap is None:
            self._clear_hooks()
            return
        dials = snap.get("dials", {}) if isinstance(snap, dict) else {}
        self._rebuild({int(k): float(v) for k, v in dials.items()})
