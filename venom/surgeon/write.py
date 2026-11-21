"""The write path — VENOM's unrestricted access to a host model's internals.

This module is deliberately different from everything else in VENOM. The rest
of the project is read-only: sensors look, dials nudge, and every
modification is bounded and reverted. Here, the user asked for the opposite:
**VENOM should be able to change anything in the model.**

So this is the full-access write surface, and it says so out loud:

* every mutating call is recorded in a **journal**;
* every mutating call snapshots the affected tensors **first**;
* a snapshot can be restored **bit-exactly**;
* nothing here hides a failure — if a shape rewrite cannot be done safely it
  raises rather than silently doing half the job.

What it does NOT do is pretend to be safe. A caller with `WriteAccess` can
damage a model. The safety this module offers is *reversibility and an audit
trail*, not *prevention*. That is the honest framing.

The two mutation families:

``zero_units(path, indices)``
    Zero the given OUTPUT units. The shape does not change. This is the
    reversible-by-construction option and it is the default.

``remove_units(path, indices)``
    A real **shape rewrite**: the units are gone, the tensors shrink, the
    downstream consumers are shrunk to match. This is what actually saves
    memory and compute. It is the dangerous one, and it is the one the
    held-out drift gate exists for.

Shape rewrite mechanics
-----------------------
Removing ``k`` output units from a module changes its output dimension from
``D`` to ``D - k``. Two kinds of other modules are affected, and both are
found automatically:

* **consumers** - modules whose *input* dimension is ``D``. Their input
  dimension must shrink to ``D - k``. (Qwen's ``down_proj``.)
* **siblings** - modules in the same parent whose *output* dimension is
  ``D``. Their output dimension must shrink to ``D - k``. (Qwen's
  ``up_proj``, which elementwise-multiplies with ``gate_proj``.)

Getting the axis right matters: HuggingFace's ``Conv1D`` stores weights as
``[in_features, out_features]`` while ``nn.Linear`` stores them as
``[out_features, in_features]``. Zeroing the wrong axis would damage a model
without any error, so the axis is resolved from the weight shape, once,
and never guessed at the call site.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# shape helpers
# --------------------------------------------------------------------------
def prunable_axes(module: nn.Module) -> Optional[Tuple[torch.Tensor, int]]:
    """(weight tensor, axis of the OUTPUT units) for a prunable module.

    ``nn.Linear``   weight [out_features, in_features]  -> output unit is a ROW
    ``Conv1D``      weight [in_features, out_features]  -> output unit is a COLUMN
    """
    weight = getattr(module, "weight", None)
    if weight is None or weight.dim() != 2:
        return None
    name = type(module).__name__
    if name == "Conv1D":
        return weight, 1
    if isinstance(module, nn.Linear):
        return weight, 0
    return None


def input_dim(module: nn.Module) -> Optional[int]:
    """The module's input dimension, or None if it has none."""
    axes = prunable_axes(module)
    if axes is None:
        return None
    weight, axis = axes
    return int(weight.shape[1 - axis])


def output_dim(module: nn.Module) -> Optional[int]:
    axes = prunable_axes(module)
    if axes is None:
        return None
    weight, axis = axes
    return int(weight.shape[axis])


def zero_units(module: nn.Module, indices: Sequence[int]) -> int:
    """Zero the given OUTPUT units of a module. Returns how many were zeroed."""
    axes = prunable_axes(module)
    if axes is None:
        return 0
    weight, axis = axes
    n = int(weight.shape[axis])
    valid = [int(i) for i in indices if 0 <= int(i) < n]
    if not valid:
        return 0
    with torch.no_grad():
        if axis == 0:
            weight[valid] = 0.0
        else:
            weight[:, valid] = 0.0
        bias = getattr(module, "bias", None)
        if bias is not None:
            bias[valid] = 0.0
    return len(valid)


# --------------------------------------------------------------------------
# journal
# --------------------------------------------------------------------------
@dataclass
class JournalEntry:
    action: str
    target: str
    detail: Dict[str, Any] = field(default_factory=dict)
    snapshot_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "target": self.target,
                "detail": self.detail, "snapshot_id": self.snapshot_id}


# --------------------------------------------------------------------------
# the write surface
# --------------------------------------------------------------------------
class WriteAccess:
    """Unrestricted, journaled, reversible mutation of a host model.

    Example
    -------
    >>> w = WriteAccess(model)                      # doctest: +SKIP
    >>> w.zero_units("model.layers.0.mlp.gate_proj", [7, 9])   # doctest: +SKIP
    >>> w.journal()                                 # doctest: +SKIP
    """

    def __init__(self, model: nn.Module, *, label: str = "host"):
        self.model = model
        self.label = label
        self._journal: List[JournalEntry] = []
        self._snapshots: Dict[str, Dict[str, torch.Tensor]] = {}
        self._snapshot_meta: Dict[str, Dict[str, Any]] = {}
        # deep copies of modules taken BEFORE a shape rewrite, so a rewrite can
        # be undone: the original module (with its original shape AND its
        # original weights) is put back in place.
        self._module_backup: Dict[str, nn.Module] = {}

    # ------------------------------------------------------------ inspection
    def modules(self) -> List[str]:
        """Every path this surface can write to."""
        out = []
        for path, module in self.model.named_modules():
            if prunable_axes(module) is not None:
                out.append(path)
        return out

    def describe(self) -> Dict[str, Any]:
        """The writable surface, in enough detail to audit before touching it."""
        rows = []
        for path in self.modules():
            module = self.model.get_submodule(path)
            rows.append({
                "path": path,
                "type": type(module).__name__,
                "in_features": input_dim(module),
                "out_features": output_dim(module),
                "has_bias": getattr(module, "bias", None) is not None,
            })
        return {
            "label": self.label,
            "writable_modules": len(rows),
            "modules": rows,
            "journal_entries": len(self._journal),
            "snapshot_count": len(self._snapshots),
            "warning": ("WriteAccess can damage the host model. It offers "
                        "reversibility and an audit trail, not prevention."),
        }

    def journal(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._journal]

    # ------------------------------------------------------------ snapshots
    def _snapshot(self, paths: Iterable[str]) -> str:
        """Snapshot the given module paths. Returns a snapshot id."""
        named = dict(self.model.named_modules())
        chosen = []
        for path in paths:
            module = named.get(path)
            if module is None:
                continue
            state: Dict[str, torch.Tensor] = {}
            weight = getattr(module, "weight", None)
            if weight is not None:
                state["weight"] = weight.detach().clone()
            bias = getattr(module, "bias", None)
            if bias is not None:
                state["bias"] = bias.detach().clone()
            if state:
                chosen.append(path)
                self._snapshots.setdefault(path, state)
        sid = f"snap{len(self._snapshot_meta) + 1}"
        self._snapshot_meta[sid] = {"paths": chosen}
        return sid

    def snapshot(self, paths: Optional[Iterable[str]] = None) -> str:
        """Snapshot the whole model (or just the given paths)."""
        if paths is None:
            paths = self.modules()
        return self._snapshot(paths)

    def restore(self) -> Dict[str, Any]:
        """Restore every snapshotted tensor, bit-exactly."""
        named = dict(self.model.named_modules())

        # undo shape rewrites first: put the pristine module back in place
        for path, backup in self._module_backup.items():
            current = named.get(path)
            if current is None:
                continue
            if tuple(current.weight.shape) != tuple(backup.weight.shape):
                parent_path, _, attr = path.rpartition(".")
                parent = (self.model.get_submodule(parent_path)
                          if parent_path else self.model)
                setattr(parent, attr, backup)
                named = dict(self.model.named_modules())

        restored, skipped = 0, []
        with torch.no_grad():
            for path, state in self._snapshots.items():
                module = named.get(path)
                if module is None:
                    skipped.append(path)
                    continue
                weight = getattr(module, "weight", None)
                if weight is not None and "weight" in state:
                    if tuple(weight.shape) != tuple(state["weight"].shape):
                        skipped.append(path)
                        continue
                    weight.copy_(state["weight"])
                bias = getattr(module, "bias", None)
                if bias is not None and "bias" in state:
                    if tuple(bias.shape) == tuple(state["bias"].shape):
                        bias.copy_(state["bias"])
                restored += 1
        self._snapshots.clear()
        self._module_backup.clear()
        self._journal.append(JournalEntry("restore", "*", {"restored": restored}))
        return {"ok": True, "restored_modules": restored, "skipped": skipped,
                "note": "a shape rewrite cannot be restored by copying - "
                        "rebuild the model instead if a rewrite was applied."}

    def is_pristine(self) -> bool:
        """True when nothing differs from the snapshot."""
        named = dict(self.model.named_modules())
        for path, state in self._snapshots.items():
            module = named.get(path)
            if module is None:
                return False
            weight = getattr(module, "weight", None)
            if weight is not None and "weight" in state:
                if tuple(weight.shape) != tuple(state["weight"].shape):
                    return False
                if not torch.equal(weight.detach(), state["weight"]):
                    return False
            bias = getattr(module, "bias", None)
            if bias is not None and "bias" in state:
                if tuple(bias.shape) == tuple(state["bias"].shape):
                    if not torch.equal(bias.detach(), state["bias"]):
                        return False
        for path, backup in self._module_backup.items():
            current = named.get(path)
            if current is None:
                return False
            if tuple(current.weight.shape) != tuple(backup.weight.shape):
                return False
        return True

    # ------------------------------------------------------------ primitives
    def set_weight(self, path: str, index: Sequence[int] | int,
                   values: Any) -> Dict[str, Any]:
        """Overwrite a row/column of a module's weight. Arbitrary values."""
        module = self.model.get_submodule(path)
        axes = prunable_axes(module)
        if axes is None:
            return {"ok": False, "error": "not_writable", "target": path}
        sid = self._snapshot([path])
        weight, axis = axes
        with torch.no_grad():
            if isinstance(index, int):
                index = [index]
            if axis == 0:
                weight[list(index)] = values
            else:
                weight[:, list(index)] = values
        self._journal.append(JournalEntry(
            "set_weight", path,
            {"indices": list(index), "shape": list(weight.shape)}, sid))
        return {"ok": True, "target": path, "snapshot_id": sid,
                "shape": list(weight.shape)}

    def set_attr(self, path: str, name: str, value: Any) -> Dict[str, Any]:
        """Set an arbitrary attribute on a module. This is the escape hatch."""
        module = self.model.get_submodule(path)
        sid = self._snapshot([path])
        setattr(module, name, value)
        self._journal.append(JournalEntry(
            "set_attr", path, {"name": name}, sid))
        return {"ok": True, "target": path, "attr": name, "snapshot_id": sid}

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> Dict[str, Any]:
        """Overwrite the model with an external state dict."""
        sid = self.snapshot()
        missing = self.model.load_state_dict(state, strict=strict)
        self._journal.append(JournalEntry(
            "load_state_dict", "*", {"strict": strict}, sid))
        return {"ok": True, "snapshot_id": sid, "missing": list(missing.missing_keys),
                "unexpected": list(missing.unexpected_keys)}

    # ------------------------------------------------------------ unit edits
    def zero_units(self, path: str, indices: Sequence[int]) -> Dict[str, Any]:
        """Zero the given output units. Shape unchanged, reversible."""
        module = self.model.get_submodule(path)
        if prunable_axes(module) is None:
            return {"ok": False, "error": "not_writable", "target": path}
        sid = self._snapshot([path])
        n = zero_units(module, indices)
        self._journal.append(JournalEntry(
            "zero_units", path, {"units": n}, sid))
        return {"ok": True, "target": path, "zeroed_units": n,
                "shape_unchanged": True, "snapshot_id": sid}

    # ------------------------------------------------------- shape rewrite
    def _consumers_and_siblings(self, path: str) -> Tuple[List[str], List[str]]:
        """The modules a shape rewrite has to keep consistent with the target.

        Both sets are scoped to the target's OWN PARENT, and that scoping is
        not cosmetic - it is the difference between a working rewrite and a
        broken model.

        An earlier version searched the WHOLE model for "modules whose input
        dim equals the target's output dim". On GPT-2 that matched the
        ``c_proj`` of every one of the 12 blocks, not just the one that
        actually consumes this ``c_fc``. Shrinking layer 0's FFN therefore
        shrank all 12 blocks' projections and the model stopped running.

        The FFN's internal wiring always lives inside the same parent module
        (GPT-2: ``mlp.c_fc`` -> ``mlp.c_proj``; Llama/Qwen:
        ``mlp.gate_proj`` -> ``mlp.down_proj``, with ``mlp.up_proj`` as a
        sibling), so the parent is the correct and sufficient scope.
        """
        target = self.model.get_submodule(path)
        d_out = output_dim(target)
        parent_path = path.rsplit(".", 1)[0] if "." in path else ""
        consumers, siblings = [], []
        for other_path, other in self.model.named_modules():
            if other_path == path or prunable_axes(other) is None:
                continue
            if parent_path:
                is_child = (other_path.startswith(parent_path + ".")
                            and "." not in other_path[len(parent_path) + 1:])
            else:
                is_child = "." not in other_path
            if not is_child:
                continue
            if input_dim(other) == d_out:
                consumers.append(other_path)
            if output_dim(other) == d_out:
                siblings.append(other_path)
        return consumers, siblings

    def remove_units(self, path: str, indices: Sequence[int]) -> Dict[str, Any]:
        """Actually REMOVE output units: a real shape rewrite.

        The target shrinks along its output axis; every consumer shrinks along
        its input axis; every sibling shrinks along its output axis. A
        snapshot is taken first, but note that a shape rewrite cannot be
        undone by copying - the caller must keep the original model.
        """
        target = self.model.get_submodule(path)
        axes = prunable_axes(target)
        if axes is None:
            return {"ok": False, "error": "not_writable", "target": path}
        weight, axis = axes
        d_out = int(weight.shape[axis])
        keep = [i for i in range(d_out) if i not in set(int(x) for x in indices)]
        if not keep or len(keep) == d_out:
            return {"ok": False, "error": "nothing_to_remove", "target": path,
                    "out_features": d_out}

        consumers, siblings = self._consumers_and_siblings(path)
        if not consumers and not siblings:
            # Shrinking this module's output would leave its consumers with a
            # mismatched shape and the model would break. Refuse rather than
            # guess which module was meant.
            return {"ok": False, "error": "no_consumer_in_parent", "target": path,
                    "out_features": d_out,
                    "note": ("nothing inside the same parent consumes this "
                             "module's output, so a shape rewrite cannot be "
                             "made consistent; use zero_units instead")}
        sid = self._snapshot([path] + consumers + siblings)

        # keep a pristine copy of every module this rewrite will touch, so the
        # rewrite can be undone even though the shapes changed
        for p in [path] + consumers + siblings:
            if p not in self._module_backup:
                self._module_backup[p] = copy.deepcopy(self.model.get_submodule(p))

        # ---- 1. shrink the target along its output axis
        with torch.no_grad():
            if axis == 0:
                new_w = weight[keep, :].detach().clone()
            else:
                new_w = weight[:, keep].detach().clone()
            bias = getattr(target, "bias", None)
            new_b = None
            if bias is not None:
                new_b = bias[keep].detach().clone()

            # ---- 2. shrink the consumers along their input axis
            consumer_changes = {}
            for c in consumers:
                cm = self.model.get_submodule(c)
                cw, caxis = prunable_axes(cm)
                # consumer's INPUT dim sits on axis 1-caxis
                in_axis = 1 - caxis
                with torch.no_grad():
                    if in_axis == 0:
                        newcw = cw[keep, :].detach().clone()
                    else:
                        newcw = cw[:, keep].detach().clone()
                    cb = getattr(cm, "bias", None)
                    newcb = cb.detach().clone() if cb is not None else None
                consumer_changes[c] = {"weight": newcw, "bias": newcb}

            # ---- 3. shrink the siblings along their output axis
            sibling_changes = {}
            for s in siblings:
                sm = self.model.get_submodule(s)
                sw, saxis = prunable_axes(sm)
                with torch.no_grad():
                    if saxis == 0:
                        newsw = sw[keep, :].detach().clone()
                    else:
                        newsw = sw[:, keep].detach().clone()
                    sb = getattr(sm, "bias", None)
                    newsb = sb[keep].detach().clone() if sb is not None else None
                sibling_changes[s] = {"weight": newsw, "bias": newsb}

        # ---- 4. commit the new shapes
        with torch.no_grad():
            _replace_weight(target, new_w)
            if new_b is not None:
                _replace_bias(target, new_b)
            for c, ch in consumer_changes.items():
                cm = self.model.get_submodule(c)
                _replace_weight(cm, ch["weight"])
                if ch["bias"] is not None:
                    _replace_bias(cm, ch["bias"])
            for s, ch in sibling_changes.items():
                sm = self.model.get_submodule(s)
                _replace_weight(sm, ch["weight"])
                if ch["bias"] is not None:
                    _replace_bias(sm, ch["bias"])

        # ---- 5. keep the config honest
        config = getattr(self.model, "config", None)
        if config is not None:
            # keep the config honest so a re-save rebuilds the SAME shape.
            # GPT-2/Llama use n_inner, Qwen uses intermediate_size.
            if getattr(config, "n_inner", None) is not None:
                config.n_inner = len(keep)
            elif hasattr(config, "intermediate_size"):
                config.intermediate_size = len(keep)
            elif hasattr(config, "hidden_size"):
                pass  # no expansion dim in the config; nothing to update

        self._journal.append(JournalEntry(
            "remove_units", path,
            {"removed": d_out - len(keep), "from": d_out, "to": len(keep),
             "consumers": consumers, "siblings": siblings,
             "note": "a shape rewrite cannot be restored by copying"},
            sid))
        return {
            "ok": True,
            "target": path,
            "removed_units": d_out - len(keep),
            "out_features_before": d_out,
            "out_features_after": len(keep),
            "consumers_reshaped": consumers,
            "siblings_reshaped": siblings,
            "snapshot_id": sid,
            "warning": ("shape rewrite: the original tensors are needed to "
                        "rebuild the model. Keep the pre-rewrite model."),
        }


def _replace_weight(module: nn.Module, new_weight: torch.Tensor) -> None:
    """Install a new weight tensor, preserving the module's dtype/device.

    A shape rewrite produces a tensor of a DIFFERENT shape, so the parameter
    object has to be replaced - ``copy_`` into the old one would raise on the
    shape mismatch, and slicing the old one in place would silently leave the
    module with the old shape.
    """
    old = module.weight
    fresh = new_weight.to(dtype=old.dtype, device=old.device).detach().clone()
    module.weight = nn.Parameter(fresh, requires_grad=old.requires_grad)
    _sync_conv1d(module)


def _sync_conv1d(module: nn.Module) -> None:
    """HuggingFace's Conv1D caches its input/output sizes in `nx`/`nf` and
    uses them to reshape the output. Shrinking the weight without updating
    them makes the forward pass raise a shape error - or worse, silently
    mis-reshape. Keep them in sync with the tensor."""
    if type(module).__name__ == "Conv1D":
        weight = module.weight
        module.nx = int(weight.shape[0])
        module.nf = int(weight.shape[1])


def _replace_bias(module: nn.Module, new_bias: torch.Tensor) -> None:
    old = module.bias
    fresh = new_bias.to(dtype=old.dtype, device=old.device).detach().clone()
    module.bias = nn.Parameter(fresh, requires_grad=old.requires_grad)
