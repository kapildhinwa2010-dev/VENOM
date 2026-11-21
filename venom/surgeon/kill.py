"""The dead-neuron killer — actually removing dead units from a live model.

This is the feature the master plan asked for under "dead neuron removal",
and it is the one operation in VENOM that permanently changes what the host
model *is*. It therefore comes with the full gate:

  1. **Census** the model on a calibration set and find units that fired on
     exactly zero samples (or on fewer than a threshold fraction).
  2. **Refuse** if there are none. "No dead units found" is a legitimate and
     frequent result — most modern models do not have exactly-dead FFN units,
     because SiLU/GELU and the biases keep things alive.
  3. **Apply** the removal: either zero the units (shape unchanged, fully
     reversible) or **remove** them (a real shape rewrite that shrinks the
     tensors and the downstream consumers).
  4. **Verify** the drift on a **HELD-OUT** set that was not used to find
     the dead units.
  5. **Restore** if the drift is over the threshold.

Two rules that are not negotiable, because breaking either one makes the
result meaningless:

* The calibration set and the held-out set must be **disjoint**. Verifying a
  prune on the prompts that justified it proves nothing.
* A prune that cannot be verified is **not kept**. If no verifier is given,
  the weights are restored before returning — "never accepted without a
  verifier" must not leave the host modified anyway.

What this does NOT do: it does not make the model smarter. Pruning buys
memory and compute, not intelligence. The intelligence path in this project
is memory + injection, and it is a separate mechanism.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .census import census, dead_units
from .write import WriteAccess


# ---------------------------------------------------------------- logit drift
def capture_logits(model: nn.Module, tokenizer: Any,
                   prompts: Sequence[str]) -> List[torch.Tensor]:
    """The last-position logits for each prompt."""
    out: List[torch.Tensor] = []
    with torch.no_grad():
        for prompt in prompts:
            if not prompt or not prompt.strip():
                continue
            enc = tokenizer(prompt, return_tensors="pt")
            result = model(**enc)
            # a HF model returns an object with `.logits`; a bare module
            # returns the tensor itself. Handle both.
            logits = getattr(result, "logits", result)
            out.append(logits[0, -1, :].float().cpu())
    return out


def compare_logits(base: Sequence[torch.Tensor],
                    current: Sequence[torch.Tensor]) -> Dict[str, Any]:
    """How far the model's outputs moved."""
    if not base or len(base) != len(current):
        return {"ok": False, "error": "baseline_mismatch",
                "baseline": len(base), "current": len(current)}
    max_abs, abs_sum, agree = 0.0, 0.0, 0
    total = 0
    for b, c in zip(base, current):
        diff = (b - c).abs()
        max_abs = max(max_abs, float(diff.max()))
        abs_sum += float(diff.sum())
        total += b.numel()
        agree += int(int(b.argmax()) == int(c.argmax()))
    n = len(base)
    return {
        "ok": True,
        "max_abs_logit_delta": round(max_abs, 8),
        "mean_abs_logit_delta": round(abs_sum / max(total, 1), 10),
        "top1_agreement": round(agree / n, 4),
        "samples": n,
    }


# ------------------------------------------------------------------- the kill
def kill_dead_neurons(model: nn.Module,
                      tokenizer: Any,
                      plan_prompts: Sequence[str],
                      heldout_prompts: Sequence[str],
                      *,
                      mode: str = "zero",
                      threshold: float = 0.0,
                      drift_threshold: float = 0.0,
                      max_units_per_module: int = 32,
                      max_modules: int = 4,
                      ffn_only: bool = True,
                      write: Optional[WriteAccess] = None) -> Dict[str, Any]:
    """Find dead units, remove them, verify on held-out data, restore if bad.

    Parameters
    ----------
    mode:
        ``"zero"``   - zero the units. Shape unchanged, bit-exactly reversible.
        ``"remove"`` - actually remove them: a shape rewrite that shrinks the
                       tensors and the downstream consumers.
    threshold:
        A unit is a candidate if it fired on fewer than this fraction of the
        calibration samples. ``0.0`` means **exactly zero** — the only claim
        a census can make without a much larger calibration set.
    drift_threshold:
        Maximum tolerated drift on the held-out set before the change is
        rolled back.
    """
    if mode not in ("zero", "remove"):
        return {"ok": False, "error": "bad_mode", "mode": mode,
                "allowed": ["zero", "remove"]}
    if not plan_prompts:
        return {"ok": False, "error": "no_calibration_set",
                "note": "a prune cannot be planned without calibration prompts"}
    if not heldout_prompts:
        return {"ok": False, "error": "no_heldout_set",
                "note": ("a prune is never accepted without a held-out "
                         "verification set that is disjoint from the "
                         "calibration set")}

    # ---- 1. census on the calibration set
    cen = census(model, tokenizer, plan_prompts, ffn_only=ffn_only)
    if not cen.get("ok"):
        return {"ok": False, "error": "census_failed", "detail": cen}
    candidates = dead_units(cen, threshold)

    report: Dict[str, Any] = {
        "ok": True,
        "mode": mode,
        "threshold": threshold,
        "calibration_samples": cen["samples"],
        "heldout_samples": len(heldout_prompts),
        "census_summary": cen["summary"],
        "census_caveat": cen["caveat"],
    }

    if not candidates:
        report.update({
            "ok": False,
            "error": "nothing_dead",
            "verdict": ("no unit fired on fewer than "
                        f"{threshold} of {cen['samples']} calibration samples; "
                        "nothing was modified"),
            "never_fired_units": cen["summary"]["never_fired_units"],
            "total_units": cen["summary"]["total_units"],
        })
        return report

    # ---- 2. baseline on the held-out set, BEFORE any edit
    baseline = capture_logits(model, tokenizer, heldout_prompts)

    # ---- 3. apply
    w = write or WriteAccess(model)
    applied: List[Dict[str, Any]] = []
    for path, indices in list(candidates.items())[:max_modules]:
        chosen = indices[:max_units_per_module]
        if mode == "zero":
            res = w.zero_units(path, chosen)
        else:
            res = w.remove_units(path, chosen)
        if res.get("ok"):
            applied.append({"path": path, "units": len(chosen), "result": res})
    report["applied"] = applied

    if not applied:
        report.update({"ok": False, "error": "apply_failed",
                       "verdict": "nothing was modified"})
        return report

    # ---- 4. verify on the held-out set.
    # A shape rewrite can leave the model unable to run at all. If the
    # forward pass raises, that is a FAILED verification, not a pass with an
    # unknown drift - restore and say so.
    try:
        current = capture_logits(model, tokenizer, heldout_prompts)
    except Exception as exc:                             # noqa: BLE001
        w.restore()
        report.update({
            "ok": False,
            "error": "forward_broken",
            "message": str(exc)[:400],
            "rolled_back": True,
            "verdict": ("the model could not even run after the change, so "
                        "the change was rolled back"),
        })
        return report
    drift = compare_logits(baseline, current)
    report["drift"] = drift
    report["drift_threshold"] = drift_threshold

    if not drift.get("ok"):
        w.restore()
        report.update({"ok": False, "error": "drift_unmeasurable",
                       "rolled_back": True,
                       "verdict": ("the drift could not be measured, so the "
                                   "change was not kept; the weights were "
                                   "restored")})
        return report

    value = drift["max_abs_logit_delta"]
    if value > drift_threshold:
        w.restore()
        report.update({
            "ok": False,
            "error": "drift_exceeded",
            "rolled_back": True,
            "verdict": ("the change exceeded the drift threshold on the "
                        "held-out set and was rolled back"),
        })
        return report

    report.update({
        "ok": True,
        "rolled_back": False,
        "verdict": ("the change is within the drift threshold on the held-out "
                    "set and has been kept"),
        "units_removed": sum(a["units"] for a in applied),
        "warning": ("a drift within the threshold on a held-out set is "
                    "evidence, not proof of losslessness. Pruning buys memory "
                    "and compute, not intelligence."),
        "journal": w.journal(),
    })
    return report


def kill_report(model: nn.Module, tokenizer: Any,
                prompts: Sequence[str], **kwargs) -> Dict[str, Any]:
    """Report-only convenience wrapper: census + plan, no modification."""
    cen = census(model, tokenizer, prompts, ffn_only=kwargs.get("ffn_only", True))
    candidates = dead_units(cen, kwargs.get("threshold", 0.0))
    return {
        "ok": True,
        "read_only": True,
        "census_summary": cen.get("summary"),
        "candidate_modules": {p: len(v) for p, v in candidates.items()},
        "total_candidates": sum(len(v) for v in candidates.values()),
        "caveat": cen.get("caveat"),
        "note": "nothing was modified",
    }
