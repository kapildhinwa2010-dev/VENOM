"""Native activation census — per-neuron, not per-layer.

The sibling repo DEAD-CELL can do this, but VENOM should not need a sibling
to answer a basic question about its own host, so this is a native
implementation. The sibling adapter in `venom/integrations/census.py` is still
there and is still called when it is available; this one always works.

What it measures
----------------
For every prunable projection module (an ``nn.Linear`` or a HuggingFace
``Conv1D``), how many of the *samples* in which the unit produced a
**non-zero** output. A unit that never fires is *exactly* dead. A unit that
fires rarely is *nearly* dead. Those are different claims, so they are
reported separately — collapsing them into one "dead" bucket is how a
prune ends up removing something that mattered.

The classic caveat applies and is repeated in every payload: **the census is
sample-dependent**. A unit that never fired in 20 prompts may fire on the
21st. That is why the kill path requires a calibration set and a held-out
verification set, and why the honest report says how many samples it saw.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from .write import prunable_axes

# The projections where "dead unit" is a meaningful concept: the FFN's
# expansion layer. A unit there that never fires is pure waste.
FFN_PATTERNS = ("c_fc", "gate_proj", "up_proj", "fc1", "fc_in", "w1",
                 "dense_h_to_4h", "up_proj")


def _is_ffn(path: str) -> bool:
    return any(p in path for p in FFN_PATTERNS)


def census(model: nn.Module,
           tokenizer: Any,
           prompts: Sequence[str],
           *,
           pattern: Optional[str] = None,
           ffn_only: bool = True,
           max_units_per_module: int = 0) -> Dict[str, Any]:
    """Count, per output unit, how many prompts made it fire.

    Parameters
    ----------
    pattern:
        Substring filter on the module path. ``None`` means "every prunable
        module", subject to ``ffn_only``.
    ffn_only:
        Restrict to the FFN expansion projections. This is the default
        because those are the units "dead neuron" refers to in practice.
    max_units_per_module:
        If > 0, only the first N units of each module are counted (a speed
        knob for very large models).

    Returns
    -------
    A dict with per-module counts and a summary. The summary always carries
    ``samples`` so a reader can see how much evidence there was.
    """
    named = dict(model.named_modules())
    targets: List[Dict[str, Any]] = []
    for path, module in named.items():
        axes = prunable_axes(module)
        if axes is None:
            continue
        if pattern is not None:
            if pattern not in path:
                continue
        elif ffn_only and not _is_ffn(path):
            continue
        _, axis = axes
        out_dim = int(axes[0].shape[axis])
        if max_units_per_module:
            out_dim = min(out_dim, max_units_per_module)
        targets.append({"path": path, "module": module, "axis": axis,
                        "out_dim": out_dim, "type": type(module).__name__})

    if not targets:
        return {"ok": False, "error": "no_targets",
                "note": "no prunable projection matched the filter",
                "modules": {}, "summary": {"samples": 0}}

    fired = {t["path"]: [0] * t["out_dim"] for t in targets}
    magnitude = {t["path"]: [0.0] * t["out_dim"] for t in targets}
    handles = []

    def make_hook(path: str, out_dim: int):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            if output.dim() == 2:
                output = output.unsqueeze(0)
            # [B, T, D] -> per-sample, per-unit "did it fire"
            was_on = (output != 0)
            fired[path][:out_dim] = [
                fired[path][i] + int(was_on[:, :, i].any(dim=1).sum().item())
                for i in range(out_dim)
            ]
            mag = output.detach().abs().sum(dim=(0, 1)).float().cpu().tolist()
            magnitude[path][:out_dim] = [
                magnitude[path][i] + float(mag[i]) for i in range(out_dim)
            ]
        return hook

    for t in targets:
        handles.append(t["module"].register_forward_hook(make_hook(t["path"], t["out_dim"])))

    processed = 0
    try:
        with torch.no_grad():
            for prompt in prompts:
                if not prompt or not prompt.strip():
                    continue
                enc = tokenizer(prompt, return_tensors="pt")
                model(**enc)
                processed += 1
    finally:
        for h in handles:
            h.remove()

    n = max(processed, 1)
    modules: Dict[str, Any] = {}
    for t in targets:
        path = t["path"]
        counts = fired[path]
        mags = magnitude[path]
        never = [i for i, c in enumerate(counts) if c == 0]
        rate = [c / n for c in counts]
        modules[path] = {
            "type": t["type"],
            "axis": t["axis"],
            "out_features": t["out_dim"],
            "fired_count": counts,
            "fired_rate": [round(r, 6) for r in rate],
            "mean_abs": [round(m / max(n, 1), 8) for m in mags],
            "never_fired": never,
            "never_fired_count": len(never),
            "rare": [i for i, r in enumerate(rate) if 0 < r < 0.001],
            "note": ("fired_count is the number of PROMPTS in which the unit "
                     "produced a non-zero output"),
        }

    total_units = sum(m["out_features"] for m in modules.values())
    total_never = sum(m["never_fired_count"] for m in modules.values())
    return {
        "ok": True,
        "samples": processed,
        "modules": modules,
        "summary": {
            "samples": processed,
            "modules_scanned": len(modules),
            "total_units": total_units,
            "never_fired_units": total_never,
            "never_fired_fraction": round(total_never / total_units, 6) if total_units else 0.0,
            "modules_with_dead": [p for p, m in modules.items() if m["never_fired_count"]],
        },
        "caveat": ("A census is only as good as its calibration set. A unit "
                   "that never fired over these prompts may still fire on "
                   "others. Nothing here is proof that a unit is dead."),
    }


def dead_units(census_result: Dict[str, Any],
               threshold: float = 0.001) -> Dict[str, List[int]]:
    """Units whose firing rate is below `threshold`, per module.

    `threshold=0.0` returns only the units that fired on **exactly zero**
    samples - the strongest claim this census can support.
    """
    out: Dict[str, List[int]] = {}
    for path, info in census_result.get("modules", {}).items():
        n = census_result.get("samples", 0)
        if not n:
            continue
        # threshold <= 0 means EXACTLY zero firings. Using a strict `<` here
        # would compare against 0.0 and never match anything, which silently
        # turns the killer into a no-op.
        if threshold <= 0:
            picks = [i for i, r in enumerate(info["fired_rate"]) if r == 0.0]
        else:
            picks = [i for i, r in enumerate(info["fired_rate"]) if r < threshold]
        if picks:
            out[path] = picks
    return out
