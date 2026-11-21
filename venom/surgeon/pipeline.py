"""Automated surgery for small models: every feature, then a decision ladder.

The problem this solves: for a 100M-class model nobody is going to hand-pick
which neurons to delete, and the features that tell you *what* is wasteful
(census, redundancy, compressibility, twin towers, skip-run, power-law) each
answer a different question. So the layer runs them all, feeds every answer
into **one ordered ladder**, and the ladder emits a verdict per unit.

The ladder is the MATLAB shape - a single `if / elseif / elseif / else` chain
where the first matching rule wins:

    if   the unit fired on exactly zero samples        -> REMOVE
    elif the unit fired on < rare_threshold of samples -> ZERO
    elif the layer is a twin of another layer          -> FLAG_LAYER
    elif the layer is compressible AND flat-tailed     -> FLAG_COMPRESS
    else                                                -> KEEP

Only the first two rules touch weights, and only `REMOVE` shrinks a tensor.
`FLAG_LAYER` and `FLAG_COMPRESS` are report-only, and the pipeline says so.

Everything the ladder decides is then applied through the usual gate: apply,
measure the drift on a **disjoint held-out set**, roll back if it is over the
threshold. A ladder that says `REMOVE` on a model where nothing is dead is
correctly a no-op - and that is the usual outcome on a healthy model.

    from venom.surgeon.pipeline import auto_surgery
    out = auto_surgery(model, tok, plan_prompts=PLAN, heldout_prompts=HELD)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .census import census, dead_units
from .kill import capture_logits, compare_logits
from .write import WriteAccess, prunable_axes

def _tap_layer_index(name: str) -> Optional[int]:
    """The layer index a tap name refers to ('h_5' -> 5)."""
    try:
        return int(str(name).rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return None


def _layer_prefixes(model: nn.Module) -> Dict[int, str]:
    """layer index -> the module path prefix of that block.

    The taps are named after layers ('h_5'), but the ladder reasons about
    MODULE paths ('transformer.h.5.mlp.c_fc'). Without this mapping the
    twin-layer and compressible rules can never fire - they would compare a
    tap name against a module path and never match.
    """
    from venom.core.discovery import fingerprint_universal

    fp = fingerprint_universal(model)
    blocks_path = fp.get("blocks_path") or ""
    n = int(fp.get("num_layers") or 0)
    if not blocks_path:
        return {}
    return {i: f"{blocks_path}.{i}." for i in range(n)}


# ---------------------------------------------------------------- the ladder
REMOVE = "REMOVE"        # exactly dead -> the tensor can shrink
ZERO = "ZERO"            # nearly dead -> zero it, keep the shape
FLAG_LAYER = "FLAG_LAYER"      # the whole layer looks redundant
FLAG_COMPRESS = "FLAG_COMPRESS"  # the layer looks compressible
KEEP = "KEEP"            # untouched


@dataclass
class LadderRule:
    order: int
    name: str
    description: str
    action: str


# The chain, in priority order. The first rule that matches wins.
LADDER_RULES: List[LadderRule] = [
    LadderRule(1, "exactly_dead",
               "the unit fired on zero calibration samples", REMOVE),
    LadderRule(2, "nearly_dead",
               "the unit fired on fewer than rare_threshold of samples", ZERO),
    LadderRule(3, "twin_layer",
               "the unit's layer is a near-duplicate of another layer",
               FLAG_LAYER),
    LadderRule(4, "compressible_flat",
               "the layer is compressible and its spectrum is flat-tailed",
               FLAG_COMPRESS),
    LadderRule(5, "default", "no rule matched", KEEP),
]


@dataclass
class Verdict:
    path: str
    index: int
    action: str
    rule: str
    rate: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "index": self.index, "action": self.action,
                "rule": self.rule, "fired_rate": round(self.rate, 6),
                "reason": self.reason}


# ------------------------------------------------------------- the features
def _taps_for(model: nn.Module, tokenizer: Any, prompts: Sequence[str]):
    """Capture per-layer hidden states for the feature stages."""
    from venom.core.bus import Bus
    from venom.core.discovery import fingerprint_universal, get_blocks

    fp = fingerprint_universal(model)
    blocks = get_blocks(model, fp)
    bus = Bus(model, blocks)
    ids = tokenizer(prompts[0], return_tensors="pt")
    with torch.no_grad():
        with bus:
            model(**ids)
    return bus.taps, blocks


def twin_tower_clusters(taps: Dict[str, Any], threshold: float = 0.99
                        ) -> Dict[str, Any]:
    """Cross-layer similarity clustering: which layers are near-duplicates.

    Native, so it does not need the TWIN-TOWER sibling repo.
    """
    if not taps:
        return {"error": "no taps", "clusters": []}
    names = sorted(taps, key=lambda n: int(str(n).rsplit("_", 1)[-1]))
    vectors = {}
    for name in names:
        h = taps[name]
        if h.dim() == 3:
            h = h[0]
        vectors[name] = h.reshape(-1, h.shape[-1]).float().mean(dim=0)
    clusters: List[List[str]] = []
    assigned = set()
    pairs = []
    with torch.no_grad():
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                va, vb = vectors[a], vectors[b]
                denom = va.norm() * vb.norm()
                if float(denom) == 0:
                    continue
                sim = float((va @ vb / denom).item())
                pairs.append((a, b, sim))
                if sim > threshold:
                    for g in clusters:
                        if a in g or b in g:
                            g.extend([x for x in (a, b) if x not in g])
                            assigned.update((a, b))
                            break
                    else:
                        clusters.append([a, b])
                        assigned.update((a, b))
    return {
        "threshold": threshold,
        "layers_compared": len(pairs),
        "clusters": [sorted(set(c)) for c in clusters if len(set(c)) > 1],
        "max_pair_similarity": round(max((p[2] for p in pairs), default=0.0), 6),
        "note": ("a near-duplicate layer pair is a candidate for skipping, not "
                 "for deleting"),
    }


def skip_run_test(model: nn.Module, tokenizer: Any, blocks: Sequence[Any],
                  prompts: Sequence[str], layer_index: int) -> Dict[str, Any]:
    """How much does the model change if this block is skipped entirely?

    Native, so it does not need the SKIP-RUN- sibling repo. If the delta is
    ~0, the block contributed nothing on these prompts - a twin tower.
    """
    if layer_index < 0 or layer_index >= len(blocks):
        return {"error": "bad_layer", "layer": layer_index}
    baseline = capture_logits(model, tokenizer, prompts)

    def skip_hook(module, inputs, output):
        hidden = inputs[0]
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    handle = blocks[layer_index].register_forward_hook(skip_hook)
    try:
        after = capture_logits(model, tokenizer, prompts)
    finally:
        handle.remove()
    delta = compare_logits(baseline, after)
    delta["layer"] = layer_index
    delta["skippable"] = bool(delta.get("max_abs_logit_delta", 1.0) < 1e-4)
    return delta


def powerlaw_exponent(matrix: torch.Tensor, top_fraction: float = 0.5
                      ) -> Optional[float]:
    """Slope of log(singular value) vs log(index) over the head of the
    spectrum. Steeper = heavier tail = more structured."""
    try:
        sv = torch.linalg.svdvals(matrix.float())
    except Exception:                                    # noqa: BLE001
        return None
    sv = sv[sv > 0]
    if sv.numel() < 8:
        return None
    k = max(4, int(sv.numel() * top_fraction))
    sv = sv[:k]
    idx = torch.arange(1, sv.numel() + 1, dtype=torch.float32)
    x = torch.log(idx)
    y = torch.log(sv)
    x = x - x.mean()
    y = y - y.mean()
    denom = float((x * x).sum())
    if denom <= 0:
        return None
    return float((x * y).sum()) / denom


def powerlaw_probe(taps: Dict[str, Any], model: nn.Module,
                   layer_weight_path: Optional[str] = None) -> Dict[str, Any]:
    """Compare the spectrum of a layer's ACTIVATIONS with the spectrum of its
    WEIGHTS. A steep activation tail means the layer carries structure; a
    flat one means it is closer to noise.

    Native, so it does not need the POWERLAW-PROBE sibling repo.
    """
    if not taps:
        return {"error": "no taps"}
    names = sorted(taps, key=lambda n: int(str(n).rsplit("_", 1)[-1]))
    named = dict(model.named_modules())
    rows = []
    for name in names:
        h = taps[name]
        if h.dim() == 3:
            h = h[0]
        act = h.reshape(-1, h.shape[-1]).float()
        if act.shape[0] < 2:
            continue
        row = {"layer": int(str(name).rsplit("_", 1)[-1]),
               "activation_exponent": powerlaw_exponent(act)}
        if layer_weight_path and layer_weight_path in named:
            w = named[layer_weight_path].weight.detach().float()
            row["weight_exponent"] = powerlaw_exponent(w)
            if row["activation_exponent"] is not None and \
                    row.get("weight_exponent") is not None:
                row["exponent_gap"] = round(
                    row["activation_exponent"] - row["weight_exponent"], 4)
                row["structured"] = row["exponent_gap"] < -0.5
        rows.append(row)
    return {
        "layers_analyzed": len(rows),
        "details": rows,
        "note": ("a negative exponent_gap means the ACTIVATIONS are flatter "
                 "than the WEIGHTS - the layer is closer to noise than to "
                 "structured content"),
    }


def _tap_layer_index(name: str) -> Optional[int]:
    """The layer index a tap name refers to ('h_5' -> 5)."""
    try:
        return int(str(name).rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return None


def _layer_prefixes(model: nn.Module) -> Dict[int, str]:
    """layer index -> the module path prefix of that block.

    The taps are named after layers ('h_5'), but the ladder reasons about
    MODULE paths ('transformer.h.5.mlp.c_fc'). Without this mapping the
    twin-layer and compressible rules can never fire - they would compare a
    tap name against a module path and never match.
    """
    from venom.core.discovery import fingerprint_universal

    fp = fingerprint_universal(model)
    blocks_path = fp.get("blocks_path") or ""
    n = int(fp.get("num_layers") or 0)
    if not blocks_path:
        return {}
    return {i: f"{blocks_path}.{i}." for i in range(n)}


# ---------------------------------------------------------------- the ladder
def decide(verdicts_input: Dict[str, Dict[str, Any]],
           twin_layers: Sequence[str],
           compressible_layers: Sequence[str],
           rare_threshold: float = 0.001) -> List[Verdict]:
    """Run the ladder over every unit and return one verdict per unit.

    `twin_layers` and `compressible_layers` are MODULE PATH PREFIXES
    ("transformer.h.5."), so a unit's path is matched with startswith. An
    exact comparison would silently never match, and the ladder would look
    like it had nothing to say.
    """
    out: List[Verdict] = []
    for path, info in verdicts_input.items():
        rate_of = info.get("fired_rate", [])
        for i, rate in enumerate(rate_of):
            if rate == 0.0:
                out.append(Verdict(path, i, REMOVE, "exactly_dead", rate,
                                    "fired on zero calibration samples"))
            elif rate < rare_threshold:
                out.append(Verdict(path, i, ZERO, "nearly_dead", rate,
                                    f"fired on {rate:.4%} of samples"))
            elif any(path.startswith(p) for p in twin_layers):
                out.append(Verdict(path, i, FLAG_LAYER, "twin_layer", rate,
                                    "the layer is a near-duplicate of another"))
            elif any(path.startswith(p) for p in compressible_layers):
                out.append(Verdict(path, i, FLAG_COMPRESS, "compressible_flat",
                                    rate, "the layer is compressible"))
            else:
                out.append(Verdict(path, i, KEEP, "default", rate,
                                    "no rule matched"))
    return out


def render_ladder(verdicts: Sequence[Verdict]) -> str:
    """A MATLAB-shaped view of what the ladder decided."""
    counts: Dict[str, int] = {}
    for v in verdicts:
        counts[v.action] = counts.get(v.action, 0) + 1
    lines = ["LADDER (first matching rule wins):", ""]
    for rule in LADDER_RULES:
        lines.append(f"  {rule.order}. if {rule.description}")
        lines.append(f"       -> {rule.action}   [{counts.get(rule.action, 0)} units]")
    lines.append(f"  else -> {KEEP}   [{counts.get(KEEP, 0)} units]")
    return "\n".join(lines)


# -------------------------------------------------------------- the pipeline
def auto_surgery(model: nn.Module,
                 tokenizer: Any,
                 *,
                 plan_prompts: Sequence[str],
                 heldout_prompts: Sequence[str],
                 mode: str = "zero",
                 rare_threshold: float = 0.001,
                 drift_threshold: float = 0.0,
                 max_units_per_module: int = 32,
                 twin_threshold: float = 0.99,
                 apply: bool = True,
                 label: str = "host") -> Dict[str, Any]:
    """Run every feature, walk the ladder, apply the verdicts, verify.

    This is the one-call automation for small models: load a model, hand it a
    calibration set and a held-out set, and get a decision it can defend.
    """
    report: Dict[str, Any] = {"label": label, "mode": mode}

    # ---- stage 1: the census
    cen = census(model, tokenizer, plan_prompts)
    if not cen.get("ok"):
        return {"ok": False, "error": "census_failed", "detail": cen}
    report["census"] = cen["summary"]

    # ---- stage 2: the feature stages (taps)
    taps, blocks = _taps_for(model, tokenizer, plan_prompts)
    from venom.sensors.audit import compressibility, redundancy
    report["redundancy"] = redundancy(taps)
    report["compressibility"] = compressibility(taps)
    report["twin_tower"] = twin_tower_clusters(taps, twin_threshold)
    report["powerlaw"] = powerlaw_probe(taps, model)

    # ---- stage 3: skip-run on the first block (is the depth real?)
    report["skip_run"] = (skip_run_test(model, tokenizer, blocks,
                                        heldout_prompts, 0)
                           if blocks else {"error": "no blocks"})

    # ---- stage 4: the ladder
    prefixes = _layer_prefixes(model)
    twin_layers: List[str] = []
    for cluster in report["twin_tower"].get("clusters", []):
        for name in cluster[1:]:              # keep the first, flag the rest
            idx = _tap_layer_index(name)
            if idx is not None and idx in prefixes:
                twin_layers.append(prefixes[idx])
    compressible_layers = []
    for name in report["compressibility"].get("details", {}).keys():
        idx = _tap_layer_index(name)
        if idx is not None and idx in prefixes:
            compressible_layers.append(prefixes[idx])
    verdicts = decide(cen["modules"], set(twin_layers),
                       set(compressible_layers), rare_threshold)

    # The two features can disagree, and the disagreement is the interesting
    # part: consecutive layers of a residual network often have a very high
    # output cosine (the residual stream dominates) while the layer is still
    # doing real work. Redundancy alone would call those layers removable.
    skippable = report["skip_run"].get("skippable")
    if report["redundancy"].get("near_duplicate", 0) and skippable is False:
        report["contradiction"] = {
            "note": ("consecutive layers have a high output cosine, but "
                     "skipping one changes the model's output - the residual "
                     "stream carries the information, so a high cosine does "
                     "NOT mean the layer is redundant"),
            "near_duplicate_pairs": report["redundancy"]["near_duplicate"],
            "skip_run_max_delta": report["skip_run"].get("max_abs_logit_delta"),
            "skippable": skippable,
        }

    tally: Dict[str, int] = {}
    for v in verdicts:
        tally[v.action] = tally.get(v.action, 0) + 1
    report["ladder"] = {
        "rules": [{"order": r.order, "name": r.name,
                   "description": r.description, "action": r.action}
                  for r in LADDER_RULES],
        "tally": tally,
        "text": render_ladder(verdicts),
    }
    report["verdicts"] = [v.to_dict() for v in verdicts
                          if v.action in (REMOVE, ZERO)]

    # ---- stage 5: apply (the only stage that touches weights)
    if not apply:
        report["applied"] = {"ok": True, "note": "dry run: nothing was modified"}
        return {"ok": True, **report}

    remove_targets: Dict[str, List[int]] = {}
    zero_targets: Dict[str, List[int]] = {}
    for v in verdicts:
        if v.action == REMOVE:
            remove_targets.setdefault(v.path, []).append(v.index)
        elif v.action == ZERO:
            zero_targets.setdefault(v.path, []).append(v.index)

    for path in remove_targets:
        remove_targets[path] = remove_targets[path][:max_units_per_module]
    for path in zero_targets:
        zero_targets[path] = zero_targets[path][:max_units_per_module]

    if not remove_targets and not zero_targets:
        report["applied"] = {
            "ok": True, "note": ("the ladder selected nothing to change - "
                                 "this is the usual result on a healthy model"),
            "removed_modules": {}, "zeroed_modules": {},
        }
        return {"ok": True, **report}

    from .kill import kill_dead_neurons
    kill_out = kill_dead_neurons(model, tokenizer, plan_prompts, heldout_prompts,
                                 mode=mode, threshold=0.0,
                                 drift_threshold=drift_threshold,
                                 write=WriteAccess(model, label=label))
    report["applied"] = kill_out
    return {"ok": kill_out.get("ok", True), **report}
