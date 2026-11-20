"""Live audit of the host — report only, never modifies weights.

`audit()` combines three analyses over the activations captured by the `Bus`:

  * dead neurons   — units that never activated across the captured samples
  * redundancy    — layers whose outputs are near-duplicates of each other
  * compressibility — effective rank of a layer's activation matrix

Everything here is **read-only**. The plan's P5 phase (actual pruning) is
deliberately deferred: pruning changes the host, and the project's trust
contract is that the host can always be restored bit-exactly. A prune that
cannot be proven lossless stays report-only, permanently.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

DEAD_THRESHOLD = 0.001
LOW_THRESHOLD = 0.05
REDUNDANCY_HI = 0.99
REDUNDANCY_LO = 0.95


def _layer_order(taps: Dict[str, Any]) -> List[str]:
    def key(name: str):
        try:
            return int(str(name).rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            return 0
    return sorted(taps, key=key)


def collect_activation_stats(taps: Dict[str, Any]) -> Dict[str, Any]:
    """Per-unit activation counts and per-layer norms from the captured taps.

    "Dead" here means: a unit whose activation was exactly zero in every
    captured sample. That is a *suspicion*, not a verdict - a unit can be
    zero simply because the prompt never needed it.
    """
    if not taps:
        return {"error": "no taps captured"}

    import torch

    unit_counts: Dict[str, int] = {}
    unit_total = 0
    layer_norms: Dict[str, float] = {}

    with torch.no_grad():
        for name in _layer_order(taps):
            hidden = taps[name]
            if hidden.dim() == 3:
                flat = hidden[0]                      # [seq, hidden]
            elif hidden.dim() == 2:
                flat = hidden
            else:
                flat = hidden.reshape(-1, hidden.shape[-1])
            nonzero = (flat != 0)
            unit_counts[name] = int(nonzero.sum().item())
            unit_total += flat.shape[0]
            layer_norms[name] = round(float(flat.float().norm().item()), 4)

    return {
        "unit_counts": unit_counts,
        "unit_total": unit_total,
        "layer_norms": layer_norms,
        "layers": len(unit_counts),
    }


def dead_neurons(taps: Dict[str, Any], threshold: float = DEAD_THRESHOLD) -> Dict[str, Any]:
    """Units that never fired, per layer."""
    stats = collect_activation_stats(taps)
    if "error" in stats:
        return stats
    total = stats["unit_total"]
    if total <= 0:
        return {"error": "no samples"}

    dead, low, active = [], [], []
    for name, count in stats["unit_counts"].items():
        frac = count / total
        if frac < threshold:
            dead.append({"layer": name, "frac": round(frac, 6)})
        elif frac < LOW_THRESHOLD:
            low.append({"layer": name, "frac": round(frac, 6)})
        else:
            active.append(name)

    return {
        "layers": stats["layers"],
        "dead_layers": len(dead),
        "low_layers": len(low),
        "active_layers": len(active),
        "threshold": threshold,
        "dead_entries": dead[:50],
        "warning": (
            "A layer that never fired is not necessarily dead - it may simply "
            "not have been needed for this prompt. Nothing was removed."
        ),
    }


def layer_similarity(taps: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Cosine similarity between consecutive layers' output directions."""
    if not taps:
        return []
    import torch

    order = _layer_order(taps)
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        vectors = {}
        for name in order:
            hidden = taps[name]
            if hidden.dim() == 3:
                hidden = hidden[0]
            flat = hidden.reshape(-1, hidden.shape[-1]).float()
            vectors[name] = flat.mean(dim=0)  # mean-pooled direction
        for i in range(len(order) - 1):
            a, b = vectors[order[i]], vectors[order[i + 1]]
            denom = a.norm() * b.norm()
            sim = float((a @ b / denom).item()) if denom > 0 else 0.0
            rows.append({
                "i": int(str(order[i]).rsplit("_", 1)[-1]),
                "j": int(str(order[i + 1]).rsplit("_", 1)[-1]),
                "cosine_sim": round(sim, 6),
            })
    return rows


def redundancy(taps: Dict[str, Any]) -> Dict[str, Any]:
    """Layers whose outputs are near-duplicates of the previous layer."""
    rows = layer_similarity(taps)
    hi = [r for r in rows if r["cosine_sim"] > REDUNDANCY_HI]
    lo = [r for r in rows if REDUNDANCY_LO < r["cosine_sim"] <= REDUNDANCY_HI]
    return {
        "pairs_compared": len(rows),
        "near_duplicate": len(hi),
        "similar": len(lo),
        "candidates": hi[:20],
        "warning": "Report only. No layers were merged.",
    }


def compressibility(taps: Dict[str, Any], max_layers: int = 6) -> Dict[str, Any]:
    """Effective rank of the captured activation matrix, per layer."""
    if not taps:
        return {"error": "no taps captured"}
    import torch

    order = _layer_order(taps)
    details: Dict[str, Any] = {}
    with torch.no_grad():
        for name in order[-max_layers:]:
            hidden = taps[name]
            if hidden.dim() == 3:
                hidden = hidden[0]
            flat = hidden.reshape(-1, hidden.shape[-1]).float()
            if flat.shape[0] < 2:
                continue
            # singular values via the eigenvalues of the Gram matrix
            gram = flat @ flat.T
            try:
                eig = torch.linalg.eigvalsh(gram)
                vals = torch.clamp(eig, min=0).sqrt().flip(0)
            except Exception:  # noqa: BLE001
                continue
            total = float((vals ** 2).sum().item())
            if total <= 0:
                continue
            cum = 0.0
            k99 = len(vals)
            for i, v in enumerate(vals.tolist()):
                cum += v * v
                if cum / total >= 0.99:
                    k99 = i + 1
                    break
            ratio = k99 / len(vals)
            details[name] = {
                "eff_rank": k99,
                "full_rank": len(vals),
                "ratio": round(ratio, 4),
                "compressible": ratio < 0.5,
            }
    return {
        "analyzed": len(details),
        "compressible": sum(1 for d in details.values() if d["compressible"]),
        "details": details,
        "warning": "Analysis only. Nothing was compressed.",
    }


def audit(taps: Dict[str, Any], threshold: float = DEAD_THRESHOLD) -> Dict[str, Any]:
    """The full report. Never modifies the host."""
    if not taps:
        return {
            "error": "no taps captured - attach the bus before auditing",
            "warning": "Report only. Nothing was modified.",
        }
    return {
        "dead_neurons": dead_neurons(taps, threshold),
        "redundancy": redundancy(taps),
        "compressibility": compressibility(taps),
        "warning": (
            "REPORT ONLY. No weights were modified, nothing was pruned, nothing "
            "was merged. Actual surgery requires an opt-in mode plus a drift "
            "proof on a held-out set (plan phase P5)."
        ),
    }
