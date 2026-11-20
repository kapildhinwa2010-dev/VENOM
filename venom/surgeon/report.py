"""Report-only analysis of the host model. Never modifies weights.

The surgeon module is deliberately report-only in v0.1: any actual pruning or
merging would change the host, and the trust contract says the host must be
restorable bit-exactly.
"""
from __future__ import annotations

from typing import Dict, List


def dead_neuron_report(activation_counts: Dict[str, int], total_samples: int,
                       threshold: float = 0.001) -> Dict:
    if total_samples <= 0:
        return {"error": "total_samples must be > 0"}
    dead, low, active = [], [], []
    for name, count in activation_counts.items():
        frac = count / total_samples
        if frac < threshold:
            dead.append({"name": name, "frac": round(frac, 6)})
        elif frac < 0.05:
            low.append({"name": name, "frac": round(frac, 6)})
        else:
            active.append(name)
    total = len(activation_counts)
    return {
        "total": total,
        "dead": len(dead),
        "dead_pct": round(len(dead) / total * 100, 2) if total else 0.0,
        "low": len(low),
        "active": len(active),
        "threshold": threshold,
        "samples": total_samples,
        "dead_entries": dead[:50],
        "warning": "NOT removed. Pruning needs an opt-in surgery mode plus a drift proof.",
    }


def redundancy_report(layer_sims: List[Dict]) -> Dict:
    above99 = [p for p in layer_sims if p.get("cosine_sim", 0) > 0.99]
    above95 = [p for p in layer_sims if 0.95 < p.get("cosine_sim", 0) <= 0.99]
    return {
        "near_dup_99": len(above99),
        "similar_95_99": len(above95),
        "candidates": above99[:20],
        "warning": "Report only. No layers merged.",
    }


def compressibility_report(svd_spectra: Dict[str, List[float]]) -> Dict:
    results: Dict[str, Dict] = {}
    for name, vals in svd_spectra.items():
        if not vals:
            continue
        total_e = sum(v * v for v in vals)
        if total_e == 0:
            results[name] = {"compressible": True}
            continue
        cum = 0.0
        k99 = len(vals)
        for i, v in enumerate(sorted(vals, reverse=True)):
            cum += v * v
            if cum / total_e >= 0.99:
                k99 = i + 1
                break
        ratio = k99 / len(vals)
        results[name] = {
            "eff_rank": k99,
            "full_rank": len(vals),
            "ratio": round(ratio, 4),
            "compressible": ratio < 0.5,
        }
    compressible = sum(1 for r in results.values() if r.get("compressible"))
    return {
        "analyzed": len(results),
        "compressible": compressible,
        "details": results,
        "warning": "Analysis only. Nothing was compressed.",
    }
