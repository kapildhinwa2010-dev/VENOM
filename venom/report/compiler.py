"""Turn raw telemetry numbers into a short natural-language report for the host."""
from __future__ import annotations

from typing import Dict


def compile_uncertainty(u: Dict) -> str:
    if u.get("is_uncertain"):
        if u.get("margin", 1.0) < 0.05:
            return (
                f"VERY UNCERTAIN: top two candidates are nearly tied "
                f"(margin={u.get('margin', 0):.2f}). Use venom.recall."
            )
        return (
            f"UNCERTAIN: confidence={u.get('confidence', 0):.2f}, "
            f"entropy={u.get('entropy', 0):.2f}."
        )
    return f"CONFIDENT: {u.get('confidence', 0):.2f} (margin={u.get('margin', 0):.2f})"


def compile_drift(d: Dict) -> str:
    if d.get("error"):
        return f"DRIFT ERROR: {d['error']}"
    parts = []
    if d.get("fixed", 0) > 0:
        parts.append(f"{d['fixed']} corrected")
    if d.get("harmed", 0) > 0:
        parts.append(f"{d['harmed']} BROKEN!")
    if d.get("abstained", 0) > 0:
        parts.append(f"{d['abstained']} abstained")
    if d.get("unchanged", 0) > 0:
        parts.append(f"{d['unchanged']} unchanged")
    net = d.get("net_gain", 0)
    verdict = "NET POSITIVE" if net > 0 else ("NET NEGATIVE!" if net < 0 else "NEUTRAL")
    return f"DRIFT: {', '.join(parts)}. Net: {net}. {verdict}."


def compile_memory_stats(s: Dict) -> str:
    capacity = s.get("capacity", 0)
    fill = (s.get("slots", 0) / capacity) if capacity else 0.0
    if s.get("corruptions", 0) == 0:
        health = "HEALTHY"
    else:
        health = f"CORRUPTED({s['corruptions']})"
    return (
        f"MEMORY: {s.get('slots', 0)}/{capacity} ({fill:.0%}), "
        f"{s.get('evictions', 0)} evictions. {health}."
    )


def compile_budget(b: Dict) -> str:
    if b.get("left", 0) == 0:
        return "BUDGET EXHAUSTED. Answer with what you have."
    return f"BUDGET: {b.get('left', 0)}/{b.get('max', 0)} calls left."


COMPILERS = {
    "uncertainty": compile_uncertainty,
    "drift": compile_drift,
    "memory_stats": compile_memory_stats,
    "budget": compile_budget,
}


def compile_all(parts: Dict) -> str:
    lines = ["=== VENOM REPORT ==="]
    for key, data in parts.items():
        fn = COMPILERS.get(key)
        lines.append(fn(data) if fn else f"{key}: {data}")
    lines.append("=== END ===")
    return "\n".join(lines)
