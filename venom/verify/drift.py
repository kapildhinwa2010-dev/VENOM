"""Compare a baseline run against a VENOM run and classify every difference."""
from __future__ import annotations

from typing import Dict, List, Optional

ABSTAIN_MARKERS = ("", "abstain", "i do not know", "unknown")


def drift_report(baseline: List[str], venom: List[str], gold: Optional[List[str]] = None) -> Dict:
    n = len(baseline)
    if len(venom) != n:
        return {"error": "length mismatch"}
    if gold is not None and len(gold) != n:
        return {"error": "gold length mismatch"}

    unchanged = fixed = harmed = shifted = abstained = 0
    details = []
    for i in range(n):
        b = baseline[i].strip().lower()
        v = venom[i].strip().lower()
        g = gold[i].strip().lower() if gold else None

        if v in ABSTAIN_MARKERS:
            abstained += 1
            category = "abstained"
        elif b == v:
            unchanged += 1
            category = "unchanged"
        elif g is not None:
            base_ok = g in b
            venom_ok = g in v
            if not base_ok and venom_ok:
                fixed += 1
                category = "fixed"
            elif base_ok and not venom_ok:
                harmed += 1
                category = "harmed"
            else:
                shifted += 1
                category = "shifted"
        else:
            shifted += 1
            category = "shifted"

        details.append({"idx": i, "category": category})

    changed = fixed + harmed + shifted + abstained
    return {
        "n": n,
        "unchanged": unchanged,
        "changed": changed,
        "fixed": fixed,
        "harmed": harmed,
        "shifted": shifted,
        "abstained": abstained,
        "change_rate": round(changed / n, 4) if n else 0.0,
        "net_gain": fixed - harmed,
        "details": details,
    }
