"""Fix rate vs harm rate. Host-agnostic: pass any host_fn(prompt) -> str."""
from __future__ import annotations

from venom.memory.store import norm


def _match(answer: str, gold: str) -> bool:
    return norm(gold) in norm(answer)


def run_bench(host_fn, facts, venom_fn=None) -> Dict:
    """facts: [{"q": str, "gold": str}, ...]

    venom_fn(prompt) -> prompt  (injects the memory context)
    """
    base_right = venom_right = fixed = harmed = 0
    rows = []
    for f in facts:
        b = host_fn(f["q"])
        v = host_fn(venom_fn(f["q"])) if venom_fn else b
        base_ok = _match(b, f["gold"])
        venom_ok = _match(v, f["gold"])
        base_right += base_ok
        venom_right += venom_ok
        fixed += (not base_ok and venom_ok)
        harmed += (base_ok and not venom_ok)
        rows.append({
            "q": f["q"], "gold": f["gold"],
            "base": b, "venom": v,
            "base_ok": base_ok, "venom_ok": venom_ok,
        })

    n = len(facts)
    wrong = n - base_right
    return {
        "n": n,
        "baseline_acc": base_right / n if n else 0.0,
        "venom_acc": venom_right / n if n else 0.0,
        "fix_rate": (fixed / wrong) if wrong else None,
        "harm_rate": (harmed / base_right) if base_right else None,
        "fixed": fixed, "harmed": harmed, "rows": rows,
    }


class MockHost:
    """Plumbing test ONLY. This is not a result about any real model."""

    def __init__(self, wrong_answers: Dict[str, str]):
        self.wrong = {norm(k): v for k, v in wrong_answers.items()}

    def __call__(self, prompt: str) -> str:
        if "CONTEXT:" in prompt:
            return prompt.split("CONTEXT:")[-1].split("|")[0].strip()
        for k, v in self.wrong.items():
            if k in norm(prompt):
                return v
        return "i do not know"
