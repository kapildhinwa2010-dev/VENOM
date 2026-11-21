"""Correction bench against a REAL host model.

The plan's P0 exit gate: *"400 facts par fix rate > harm rate"*. This module
runs that gate honestly:

  * the host is asked each fact **twice** - once bare (baseline), once with
    the memory context injected (VENOM);
  * the SAME seed / deterministic decoding is used for both, so the only
    difference is the injected context;
  * a **neutral-memory baseline** is included, because the tool-drift
    literature shows that injecting an irrelevant memory shifts behaviour
    (arXiv 2605.24941). Without that control, a fix-rate improvement could
    just be "any extra text helps";
  * the four outcome categories (fixed / harmed / shifted / abstained) come
    from `venom.verify.drift`, where a correct answer replaced by a refusal
    counts as HARM.

What this does NOT prove: that the injection helps on any real workload. The
fact set is mundane world knowledge and the host is a small open model. This
is a gate, not a product claim.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from venom.bench.facts import facts, facts_for_model
from venom.memory.store import MemoryStore
from venom.reflex.injector import Injector
from venom.verify.drift import drift_report


def _match(answer: str, gold: str) -> bool:
    """Word-bounded match - 'uk' must not match 'ukraine'."""
    import re
    from venom.memory.store import norm
    g, a = norm(gold), norm(answer)
    if not g:
        return False
    return re.search(rf"(?<!\w){re.escape(g)}(?!\w)", a) is not None


def seed_memory(store: MemoryStore, fact_list: List[Dict[str, str]]) -> None:
    """Store every fact under its key."""
    for f in fact_list:
        store.remember(f["key"], f["gold"])


def seed_neutral_memory(store: MemoryStore, fact_list: List[Dict[str, str]]) -> None:
    """Store the SAME facts under keys that do not appear in any question.

    This is the control: the memory bank has the same size and the same kind
    of content, but nothing matches, so the reflex must not fire. Any change
    in the host's answers here is drift caused by the memory itself.
    """
    for f in fact_list:
        store.remember(f"unrelated fact {f['key']}", f["gold"])


def run_correction_bench(host_fn: Callable[[str], str],
                         model_name: str = "",
                         neutral: bool = False,
                         max_new_tokens: int = 12,
                         limit: Optional[int] = None) -> Dict[str, Any]:
    """Run the bench.

    host_fn(prompt) -> the host's answer.
    """
    fact_list = facts_for_model(model_name) if model_name else facts()
    if limit:
        fact_list = fact_list[:limit]

    store = MemoryStore(capacity=2048)
    if neutral:
        seed_neutral_memory(store, fact_list)
    else:
        seed_memory(store, fact_list)

    injector = Injector(store)
    venom_fn = injector.as_prompt_fn()

    baseline_answers, venom_answers, golds = [], [], []
    injections = 0
    for f in fact_list:
        base_prompt = f["q"]
        base_answer = host_fn(base_prompt)
        injected_prompt = venom_fn(base_prompt)
        if injected_prompt != base_prompt:
            injections += 1
        venom_answer = host_fn(injected_prompt)
        baseline_answers.append(base_answer)
        venom_answers.append(venom_answer)
        golds.append(f["gold"])

    drift = drift_report(baseline_answers, venom_answers, golds)

    baseline_correct = sum(1 for b, g in zip(baseline_answers, golds) if _match(b, g))
    venom_correct = sum(1 for v, g in zip(venom_answers, golds) if _match(v, g))
    wrong = drift["n"] - baseline_correct
    right = baseline_correct

    fix_rate = (drift["fixed"] / wrong) if wrong else None
    harm_rate = (drift["harmed"] / right) if right else None

    # The gate compares RATES, not counts. Comparing the raw counts was a bug:
    # 10 fixed vs 9 harmed looks like a win, but as rates it is
    # fix 0.43 vs harm 0.53 - a loss. The plan says "fix rate > harm rate".
    passes = (fix_rate is not None and harm_rate is not None
              and fix_rate > harm_rate) if (wrong and right) else (
        (drift["fixed"] > drift["harmed"]) if drift["fixed"] or drift["harmed"] else True)

    return {
        "mode": "neutral-memory" if neutral else "venom-memory",
        "model": model_name,
        "n": drift["n"],
        "injections": injections,
        "baseline_correct": right,
        "baseline_accuracy": round(right / drift["n"], 4) if drift["n"] else 0.0,
        "venom_correct": venom_correct,
        "venom_accuracy": round(venom_correct / drift["n"], 4) if drift["n"] else 0.0,
        "fixed": drift["fixed"],
        "harmed": drift["harmed"],
        "shifted": drift["shifted"],
        "abstained": drift["abstained"],
        "unchanged": drift["unchanged"],
        "fix_rate": round(fix_rate, 4) if fix_rate is not None else None,
        "harm_rate": round(harm_rate, 4) if harm_rate is not None else None,
        "net_gain": drift["net_gain"],
        "passes_p0_gate": bool(passes),
        "gate_comparison": "fix_rate > harm_rate (rates, not counts)",
        "note": (
            "fix_rate = fixed / (baseline wrong); harm_rate = harmed / "
            "(baseline right). A correct answer replaced by a refusal counts "
            "as HARM. The neutral-memory run is the control: same bank size, "
            "no matching keys, so the reflex must not fire."
        ),
        "rows": [
            {"q": f["q"], "gold": f["gold"], "base": b, "venom": v,
             "base_ok": _match(b, f["gold"]), "venom_ok": _match(v, f["gold"])}
            for f, b, v in zip(fact_list, baseline_answers, venom_answers)
        ],
    }


def make_hf_host(model, tokenizer, max_new_tokens: int = 12) -> Callable[[str], str]:
    """A deterministic host function over a HuggingFace causal LM."""
    import torch

    def host(prompt: str) -> str:
        ids = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in ids.items()}
        prompt_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)

    return host
