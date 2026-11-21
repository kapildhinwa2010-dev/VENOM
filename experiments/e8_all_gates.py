"""E8: the plan's remaining exit gates, run in one place.

    python experiments/e8_all_gates.py --gates recall,abstain,latency
    python experiments/e8_all_gates.py --gates facts --model Qwen/Qwen2.5-0.5B-Instruct --limit 400
    python experiments/e8_all_gates.py --gates multimodel

The gates, and what each one actually measures:

  facts       400 facts, fix rate > harm rate (the P0 gate)
  multimodel  the same gate across 5+ models
  latency     the layer's own overhead must stay under 15%
  recall      the tool path must return the right fact >= 60% of the time
  abstain     under memory corruption the store must abstain >= 95% of the time
  agnostic    the content-agnostic control (already measured in E5)

Everything is reported with the denominator, because a rate without one is
not a result.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

MODELS = [
    "gpt2",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/pythia-70m",
    "facebook/opt-125m",
    "Qwen/Qwen2.5-0.5B-Instruct",
]


def _load(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name).eval()
    return model, tok


def _p(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- gate: facts
def gate_facts(model_name: str, limit: int, out: Dict[str, Any]) -> Dict[str, Any]:
    from venom.bench.real_bench import make_hf_host, run_correction_bench

    model, tok = _load(model_name)
    host = make_hf_host(model, tok)
    res = run_correction_bench(host, model_name=model_name, limit=limit)
    neutral = run_correction_bench(host, model_name=model_name, limit=limit,
                                   neutral=True)
    _p(f"  [{model_name}] n={res['n']} injections={res['injections']}")
    _p(f"    baseline acc {res['baseline_accuracy']} -> venom acc {res['venom_accuracy']}")
    _p(f"    fix {res['fix_rate']}  harm {res['harm_rate']}  "
       f"fixed={res['fixed']} harmed={res['harmed']}  gate={res['passes_p0_gate']}")
    _p(f"    control (neutral memory): injections={neutral['injections']} "
       f"fixed={neutral['fixed']} harmed={neutral['harmed']}")
    return {
        "model": model_name,
        "limit": limit,
        "n": res["n"],
        "injections": res["injections"],
        "baseline_accuracy": res["baseline_accuracy"],
        "venom_accuracy": res["venom_accuracy"],
        "fix_rate": res["fix_rate"],
        "harm_rate": res["harm_rate"],
        "fixed": res["fixed"],
        "harmed": res["harmed"],
        "passes_p0_gate": res["passes_p0_gate"],
        "control": {"injections": neutral["injections"],
                    "fixed": neutral["fixed"],
                    "harmed": neutral["harmed"]},
        "gate": "fix_rate > harm_rate",
    }


# ------------------------------------------------------------ gate: multimodel
def gate_multimodel(limit: int, out: Dict[str, Any]) -> Dict[str, Any]:
    from venom.bench.real_bench import make_hf_host, run_correction_bench

    rows = []
    for name in MODELS:
        try:
            model, tok = _load(name)
        except Exception as exc:                          # noqa: BLE001
            _p(f"  [{name}] SKIPPED: {str(exc)[:90]}")
            rows.append({"model": name, "error": str(exc)[:200]})
            continue
        host = make_hf_host(model, tok)
        res = run_correction_bench(host, model_name=name, limit=limit)
        _p(f"  [{name}] n={res['n']} fix={res['fix_rate']} harm={res['harm_rate']} "
           f"gate={res['passes_p0_gate']} base_acc={res['baseline_accuracy']}")
        rows.append({
            "model": name, "n": res["n"], "injections": res["injections"],
            "baseline_accuracy": res["baseline_accuracy"],
            "venom_accuracy": res["venom_accuracy"],
            "fix_rate": res["fix_rate"], "harm_rate": res["harm_rate"],
            "fixed": res["fixed"], "harmed": res["harmed"],
            "passes_p0_gate": res["passes_p0_gate"],
        })
        del model, tok, host
    passed = [r for r in rows if r.get("passes_p0_gate")]
    _p(f"  models run: {len(rows)}, passing the gate: {len(passed)}")
    return {"models": rows, "models_run": len(rows), "models_passed": len(passed),
            "gate": "fix rate > harm rate on 5+ models"}


# -------------------------------------------------------------- gate: latency
def gate_latency(model_name: str, trials: int, out: Dict[str, Any]) -> Dict[str, Any]:
    """The layer's own overhead.

    Two numbers are reported and they are NOT the same thing:

    * `hook_overhead_pct`  - the layer attached, same prompt, same token
                             count. This is the cost of the layer itself.
    * `reflex_eval_pct`    - the time the trigger scan + injection costs,
                             as a fraction of one generation.
    * `end_to_end_pct`     - attached + injected prompt vs the bare prompt.
                             This includes generating MORE tokens because
                             the injected context is longer, which is a
                             cost of the extra context, not overhead.

    The gate is on the layer's own overhead: hooks + reflex evaluation.
    """
    import torch

    from venom.core.runtime import VenomConfig, VenomRuntime
    from venom.memory.store import MemoryStore
    from venom.reflex.injector import Injector
    from venom.shell.attach import attach

    model, tok = _load(model_name)
    prompt = "The capital of France is"

    def gen_once(max_new_tokens: int = 16) -> None:
        """A REAL generation, not a single forward pass. A single pass over a
        5-token prompt is dominated by dispatch overhead, so the layer's
        cost would be lost in the noise."""
        enc = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tok.pad_token_id)

    for _ in range(2):
        gen_once()

    # --- A/B, alternating order so a machine that is warming up or slowing
    # down cannot fake a result
    def attached_once(call_tool: bool) -> float:
        """The SAME generation with the layer attached.

        `call_tool=True` adds one tool call. That tool call runs its own
        forward pass, so its cost is the PRICE OF THE TOOL, not the layer's
        overhead - the two must not be added together.
        """
        runtime = VenomRuntime(VenomConfig(session_id="latency"))
        store = MemoryStore(capacity=64)
        store.remember("capital of france", "Paris")
        injector = Injector(store)
        t0 = time.perf_counter()
        with attach(model, tok, runtime=runtime) as host:
            gen_once()
            if call_tool:
                host.call_tool("venom.uncertainty", {})
        return time.perf_counter() - t0

    base_times, hook_times, tool_times = [], [], []
    # ABBA ordering: each round runs base, hook, hook, base (and the mirror
    # in odd rounds), so any monotone drift cancels out of the comparison.
    for i in range(trials):
        def timed_base():
            t0 = time.perf_counter(); gen_once()
            return time.perf_counter() - t0

        if i % 2 == 0:
            base_times.append(timed_base())
            hook_times.append(attached_once(False))
            tool_times.append(attached_once(True))
            base_times.append(timed_base())
        else:
            tool_times.append(attached_once(True))
            hook_times.append(attached_once(False))
            base_times.append(timed_base())
            hook_times.append(attached_once(False))
            base_times.append(timed_base())

    # --- reflex evaluation cost on its own
    store = MemoryStore(capacity=64)
    for i in range(64):
        store.remember(f"key number {i}", f"value {i}")
    injector = Injector(store)
    t0 = time.perf_counter()
    for _ in range(200):
        injector.maybe_inject("what is the capital of france?", auto_recall=True)
    reflex_per_call = (time.perf_counter() - t0) / 200

    def median(xs):
        xs = sorted(xs)
        n = len(xs)
        return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2

    def spread(xs):
        xs = sorted(xs)
        return (round(xs[len(xs) // 4] * 1000, 2), round(xs[-1 - len(xs) // 4] * 1000, 2))

    mean_base = median(base_times)
    mean_hook = median(hook_times)
    mean_tool = median(tool_times)
    hook_pct = (mean_hook - mean_base) / mean_base * 100
    tool_pct = (mean_tool - mean_hook) / mean_hook * 100
    reflex_pct = reflex_per_call / mean_base * 100
    total_pct = hook_pct + reflex_pct

    _p(f"  base {mean_base*1000:.1f} ms | attached {mean_hook*1000:.1f} ms "
       f"(hook {hook_pct:+.2f}%) | +1 tool call {mean_tool*1000:.1f} ms "
       f"(tool {tool_pct:+.2f}%) | reflex eval {reflex_pct:.3f}%")
    _p(f"  LAYER OVERHEAD (hooks + reflex) = {total_pct:+.2f}%  "
       f"gate <= 15% -> {'PASS' if total_pct <= 15 else 'FAIL'}")
    return {
        "model": model_name,
        "trials": trials,
        "median_base_ms": round(mean_base * 1000, 3),
        "median_attached_ms": round(mean_hook * 1000, 3),
        "median_with_tool_ms": round(mean_tool * 1000, 3),
        "hook_overhead_pct": round(hook_pct, 3),
        "tool_call_cost_pct": round(tool_pct, 3),
        "reflex_eval_ms": round(reflex_per_call * 1000, 5),
        "reflex_eval_pct": round(reflex_pct, 4),
        "layer_overhead_pct": round(total_pct, 3),
        "base_ms_iqr": spread(base_times),
        "attached_ms_iqr": spread(hook_times),
        "gate": "layer overhead (hooks + reflex) <= 15%",
        "passes": bool(total_pct <= 15.0),
        "note": ("the tool-call arm is reported SEPARATELY: a tool call runs "
                 "its own forward pass, so its cost is the price of the tool, "
                 "not the layer's overhead. Medians are used because a "
                 "shared CPU makes the mean noisy"),
    }


# -------------------------------------------------------------- gate: recall
def gate_recall(limit: int, out: Dict[str, Any]) -> Dict[str, Any]:
    """The tool path must return the right fact often enough to be useful.

    Two levels:
      exact  - recall by the stored key. The tool's own accuracy.
      fuzzy  - the question as the user would type it, so the trigger has to
               match it to the right entry. This is the level that matters,
               and the one the 60% gate is judged on.
    """
    from venom.bench.facts import facts
    from venom.memory.store import MemoryStore
    from venom.reflex.injector import Injector

    fact_list = facts()[:limit]
    store = MemoryStore(capacity=4096)
    for f in fact_list:
        store.remember(f["key"], f["gold"])

    exact_hits = 0
    for f in fact_list:
        r = store.recall(f["key"])
        if r.status == "hit" and r.value == f["gold"]:
            exact_hits += 1

    injector = Injector(store)
    fuzzy_hits = 0
    injected_any = 0
    for f in fact_list:
        out_dict = injector.maybe_inject(f["q"], auto_recall=True)
        if out_dict.get("changed"):
            injected_any += 1
            # `recalls` is a list of dicts: {status, key, value | reason}
            values = [str(r.get("value", "")) for r in out_dict.get("recalls", [])
                      if r.get("status") == "hit"]
            text = " ".join(values)
            if f["gold"].lower() in text.lower():
                fuzzy_hits += 1

    n = len(fact_list)
    exact_rate = exact_hits / n
    fuzzy_rate = fuzzy_hits / n
    _p(f"  n={n} exact {exact_hits}/{n} ({exact_rate:.1%})  "
       f"fuzzy {fuzzy_hits}/{n} ({fuzzy_rate:.1%})  injected {injected_any}")
    return {
        "n": n,
        "exact_hits": exact_hits, "exact_rate": round(exact_rate, 4),
        "fuzzy_hits": fuzzy_hits, "fuzzy_rate": round(fuzzy_rate, 4),
        "injections": injected_any,
        "gate": "tool-path recall >= 60%",
        "passes": bool(fuzzy_rate >= 0.60),
    }


# ------------------------------------------------------------- gate: abstain
def gate_abstain(limit: int, out: Dict[str, Any]) -> Dict[str, Any]:
    """Under memory corruption the store must ABSTAIN, not return a plausible
    guess. Corrupted = the value was changed without the digest being
    recomputed, which is exactly what a hand-edit or a tamper looks like."""
    import tempfile

    from venom.bench.facts import facts
    from venom.memory.store import MemoryStore

    fact_list = facts()[:limit]
    tmp = tempfile.mkdtemp(prefix="venom-abstain-")
    path = os.path.join(tmp, "memory.json")

    store = MemoryStore(capacity=4096, persist_path=path)
    for f in fact_list:
        store.remember(f["key"], f["gold"])

    # corrupt every entry's value without touching the digest
    corrupted = 0
    for f in fact_list:
        try:
            store.corrupt_for_test(f["key"], "TAMPERED VALUE")
            corrupted += 1
        except KeyError:
            pass

    hits, abstains = 0, 0
    reasons: Dict[str, int] = {}
    for f in fact_list:
        r = store.recall(f["key"])
        if r.status == "hit":
            hits += 1
        else:
            abstains += 1
            reasons[r.reason or "unknown"] = reasons.get(r.reason or "unknown", 0) + 1

    n = len(fact_list)
    abstain_rate = abstains / n

    # also check the persistence boundary: a hand-edited FILE must not load
    # as a valid hit
    file_abstain = None
    try:
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        for entry in blob.get("memory", []):
            entry["value"] = "HAND EDITED"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(blob, fh)
        store2 = MemoryStore(capacity=4096, persist_path=path)
        file_abstain = sum(1 for f in fact_list
                           if store2.recall(f["key"]).status != "hit")
        file_abstain_rate = file_abstain / n
    except Exception as exc:                              # noqa: BLE001
        file_abstain_rate = None
        _p(f"  (persistence check skipped: {str(exc)[:80]})")

    _p(f"  n={n} corrupted={corrupted} abstained={abstains} ({abstain_rate:.1%}) "
       f"leaked={hits} reasons={reasons}")
    if file_abstain_rate is not None:
        _p(f"  hand-edited FILE: abstained {file_abstain}/{n} ({file_abstain_rate:.1%})")
    return {
        "n": n, "corrupted": corrupted,
        "abstained": abstains, "abstain_rate": round(abstain_rate, 4),
        "leaked_hits": hits, "reasons": reasons,
        "file_abstain_rate": file_abstain_rate,
        "gate": "abstain rate >= 95% under corruption",
        "passes": bool(abstain_rate >= 0.95),
    }


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", default="recall,abstain,latency",
                    help="comma separated: facts,multimodel,latency,recall,abstain")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--out", default="e8_gates.json")
    args = ap.parse_args()

    gates = [g.strip() for g in args.gates.split(",") if g.strip()]
    report: Dict[str, Any] = {"gates": {}}
    if os.path.exists(args.out):
        try:
            with open(args.out, encoding="utf-8") as fh:
                report = json.load(fh)
            report.setdefault("gates", {})
        except Exception:                                   # noqa: BLE001
            report = {"gates": {}}

    for gate in gates:
        _p(f"\n=== GATE: {gate} ===")
        t0 = time.perf_counter()
        try:
            if gate == "facts":
                res = gate_facts(args.model, args.limit, report)
            elif gate == "multimodel":
                res = gate_multimodel(args.limit, report)
            elif gate == "latency":
                res = gate_latency(args.model, args.trials, report)
            elif gate == "recall":
                res = gate_recall(args.limit, report)
            elif gate == "abstain":
                res = gate_abstain(args.limit, report)
            else:
                _p(f"  unknown gate {gate}")
                continue
        except Exception as exc:                           # noqa: BLE001
            import traceback
            traceback.print_exc()
            res = {"error": str(exc)[:400]}
        res["seconds"] = round(time.perf_counter() - t0, 2)
        report["gates"][gate] = res

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    _p("\n" + "=" * 72)
    _p("GATE SUMMARY")
    _p("=" * 72)
    for gate, res in report["gates"].items():
        status = res.get("passes")
        if status is None and "passes_p0_gate" in res:
            status = res["passes_p0_gate"]
        mark = "PASS" if status else ("FAIL" if status is not None else "-")
        _p(f"  [{mark}] {gate:12s} {res.get('gate', '')}")
        for k in ("n", "fix_rate", "harm_rate", "fuzzy_rate", "exact_rate",
                  "abstain_rate", "layer_overhead_pct", "models_run",
                  "models_passed"):
            if k in res:
                _p(f"          {k}: {res[k]}")
    _p(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
