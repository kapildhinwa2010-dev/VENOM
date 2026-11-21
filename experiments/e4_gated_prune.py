"""E4: gated self-surgery with a real drift proof.

The plan's P5 exit gate: *"10k prompts on, drift = 0, warna prune radd"* —
prune only if the drift is zero on a held-out set, otherwise cancel.

This script does that on a real model:

  1. capture activations over a **PLAN set** of prompts,
  2. find the units that never fired,
  3. build a prune plan from them,
  4. apply the prune,
  5. measure the drift on a **HELD-OUT set** of different prompts,
  6. keep the prune only if the drift is within the threshold, otherwise
     restore the weights.

The plan set and the held-out set are disjoint on purpose: verifying a prune
on the same prompts that justified it proves nothing.

    python experiments/e4_gated_prune.py --model gpt2
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from venom.core.dials import logit_drift
from venom.surgeon.prune import PruneManager, PrunePlan

PLAN_PROMPTS = [
    "The capital of France is",
    "The capital of Italy is",
    "The capital of Spain is",
    "The capital of Japan is",
    "The capital of Brazil is",
    "The capital of Canada is",
    "The capital of Germany is",
    "The capital of Egypt is",
    "The capital of Kenya is",
    "The capital of Peru is",
]

HELDOUT_PROMPTS = [
    "The largest ocean on Earth is",
    "The chemical symbol for gold is",
    "The author of Hamlet wrote",
    "The tallest mountain in the world is",
    "The planet closest to the Sun is",
    "The inventor of the telephone was",
    "The largest mammal on Earth is",
    "The currency of Japan is",
    "The number of continents on Earth is",
    "The first human to walk on the Moon was",
]


def _logits(model, tok, prompts):
    out = []
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt")
            out.append(model(**ids).logits.detach().clone())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--threshold", type=float, default=0.001)
    ap.add_argument("--drift-threshold", type=float, default=0.01)
    ap.add_argument("--max-units", type=int, default=16)
    ap.add_argument("--out", default="e4_gated_prune.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).eval()

    baseline = _logits(model, tok, HELDOUT_PROMPTS)

    # --- 1. observe activations over the PLAN set ---
    counts = {}
    samples = 0
    with torch.no_grad():
        for p in PLAN_PROMPTS:
            ids = tok(p, return_tensors="pt")
            model(**ids)
            samples += 1

    # per-output-unit activation counts for every Linear in the model
    unit_counts = {}
    handles = []

    def make_hook(path):
        def fn(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            if out.dim() < 2:
                return None
            flat = out.reshape(-1, out.shape[-1])
            nonzero = (flat != 0).any(dim=0)
            unit_counts.setdefault(path, torch.zeros(out.shape[-1]))
            unit_counts[path] += nonzero.float().cpu()
        return fn

    from venom.surgeon.prune import _prunable_axes
    for path, module in model.named_modules():
        if _prunable_axes(module) is not None:
            handles.append(module.register_forward_hook(make_hook(path)))
    with torch.no_grad():
        for p in PLAN_PROMPTS:
            model(**tok(p, return_tensors="pt"))
    for h in handles:
        h.remove()

    # --- 2. build the plan ---
    pm = PruneManager(model, drift_threshold=args.drift_threshold,
                      max_units=args.max_units)
    plan_info = pm.plan({k: v.tolist() for k, v in unit_counts.items()},
                         total_samples=samples, threshold=args.threshold)

    # how rare are the rarest units? this is the informative part even when
    # nothing is prunable.
    rarity = []
    for path, counts in unit_counts.items():
        rates = sorted((c / samples) for c in counts.tolist())
        rarity.append({
            "path": path,
            "rarest_rate": round(rates[0], 6),
            "p10_rate": round(rates[max(0, len(rates) // 10)], 6),
            "median_rate": round(rates[len(rates) // 2], 6),
            "units": len(rates),
        })
    rarity.sort(key=lambda r: r["rarest_rate"])

    plans = plan_info.get("plans", [])
    if not plans:
        summary = {
            "model": args.model,
            "result": "no_prunable_units",
            "plan_set_size": len(PLAN_PROMPTS),
            "heldout_set_size": len(HELDOUT_PROMPTS),
            "threshold": args.threshold,
            "rarest_units": rarity[:8],
            "interpretation": (
                "No unit stayed below the activation threshold over the plan "
                "set. This is a finding, not a failure: a small model with "
                "dense activations has no dead units to prune at this sample "
                "count. The plan's P5 gate assumes 10k prompts; 10 prompts "
                "cannot establish that a unit is dead."
            ),
            "weights_modified": False,
        }
        print(json.dumps(summary, indent=2, default=str))
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        return 0

    # pick the largest plan
    biggest = max(plans, key=lambda p: p["units"])
    plan = PrunePlan(
        target_path=biggest["target_path"],
        layer_name=biggest["layer_name"],
        unit_indices=[],
        total_units=biggest["total_units"],
        fraction=biggest["fraction"],
        reason=biggest["reason"],
    )
    # recover the actual indices from the counts
    counts_for_target = unit_counts[biggest["target_path"]].tolist()
    plan.unit_indices = [i for i, c in enumerate(counts_for_target)
                         if (c / samples) < args.threshold][: args.max_units]

    # --- 3. apply + verify on the HELD-OUT set ---
    def verifier():
        current = _logits(model, tok, HELDOUT_PROMPTS)
        drifts = [logit_drift(b, c) for b, c in zip(baseline, current)]
        return {"drift": max(drifts) if drifts else 0.0,
                "mean_drift": sum(drifts) / len(drifts) if drifts else 0.0}

    result = pm.prune_with_gate(plan, verifier)
    final = verifier()

    weights_restored = pm.is_pristine()

    summary = {
        "model": args.model,
        "plan_set_size": len(PLAN_PROMPTS),
        "heldout_set_size": len(HELDOUT_PROMPTS),
        "target": biggest["target_path"],
        "units_pruned": len(plan.unit_indices),
        "total_units": plan.total_units,
        "fraction_pruned": round(len(plan.unit_indices) / max(1, plan.total_units), 6),
        "drift_threshold": args.drift_threshold,
        "gate": result,
        "heldout_drift_after": final,
        "weights_pristine_after": weights_restored,
        "note": (
            "The plan set and the held-out set are disjoint. A drift within the "
            "threshold is evidence, not proof of losslessness. The plan's P5 "
            "gate asks for drift == 0 over 10k prompts; this run uses 10 "
            "held-out prompts, which is far short of that bar."
        ),
    }
    print(json.dumps(summary, indent=2, default=str))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
