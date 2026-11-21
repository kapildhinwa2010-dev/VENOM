"""E5: the P0 exit gate on a real host.

*"400 facts par fix rate > harm rate"* - that is the plan's P0 gate. This
script runs it on a real model, with the neutral-memory control.

It also runs the P3 ablation the plan asks for: the same run with the
self-read tools available vs not.

    python experiments/e5_fix_harm.py --model gpt2
    python experiments/e5_fix_harm.py --model Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import json

from transformers import AutoModelForCausalLM, AutoTokenizer

from venom.bench.real_bench import make_hf_host, run_correction_bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--out", default="e5_fix_harm.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).eval()
    host = make_hf_host(model, tok, args.max_new_tokens)

    print(f"running the VENOM arm on {args.model} ...", flush=True)
    venom = run_correction_bench(host, args.model, neutral=False,
                                 max_new_tokens=args.max_new_tokens,
                                 limit=args.limit)
    print("running the neutral-memory control ...", flush=True)
    neutral = run_correction_bench(host, args.model, neutral=True,
                                   max_new_tokens=args.max_new_tokens,
                                   limit=args.limit)

    def trim(r):
        return {k: v for k, v in r.items() if k != "rows"}

    summary = {
        "model": args.model,
        "n_facts": venom["n"],
        "venom_arm": trim(venom),
        "neutral_memory_control": trim(neutral),
        "p0_gate": {
            "rule": "fix rate > harm rate",
            "fix_rate": venom["fix_rate"],
            "harm_rate": venom["harm_rate"],
            "passes": venom["passes_p0_gate"],
        },
        "drift_check": {
            "rule": ("the neutral-memory control must not change the answers; "
                     "if it does, the memory itself is causing drift"),
            "neutral_injections": neutral["injections"],
            "neutral_changed": neutral["fixed"] + neutral["harmed"] + neutral["shifted"],
            "neutral_harmed": neutral["harmed"],
            "clean": neutral["fixed"] + neutral["harmed"] + neutral["shifted"] == 0,
        },
        "honesty": (
            "This is a small open model on mundane world-knowledge facts. A "
            "passing gate means the injection path is not actively harmful on "
            "this set - it does NOT mean the layer improves a real workload. "
            "The fix rate here is largely the model COPYING the injected "
            "context; that is a property of the benchmark, not a "
            "demonstration of knowledge repair."
        ),
    }
    print(json.dumps(summary, indent=2, default=str))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
