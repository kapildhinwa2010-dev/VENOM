"""E7: the automated surgery pipeline on a small model.

One call runs every feature, walks the decision ladder, applies the verdicts
and verifies the result on a held-out set.

    python experiments/e7_auto_surgery.py --model gpt2
"""
from __future__ import annotations

import argparse
import json

from transformers import AutoModelForCausalLM, AutoTokenizer

from venom.surgeon.pipeline import auto_surgery

PLAN = [
    "The capital of France is", "Hello world", "In 1969 humans landed on",
    "The largest ocean is", "Photosynthesis converts light into",
    "My favourite colour is", "Water boils at", "Shakespeare wrote",
    "In the morning I drink", "The second planet from the sun is",
    "The tallest mountain on Earth is", "Water is made of",
]
HELD = [
    "Rome is the capital of", "The Red Sea is in", "Venus is the",
    "Hamlet was written by", "Mount Everest is in",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--mode", default="zero", choices=["zero", "remove"])
    ap.add_argument("--drift-threshold", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="e7_auto_surgery.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).eval()

    out = auto_surgery(model, tok, plan_prompts=PLAN, heldout_prompts=HELD,
                       mode=args.mode, drift_threshold=args.drift_threshold,
                       apply=not args.dry_run, label=args.model)

    print(f"\n{'=' * 72}\nAUTOMATED SURGERY PIPELINE - {args.model}\n{'=' * 72}")
    print(f"  census          : {out.get('census')}")
    print(f"  redundancy      : near_duplicate={out.get('redundancy', {}).get('near_duplicate')}"
          f" similar={out.get('redundancy', {}).get('similar')}")
    print(f"  compressibility : {out.get('compressibility', {}).get('compressible')}"
          f" of {out.get('compressibility', {}).get('analyzed')} layers")
    tt = out.get("twin_tower", {})
    print(f"  twin towers     : {len(tt.get('clusters', []))} cluster(s), "
          f"max pair cos={tt.get('max_pair_similarity')}")
    sr = out.get("skip_run", {})
    print(f"  skip-run (L0)   : max_delta={sr.get('max_abs_logit_delta')} "
          f"skippable={sr.get('skippable')}")
    pl = out.get("powerlaw", {})
    print(f"  power-law       : {pl.get('layers_analyzed')} layers analysed")
    print()
    print(out.get("ladder", {}).get("text", ""))
    app = out.get("applied", {})
    print(f"\n  applied         : ok={app.get('ok')} {app.get('note', '')}")
    if app.get("drift"):
        print(f"  drift           : {app['drift']}")
    if app.get("verdict"):
        print(f"  verdict         : {app['verdict']}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
