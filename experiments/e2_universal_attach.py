"""Real-model proof of the universal layer.

Attaches VENOM to three different architectures, runs the reflex, and proves
the host is bit-exact identical after detach. This is the file that backs the
"plug into any AI" claim - if it fails for an architecture, the layer does
not support that architecture.

    python experiments/e2_universal_attach.py --models gpt2,Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from venom.core.bus import logits_hash
from venom.core.runtime import VenomConfig, VenomRuntime
from venom.shell.attach import attach

DEFAULT_MODELS = [
    "gpt2",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
]

PROMPT = "The capital of France is"


def run_one(name: str, prompt: str, max_new_tokens: int = 24) -> dict:
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name).eval()
    ids = tok(prompt, return_tensors="pt")

    with torch.no_grad():
        before = model(**ids).logits
    hash_before = logits_hash(before)

    runtime = VenomRuntime(VenomConfig(session_id=f"exp-{name}"))
    runtime.remember("capital of france", "Paris")

    t0 = time.perf_counter()
    with attach(model, tok, runtime=runtime, name_hint=name) as host:
        injected = host.prompt_for(prompt)
        with torch.no_grad():
            during = model(**ids).logits
        fp = host.stats()["fingerprint"]
        telemetry = host.stats()
    attach_s = time.perf_counter() - t0

    with torch.no_grad():
        after = model(**ids).logits

    return {
        "model": name,
        "family": fp.get("family"),
        "model_type": fp.get("model_type"),
        "blocks_path": fp.get("blocks_path"),
        "num_layers": fp.get("num_layers"),
        "confidence": fp.get("confidence"),
        "memory_hit": injected != prompt,
        "injected_prefix": injected[:60] if injected != prompt else "",
        "bit_exact_during_attach": bool(torch.equal(before, during)),
        "bit_exact_after_detach": bool(torch.equal(before, after)),
        "hash_match": hash_before == logits_hash(after),
        "attach_seconds": round(attach_s, 3),
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "telemetry": {k: v for k, v in telemetry.items() if k != "fingerprint"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--out", default="e2_universal.json")
    args = ap.parse_args()

    results = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"--- {name} ---", flush=True)
        try:
            r = run_one(name, args.prompt)
        except Exception as exc:  # noqa: BLE001
            import traceback
            r = {"model": name, "error": str(exc),
                 "traceback": traceback.format_exc()[-1500:]}
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "telemetry"}, indent=2),
              flush=True)

    ok = [r for r in results if r.get("bit_exact_after_detach")]
    summary = {
        "models_tested": len(results),
        "models_bit_exact_after_detach": len(ok),
        "all_bit_exact": len(ok) == len(results) and len(results) > 0,
        "families": sorted({r.get("family") for r in results if r.get("family")}),
        "results": results,
    }
    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return 0 if summary["all_bit_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
