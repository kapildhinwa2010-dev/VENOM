"""E1: does attaching the bus change the model at all? Plus latency cost.

Requires torch + transformers. Run on a machine that can download gpt2 (124M).

    pip install torch transformers
    python experiments/e1_bus_detach.py

Trust contract: after detach the logits must be bit-exact identical to the
baseline. If `bit_exact_after_detach` is False, stop - something writes.
"""
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from venom.core.bus import Bus, gpt2_blocks, logits_hash

NAME = "gpt2"
PROMPT = "The capital of France is"
N = 20


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME).eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids

    with torch.no_grad():
        before = model(ids).logits
    hash_before = logits_hash(before)

    bus = Bus(model, gpt2_blocks(model))
    with torch.no_grad():
        with bus:
            during = model(ids).logits
            taps = {k: list(v.shape) for k, v in bus.taps.items()}
            calls = bus.calls
        after = model(ids).logits

    def bench(fn, n=N):
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n):
                fn()
        return (time.perf_counter() - t0) / n * 1000

    baseline_ms = bench(lambda: model(ids))
    b = Bus(model, gpt2_blocks(model)); b.attach()
    bus_ms = bench(lambda: model(ids))
    b.detach()

    out = {
        "model": NAME,
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "blocks_tapped": len(taps),
        "hook_calls": calls,
        "bit_exact_during": bool(torch.equal(before, during)),
        "bit_exact_after_detach": bool(torch.equal(before, after)),
        "hash_before": hash_before[:16],
        "hash_after": logits_hash(after)[:16],
        "baseline_ms": round(baseline_ms, 3),
        "bus_ms": round(bus_ms, 3),
        "latency_ratio": round(bus_ms / baseline_ms, 4),
        "bus_detached_cleanly": not b.attached,
    }
    print(json.dumps(out, indent=2))
    with open("e1_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
