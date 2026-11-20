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

    # Interleaved A/B re-measurement. Measuring all of A then all of B lets
    # CPU frequency scaling bias the result - the first version of this
    # script reported a ratio of 0.92 (bus "faster" than no bus), which is
    # impossible. Alternating the two conditions cancels the drift.
    base_samples, bus_samples = [], []
    for _ in range(5):
        base_samples.append(bench(lambda: model(ids)))
        b.attach()
        bus_samples.append(bench(lambda: model(ids)))
        b.detach()
    base_ms = min(base_samples)
    bus_ms = min(bus_samples)

    # v0.1.2: release() detaches AND frees the captured activations. Without
    # it, the taps keep every layer's full activation on the host device - for
    # a 70B model with a 4k context that is tens of GB.
    b.detach()
    tap_bytes = sum(t.numel() * t.element_size() for t in b.taps.values())
    b.release()

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
        "tap_bytes_retained": tap_bytes,
        "tap_bytes_after_release": sum(t.numel() * t.element_size() for t in b.taps.values()),
    }
    print(json.dumps(out, indent=2))
    with open("e1_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
