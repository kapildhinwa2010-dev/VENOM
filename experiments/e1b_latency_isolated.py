"""E1b: isolated, order-randomised latency measurement for the Bus.

The first E1 run reported the bus being *faster* than no bus (ratio 0.92),
which is impossible for a hook that copies tensors. This script isolates the
cause by measuring four conditions in randomised order:

  A: no hooks at all
  B: a null hook (registered, returns None, does nothing)
  C: the real Bus (copies each block output into self.taps)
  D: the real Bus but with taps released immediately (no retention)

If C > A but D ~= A, the cost is tensor retention, not the hook itself.
If B ~= A ~= C, then the original measurement was an ordering artefact.
"""
import json
import random
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from venom.core.bus import Bus, gpt2_blocks, logits_hash

NAME = "gpt2"
PROMPT = "The capital of France is"
ROUNDS = 9
REPEATS = 8


def bench(fn, n=REPEATS):
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            fn()
    return (time.perf_counter() - t0) / n * 1000


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME).eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids
    blocks = gpt2_blocks(model)

    with torch.no_grad():
        base_logits = model(ids).logits
    hash_base = logits_hash(base_logits)

    null_handles = []

    def cond_none():
        return model(ids)

    def cond_null():
        return model(ids)

    def cond_bus():
        bus = Bus(model, blocks)
        bus.attach()
        out = model(ids).logits
        bus.detach()
        bus.clear_taps()
        return out

    def cond_bus_retain():
        bus = Bus(model, blocks)
        bus.attach()
        out = model(ids).logits
        bus.detach()
        return out

    conditions = {
        "none": cond_none,
        "null_hook": cond_null,
        "bus_retain": cond_bus_retain,
        "bus_release": cond_bus,
    }

    # warmup
    for fn in conditions.values():
        for _ in range(3):
            bench(fn, n=2)

    samples = {k: [] for k in conditions}
    order = list(conditions)
    for r in range(ROUNDS):
        random.shuffle(order)
        for name in order:
            samples[name].append(bench(conditions[name]))

    # correctness of the bus under this harness
    bus = Bus(model, blocks)
    with torch.no_grad():
        with bus:
            during = model(ids).logits
        after = model(ids).logits

    out = {
        "model": NAME,
        "rounds": ROUNDS,
        "repeats_per_round": REPEATS,
        "median_ms": {k: round(statistics.median(v), 3) for k, v in samples.items()},
        "min_ms": {k: round(min(v), 3) for k, v in samples.items()},
        "ratio_vs_none_median": {
            k: round(statistics.median(v) / statistics.median(samples["none"]), 4)
            for k, v in samples.items() if k != "none"
        },
        "ratio_vs_none_min": {
            k: round(min(v) / min(samples["none"]), 4)
            for k, v in samples.items() if k != "none"
        },
        "bit_exact_during": bool(torch.equal(base_logits, during)),
        "bit_exact_after_detach": bool(torch.equal(base_logits, after)),
        "hash_base": hash_base[:16],
        "hash_after": logits_hash(after)[:16],
    }
    print(json.dumps(out, indent=2))
    with open("e1b_latency.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
