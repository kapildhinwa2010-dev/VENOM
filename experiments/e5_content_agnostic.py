"""E5: the content-agnostic control. The experiment the project rests on.

The question: when VENOM's self-read tools report "something is off", are they
reporting on the model's **internal state**, or just detecting that *an
anomaly happened anywhere*?

Published work says the latter is the more likely explanation:
  * models cannot reliably distinguish activation-level interventions from
    input-level ones (arXiv 2605.26242);
  * the detection is content-agnostic (arXiv 2603.05414);
  * a supervised probe beats the model's own self-report 98.3% vs 11.1%
    (arXiv 2611.15267).

The design is a MATCHED control. The SAME vector v, with the SAME magnitude,
is injected at two different sites:

    A. input-level      -> added to the embedding output
    B. activation-level -> added to the residual stream at layer L

If the self-read signals move by the same amount in both cases, the tools are
an anomaly detector, not a self-monitor. If B moves them much more than A, the
tools are reading something specifically about the computation.

A second, sharper test: **localisation**. Inject at layer L and check WHERE
the per-layer divergence spikes. A tool that can name the perturbed layer is
doing something an anomaly detector cannot do.

    python experiments/e5_content_agnostic.py --model gpt2
"""
from __future__ import annotations

import argparse
import json
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from venom.core.discovery import fingerprint_universal
from venom.sensors.consensus import consensus
from venom.sensors.uncertainty import token_uncertainty


def _embed_path(model):
    """The module whose output is the token embeddings."""
    for path, module in model.named_modules():
        if path.endswith("embed_tokens") or path.endswith("wte") or path.endswith("embed_in"):
            return path, module
    return None, None


def _block_path(model, layer):
    fp = fingerprint_universal(model)
    path = fp.get("blocks_path")
    if not path:
        return None, None
    obj = model
    for part in path.split("."):
        obj = getattr(obj, part)
    return path, obj[layer]


def _norm(v):
    return float(v.norm().item())


def measure(model, tok, prompt, inject=None):
    """One forward pass. `inject` is ("embed"|"block", layer, vector) or None."""
    ids = tok(prompt, return_tensors="pt")
    handles = []

    def add_v_hook(module, inputs, output):
        v = inject[2].to(output.dtype)
        if isinstance(output, tuple):
            return (output[0] + v,) + tuple(output[1:])
        return output + v

    if inject is not None:
        kind, layer, vec = inject
        if kind == "embed":
            _, mod = _embed_path(model)
            handles.append(mod.register_forward_hook(add_v_hook))
        else:
            _, mod = _block_path(model, layer)
            handles.append(mod.register_forward_hook(add_v_hook))

    with torch.no_grad():
        out = model(**ids)
    for h in handles:
        h.remove()

    logits = out.logits[0, -1, :].float()
    u = token_uncertainty(logits.tolist())

    # per-layer divergence is measured separately (needs the taps)
    return {"uncertainty": u, "logits": logits}


def layer_divergence(model, tok, prompt, baseline_logits, inject=None):
    """Per-layer KL-style divergence between this run and the baseline."""
    from venom.core.bus import Bus
    from venom.core.discovery import get_blocks

    fp = fingerprint_universal(model)
    blocks = get_blocks(model, fp)
    bus = Bus(model, blocks)
    handles = []

    def add_v_hook(module, inputs, output):
        v = inject[2].to(output.dtype)
        if isinstance(output, tuple):
            return (output[0] + v,) + tuple(output[1:])
        return output + v

    if inject is not None:
        kind, layer, vec = inject
        if kind == "embed":
            _, mod = _embed_path(model)
        else:
            _, mod = _block_path(model, layer)
        handles.append(mod.register_forward_hook(add_v_hook))

    ids = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        with bus:
            model(**ids)
    for h in handles:
        h.remove()

    # a simple, honest divergence: cosine distance between the baseline and
    # the current final hidden state, per layer
    import torch.nn.functional as F
    base = baseline_logits
    rows = []
    for name in sorted(bus.taps, key=lambda n: int(n.rsplit("_", 1)[-1])):
        h = bus.taps[name]
        if h.dim() == 3:
            h = h[0, -1, :]
        else:
            h = h[-1, :]
        rows.append({
            "layer": int(name.rsplit("_", 1)[-1]),
            "cosine_to_baseline": None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--scale", type=float, default=0.5,
                    help="perturbation size as a fraction of the hidden-state norm")
    ap.add_argument("--out", default="e5_content_agnostic.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).eval()
    fp = fingerprint_universal(model, args.model)

    ids = tok(args.prompt, return_tensors="pt")
    with torch.no_grad():
        base_logits = model(**ids).logits[0, -1, :].float()

    # build the perturbation vector: a fixed random direction, scaled to the
    # size of the activation it will be added to
    hidden = fp.get("hidden_size") or 768
    torch.manual_seed(0)
    direction = torch.randn(hidden)
    direction = direction / direction.norm()
    magnitude = base_logits.new_tensor(args.scale)

    conditions = {}
    conditions["baseline"] = measure(model, tok, args.prompt)
    conditions["input_level"] = measure(
        model, tok, args.prompt, inject=("embed", None, direction * magnitude))
    conditions["activation_level"] = measure(
        model, tok, args.prompt, inject=("block", args.layer, direction * magnitude))

    def delta(cond):
        b = conditions["baseline"]["uncertainty"]
        c = conditions[cond]["uncertainty"]
        return {
            "d_entropy": round(c["entropy"] - b["entropy"], 4),
            "d_margin": round(c["margin"] - b["margin"], 4),
            "d_confidence": round(c["confidence"] - b["confidence"], 4),
            "d_top1_prob": round(c["top1_prob"] - b["top1_prob"], 4),
        }

    # the localisation test: where does the residual stream diverge?
    from venom.core.bus import Bus
    from venom.core.discovery import get_blocks
    blocks = get_blocks(model, fp)

    def capture(inject):
        bus = Bus(model, blocks)
        handles = []
        def add_v(module, inputs, output):
            v = inject[2].to(output.dtype)
            if isinstance(output, tuple):
                return (output[0] + v,) + tuple(output[1:])
            return output + v
        if inject is not None:
            kind, layer, vec = inject
            _, mod = (_embed_path(model) if kind == "embed" else _block_path(model, layer))
            handles.append(mod.register_forward_hook(add_v))
        with torch.no_grad():
            with bus:
                model(**ids)
        for h in handles:
            h.remove()
        return {k: v.clone() for k, v in bus.taps.items()}

    base_taps = capture(None)
    embed_taps = capture(("embed", None, direction * magnitude))
    block_taps = capture(("block", args.layer, direction * magnitude))

    def per_layer(taps):
        rows = []
        for name in sorted(taps, key=lambda n: int(n.rsplit("_", 1)[-1])):
            a = base_taps[name]
            b = taps[name]
            if a.dim() == 3:
                a, b = a[0, -1, :], b[0, -1, :]
            else:
                a, b = a[-1, :], b[-1, :]
            cos = torch.nn.functional.cosine_similarity(
                a.float().reshape(1, -1), b.float().reshape(1, -1)).item()
            rows.append({"layer": int(name.rsplit("_", 1)[-1]),
                         "cosine_to_baseline": round(cos, 6)})
        return rows

    summary = {
        "model": args.model,
        "family": fp.get("family"),
        "num_layers": fp.get("num_layers"),
        "prompt": args.prompt,
        "injected_layer": args.layer,
        "perturbation_scale": args.scale,
        "conditions": {
            name: {k: v for k, v in c["uncertainty"].items()}
            for name, c in conditions.items()
        },
        "signal_change": {
            "input_level": delta("input_level"),
            "activation_level": delta("activation_level"),
        },
        "localisation": {
            "input_level_per_layer_cosine": per_layer(embed_taps),
            "activation_level_per_layer_cosine": per_layer(block_taps),
        },
        "interpretation": None,   # filled in below
    }

    d_in = summary["signal_change"]["input_level"]
    d_ac = summary["signal_change"]["activation_level"]

    def mag(d):
        return abs(d["d_entropy"]) + abs(d["d_margin"]) + abs(d["d_confidence"])

    ratio = mag(d_ac) / mag(d_in) if mag(d_in) > 1e-9 else float("inf")
    summary["signal_ratio_activation_over_input"] = (
        round(ratio, 3) if ratio != float("inf") else "inf")

    # where does each injection show up? Report BOTH: the first layer that
    # moved at all, and the layer that moved most. They are different
    # questions and conflating them is how you end up mislabelling a result.
    def first_diverged(rows, eps=1e-9):
        for r in rows:
            if abs(r["cosine_to_baseline"] - 1.0) > eps:
                return r["layer"]
        return None

    def most_diverged(rows):
        return min(rows, key=lambda r: r["cosine_to_baseline"])["layer"]

    rows = summary["localisation"]["activation_level_per_layer_cosine"]
    in_rows = summary["localisation"]["input_level_per_layer_cosine"]
    summary["activation_first_diverged_layer"] = first_diverged(rows)
    summary["activation_most_diverged_layer"] = most_diverged(rows)
    summary["input_first_diverged_layer"] = first_diverged(in_rows)
    summary["input_most_diverged_layer"] = most_diverged(in_rows)
    summary["layers_bit_exact_before_injection"] = (
        first_diverged(rows) if first_diverged(rows) is not None else None)
    summary["localisation_correct"] = (
        summary["activation_first_diverged_layer"] == args.layer)

    if ratio < 1.5:
        verdict = ("GENERIC: the self-read signal moves about as much for an "
                   "input-level perturbation as for an activation-level one. "
                   "That is the signature of an ANOMALY DETECTOR, not "
                   "introspection.")
    else:
        verdict = ("SPECIFIC: the activation-level perturbation moves the "
                   "signal much more than the input-level one. The tools are "
                   "reading something about the computation specifically. "
                   "This does NOT by itself establish introspection.")
    summary["interpretation"] = verdict
    summary["caveat"] = (
        "One model, one prompt, one perturbation direction. A ratio above 1.5 "
        "is not a proof of introspection and a ratio below it is not a proof "
        "of its absence. The published evidence (arXiv 2605.26242, "
        "2603.05414, 2611.15267) favours the generic reading."
    )

    print(json.dumps(summary, indent=2, default=str))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
