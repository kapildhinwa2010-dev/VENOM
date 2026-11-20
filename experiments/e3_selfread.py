"""E3: the self-read tools on a real model.

Runs the full Tier-1 tool surface against a real HuggingFace model and
records what each tool actually returns. This is the first time the
self-read tools touch a real forward pass.

It also runs the P3 ablation the plan calls for: with the self-read tools
available vs not. The ablation measures whether a host that can consult its
own uncertainty does anything different - which is the question the whole
project rests on.

    python experiments/e3_selfread.py --model gpt2
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from venom.core.runtime import VenomConfig, VenomRuntime
from venom.shell.attach import attach

PROMPT = "The capital of France is"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--out", default="e3_selfread.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).eval()
    ids = tok(args.prompt, return_tensors="pt")

    # The production budget is 3 calls per ANSWER. This experiment asks for
    # 7 tools in one go, so it uses a larger cap - each tool call here is
    # treated as its own "answer".
    runtime = VenomRuntime(VenomConfig(session_id="e3", max_calls=20))
    runtime.remember("capital of france", "Paris")

    with attach(model, tok, runtime=runtime, name_hint=args.model) as host:
        # one forward pass so the probe and the bus have something to report
        with torch.no_grad():
            model(**ids)

        results = {}
        for tool, tool_args in [
            ("venom.uncertainty", {}),
            ("venom.check_myself", {}),
            ("venom.why", {"token": " Paris"}),
            ("venom.where", {"concept": " France"}),
            ("venom.audit", {}),
            ("venom.verify", {"answer": "the capital is Paris",
                              "key": "capital of france"}),
            ("venom.report", {}),
        ]:
            try:
                out = host.call_tool(tool, tool_args)
            except Exception as exc:  # noqa: BLE001
                out = {"ok": False, "error": str(exc)}
            results[tool] = out

        # the ablation: same prompt, with and without a self-read hint
        base = host.generate(args.prompt, max_new_tokens=12, do_sample=False)
        unc = host.call_tool("venom.uncertainty", {})
        hinted = ("CONTEXT: The answer is uncertain (confidence "
                  f"{unc.get('confidence')}, entropy {unc.get('entropy')}). "
                  f"| {args.prompt}") if unc.get("ok") else args.prompt
        with_self_read = host.generate(hinted, max_new_tokens=12, do_sample=False)

    def trim(obj, depth=0):
        """keep the output small but keep the numbers"""
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in ("caveat", "warning", "note"):
                    continue
                out[k] = trim(v, depth + 1)
            return out
        if isinstance(obj, list):
            return [trim(v, depth + 1) for v in obj[:12]] + ([f"...+{len(obj)-12}"] if len(obj) > 12 else [])
        if isinstance(obj, str) and len(obj) > 160:
            return obj[:160] + "..."
        return obj

    summary = {
        "model": args.model,
        "prompt": args.prompt,
        "ablation": {
            "note": ("P3 ablation: does a host that can consult its own "
                     "uncertainty produce a different continuation? This is "
                     "one prompt, not a benchmark - it is a plumbing check, "
                     "not a result."),
            "without_self_read": base,
            "with_self_read_hint": with_self_read,
            "identical": base == with_self_read,
        },
        "tools": {k: trim(v) for k, v in results.items()},
        "caveats": {
            k: v.get("caveat") or v.get("warning")
            for k, v in results.items()
            if isinstance(v, dict) and (v.get("caveat") or v.get("warning"))
        },
    }
    print(json.dumps(summary, indent=2, default=str))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
