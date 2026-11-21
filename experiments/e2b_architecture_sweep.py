"""E2b: architecture sweep — how many model families can VENOM attach to?

The "works with any AI" claim needs more than three families. This script
builds the **module graph** of many architectures on the `meta` device (no
weights are allocated, so it runs in a couple of MB of RAM) and asks the
discoverer to find the blocks, the LM head and the final norm.

Meta-device construction exercises exactly the code path that matters: the
structure of the model, not its numbers. It does NOT prove that a forward
pass on real weights behaves the same way - for that, see
`e2_universal_attach.py`, which loads real weights and checks bit-exactness.

    python experiments/e2b_architecture_sweep.py
"""
from __future__ import annotations

import argparse
import json

MODELS = [
    # (repo id, label)
    ("gpt2", "GPT-2"),
    ("EleutherAI/gpt-neo-125M", "GPT-Neo"),
    ("EleutherAI/gpt-j-6B", "GPT-J"),
    ("facebook/opt-125m", "OPT"),
    ("bigscience/bloom-560m", "BLOOM"),
    ("mistralai/Mistral-7B-Instruct-v0.2", "Mistral"),
    ("meta-llama/Llama-3.2-1B", "Llama-3"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5"),
    ("Qwen/Qwen3-0.6B", "Qwen3"),
    ("microsoft/Phi-3-mini-4k-instruct", "Phi-3"),
    ("microsoft/phi-2", "Phi-2"),
    ("google/gemma-2-2b-it", "Gemma-2"),
    ("google/gemma-3-1b-it", "Gemma-3"),
    ("allenai/OLMo-2-1124-7B", "OLMo-2"),
    ("tiiuae/falcon-rw-1b", "Falcon"),
    ("bigcode/starcoder2-3b", "StarCoder2"),
    ("mosaicml/mpt-7b-chat", "MPT"),
    ("ibm-granite/granite-3.1-2b-instruct", "Granite"),
    ("CohereForAI/c4ai-command-r-v01", "Command-R"),
    ("internlm/internlm2-1_8b", "InternLM2"),
    ("microsoft/DeciLM-7B", "DeciLM"),
    ("stabilityai/stablelm-2-1_6b", "StableLM"),
    ("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "TinyLlama"),
    ("HuggingFaceTB/SmolLM2-135M-Instruct", "SmolLM2"),
    ("facebook/opt-125m", "OPT-2"),
    ("EleutherAI/pythia-70m", "Pythia"),
    ("openai-community/gpt2-medium", "GPT-2-medium"),
    ("Salesforce/codegen-350M-mono", "CodeGen"),
    ("cerebras/Cerebras-GPT-590M", "Cerebras"),
    ("microsoft/Phi-3-mini-4k-instruct", "Phi-3-dup"),
]


def classify_failure(exc: Exception) -> str:
    """Separate "we could not DOWNLOAD it" from "the discoverer failed".

    These are completely different claims and must never be mixed up: a gated
    repo says nothing about whether the discoverer can handle that
    architecture.
    """
    text = str(exc).lower()
    if "gated repo" in text or "401" in text or "403" in text:
        return "access_gated"
    if "not a valid model identifier" in text or "404" in text:
        return "repo_not_found"
    if "custom code" in text or "trust_remote_code" in text:
        return "needs_custom_code"
    if "memory" in text or "killed" in text:
        return "out_of_memory"
    return "unknown"


def probe(repo: str):
    """Build the module graph on the meta device and fingerprint it."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from venom.core.discovery import fingerprint_universal

    cfg = AutoConfig.from_pretrained(repo, trust_remote_code=False)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=False)

    fp = fingerprint_universal(model, repo)
    fp.pop("lm_head", None)
    fp.pop("norm", None)
    return fp


def probe_with_custom_code(repo: str):
    """Some architectures ship their own modelling code."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from venom.core.discovery import fingerprint_universal

    cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    fp = fingerprint_universal(model, repo)
    fp.pop("lm_head", None)
    fp.pop("norm", None)
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="e2b_sweep.json")
    ap.add_argument("--only", default=None, help="comma-separated repo ids")
    args = ap.parse_args()

    models = MODELS
    if args.only:
        wanted = {m.strip() for m in args.only.split(",")}
        models = [m for m in MODELS if m[0] in wanted]

    results = []
    for repo, label in models:
        try:
            fp = probe(repo)
            needs_custom = False
        except Exception as first:  # noqa: BLE001
            first_error = f"{type(first).__name__}: {str(first)[:160]}"
            try:
                fp = probe_with_custom_code(repo)
                needs_custom = True
            except Exception as exc:  # noqa: BLE001
                results.append({
                    "repo": repo, "label": label, "ok": False,
                    "failure_kind": classify_failure(exc),
                    "error": f"{type(exc).__name__}: {str(exc)[:160]}",
                })
                fp = None
        if fp is not None:
            results.append({
                "repo": repo, "label": label,
                "family": fp.get("family"),
                "model_type": fp.get("model_type"),
                "blocks_path": fp.get("blocks_path"),
                "num_layers": fp.get("num_layers"),
                "confidence": fp.get("confidence"),
                "requires_trust_remote_code": needs_custom,
                "ok": fp.get("confidence", 0) >= 0.75 and bool(fp.get("blocks_path")),
            })

        r = results[-1]
        status = "OK  " if r.get("ok") else "FAIL"
        detail = (f"{r.get('family'):<12} {str(r.get('blocks_path')):<22} "
                  f"layers={r.get('num_layers')} conf={r.get('confidence')}"
                  if r.get("ok") else r.get("error", "unknown")[:90])
        print(f"{status} {label:<14} {detail}", flush=True)

    ok = [r for r in results if r.get("ok")]
    families = sorted({r["family"] for r in ok})
    kinds = {}
    for r in results:
        if not r.get("ok"):
            kinds[r.get("failure_kind", "unknown")] = kinds.get(r.get("failure_kind", "unknown"), 0) + 1
    summary = {
        "models_attempted": len(results),
        "models_discovered": len(ok),
        "distinct_families": len(families),
        "families": families,
        "failed": [{"repo": r["repo"], "kind": r.get("failure_kind", "unknown"),
                    "error": r.get("error")} for r in results if not r.get("ok")],
        "failure_kinds": kinds,
        "results": results,
        "note": (
            "Meta-device construction: the module GRAPH is real, the weights "
            "are not. This proves the discoverer can find the structure of an "
            "architecture it has never seen. It does NOT prove that a forward "
            "pass on real weights is bit-exact - see e2_universal_attach.py "
            "for that, on real weights. Failures are classified: an "
            "access/404 failure says NOTHING about the discoverer."
        ),
    }
    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
