#!/usr/bin/env python
"""Regenerate `configs/*.json` from real model fingerprints.

The old configs were hand-written and guessed the block paths. These are
measured: the model's module graph is built on the meta device (no weights)
and the discoverer is asked where the blocks, the head and the norm are.

    python scripts/gen_configs.py
"""
import json
import os
import sys

TARGETS = {
    "qwen2.5": "Qwen/Qwen2.5-0.5B-Instruct",
    "llama3.2": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "gpt2": "gpt2",
}


def main():
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        sys.exit("this script needs torch + transformers")

    from venom.core.discovery import fingerprint_universal

    os.makedirs("configs", exist_ok=True)
    for name, repo in TARGETS.items():
        cfg = AutoConfig.from_pretrained(repo)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(cfg)
        fp = fingerprint_universal(model, repo)
        out = {
            "model": repo,
            "family": fp.get("family"),
            "model_type": fp.get("model_type"),
            "blocks_path": fp.get("blocks_path"),
            "num_layers": fp.get("num_layers"),
            "lm_head_path": fp.get("lm_head_path"),
            "norm_path": fp.get("norm_path"),
            "confidence": fp.get("confidence"),
            "note": ("Generated from a real fingerprint (meta-device build) by "
                     "scripts/gen_configs.py"),
            "mcp": {"transport": "stdio", "command": "venom", "args": ["--host", name]},
            "budget": {"max_calls": 3},
            "memory": {"capacity": 1024},
        }
        with open(f"configs/{name}.json", "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"configs/{name}.json -> {out['family']} {out['blocks_path']}")


if __name__ == "__main__":
    main()
