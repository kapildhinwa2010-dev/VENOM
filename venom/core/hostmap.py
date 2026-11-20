"""Auto-detect the architecture of a host model so tools can attach generically."""
from __future__ import annotations

from typing import Any, Dict, List

FAMILIES = {
    "gpt2": {"blocks_attr": "transformer.h", "lm_head": "lm_head", "norm": "transformer.ln_f"},
    "llama": {"blocks_attr": "model.layers", "lm_head": "lm_head", "norm": "model.norm"},
    "qwen2": {"blocks_attr": "model.layers", "lm_head": "lm_head", "norm": "model.norm"},
    "mistral": {"blocks_attr": "model.layers", "lm_head": "lm_head", "norm": "model.norm"},
    "phi": {"blocks_attr": "model.layers", "lm_head": "lm_head", "norm": "model.final_layernorm"},
    "gemma": {"blocks_attr": "model.layers", "lm_head": "lm_head", "norm": "model.norm"},
}


def _resolve(model: Any, path: str):
    obj = model
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def fingerprint(model: Any, name_hint: str = "") -> Dict[str, Any]:
    result: Dict[str, Any] = {"name_hint": name_hint, "family": "unknown"}

    for family, attrs in FAMILIES.items():
        try:
            blocks = _resolve(model, attrs["blocks_attr"])
        except AttributeError:
            continue
        result.update({
            "family": family,
            "blocks_path": attrs["blocks_attr"],
            "lm_head_path": attrs["lm_head"],
            "norm_path": attrs["norm"],
            "num_layers": len(list(blocks)),
        })
        break

    cfg = getattr(model, "config", None)
    if cfg is not None:
        result["hidden_size"] = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))
        result["vocab_size"] = getattr(cfg, "vocab_size", None)
        result["num_heads"] = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None))
        result["model_type"] = getattr(cfg, "model_type", None)

    return result


def get_blocks(model: Any, fp: Dict[str, Any]) -> List[Any]:
    return list(_resolve(model, fp["blocks_path"]))
