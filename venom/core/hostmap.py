"""Auto-detect the architecture of a host model so tools can attach generically.

v0.1.2 fix: the family was decided by the FIRST family whose `blocks_attr`
resolved. Llama, Qwen2, Mistral, Phi and Gemma all expose `model.layers`, so
every non-GPT-2 model was labelled "llama" - and for Phi the returned
`norm_path` ("model.norm") does not even exist on the model.
`config.model_type` is now authoritative; path probing is only the fallback.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

FAMILIES = {
    "gpt2": {"blocks_attr": "transformer.h", "lm_head": "lm_head", "norm": "transformer.ln_f"},
    "llama": {"blocks_attr": "model.layers", "lm_head": "lm_head", "norm": "model.norm"},
    "qwen2": {"blocks_attr": "model.layers", "lm_head": "lm_head", "norm": "model.norm"},
    "mistral": {"blocks_attr": "model.layers", "lm_head": "lm_head", "norm": "model.norm"},
    "phi": {"blocks_attr": "model.layers", "lm_head": "lm_head", "norm": "model.final_layernorm"},
    "gemma": {"blocks_attr": "model.layers", "lm_head": "lm_head", "norm": "model.norm"},
}


def _resolve(model: Any, path: str, optional: bool = False):
    obj = model
    for part in path.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            if optional:
                return None
            raise
    return obj


def fingerprint(model: Any, name_hint: str = "") -> Dict[str, Any]:
    result: Dict[str, Any] = {"name_hint": name_hint, "family": "unknown"}

    cfg = getattr(model, "config", None)
    model_type = getattr(cfg, "model_type", None) if cfg is not None else None
    if model_type:
        result["model_type"] = model_type

    # v0.1.2: `config.model_type` is authoritative when it is one we know.
    # Probing is only the fallback for unknown/absent model_type, and then the
    # candidate must expose BOTH the blocks path and the norm path - otherwise
    # Phi would be matched as Llama (whose "model.norm" Phi does not have).
    if model_type in FAMILIES:
        attrs = FAMILIES[model_type]
        try:
            blocks = _resolve(model, attrs["blocks_attr"])
        except AttributeError:
            blocks = None
        if blocks is not None:
            result.update({
                "family": model_type,
                "blocks_path": attrs["blocks_attr"],
                "lm_head_path": attrs["lm_head"],
                "norm_path": attrs["norm"],
                "num_layers": len(list(blocks)),
            })
    else:
        for family, attrs in FAMILIES.items():
            try:
                blocks = _resolve(model, attrs["blocks_attr"])
            except AttributeError:
                continue
            if _resolve(model, attrs["norm"], optional=True) is None:
                continue
            result.update({
                "family": family,
                "blocks_path": attrs["blocks_attr"],
                "lm_head_path": attrs["lm_head"],
                "norm_path": attrs["norm"],
                "num_layers": len(list(blocks)),
            })
            break

    if cfg is not None:
        result["hidden_size"] = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))
        result["vocab_size"] = getattr(cfg, "vocab_size", None)
        result["num_heads"] = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None))

    return result


def get_blocks(model: Any, fp: Dict[str, Any]) -> List[Any]:
    path: Optional[str] = fp.get("blocks_path")
    if not path:
        raise KeyError(
            f"no blocks_path in fingerprint (family={fp.get('family')!r}); "
            "the host architecture was not recognised"
        )
    return list(_resolve(model, path))
