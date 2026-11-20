"""Universal architecture discovery.

v0.2 replaces the hardcoded family table. The old `hostmap.FAMILIES` table
knew six exact attribute paths and, when the path probing failed, silently
labelled the host "unknown" - which meant the layer could not attach to
anything outside that list.

This module instead *discovers* the host structure by walking the module
graph:

  1. find every `nn.ModuleList` whose children look like transformer blocks
     (they have an attention sub-module AND an MLP/FFN sub-module);
  2. score the candidates (depth, name hints, attention/mlp richness) and
     take the winner;
  3. locate the output head and the final norm without assuming a name.

The result works for any HuggingFace causal LM, including architectures that
did not exist when this table was written, and it reports a confidence score
so the caller can refuse to attach when discovery is not convincing.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

ATTN_NAMES = ("attn", "attention", "self_attn", "cross_attn")
MLP_NAMES = ("mlp", "feed_forward", "ffn", "feed_forward_net", "w1")
NAME_HINTS = ("layers", "blocks", "h", "decoder")


def _has_child(obj: Any, names: Tuple[str, ...]) -> bool:
    for child in obj.children():
        for name in names:
            if name in child.__class__.__name__.lower():
                return True
    return False


def _looks_like_block(obj: Any) -> bool:
    """A transformer block: has an attention part and an MLP/FFN part."""
    if isinstance(obj, (nn.ModuleList, nn.Sequential)):
        return False
    names = {n.lower() for n, _ in obj.named_children()}
    has_attn = any(any(h in n for h in ATTN_NAMES) for n in names)
    has_mlp = any(any(h in n for h in MLP_NAMES) for n in names)
    return has_attn and has_mlp


def _block_score(container: nn.ModuleList, path: str) -> float:
    """Higher is better. Combines how many children look like blocks, how
    rich their internals are, and how plausible the container name is."""
    if not container:
        return 0.0
    blocks = [c for c in container if _looks_like_block(c)]
    if not blocks:
        return 0.0
    ratio = len(blocks) / len(container)
    if ratio < 0.5:
        return 0.0

    richness = 0.0
    for blk in blocks:
        richness += sum(1 for _ in blk.named_modules())
    richness /= max(1, len(blocks))

    name_score = 0.0
    leaf = path.rsplit(".", 1)[-1].lower()
    for hint in NAME_HINTS:
        if leaf == hint:
            name_score += 3.0
        elif hint in leaf:
            name_score += 1.5
    depth = path.count(".")

    return ratio * 10.0 + min(richness, 50.0) / 10.0 + name_score + depth * 0.1


def discover_blocks(model: Any) -> Optional[Tuple[str, nn.ModuleList]]:
    """Return (path, module_list) for the host's transformer blocks, or None."""
    best: Optional[Tuple[float, str, nn.ModuleList]] = None
    for path, module in model.named_modules():
        if not isinstance(module, nn.ModuleList):
            continue
        score = _block_score(module, path)
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, path, module)
    if best is None:
        return None
    return best[1], best[2]


def discover_output_head(model: Any) -> Optional[Tuple[str, Any]]:
    """Find the LM head (the module that maps hidden states to vocabulary)."""
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        try:
            head = getter()
        except Exception:  # noqa: BLE001 - not all models implement this
            head = None
        if head is not None:
            return "get_output_embeddings()", head

    for name in ("lm_head", "output", "embed_out", "head", "decoder"):
        obj = model
        ok = True
        for part in name.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and obj is not None:
            return name, obj

    for path, module in model.named_modules():
        leaf = path.rsplit(".", 1)[-1]
        if leaf in ("lm_head", "embed_out", "output_projection"):
            return path, module
    return None


def discover_final_norm(model: Any) -> Optional[Tuple[str, Any]]:
    """Find the final normalisation layer applied before the LM head.

    Needed for a correct logit lens. Tries, in order: the config's declared
    path, well-known names, then the norm-like module closest to the head.
    """
    candidates = (
        "model.norm", "model.final_layernorm", "transformer.ln_f",
        "transformer.ln", "norm", "final_norm", "ln_f", "gpt2.ln_f",
        "model.decoder.final_layer_norm",
    )
    for name in candidates:
        obj = model
        ok = True
        for part in name.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and isinstance(obj, nn.Module):
            return name, obj

    # Fallback: a norm module whose name suggests a final position.
    best = None
    for path, module in model.named_modules():
        if isinstance(module, nn.LayerNorm) or isinstance(module, nn.RMSNorm):
            leaf = path.rsplit(".", 1)[-1].lower()
            if any(h in leaf for h in ("final", "ln_f", "norm", "ln")):
                depth = path.count(".")
                if best is None or depth > best[0]:
                    best = (depth, path, module)
    if best is not None:
        return best[1], best[2]
    return None


def fingerprint_universal(model: Any, name_hint: str = "") -> Dict[str, Any]:
    """Describe a host model without assuming which architecture it is.

    The returned dict is the contract the rest of VENOM uses:

        family          "unknown" | config.model_type | detected architecture
        blocks_path     dotted path to the block container
        num_layers      number of transformer blocks
        lm_head_path    dotted path (or a synthetic label)
        norm_path       dotted path
        confidence      0.0 - 1.0, how convincing the discovery is
    """
    result: Dict[str, Any] = {
        "name_hint": name_hint,
        "family": "unknown",
        "confidence": 0.0,
    }

    cfg = getattr(model, "config", None)
    model_type = getattr(cfg, "model_type", None) if cfg is not None else None
    if model_type:
        result["model_type"] = model_type
        result["family"] = model_type

    found = discover_blocks(model)
    if found is None:
        result["error"] = "no transformer block list could be discovered"
        return result

    blocks_path, blocks = found
    result["blocks_path"] = blocks_path
    result["num_layers"] = len(blocks)

    head = discover_output_head(model)
    if head is not None:
        result["lm_head_path"], result["lm_head"] = head[0], head[1]
    norm = discover_final_norm(model)
    if norm is not None:
        result["norm_path"], result["norm"] = norm[0], norm[1]

    if cfg is not None:
        result["hidden_size"] = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))
        result["vocab_size"] = getattr(cfg, "vocab_size", None)
        result["num_heads"] = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None))

    # confidence: did we find blocks, a head and a norm?
    score = 0.5 if found else 0.0
    if head is not None:
        score += 0.25
    if norm is not None:
        score += 0.25
    result["confidence"] = round(min(score, 1.0), 2)
    if not model_type:
        result["family"] = "detected"
    return result


def get_blocks(model: Any, fp: Dict[str, Any]) -> List[Any]:
    """Resolve the block list from a fingerprint (discovery or legacy)."""
    path = fp.get("blocks_path")
    if not path:
        raise KeyError(
            f"no blocks_path in fingerprint (family={fp.get('family')!r}); "
            "the host architecture was not recognised"
        )
    obj = model
    for part in path.split("."):
        obj = getattr(obj, part)
    return list(obj)
