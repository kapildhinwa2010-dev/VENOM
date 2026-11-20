"""Layer-wise consensus — "check myself".

The idea: a token chosen only by the last layer (or by a single outlier layer)
is fragile; a token that most layers' logit lens already preferred is broadly
supported by the network.

`consensus()` reads the per-layer hidden states captured by the `Bus`, projects
each through the final norm + LM head, and measures how many layers agree with
the final top-1 token.

This is a *self-report about the computation*, NOT a claim of introspection.
The published literature is explicit that a supervised probe beats a model's
own self-report by a wide margin (arXiv 2611.15267), and that activation-level
detection is content-agnostic (arXiv 2603.05414). Read `RESEARCH.md` §2.2
before drawing conclusions from these numbers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _layer_order(taps: Dict[str, Any]) -> List[str]:
    """Sort tap keys by their numeric layer index."""
    def key(name: str):
        try:
            return int(str(name).rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            return 0
    return sorted(taps, key=key)


def consensus(taps: Dict[str, Any], norm_module, lm_head, tokenizer,
              top_k: int = 5) -> Dict[str, Any]:
    """How broadly supported is the final top-1 token?

    Returns:
        {
          "layers_agreeing": int,
          "layers_total": int,
          "agreement": float,          # agreeing / total, 0..1
          "final_top1": str,
          "per_layer": [{"layer": int, "top1": str, "is_final_top1": bool,
                         "rank_of_final": int}],
          "verdict": str,              # one line
          "caveat": str,
        }
    """
    caveat = (
        "Logit-lens agreement measures how broadly the network's internal "
        "readouts support the final token. It is NOT a measure of "
        "introspection or understanding: published work shows an external "
        "probe beats a model's own self-report (arXiv 2611.15267) and that "
        "activation-level detection is content-agnostic (arXiv 2603.05414)."
    )

    if not taps:
        return {"error": "no taps captured - attach the bus before calling this", "caveat": caveat}
    if norm_module is None or lm_head is None:
        return {"error": "norm_module and lm_head are required", "caveat": caveat}

    import torch

    order = _layer_order(taps)
    per_layer: List[Dict[str, Any]] = []
    final_top1_id: Optional[int] = None
    final_layer = order[-1]

    with torch.no_grad():
        # the final layer defines what the host actually chose
        final_hidden = taps[final_layer]
        if final_hidden.dim() >= 3:
            final_hidden = final_hidden[0, -1, :]
        else:
            final_hidden = final_hidden[-1, :]
        final_logits = lm_head(norm_module(final_hidden))
        final_top1_id = int(final_logits.argmax().item())
        final_top1 = tokenizer.decode([final_top1_id])

        for name in order:
            hidden = taps[name]
            if hidden.dim() >= 3:
                hidden = hidden[0, -1, :]
            else:
                hidden = hidden[-1, :]
            logits = lm_head(norm_module(hidden))
            values, indices = logits.topk(min(top_k, logits.shape[-1]))
            layer_top1_id = int(indices[0].item())
            rank = int((logits > logits[final_top1_id]).sum().item())
            per_layer.append({
                "layer": int(str(name).rsplit("_", 1)[-1]),
                "top1": tokenizer.decode([layer_top1_id]),
                "is_final_top1": layer_top1_id == final_top1_id,
                "rank_of_final": rank,
                "top_tokens": [tokenizer.decode([int(i)]) for i in indices[:top_k]],
            })

    agreeing = sum(1 for row in per_layer if row["is_final_top1"])
    total = len(per_layer)
    agreement = agreeing / total if total else 0.0

    if agreement >= 0.7:
        verdict = "BROAD: most layers already preferred the final token"
    elif agreement >= 0.3:
        verdict = "MIXED: the final token was decided late"
    else:
        verdict = "LATE: the final token was decided by the last layers only"

    return {
        "layers_agreeing": agreeing,
        "layers_total": total,
        "agreement": round(agreement, 4),
        "final_top1": final_top1,
        "per_layer": per_layer,
        "verdict": verdict,
        "caveat": caveat,
    }


def uncertainty_summary(logits_row: List[float], k: int = 5) -> Dict[str, Any]:
    """Convenience wrapper used by the tool layer."""
    from venom.sensors.uncertainty import token_uncertainty
    return token_uncertainty(logits_row, k)
