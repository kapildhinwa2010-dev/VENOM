"""Logit lens: read the intermediate representation through the final norm + head.

Requires torch. Kept separate so the torch-free core stays installable.
"""
from __future__ import annotations


def logit_lens(model, hidden, norm_module=None, lm_head=None, apply_final_norm: bool = True):
    """Project an intermediate hidden state into vocabulary logits.

    hidden: tensor [batch, seq, hidden]
    returns: tensor [batch, seq, vocab]
    """
    import torch

    if norm_module is not None and apply_final_norm:
        hidden = norm_module(hidden)
    if lm_head is None:
        lm_head = getattr(model, "lm_head")
    return lm_head(hidden)


def top_tokens(logits_row, tokenizer, k: int = 5):
    """Return the top-k token strings for one logit row."""
    values, indices = logits_row.topk(k)
    return [
        {"token": tokenizer.decode([int(i)]), "id": int(i), "logit": round(float(v), 4)}
        for v, i in zip(values, indices)
    ]
