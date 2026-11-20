"""Logit-lens attribution: *why* a token, and *where* a concept lives.

Requires torch. Read-only: the taps dict is never mutated.

Honesty contract (see LIMITATIONS.md §5): this is a **readout**, not a proof.
Reading an intermediate residual stream through the final norm + head shows
what the forward pass had already written into that stream at the moment the
logit was computed. It does **not** show that the layer "caused" the token,
that the representation is what the model "knows", or that the model is
"self-aware". Published evidence (arXiv 2611.15267) reports that a supervised
probe beats a model's own self-report by a wide margin, so activation
readouts are not a substitute for a trained probe.

Two entry points:

    why(token, taps, norm_module, lm_head, tokenizer, ...)  -> why the host
                                                               picked a token
    where(concept, taps, norm_module, lm_head, tokenizer, ...) -> where a
                                                               concept is
                                                               represented
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

CAVEAT = (
    "This is a logit-lens readout of the forward pass, not a causal proof: it "
    "shows what the residual stream contained at each layer, not that any "
    "layer caused the token, and not that the model 'knows' the concept. "
    "A supervised probe has been shown to beat a model's own self-report "
    "(arXiv 2611.15267)."
)


# --------------------------------------------------------------------------- #
# internal helpers
# --------------------------------------------------------------------------- #

def _layer_names(taps: Dict[str, Any], layer_ids: Optional[List[int]]) -> List[str]:
    """Return the tap names to read, in ascending layer order.

    Tap keys look like "block_0", "block_1", ... A non-numeric key is kept
    after the numeric ones, ordered by its original insertion order, so an
    unusual naming scheme still produces a deterministic result.
    """
    names = list(taps.keys())

    def key(name: str):
        if name.startswith("block_"):
            tail = name[len("block_"):]
            if tail.isdigit():
                return (0, int(tail), "")
        return (1, 0, name)

    names.sort(key=key)

    if layer_ids is not None:
        wanted = {int(i) for i in layer_ids}
        keep = []
        for name in names:
            if name.startswith("block_"):
                tail = name[len("block_"):]
                if tail.isdigit() and int(tail) in wanted:
                    keep.append(name)
            elif name in wanted:
                keep.append(name)
        return keep
    return names


def _layer_index(name: str, fallback: int) -> int:
    if name.startswith("block_"):
        tail = name[len("block_"):]
        if tail.isdigit():
            return int(tail)
    return fallback


def _last_hidden(tap) -> Any:
    """Reduce a [batch, seq, hidden] tap to [batch, hidden] at the last position."""
    import torch

    tensor = tap
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    if tensor.dim() == 2:
        # already [batch, hidden]
        return tensor
    if tensor.dim() != 3:
        raise ValueError(f"tap must be [batch, seq, hidden], got shape {tuple(tensor.shape)}")
    return tensor[:, -1, :]


def _logit_row(tap, norm_module, lm_head):
    """Apply final norm + head to a tap and return the logits for batch item 0.

    Returns a 1-D tensor of shape [vocab].
    """
    import torch

    hidden = _last_hidden(tap)
    with torch.no_grad():
        normed = norm_module(hidden) if norm_module is not None else hidden
        logits = lm_head(normed)
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    logits = logits[0]
    return logits.detach().to(torch.float32)


def _vocab_rank(logits, token_id: int) -> int:
    """0-based rank of `token_id` in the full vocabulary ordering.

    The number of tokens with a strictly greater logit. Computed against the
    whole logit vector - never approximated by the top-k window.
    """
    import torch

    target = logits[token_id]
    return int((logits > target).sum().item())


def _decode(tokenizer, token_id: int) -> str:
    try:
        text = tokenizer.decode([int(token_id)])
    except Exception:  # noqa: BLE001 - a broken tokenizer must not kill the readout
        text = ""
    return text if isinstance(text, str) else str(text)


def _encode(tokenizer, text: str) -> List[int]:
    ids = tokenizer.encode(text)
    if isinstance(ids, tuple) or not isinstance(ids, list):
        try:
            ids = list(ids)
        except TypeError:
            ids = []
    return [int(i) for i in ids]


def _error(message: str, **extra) -> Dict[str, Any]:
    out: Dict[str, Any] = {"error": message}
    out.update(extra)
    return out


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def why(token: str, taps: dict, norm_module, lm_head, tokenizer,
        layer_ids: Optional[list] = None, top_k: int = 5) -> dict:
    """Explain, via the logit lens, WHY the host picked `token`.

    For every captured block (ascending), the final norm and the LM head are
    applied to that block's output at the last sequence position, and the
    logit of the target token is read out.

    What this does NOT prove: it does not show that any layer caused the
    decision, that the residual stream encodes the reason, or that the model
    is "self-aware". It is a readout of what the forward pass wrote into the
    residual stream; a supervised probe has been shown to beat a model's own
    self-report (arXiv 2611.15267). See LIMITATIONS.md §5.

    Returns a dict with keys:
        token, token_id, tokenized_as, layers, decision_layer, final_top1,
        caveat. On error: {"error": ...}.
    """
    if not taps:
        return _error("no taps captured: attach the bus and run a forward pass before calling why()")
    if norm_module is None:
        return _error("no norm_module: the final normalisation layer is required for a logit-lens readout")
    if lm_head is None:
        return _error("no lm_head: the output projection is required for a logit-lens readout")

    tokenized = _encode(tokenizer, token)
    if not tokenized:
        return _error(f"tokenizer produced no ids for token {token!r}")
    token_id = tokenized[0]

    names = _layer_names(taps, layer_ids)
    if not names:
        return _error("no taps matched the requested layer_ids")

    layers: List[Dict[str, Any]] = []
    decision_layer: Optional[int] = None
    final_top1 = None

    for fallback, name in enumerate(names):
        idx = _layer_index(name, fallback)
        try:
            logits = _logit_row(taps[name], norm_module, lm_head)
        except Exception as exc:  # noqa: BLE001 - one bad tap must not kill the sweep
            layers.append({"layer": idx, "error": f"{type(exc).__name__}: {exc}"})
            continue

        target_logit = float(logits[token_id].item())
        top1_id = int(logits.argmax().item())
        top1_logit = float(logits[top1_id].item())
        is_top1 = top1_id == token_id
        if is_top1 and decision_layer is None:
            decision_layer = idx

        k = max(1, int(top_k))
        _, top_indices = logits.topk(min(k, logits.shape[-1]))
        top_tokens = [_decode(tokenizer, int(i)) for i in top_indices]

        layers.append({
            "layer": idx,
            "target_logit": round(target_logit, 4),
            "top1_token": _decode(tokenizer, top1_id),
            "top1_logit": round(top1_logit, 4),
            "is_top1": bool(is_top1),
            "top_tokens": top_tokens,
        })
        final_top1 = layers[-1].get("top1_token", final_top1)

    return {
        "token": token,
        "token_id": token_id,
        "tokenized_as": tokenized,
        "multi_token": len(tokenized) > 1,
        "layers": layers,
        "decision_layer": decision_layer,
        "final_top1": final_top1,
        "caveat": CAVEAT,
    }


def where(concept: str, taps: dict, norm_module, lm_head, tokenizer,
          layer_ids: Optional[list] = None, top_k: int = 5) -> dict:
    """Find WHERE in the network the information about `concept` lives.

    Same logit-lens machinery as `why`, but the probe is the concept's token
    ids: for every captured block, the full-vocabulary rank of the concept's
    best token and the mean logit of the concept's tokens are recorded.

    What this does NOT prove: it does not show that the layer stores the
    concept, that the concept is decoded from that layer, or that the model
    "knows" the concept. Ranks and logits are readout statistics; a
    supervised probe has been shown to beat a model's own self-report
    (arXiv 2611.15267). See LIMITATIONS.md §5.

    Returns a dict with keys:
        concept, concept_ids, layers, best_layer, summary, caveat.
        On error: {"error": ...}.
    """
    if not taps:
        return _error("no taps captured: attach the bus and run a forward pass before calling where()")
    if norm_module is None:
        return _error("no norm_module: the final normalisation layer is required for a logit-lens readout")
    if lm_head is None:
        return _error("no lm_head: the output projection is required for a logit-lens readout")

    concept_ids = _encode(tokenizer, concept)
    if not concept_ids:
        return _error(f"tokenizer produced no ids for concept {concept!r}")

    names = _layer_names(taps, layer_ids)
    if not names:
        return _error("no taps matched the requested layer_ids")

    layers: List[Dict[str, Any]] = []
    best_layer: Optional[int] = None
    best_rank: Optional[int] = None

    for fallback, name in enumerate(names):
        idx = _layer_index(name, fallback)
        try:
            logits = _logit_row(taps[name], norm_module, lm_head)
        except Exception as exc:  # noqa: BLE001 - one bad tap must not kill the sweep
            layers.append({"layer": idx, "error": f"{type(exc).__name__}: {exc}"})
            continue

        concept_logits = [float(logits[c].item()) for c in concept_ids if c < logits.shape[-1]]
        if not concept_logits:
            layers.append({"layer": idx, "error": "concept token id is outside the vocabulary"})
            continue

        # Full-vocabulary rank of every concept token; the best (lowest) wins.
        ranks = [_vocab_rank(logits, c) for c in concept_ids if c < logits.shape[-1]]
        layer_best_rank = min(ranks)

        k = max(1, int(top_k))
        _, top_indices = logits.topk(min(k, logits.shape[-1]))
        top_tokens = [_decode(tokenizer, int(i)) for i in top_indices]

        layers.append({
            "layer": idx,
            "best_rank": int(layer_best_rank),
            "mean_logit": round(sum(concept_logits) / len(concept_logits), 4),
            "top_tokens": top_tokens,
        })

        if best_rank is None or layer_best_rank < best_rank:
            best_rank = layer_best_rank
            best_layer = idx

    if best_layer is None:
        summary = f"concept '{concept}' could not be read out from any layer"
    else:
        summary = (
            f"concept '{concept}' is most strongly represented at layer "
            f"{best_layer} (rank {best_rank})"
        )

    return {
        "concept": concept,
        "concept_ids": concept_ids,
        "layers": layers,
        "best_layer": best_layer,
        "summary": summary,
        "caveat": CAVEAT,
    }
