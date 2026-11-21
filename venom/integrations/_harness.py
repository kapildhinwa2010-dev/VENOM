"""A tiny, local, randomly-initialised model + a stub tokenizer.

Three sibling repos (POWERLAW-PROBE, VARIANCE-LIES, Draft_ponit_harness)
need a real HuggingFace model and a tokenizer.  Downloading one would make
the integration depend on the network, so instead VENOM builds a **tiny
GPT-2 from a config** (no weights are downloaded, the weights are random)
and pairs it with a deterministic stub tokenizer.

What this does: it lets the sibling's own code run for real - the forward
hooks, the SVD bases and the replay are the sibling's, unmodified.

What this does NOT prove: anything about a trained model.  The numbers
these integrations report describe a random-weight toy model; they are a
plumbing check that the adapter wires the sibling up correctly, never a
result about language models.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class StubTokenizer:
    """A deterministic character-level tokenizer.

    It implements only what the sibling repos call: ``tok(text,
    return_tensors="pt").input_ids`` and ``tok.decode(ids)``.  It is not a
    real vocabulary and it is not compatible with any pretrained model.
    """

    def __init__(self, vocab_size: int = 64):
        self.vocab_size = int(vocab_size)
        self._ids: Dict[str, List[int]] = {}

    def _encode(self, text: str) -> List[int]:
        cached = self._ids.get(text)
        if cached is not None:
            return list(cached)
        ids = [1 + (ord(ch) % (self.vocab_size - 2)) for ch in text][:64] or [1]
        self._ids[text] = list(ids)
        return list(ids)

    def __call__(self, text, return_tensors=None, **kwargs):
        import torch

        ids = self._encode(str(text))
        tensor = torch.tensor([ids], dtype=torch.long)
        return type("Encoding", (), {"input_ids": tensor})()

    def encode(self, text, **kwargs):
        return self._encode(str(text))

    def decode(self, ids, **kwargs) -> str:
        out = []
        for i in ids:
            i = int(i)
            out.append(chr(32 + (i % 95)))
        return "".join(out)


def tiny_model(vocab_size: int = 64, layers: int = 2, width: int = 32,
               seed: int = 0):
    """Build a tiny randomly-initialised GPT-2.

    Returns ``(model, tokenizer)``.  Requires torch + transformers, which
    are optional dependencies of VENOM; the caller must handle their
    absence.
    """
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=128,
        n_embd=width,
        n_layer=layers,
        n_head=2,
        n_inner=4 * width,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model, StubTokenizer(vocab_size=vocab_size)


def tiny_states(layers: int = 3, samples: int = 8, dim: int = 6, seed: int = 0):
    """Deterministic per-layer hidden states ``(layers, samples, dim)``."""
    import numpy as np

    rng = np.random.default_rng(seed)
    base = rng.normal(size=(samples, dim))
    out = []
    for i in range(layers):
        noise = 0.05 * (i + 1)
        out.append(base + rng.normal(scale=noise, size=(samples, dim)))
    return out
