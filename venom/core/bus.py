"""Read-only forward-hook bus: reach every transformer block without changing anything.

The hook returns None, so PyTorch keeps the original output. `detach()` must
restore the host to bit-exact original behaviour - that is the trust contract.
"""
from __future__ import annotations

import hashlib


class Bus:
    """Taps every transformer block. Hooks return None -> output untouched."""

    def __init__(self, model, blocks):
        self.model = model
        self.blocks = list(blocks)
        self.taps: dict = {}
        self._handles = []
        self.calls = 0

    def _hook(self, name: str):
        def fn(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            self.taps[name] = tensor.detach()
            self.calls += 1
            return None
        return fn

    def attach(self) -> "Bus":
        if self._handles:
            raise RuntimeError("bus already attached")
        for i, blk in enumerate(self.blocks):
            self._handles.append(blk.register_forward_hook(self._hook(f"block_{i}")))
        return self

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self.taps.clear()

    @property
    def attached(self) -> bool:
        return bool(self._handles)

    def __enter__(self) -> "Bus":
        return self.attach()

    def __exit__(self, *exc):
        self.detach()


def gpt2_blocks(model):
    return list(model.transformer.h)


def logits_hash(logits) -> str:
    import torch
    raw = logits.detach().to(torch.float32).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()
