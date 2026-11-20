"""Read-only forward-hook bus: reach every transformer block without changing anything.

The hook returns None, so PyTorch keeps the original output. `detach()` must
restore the host to bit-exact original behaviour - that is the trust contract.
"""
from __future__ import annotations

import hashlib


class Bus:
    """Taps every transformer block. Hooks return None -> output untouched."""

    def __init__(self, model, blocks, offload_to_cpu: bool = False):
        self.model = model
        self.blocks = list(blocks)
        self.offload_to_cpu = offload_to_cpu
        self.taps: dict = {}
        self._handles = []
        self._attached = False
        self.calls = 0

    def _hook(self, name: str):
        def fn(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            # v0.1.2: clone. `tensor.detach()` is a LIVE ALIAS of the block
            # output - any later in-place write to the residual stream (a
            # future dial, an in-place residual add) silently rewrites the
            # recorded history. A tap must be a snapshot.
            snapshot = tensor.detach().clone()
            if self.offload_to_cpu:
                snapshot = snapshot.cpu()
            self.taps[name] = snapshot
            self.calls += 1
            return None
        return fn

    def attach(self) -> "Bus":
        if self._attached:
            raise RuntimeError("bus already attached")
        self._attached = True
        for i, blk in enumerate(self.blocks):
            self._handles.append(blk.register_forward_hook(self._hook(f"block_{i}")))
        return self

    def detach(self) -> None:
        """Remove the hooks and restore the host.

        v0.1.2: detach() no longer clears `self.taps`. Previously the taps were
        wiped on exit, which destroyed the very data the bus was attached to
        collect - the caller had to read `bus.taps` from inside the `with`
        block or lose everything. Memory is now released explicitly via
        `clear_taps()` or `release()`.
        """
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._attached = False

    def clear_taps(self) -> None:
        """Release the captured activation tensors."""
        self.taps.clear()

    def release(self) -> None:
        """Detach and free everything. The 'I am done' call."""
        self.detach()
        self.clear_taps()

    @property
    def attached(self) -> bool:
        return self._attached

    def __enter__(self) -> "Bus":
        return self.attach()

    def __exit__(self, *exc):
        self.detach()


def gpt2_blocks(model):
    return list(model.transformer.h)


def logits_hash(logits) -> str:
    """Content hash of a logit tensor.

    numpy-free on purpose: `torch` does not depend on numpy, so a torch-only
    environment must still be able to run the bit-exactness gate.
    """
    import struct

    import torch
    flat = logits.detach().to(torch.float32).cpu().contiguous().flatten()
    values = flat.view(torch.int32).tolist()
    return hashlib.sha256(struct.pack(f"<{len(values)}i", *values)).hexdigest()
