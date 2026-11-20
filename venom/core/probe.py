"""Live logits capture — the sensor that makes self-reading possible.

The `Bus` taps *hidden states* (per block). A self-read tool also needs the
**logits** — the actual distribution over the vocabulary — so it can answer
"how uncertain are you?" and "which token did you pick?".

`LogitProbe` hooks the top-level model and keeps the last position's logits.
It is deliberately cheap: one tensor per forward pass, not per token, unless
`stream=True` is requested.

Contract used by the tools:

    probe = LogitProbe(model)
    probe.attach()
    ... model(...) ...
    probe.last_logits()      # tensor [vocab] or None
    probe.last_uncertainty()  # dict from venom.sensors.uncertainty
    probe.detach()
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from venom.sensors.uncertainty import token_uncertainty


class LogitProbe:
    """Captures the logits of the last position of every forward pass."""

    def __init__(self, model, stream: bool = False):
        self.model = model
        self.stream = stream
        self._logits: Optional[Any] = None
        self._stream: List[Any] = []
        self._handle = None
        self._attached = False
        self.calls = 0

    # ------------------------------------------------------------- lifecycle
    def attach(self) -> "LogitProbe":
        if self._attached:
            raise RuntimeError("probe already attached")
        self._handle = self.model.register_forward_hook(self._hook)
        self._attached = True
        return self

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._attached = False

    def release(self) -> None:
        self.detach()
        self._logits = None
        self._stream = []

    @property
    def attached(self) -> bool:
        return self._attached

    def __enter__(self) -> "LogitProbe":
        return self.attach()

    def __exit__(self, *exc) -> None:
        self.release()

    # ----------------------------------------------------------------- hook
    def _hook(self, module, inputs, output):
        logits = None
        if isinstance(output, tuple):
            for item in output:
                if hasattr(item, "shape") and len(item.shape) >= 2:
                    logits = item
                    break
        elif hasattr(output, "shape"):
            logits = output
        elif hasattr(output, "logits"):
            logits = output.logits

        if logits is None:
            return None
        try:
            last = logits[0, -1, :] if logits.dim() >= 3 else logits[-1, :]
        except Exception:  # noqa: BLE001
            return None

        self._logits = last.detach()
        self.calls += 1
        if self.stream:
            self._stream.append(self._logits)
        return None

    # --------------------------------------------------------------- reading
    def last_logits(self):
        """Logits of the most recent forward pass, or None."""
        return self._logits

    def last_uncertainty(self, k: int = 5) -> Optional[Dict[str, float]]:
        """Per-token uncertainty of the most recent forward pass."""
        if self._logits is None:
            return None
        return token_uncertainty(self._logits.float().tolist(), k)

    def stream_uncertainty(self, k: int = 5) -> List[Dict[str, float]]:
        """Uncertainty for every captured position (requires stream=True)."""
        return [token_uncertainty(row.float().tolist(), k) for row in self._stream]

    def top_tokens(self, tokenizer, k: int = 5) -> List[Dict[str, Any]]:
        """Top-k tokens of the most recent forward pass, decoded."""
        if self._logits is None:
            return []
        values, indices = self._logits.topk(min(k, self._logits.shape[-1]))
        return [
            {"token": tokenizer.decode([int(i)]), "id": int(i), "logit": round(float(v), 4)}
            for v, i in zip(values, indices)
        ]
