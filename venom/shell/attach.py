"""Attach VENOM to any HuggingFace causal LM.

The attach contract:

  * **discovery is automatic** - no hardcoded attribute paths
  * **attach never changes the output** - the hook returns None
  * **detach restores the host bit-exactly** - proven, see `tests/`
  * **detach always happens** - use it as a context manager

    with attach(model, tokenizer) as host:
        host.generate(prompt)          # VENOM is in the loop
    # model is bit-exact identical to before the `with` block

Usage:

    from venom.shell import attach

    with attach(model, tokenizer) as host:
        out = host.chat("what is the capital of france?")
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from venom.core.bus import Bus
from venom.core.discovery import fingerprint_universal, get_blocks
from venom.core.runtime import VenomConfig, VenomRuntime


class AttachedHost:
    """A host model with VENOM attached.

    Wraps the model; `generate` and `chat` run the reflex first and apply the
    memory context. The raw model is still reachable as `.model`.
    """

    def __init__(self, model, tokenizer=None, runtime: Optional[VenomRuntime] = None,
                 fingerprint: Optional[Dict[str, Any]] = None, bus: Optional[Bus] = None,
                 config: Optional[VenomConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.runtime = runtime or VenomRuntime(config)
        self.fingerprint = fingerprint or {}
        self.bus = bus
        self.probe = None
        self.dials = None
        self._attached = False
        self._baseline_logits = None
        self._last_inputs = None

    # ------------------------------------------------------------- lifecycle
    def attach(self) -> "AttachedHost":
        if self._attached:
            raise RuntimeError("already attached")
        if self.bus is None:
            blocks = get_blocks(self.model, self.fingerprint)
            self.bus = Bus(self.model, blocks)
        self.bus.attach()

        # a live logits probe so the self-read tools work
        from venom.core.probe import LogitProbe
        self.probe = LogitProbe(self.model)
        self.probe.attach()

        # bounded self-edit (only if the host opted in)
        if self.runtime.config.enable_dials:
            from venom.core.dials import DialManager
            self.dials = DialManager(
                self.model,
                get_blocks(self.model, self.fingerprint),
                max_delta=self.runtime.config.max_delta,
                max_active_dials=self.runtime.config.max_active_dials,
                drift_threshold=self.runtime.config.drift_threshold,
                verifier=self._drift_verifier,
            )

        self.runtime.bind_host(
            probe=self.probe,
            taps_provider=lambda: (self.bus.taps if self.bus else {}),
            norm=self.fingerprint.get("norm"),
            lm_head=self.fingerprint.get("lm_head"),
            tokenizer=self.tokenizer,
            dials=self.dials,
        )
        self._attached = True
        return self

    def detach(self) -> None:
        if self.probe is not None:
            self.probe.release()
            self.probe = None
        if self.dials is not None:
            self.dials.revert_all()
        if self.bus is not None and self.bus.attached:
            self.bus.detach()
        self._attached = False

    @property
    def attached(self) -> bool:
        return self._attached

    def __enter__(self) -> "AttachedHost":
        return self.attach()

    def __exit__(self, *exc) -> None:
        self.detach()

    # --------------------------------------------------------------- dials
    def _drift_verifier(self) -> dict:
        """Measure how far the logits moved since the last clean state.

        The verifier is what makes a self-edit safe: a dial that moves the
        host too far is rolled back automatically. It measures MAGNITUDE, not
        quality - a small drift is not automatically a better model.
        """
        if self._baseline_logits is None:
            # Deliberately NO "drift" key. The DialManager treats a verifier
            # without a numeric drift as a failure and rolls the dial back.
            # Reporting 0.0 here would be a lie: nothing was measured.
            return {"note": "no baseline captured yet - call capture_baseline() first"}
        import torch
        with torch.no_grad():
            current = self.model(**self._last_inputs).logits
        from venom.core.dials import logit_drift
        return {"drift": logit_drift(self._baseline_logits, current)}

    def capture_baseline(self, inputs) -> None:
        """Capture the reference logits the dial verifier compares against."""
        import torch
        with torch.no_grad():
            self._baseline_logits = self.model(**inputs).logits.detach().clone()
        self._last_inputs = inputs

    # ------------------------------------------------------------ inference
    def prompt_for(self, text: str) -> str:
        return self.runtime.prompt_for(text)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate with the reflex applied to the prompt."""
        return self._generate(self.prompt_for(prompt), **kwargs)

    def chat(self, user_text: str, system: Optional[str] = None, **kwargs) -> str:
        """Chat convenience wrapper: applies the reflex, then generates."""
        if self.tokenizer is None:
            raise RuntimeError("chat() needs a tokenizer; use generate() instead")
        prompt = user_text
        if system:
            prompt = f"{system}\n\n{user_text}"
        return self._generate(self.prompt_for(prompt), **kwargs)

    def _generate(self, prompt: str, **kwargs) -> str:
        if self.tokenizer is None:
            raise RuntimeError("generate() needs a tokenizer")
        ids = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in ids.items()}
        prompt_len = inputs["input_ids"].shape[-1]
        if self.dials is not None and self._baseline_logits is None:
            self.capture_baseline(inputs)
        out = self.model.generate(**inputs, **kwargs)
        # decode only the NEW tokens - the caller asked for a continuation,
        # not a repeat of the prompt
        text = self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        self.runtime.note_answer()
        return text

    # -------------------------------------------------------------- tools
    def call_tool(self, name: str, args: Optional[Dict[str, Any]] = None) -> dict:
        """Call one of VENOM's tools against this host."""
        return self.runtime.call_tool(name, args or {})

    def tool_defs(self) -> List[Dict[str, Any]]:
        return self.runtime.tool_defs()

    # ------------------------------------------------------------ telemetry
    def stats(self) -> dict:
        out = self.runtime.stats()
        out["attached"] = self._attached
        out["fingerprint"] = {
            k: v for k, v in self.fingerprint.items() if k not in ("lm_head", "norm")
        }
        return out


def attach(model, tokenizer=None, runtime: Optional[VenomRuntime] = None,
           config: Optional[VenomConfig] = None, min_confidence: float = 0.5,
           name_hint: str = "", capture: bool = False) -> AttachedHost:
    """Attach VENOM to a model. Returns an unattached handle - use it as a
    context manager so the host is always restored.

    `min_confidence` guards against attaching to a model whose structure could
    not be discovered confidently. Set it to 0.0 to attach anyway.
    """
    fp = fingerprint_universal(model, name_hint)
    if fp.get("confidence", 0.0) < min_confidence:
        raise RuntimeError(
            f"host discovery confidence {fp.get('confidence')} < {min_confidence}; "
            f"fingerprint={ {k: v for k, v in fp.items() if k not in ('lm_head', 'norm')} }. "
            "Pass min_confidence=0.0 to attach anyway."
        )
    host = AttachedHost(model, tokenizer, runtime, fp, config=config)
    if capture:
        host.attach()
    return host


def fingerprint(model, name_hint: str = "") -> Dict[str, Any]:
    """Describe a host without attaching. Safe to call on any model."""
    fp = fingerprint_universal(model, name_hint)
    return {k: v for k, v in fp.items() if k not in ("lm_head", "norm")}
