"""Time integration: TMU (Time Management Unit).

VENOM already serves a UTC timestamp (`venom.time_now`).  TMU adds two
things VENOM does not have:

* an **internal clock** that measures the time a host spends between
  steps, with an optional per-step budget and pressure signal;
* a **time management unit**, a small torch module that turns a timestamp
  into a conditioning vector and modulates the hidden states.

What this does NOT prove: that conditioning a model on the time improves
it.  The TMU's projection is zero-initialised, so at initialisation it is
an identity - the integration shows the wiring, not an effect.  Nothing
here claims a latency or quality benefit.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from venom.integrations._base import guard, ok, require

SOURCE_TMU = "TMU"


def tmu_available() -> bool:
    """Is the TMU sibling importable?"""
    from venom.integrations._loader import load
    return load("tmu") is not None


def tmu_run(ticks: int = 2, hidden_dim: int = 16, seq_len: int = 4,
            timestamp: Optional[float] = None,
            device: str = "cpu") -> Dict[str, Any]:
    """Run the sibling's internal clock and time-management unit.

    What it does: ticks TMU's ``AdvancedInternalClock`` a few times (each
    tick reports the elapsed and delta time, the token count and the
    budget pressure) and runs its ``TimeManagementUnit`` forward on a
    random hidden-state tensor with a timestamp, reporting the shape of
    the time embedding and whether the modulated output equals its input.

    What it does NOT prove: that time conditioning helps a model.  The
    unit's output projection is zero-initialised, so at initialisation the
    modulation is the identity - the numbers show the module was wired
    up, not that it changes behaviour.  It is also not a clock for the
    host model: it measures the adapter's own step time.
    """
    def _run():
        module, failure = require("tmu", SOURCE_TMU)
        if failure is not None:
            return failure
        import torch

        clock_mod = module
        try:  # the clock lives in tmu.clock; the package re-exports it
            from tmu.clock import AdvancedInternalClock
        except Exception:  # noqa: BLE001 - fall back to the re-export
            AdvancedInternalClock = clock_mod.AdvancedInternalClock

        clock = AdvancedInternalClock(device=device)
        snapshots = []
        for _ in range(max(1, int(ticks))):
            snap = clock.tick(tokens=1)
            snapshots.append({
                "elapsed_s": round(float(snap.elapsed_s), 6),
                "delta_ms": round(float(snap.delta_ms), 6),
                "tokens": int(snap.tokens),
                "step": int(snap.step),
                "pressure": round(float(snap.pressure), 6),
                "confidence": round(float(snap.confidence), 6),
            })

        tmu_state: Dict[str, Any] = {"built": False}
        try:
            from tmu.core import TimeManagementUnit

            unit = TimeManagementUnit(hidden_dim=int(hidden_dim))
            torch.manual_seed(0)
            x = torch.randn(1, int(seq_len), int(hidden_dim))
            stamp = timestamp if timestamp is not None else 0.0
            embed = unit.encode_time(stamp, batch_size=1)
            out = unit(x, stamp)
            tmu_state = {
                "built": True,
                "time_feature_dim": int(embed.shape[-1]),
                "input_shape": list(x.shape),
                "output_shape": list(out.shape),
                "identity_at_init": bool(torch.allclose(out, x)),
                "note": (
                    "The scale projection is zero-initialised, so at "
                    "initialisation the unit is the identity. That is the "
                    "wiring check, not an effect."
                ),
            }
        except Exception as exc:  # noqa: BLE001 - the clock part still ran
            tmu_state = {"built": False, "error": "{}: {}".format(type(exc).__name__, exc)}

        return ok(
            SOURCE_TMU,
            ticks=max(1, int(ticks)),
            clock_snapshots=snapshots,
            time_management_unit=tmu_state,
            note=(
                "The clock measures the adapter's own step time on this "
                "machine; it is not a measurement of the host model's "
                "latency. The time-conditioning unit is an identity at "
                "initialisation."
            ),
        )
    return guard(SOURCE_TMU, _run)
