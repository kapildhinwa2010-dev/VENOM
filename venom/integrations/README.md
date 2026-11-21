# Sibling integrations

The master plan's TABLE 2 named 20 sibling repositories as the sources for
the consolidation. This package **calls those repos** instead of
re-implementing them. Every adapter has the same shape:

```python
<name>_available() -> bool
<name>_run(...) -> {"ok": bool, "source": "<REPO>", ...}
```

`ok: False` means the sibling is missing or failed — never an exception.
VENOM installs and runs with **none** of the siblings present; the built-in
implementations under `venom/sensors/` and `venom/memory/` are the fallback.

## The repos and what they provide

| Sibling repo | Integration key | Provides | VENOM uses it for |
|---|---|---|---|
| NEVER_LIE | `never_lie` | digest-guarded memory + fuzz | `venom/memory/` |
| FactBase | `factbase` | fact triples | `venom/memory/` |
| KNOW-VAULT | `know_vault` | key/activation store | `venom/memory/` |
| fluidtech | `fluidlayer` | combined layer, deployment | `venom/memory/`, `venom/shell/` |
| INNER-VOTE- | `inner_vote` | layer-wise consensus | `venom.check_myself` |
| LOGIT-STREAM | `logit_stream` | logit tracing | `venom.why` |
| STATE-MIRROR | `state_mirror` | state comparison | `venom.check_myself` |
| VARIEANCE-LIES | `variance_lies` | variance / rank analysis | `venom/report/` |
| LAYER-CENSUS- | `layer_census` | effective rank, layer stats | `venom/sensors/audit.py` |
| DEAD-CELL | `dead_cell` | dead rows/columns/neurons | `venom.audit` |
| TWIN-TOWER | `twin_tower` | weight redundancy | `venom/sensors/audit.py` |
| SKIP-RUN- | `skip_run` | layer skipping analysis | `venom/surgeon/` |
| POWERLAW-PROBE | `powerlaw_probe` | scaling probes | `venom/surgeon/` |
| BUS-TRACE | `bus_trace` | hook bus analysis | `venom/core/bus.py` |
| TRACE-PATH | `trace_path` | activation patching | `venom.where` |
| GEOMETRY-SCAN | `geometry_scan` | representation geometry | `venom/sensors/attribution.py` |
| Draft_ponit_harness | `drift_point` | verification harness | `venom.verify` |
| NOISE-STORM | `noise_storm` | robustness under noise | `venom/verify/drift.py` |
| TMU | `tmu` | internal clock | `venom.time_now` |
| ComplexityRouter | `complexityrouter` | routing by complexity | `venom/reflex/` |

## Install

```bash
pip install -e ".[integrations]"          # from PyPI-style git URLs
# or, from a checkout of the siblings:
export VENOM_SIBLINGS_DIR=/path/to/siblings
```

`VENOM_SIBLINGS_DIR` points at a directory holding the sibling checkouts; the
loader falls back to it when a package is not pip-installed.

## What these integrations do NOT prove

- **NEVER_LIE's digest guard is corruption detection, not security.** The
  sibling says so itself: use an HMAC for adversarial settings.
- **TRACE-PATH's activation patching localises a *fact*, not a *decision*.**
  The logit-lens `why`/`where` in VENOM are readouts, not causal proofs.
- **DEAD-CELL's "dead" neurons are dead only for the prompts that were
  run.** A unit that never fired on 10 prompts has not been shown to be
  prunable — see `experiments/e4_gated_prune.py`, where the gate correctly
  refuses.
- **No integration inherits a performance claim.** They are analysis tools.
