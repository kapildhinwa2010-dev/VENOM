# 🐍 THE VENOM-PARASITE — v0.1.1

**Training-free middleware that gives an open-weight LLM tool-access to its own internals.**

The host model is the agent. VENOM knows nothing on its own — it only serves tools.
Every intervention is measured, bounded, and reversible.

> **Research prototype.** The core logic is verified by unit tests. The model-level
> claims (fix rate, harm rate, latency, bit-exact detach) are **not yet measured** —
> they need a GPU/CPU host with `torch` + `transformers`. See [`LIMITATIONS.md`](LIMITATIONS.md)
> for exactly what is and is not proven. Do not quote percentages from the plan as results.

---

## What this is (and is not)

| | |
|---|---|
| **Host = agent** | The host model decides, reasons and calls tools. VENOM never answers. |
| **Zero gradient training** | No fine-tuning, no backprop through the host. Only prompt rewriting + closed-form statistics. |
| **Reflex layer** | Rule-based auto-trigger that runs *without* host permission (memory hit / low margin). |
| **Reversible** | `Bus.detach()` must restore bit-exact original logits. `RollbackManager` snapshots any state. |
| **Abstain, don't guess** | Unknown or corrupted memory → explicit `abstain`, never a plausible fabrication. |

**Not this:** a trained parasite model, a model that overwrites host weights, a jailbreak,
a self-modifying agent. The `surgeon/` module is **report-only** — it never touches weights.

---

## Status

| Item | State |
|---|---|
| Core modules (no torch) | ✅ implemented, **42/42 unit tests pass** |
| MCP stdio server (`venom/server.py`) | ✅ implemented, tested |
| Attach/detach bus | ⚠️ implemented, **runtime unverified** (needs torch) |
| E1–E6 model experiments | ❌ **not run** — no torch/model in this environment |
| Dead-neuron / pruning surgery | ⏳ v0.3, report-only in v0.1 |

---

## Install & test

```bash
pip install -e ".[dev]"
pytest -q          # 42 tests, no torch, no network, no model download
```

Torch-dependent extras (only needed for model-level experiments):

```bash
pip install -e ".[torch]"
```

---

## Module map

```
venom/
├── memory/store.py       exact recall + digest guard + bounded LRU + abstain
├── reflex/triggers.py    memory-hit / low-margin triggers (rule-based, per-token cheap)
├── reflex/budget.py      hard cap on tool calls per answer
├── reflex/injector.py    memory→prompt rewrite for hosts that cannot call tools
├── tools/schema.py       MCP tool schemas (3 P0 tools + planned later tools)
├── tools/handlers.py     tool dispatch + budget enforcement
├── server.py             MCP-style JSON-RPC 2.0 over stdio, zero dependencies
├── core/bus.py           read-only forward hooks on every transformer block
├── core/hostmap.py       auto-detects GPT-2 / Llama / Qwen2 / Mistral / Phi / Gemma
├── core/session.py       JSON-serialisable session state (memory + budget + telemetry)
├── sensors/uncertainty.py  per-token entropy / margin / confidence (pure python)
├── sensors/logitlite.py    logit lens (requires torch)
├── report/compiler.py    raw telemetry → one-line natural language
├── verify/drift.py       baseline vs VENOM: fixed / harmed / shifted / abstained
├── verify/rollback.py    snapshot + restore for any state
├── surgeon/report.py     dead-neuron, redundancy, compressibility — report only
└── bench/correction_bench.py  fix-rate vs harm-rate harness
```

---

## P0 tool API

| Tool | Tier | Purpose |
|---|---|---|
| `venom.recall(query)` | 1+2 | Exact lookup. Returns `status: abstain` when unknown or corrupted. |
| `venom.remember(key, value)` | 1+2 | Store a fact. Bounded LRU, digest-guarded. |
| `venom.time_now()` | 1+2 | UTC timestamp (hosts have no clock). |

**Tier 1** = open-weight hosts (all tools) · **Tier 2** = API-only hosts (memory + verify).

Planned (not implemented): `uncertainty`, `check_myself`, `verify`, `why`, `where`,
`dial`, `audit`.

---

## Reflex vs agent

| | Reflex | Agent |
|---|---|---|
| Driven by | VENOM, automatically | The host, via a tool call |
| Trigger | factual span + low margin + memory hit | the model's own choice |
| Intelligence | none (rule-based) | the host's |
| Cost | ~ms | extra forward passes |
| Why needed | the model does not know what it does not know | depth and reasoning |
| Budget | every token, cheap | max 3 calls per answer |

---

## Metrics (targets, not results)

| Metric | How measured | Target |
|---|---|---|
| Fix rate | 200 wrong facts → how many became right | ≥ 40% |
| Harm rate | 200 right facts → how many broke | ≤ 2% |
| Tool-call recall | needed a call, called? | ≥ 60% |
| Tool-call precision | called, needed? | ≥ 50% |
| Abstain correctness | corrupted fact → stayed quiet? | ≥ 95% |
| Latency ratio | venom / baseline | reflex ≤ 1.15×, agent ≤ 4× |
| Detach exactness | attach→detach→logit hash | bit-exact |

None of these are measured yet. `experiments/e1_bus_detach.py` is the first one that can be.

---

## Kill criteria

Honest failure conditions, written down before the results so they cannot be quietly moved:

| When | Signal | Action |
|---|---|---|
| P0 | fix ≤ harm | the memory injection is wrong — debug, do not build further |
| P1 | tool-call recall < 30% | thicken the reflex, demote the agent path |
| P3 | introspection ablation gain = 0 | remove the self-read tools from the product, keep them offline |
| P4 | rollback fails | self-edit stays off, permanently |
| P5 | drift ≠ 0 after pruning | pruning is not lossless → report-only mode, permanently |

---

## Roadmap

| Phase | Scope | Exit gate |
|---|---|---|
| P0 | 3 tools + MCP server + bench | fix rate > harm rate on 400 facts |
| P1 | reflex auto-trigger + abstain + budget | tool-call-when-needed ≥ 60%, harm ≤ 2% |
| P2 | live bus + hostmap + uncertainty sensors | reflex path ≤ 15% extra latency |
| P3 | report compiler + `why()` / `where()` | ablation: self-read helps or it gets cut |
| P4 | bounded self-edit dials + auto-rollback | detach → bit-exact logits |
| P5 | dead-cell report → prune → drift proof | 10k prompts, drift = 0 or pruning is cancelled |
| P6 | 6 model families + packaging | attach to a new model in ≤ 5 minutes |

---

## License

MIT — see [`LICENSE`](LICENSE).
