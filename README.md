# 🐍 THE VENOM-PARASITE — v0.1.2

**Training-free middleware that gives an open-weight LLM tool-access to its own internals.**

The host model is the agent. VENOM knows nothing on its own — it only serves tools.
Every intervention is measured, bounded, and reversible.

> **Research prototype.** The core logic is verified by unit tests, and the central
> trust contract — *attach → detach → bit-exact original logits* — is **measured on a
> real GPT-2**. Everything else (fix rate, harm rate, tool-call recall, latency under
> a real reflex) is **not yet measured** and needs a host model + a tool-calling
> model. See [`LIMITATIONS.md`](LIMITATIONS.md) for exactly what is and is not proven,
> and [`RESEARCH.md`](RESEARCH.md) for the prior art that bounds the novelty claim.

---

## What this is (and is not)

| | |
|---|---|
| **Host = agent** | The host model decides, reasons and calls tools. VENOM never answers. |
| **Zero gradient training** | No fine-tuning, no backprop through the host. Only prompt rewriting + closed-form statistics. |
| **Reflex layer** | Rule-based auto-trigger that runs *without* host permission (memory hit / low margin). |
| **Reversible** | `Bus.detach()` restores bit-exact original logits — **verified on GPT-2 124M**. `RollbackManager` snapshots any state. |
| **Abstain, don't guess** | Unknown or corrupted memory → explicit `abstain`, never a plausible fabrication. |

**Not this:** a trained parasite model, a model that overwrites host weights, a jailbreak,
a self-modifying agent, or a source of metacognition. The `surgeon/` module is
**report-only** — it never touches weights.

---

## Status

| Item | State |
|---|---|
| Core modules (no torch) | ✅ **44/44** unit tests pass |
| Regression tests for the self-review bug batch | ✅ **37/37** pass |
| Torch-level trust-contract tests | ✅ **8/8** pass |
| **Total** | ✅ **89/89** pass |
| Bit-exact attach/detach on a **real GPT-2 124M** | ✅ **verified** — `bit_exact_during=true`, `bit_exact_after_detach=true`, identical logits hash |
| Latency overhead of the bus (GPT-2, CPU) | ✅ **measured ≈ +2%** (median of randomised A/B; see caveat below) |
| MCP stdio server (`venom/server.py`) | ✅ implemented, JSON-RPC 2.0 conformance tested |
| Fix rate / harm rate on a real host | ❌ **not measured** — the bench pipeline is now correct and end-to-end, but needs a real host |
| Tool-call recall / precision | ❌ **not measured** — needs a tool-calling host (GPT-2 cannot call tools) |
| Dead-neuron / pruning surgery | ⏳ v0.3, report-only in v0.1 |

**Latency caveat:** on this shared CPU the measurement is noise-dominated — a *null*
hook (which does nothing) also measured between 0.87× and 1.09× across runs. The
defensible statement is "the full 12-layer tap costs about 2% on this machine",
**not** a precise number. See `experiments/e1b_latency_isolated.py`.

---

## Install & test

```bash
pip install -e ".[dev]"
pytest -q          # 89 tests; 81 of them need no torch and no network
```

Torch-dependent tests (the trust contract):

```bash
pip install -e ".[torch]"
pytest -q tests/test_bus_torch.py -v
```

Real-model experiments:

```bash
pip install -e ".[torch]"
python experiments/e1_bus_detach.py         # bit-exactness + latency, GPT-2
python experiments/e1b_latency_isolated.py   # randomised A/B latency isolation
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
| Budget | every token, cheap | max 3 calls per answer (re-armed per turn) |

The injector refuses **generic keys** (too short to be evidence of relevance) because
memory-induced tool-drift is a *measured* failure mode — see `RESEARCH.md` §2.3.

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
| Detach exactness | attach→detach→logit hash | bit-exact — ✅ **verified** |

None of these except detach exactness are measured yet.

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

`RESEARCH.md` §5 argues that **two of these are more likely to fire than not**,
based on published results.

---

## Roadmap

| Phase | Scope | Exit gate |
|---|---|---|
| P0 | 3 tools + MCP server + bench | fix rate > harm rate on 400 facts |
| P1 | reflex auto-trigger + abstain + budget | tool-call-when-needed ≥ 60%, harm ≤ 2% |
| P2 | live bus + hostmap + uncertainty sensors | ✅ *bus done, bit-exact verified*; reflex path ≤ 15% extra latency |
| P3 | report compiler + `why()` / `where()` | ablation: self-read helps or it gets cut |
| P4 | bounded self-edit dials + auto-rollback | detach → bit-exact logits |
| P5 | dead-cell report → prune → drift proof | 10k prompts, drift = 0 or pruning is cancelled |
| P6 | 6 model families + packaging | attach to a new model in ≤ 5 minutes |

---

## License

MIT — see [`LICENSE`](LICENSE).
