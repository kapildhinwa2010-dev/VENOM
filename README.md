# 🐍 THE VENOM-PARASITE — v0.4.0

**A plug-in layer for any AI.** Training-free middleware that gives a model
memory, a clock, and read-access to its own forward pass — without changing
the model, and without training anything.

The host model stays the agent. VENOM serves tools. Every intervention is
measured, bounded, and reversible.

> **Research prototype.** The core logic is verified by unit tests. The central
> trust contract — *attach → detach → bit-exact original logits* — is
> **measured on real models across three architectures**. The remaining
> model-level claims (fix rate, harm rate, tool-call recall) are **not yet
> measured**. See [`LIMITATIONS.md`](LIMITATIONS.md) for exactly what is and is
> not proven, and [`RESEARCH.md`](RESEARCH.md) for the prior art that bounds
> the novelty claim.

---

## The tool surface (v0.3.0)

All fourteen tools the plan's TABLE 4 asked for, plus state control:

| Tool | Tier | What it does |
|---|---|---|
| `venom.recall(query)` | 1+2 | Exact lookup. Abstains when unknown or corrupted. |
| `venom.remember(key, value)` | 1+2 | Store a fact. Bounded LRU, digest-guarded. |
| `venom.time_now()` | 1+2 | UTC timestamp (hosts have no clock). |
| `venom.uncertainty()` | 1 | Entropy / margin / confidence of the last token. |
| `venom.check_myself()` | 1 | Layer-wise consensus: how many layers' logit lens agree with the final top-1. |
| `venom.why(token)` | 1 | Per-layer logit-lens trajectory of a token, and the layer where it overtook the competitor. |
| `venom.where(concept)` | 1 | Which layer carries the information about a concept. |
| `venom.dial(layer, delta)` | 1 | Bounded self-edit. Auto-rolls-back if the drift is too large. |
| `venom.audit()` | 1 | Dead neurons, redundancy, compressibility. **Report only.** |
| `venom.verify(answer)` | 1+2 | verified / contradicted / unknown against memory. |
| `venom.forget(key)` | 1+2 | Retract a fact — the memory is not append-only. |
| `venom.snapshot(tag)` | 1+2 | Take a rollback snapshot. |
| `venom.restore(id)` | 1+2 | Restore a snapshot. |
| `venom.report()` | 1+2 | A one-line human-readable state report. |

**Tier 1** = open-weight hosts (everything) · **Tier 2** = API-only hosts
(memory + verify + state). The proxy serves Tier 2 by default, because an API
host has no access to its own internals — offering it `venom.uncertainty`
would be a lie.

### The self-read tools, measured on a real GPT-2

`python experiments/e3_selfread.py --model gpt2`:

| Tool | Result |
|---|---|
| `venom.uncertainty` | entropy **5.9984**, margin **0.0366**, confidence **0.0846**, `is_uncertain: true` — GPT-2 really is unsure after "The capital of France is" |
| `venom.check_myself` | **1 of 12 layers** agree with the final top-1 → verdict "LATE: the final token was decided by the last layers only" |
| `venom.where(" France")` | best represented at **layer 9** (rank 0) |
| `venom.why(" Paris")` | target logit **−8.84** at layer 0 and never overtakes → `decision_layer: null` — GPT-2 did not predict "Paris" at all |
| `venom.audit` | **8 of 11** layer pairs are near-duplicates (cosine > 0.99); 5 of 6 sampled layers are compressible |
| `venom.verify` | `verified` |

Every one of these carries its own caveat in the payload. `check_myself` in
particular states that layer agreement is **not** introspection — a supervised
probe beats a model's own self-report 98.3% to 11.1% (arXiv 2611.15267) and
activation-level detection is content-agnostic (arXiv 2603.05414).

### The P0 gate, measured on a real host

`python experiments/e5_fix_harm.py --model gpt2`:

| | fix rate | harm rate | baseline acc | venom acc |
|---|---|---|---|---|
| VENOM arm | **0.417** | **0.375** | 0.40 | 0.45 |
| neutral-memory control | 0.0 | 0.0 | 0.40 | 0.40 |

The gate is *"fix rate > harm rate"* — it passes. The **neutral-memory
control is clean**: the same bank of facts, stored under keys that match
nothing, changed **zero** answers. Without that control, the fix rate could
just be "any extra text helps".

**What this does not show:** that the layer improves a real workload. A small
model on mundane world-knowledge facts, where the fix rate is largely the
model *copying the injected context*. That is a property of the benchmark.

### Gated self-surgery

`python experiments/e4_gated_prune.py --model gpt2` — **no unit was prunable.**
Every unit of every projection fired at least once over the 10 plan prompts,
so the gate refused to prune. That is the correct outcome: the plan's P5 gate
assumes 10k prompts, and 10 prompts cannot establish that a unit is dead.

### Bounded self-edit, measured on a real GPT-2

`python experiments/e3_selfread.py` → `venom dial --model gpt2 --layer 3`:

| delta | measured drift | outcome |
|---|---|---|
| 0.01 | 0.00646 | applied, verified |
| 0.02 | 0.01240 | applied, verified |
| 0.03 | 0.01794 | applied, verified |
| 0.05 | 0.02809 | **auto-rolled back** (threshold 0.02), host restored bit-exactly |

A dial that moves the host too far is undone automatically. A verifier that
cannot measure is treated as a failure, not as "no drift".

---

## Plug it into anything

**Open-weight models (local):**

```python
from venom.shell import attach

with attach(model, tokenizer) as host:
    host.remember("capital of france", "Paris")
    print(host.chat("what is the capital of france?"))
# model is bit-exact identical to before the `with` block
```

**Any API model (closed — GPT, Claude, Gemini, Llama-API, vLLM, Ollama):**

```bash
venom proxy --upstream https://api.openai.com/v1 --upstream-model gpt-4o-mini
# point your client at http://127.0.0.1:8765/v1
```

**No install at all — one file:**

```python
import venom                      # a single copied file, no dependencies
host = venom.wrap(model, tokenizer)
```

**As MCP tools for any MCP client:**

```bash
venom serve              # all 14 tools over MCP stdio
```

**Drive it from the terminal:**

```bash
venom probe --model gpt2        # one pass, then report what the self-read tools see
venom chat --model gpt2         # interactive chat with VENOM attached
venom dial --model gpt2 --layer 3 --delta 0.05   # bounded self-edit + drift
```

**From the command line:**

```bash
venom fingerprint --model gpt2      # describe a host without attaching
venom attach --model gpt2           # attach, generate, detach, prove bit-exact
venom proxy --upstream URL          # wrap any OpenAI-compatible host
venom tools                         # print the MCP schemas
```

---

## What is verified

| | |
|---|---|
| Test suite | ✅ **179/179 pass** (core + universal + tools + dials + prune + integrations) |
| **Architecture sweep** | ✅ **22 of 30** models discovered across **18 distinct families** — every failure was an access problem (gated/renamed repo), **zero discovery failures** |
| **Sibling-repo integration** | ✅ **20 sibling repos** wired as optional adapters; VENOM still runs with none installed |
| **Bit-exact attach → detach on real models** | ✅ **GPT-2 124M, Qwen2.5-0.5B-Instruct, SmolLM2-135M-Instruct** — `bit_exact_during=true`, `bit_exact_after_detach=true`, identical logits hash |
| Architecture discovery | ✅ **no hardcoded paths** — blocks/head/norm found by walking the module graph; `confidence` reported |
| MCP stdio server | ✅ JSON-RPC 2.0 conformance tested (notification vs request, id preservation, error codes) |
| API proxy | ✅ end-to-end tested against a fake upstream (injects AND forwards) |
| Single-file standalone | ✅ bit-exact on all three models, zero install |
| Bus latency overhead (GPT-2, CPU) | ≈ **+2%** (randomised A/B, null-hook control — noise-dominated, see caveat) |
| Fix rate / harm rate on a real host | ❌ **not measured** |
| Tool-call recall / precision | ❌ **not measured** — needs a tool-calling host |
| Introspection gain | ❌ **not measured**, and the literature says be sceptical (see `RESEARCH.md` §2.2) |

**Latency caveat:** on a shared CPU the measurement is noise-dominated — a *null*
hook also measured between 0.87× and 1.09×. The defensible statement is "the
full tap costs about 2% on this machine", **not** a precise number.

---

## How the universality works

`venom/core/discovery.py` replaces the old hardcoded family table. Instead of
knowing that GPT-2 has `transformer.h` and Llama has `model.layers`, it walks
the module graph:

1. find every `nn.ModuleList` whose children look like transformer blocks
   (they have an attention child **and** an MLP/FFN child);
2. score the candidates — block ratio, internal richness, name hints, depth;
3. take the winner, and find the LM head and the final norm without assuming
   a name.

The result: a model whose architecture this code has never seen still attaches.
`fingerprint()` reports a `confidence` score, and `attach()` **refuses** to
attach below a threshold rather than guessing.

```bash
$ venom fingerprint --model gpt2
{
  "family": "gpt2",
  "confidence": 1.0,
  "blocks_path": "transformer.h",
  "num_layers": 12,
  "lm_head_path": "get_output_embeddings()",
  "norm_path": "transformer.ln_f"
}
```

---

## Architecture

```
venom/
├── integrations/        ★ 20 sibling repos as optional adapters
├── core/
│   ├── discovery.py     ★ universal architecture discovery (no hardcoded paths)
│   ├── runtime.py       ★ the framework-agnostic core (VenomRuntime)
│   ├── bus.py           read-only forward hooks; detach restores bit-exactly
│   ├── hostmap.py       legacy family table (kept for compatibility)
│   └── session.py       JSON-serialisable session state
├── shell/
│   ├── attach.py        ★ attach/detach any HF causal LM
│   ├── proxy.py         ★ OpenAI-compatible proxy for any API host
│   └── cli.py           ★ the `venom` command
├── core/
│   ├── probe.py         ★ live logits capture (the self-read sensor)
│   └── dials.py         ★ bounded, verified, revertible self-edit
├── memory/store.py      exact recall + digest guard + bounded LRU + abstain
├── reflex/              triggers, budget, injector (memory → prompt)
├── sensors/             uncertainty, logit lens
├── report/compiler.py   telemetry → natural language
├── verify/              drift (fixed/harmed/shifted/abstained), rollback
├── surgeon/report.py    dead-neuron analysis — report only, never edits weights
├── surgeon/prune.py     ★ gated self-surgery: apply -> verify -> restore
├── bench/               fix-rate vs harm-rate harness + the fact set
├── server.py            MCP JSON-RPC 2.0 stdio server
└── standalone.py        ★ single-file drop-in, no install
```

★ = new in v0.2.0 / v0.3.0

---

## The runtime

`VenomRuntime` is the whole layer minus the model. It owns the memory, the
budget, the reflex and the telemetry — and it produces a *prompt* and reads
*numbers*. It never generates the answer and never touches host weights.

```python
from venom.core.runtime import VenomRuntime, VenomConfig

rt = VenomRuntime(VenomConfig(max_calls=3))
rt.remember("capital of france", "Paris")

rt.observe("what is the capital of france?")
# {'fire': True, 'reason': 'memory_hit', 'key': 'capital of france', ...}

rt.prompt_for("what is the capital of france?")
# 'CONTEXT: Paris | what is the capital of france?'

rt.call_tool("venom.recall", {"query": "capital of france"})
# {'ok': True, 'status': 'hit', 'value': 'Paris', ...}
```

---

## The reflex vs the agent

| | Reflex | Agent |
|---|---|---|
| Driven by | VENOM, automatically | The host, via a tool call |
| Trigger | factual span + low margin + memory hit | the model's own choice |
| Intelligence | none (rule-based) | the host's |
| Cost | ~ms | extra forward passes |
| Budget | every token, cheap | max 3 calls per answer (re-armed per turn) |

The injector refuses **generic keys** (too short to be evidence of relevance)
because memory-induced tool-drift is a *measured* failure mode — see
`RESEARCH.md` §2.3.

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
| Detach exactness | attach→detach→logit hash | bit-exact — ✅ **verified on 3 architectures** |

---

## Kill criteria

Written down before the results so they cannot be quietly moved:

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

## Install

```bash
pip install -e ".[dev]"          # core, zero dependencies
pytest -q                        # 108 tests; 100 need no torch and no network

pip install -e ".[torch]"        # + torch, transformers, numpy
pip install -e ".[dev,torch]"
```

Experiments:

```bash
python experiments/e1_bus_detach.py         # bit-exactness + latency (GPT-2)
python experiments/e1b_latency_isolated.py   # randomised A/B latency isolation
python experiments/e2_universal_attach.py    # universal attach across 3 architectures
python experiments/e3_selfread.py            # the self-read tools on a real model
python experiments/e2b_architecture_sweep.py # 30 architectures, meta-device
python experiments/e4_gated_prune.py         # gated self-surgery + drift proof
python experiments/e5_fix_harm.py           # the P0 gate + neutral-memory control
```

---

## The surgery stack (new in v0.5.0)

The plan's "dead neuron removal" is now implemented, end to end, and it comes
with the write surface it needs.

```bash
venom census  --model gpt2 --prompts "a||b||c"      # per-neuron census
venom kill    --model gpt2 --prompts "calib||..." \
              --heldout "held||..." --mode remove   # kill + verify + roll back
venom install /path/to/model                        # make it addressable
venom write   --model gpt2 --action describe        # the full write surface
```

| Module | What it does |
|---|---|
| `venom/surgeon/census.py` | Per-**neuron** activation census. Counts, for every FFN unit, how many prompts made it fire. Separates exactly-dead from rarely-active. |
| `venom/surgeon/write.py` | The **full write surface**. `zero_units`, `remove_units` (a real shape rewrite), `set_weight`, `set_attr`, `load_state_dict`. Every mutation is journaled and snapshotted; a shape rewrite is undoable. |
| `venom/surgeon/kill.py` | The killer: census → refuse if nothing is dead → apply → verify the drift on a **disjoint held-out set** → roll back if it is over the threshold. |
| `venom/install.py` | Installs a `venom.manifest.json` into a model directory: the fingerprint, the writable surface, the capabilities, the guarantees and the non-guarantees. |

Three MCP tools were added: `venom.census`, `venom.kill`, `venom.write`
(Tier 1 only - they need the host's internals).

**What it is not.** `venom.write` can damage a model. It offers
*reversibility and an audit trail*, not *prevention*. Pruning buys memory and
compute, not intelligence.

### Measured on real GPT-2

| | Result |
|---|---|
| Census, 12 calibration prompts, 36 864 FFN units | **0** exactly-dead units, **0** rarely-active |
| Kill on real GPT-2 | correctly **refused** - nothing was dead, nothing was modified |
| Kill on a controlled fixture (12 units zeroed on purpose) | found all 12, removed them, drift on the held-out set **0.0**, top-1 agreement **1.0** |
| Shape rewrite, `mode="remove"`, 8 units of `mlp.c_fc` | `3072 → 3064`, consumer `mlp.c_proj` reshaped with it, model still runs, drift **0.2255**, top-1 agreement **1.0** |
| Restore after the shape rewrite | **bit-exact** - drift back to **0.0** |
| Install | manifest written, **49** writable modules, **no weight changed**, reload through the manifest gives a working write surface |

## The automated pipeline (v0.6.0)

One call runs every feature, feeds the answers into **one ordered ladder**,
and applies what the ladder decides. Nobody hand-picks neurons.

```bash
venom auto --model gpt2 \
  --prompts "calib one||calib two" \
  --heldout "held one||held two" \
  --mode remove --dry-run
```

The ladder is a MATLAB-shaped chain — **the first matching rule wins**:

```
1. if the unit fired on exactly zero calibration samples  -> REMOVE
2. elif it fired on < 0.1% of samples                    -> ZERO
3. elif its layer is a near-duplicate of another layer    -> FLAG_LAYER
4. elif its layer is compressible and flat-tailed        -> FLAG_COMPRESS
5. else                                                  -> KEEP
```

Only `REMOVE` and `ZERO` touch weights, and both go through the usual gate:
apply → measure the drift on a **disjoint held-out set** → roll back if it is
over the threshold. `FLAG_LAYER` and `FLAG_COMPRESS` are report-only.

Every stage is **native** — no sibling repo needed:

| Stage | Native implementation | What it asks |
|---|---|---|
| census | `venom/surgeon/census.py` | which units fired on how many prompts |
| redundancy | `venom/sensors/audit.py` | which consecutive layers look alike |
| compressibility | `venom/sensors/audit.py` | which layers have a low effective rank |
| twin towers | `venom/surgeon/pipeline.py` | which layers are near-duplicates (cross-layer) |
| skip-run | `venom/surgeon/pipeline.py` | does skipping the block change the output? |
| power-law | `venom/surgeon/pipeline.py` | is the spectrum structured or flat? |

### Measured on two small models

| | GPT-2 124M | SmolLM2-135M-Instruct |
|---|---|---|
| FFN units scanned | 36 864 | 92 160 |
| exactly-dead units | **0** | **0** |
| rarely-active units | **0** | **0** |
| near-duplicate layer pairs | 8 | 22 |
| twin-tower clusters | 1 (max cos **0.9999**) | 2 (max cos **0.99999**) |
| compressible layers | 5 of 6 | 5 of 6 |
| skip-run on layer 0 | Δ **83.56**, **not** skippable | Δ **37.81**, **not** skippable |
| ladder: REMOVE / ZERO | **0 / 0** | **0 / 0** |
| ladder: FLAG_LAYER / FLAG_COMPRESS | 24 576 / 6 144 | 67 584 / 6 144 |
| ladder: KEEP | 6 144 | 18 432 |
| weights changed | **none** | **none** |

**The interesting part is a contradiction the pipeline surfaces.** Redundancy
says 8 near-duplicate layer pairs (cosine > 0.99), but the skip-run test says
skipping a block changes the output by **83.56** in the logits — the layer is
doing real work. A pipeline that trusted redundancy alone would have flagged
8 layers as removable. That is the trap the plan's "redundant layers" feature
would have walked into, and it is why the ladder treats redundancy as
`FLAG_LAYER` (report-only) rather than as a licence to cut.

## The plan's exit gates — measured

Run them yourself:

```bash
python experiments/e8_all_gates.py --gates recall,abstain,latency --limit 400
python experiments/e8_all_gates.py --gates facts --model gpt2 --limit 400
python experiments/e8_all_gates.py --gates multimodel --limit 120
```

| Gate | Target | Measured | |
|---|---|---|---|
| Content-agnostic control | the scientific claim | **GENERIC** — an input-level perturbation moved the self-read signal **5× (GPT-2) to 48× (Qwen)** more than an activation-level one. The self-report tracks the input, not the computation. **Anomaly detector, not introspection.** (E5) | ✅ |
| Tool-call recall | ≥ 60% | exact **400/400 (100.0%)**, fuzzy (the question as a user would type it) **348/400 = 87.0%** | ✅ |
| Abstain under corruption | ≥ 95% | **400/400 = 100%** abstained, every one `digest_mismatch`; a **hand-edited file** also abstained **400/400 = 100%** | ✅ |
| Reflex latency overhead | ≤ 15% | real 16-token generation, ABBA-interleaved, medians: base 1359.3 ms → attached 1369.2 ms → hook **+0.72%**, one tool call **+1.77%**, reflex evaluation **0.002%** → **layer overhead +0.73%** | ✅ |
| 400 facts, fix rate > harm rate | the P0 gate | see `e8_gates.json` | ⏳ |
| Multi-model | 5+ models | see `e8_gates.json` | ⏳ |

**A measurement note worth keeping.** The latency gate first reported
**28.65%** and failed. The measurement was wrong, not the layer: the hooked
arm ran a tool call as well, and a tool call runs *its own forward pass*, so
the tool's price and the layer's overhead were being added together. Split
apart, the layer's own cost is **+0.73%**. An earlier version of the same
measurement reported **−98%** — the hooked arm never generated at all, so it
was "faster than no layer", which is how you spot a measurement that is not
measuring anything.

## The write-up

[`PAPER.md`](PAPER.md) is the honest research write-up: what is measured, what
is *not* claimed, the prior art that constrains the project, the results that
land directly on our mechanisms, and the experiments we still owe — including
the one we predict will go against us.

[`ARCHIVE.md`](ARCHIVE.md) indexes all 131 sibling repositories (20 consumed
by VENOM, 111 archived). [`RESEARCH.md`](RESEARCH.md) is the full literature
scan. [`LIMITATIONS.md`](LIMITATIONS.md) is the long-form honesty file.

## Packaging

```bash
pip install venomlayer          # or: pip install -e ".[torch]"
venom --help
```

`pyproject.toml` ships a `venom` console script, an optional `torch` extra
and an optional `integrations` extra (the 20 sibling repos).

## License

MIT — see [`LICENSE`](LICENSE).
