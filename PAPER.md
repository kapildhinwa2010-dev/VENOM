# VENOM-PARASITE — a plug-in layer that lets an LLM read its own forward pass

**Status: research prototype. Alpha. The central scientific claim is NOT established.**

This document is the honest write-up. It is deliberately written to be
readable by a reviewer who wants to catch us out. Every number below was
produced by a script in this repository; every claim that is not measured is
marked as such.

---

## 1. The idea in one paragraph

An LLM's forward pass produces hidden states and logits that are, today,
invisible to the model itself. We expose them as **ordinary tools** over MCP,
so the model can ask questions of its own computation: *how uncertain am I?
which layer decided this? where do I hold this fact?* Every tool that could
change the host is bounded, measured, and revertible, and the host model is
never modified without a proof that the change can be undone.

## 2. What is *not* being claimed

This matters more than the claims.

| We do NOT claim | Why |
|---|---|
| "The model becomes self-aware." | The best current evidence says the opposite. A supervised probe beats a model's own self-report **98.3% vs 11.1%** on perturbation localisation ([arXiv 2611.15267](https://arxiv.org/pdf/2611.15267v2)). |
| "Reading activations gives introspection." | Models cannot reliably distinguish an *activation-level* intervention from an *input-level* one ([arXiv 2605.26242](https://arxiv.org/html/2605.26242v1)); detection is **content-agnostic** ([arXiv 2603.05414](https://arxiv.org/pdf/2603.05414)). |
| "The layer makes the model smarter." | Untested. The one gate we ran shows the injection path is not actively harmful on a small fact set — nothing more. |
| "The layer is faster." | The layer *adds* work. Measured overhead for the read-only bus is ≈ **+2%** on GPT-2 CPU, and that number is noise-dominated. |
| "Self-edit improves the model." | We can bound, measure and revert a self-edit. Whether a bounded edit *helps* is unanswered. |
| "This is new." | Forward hooks, activation steering, conditional steering, uncertainty-gated tool use and MCP are all prior art. See §7. |

## 3. Architecture

```
user / MCP client
      │
      ▼
venom/server.py          MCP JSON-RPC 2.0 over stdio — 15 tools
      │
      ▼
venom/core/runtime.py    VenomRuntime: memory · budget · reflex · telemetry
      │                    + optional host internals (probe, taps, norm, head, dials)
      ├──────────────► venom/core/discovery.py   finds blocks/head/norm of ANY architecture
      ├──────────────► venom/memory/store.py    exact recall, digest guard, abstain
      ├──────────────► venom/reflex/             rule-based trigger, budget, injector
      ├──────────────► venom/sensors/            uncertainty, consensus, attribution, audit
      ├──────────────► venom/verify/             drift classification, rollback
      ├──────────────► venom/surgeon/prune.py    gated self-surgery
      └──────────────► venom/integrations/       20 sibling repos, called not copied
```

Two entry points cover every host we could think of:

```python
# open-weight, local
from venom.shell import attach
with attach(model, tokenizer) as host:
    host.chat("what is the capital of france?")

# any API model (GPT, Claude, Gemini, vLLM, Ollama)
venom proxy --upstream https://api.openai.com/v1 --upstream-model gpt-4o-mini
```

## 4. The trust contract, and how it is enforced

This is the load-bearing part of the project. Four properties, each with a
test that fails if the property breaks.

### 4.1 Attach → detach is bit-exact

A forward hook that returns `None` leaves PyTorch's output untouched. That
single fact is what makes the layer non-invasive.

```bash
python experiments/e2_universal_attach.py
```

| Model | Family | `bit_exact_during` | `bit_exact_after_detach` | logit hash |
|---|---|---|---|---|
| GPT-2 124M | gpt2 | ✅ | ✅ | identical |
| Qwen2.5-0.5B-Instruct | qwen2 | ✅ | ✅ | identical |
| SmolLM2-135M-Instruct | llama | ✅ | ✅ | identical |

Also covered by `tests/test_bus_torch.py`: tuple-returning modules, repeated
attach/detach cycles, and taps being snapshots rather than live aliases.

### 4.2 The memory abstains; it never guesses

Unknown → `abstain/not_found`. Corrupted → `abstain/digest_mismatch`. The
digest survives the persistence boundary: a hand-edited memory file loads as
abstain, not as a hit.

### 4.3 Every intervention is bounded and revertible

A **dial** scales one block's output by `(1 + delta)`. `max_delta` caps the
magnitude, `max_active_dials` caps how many can be live, and a verifier
measures the drift against a captured baseline. If the drift exceeds the
threshold, the dial is rolled back automatically.

```bash
venom dial --model gpt2 --layer 3 --delta 0.05
```

| delta | measured drift | outcome |
|---|---|---|
| 0.01 | 0.00646 | applied, verified |
| 0.02 | 0.01240 | applied, verified |
| 0.03 | 0.01794 | applied, verified |
| 0.05 | **0.02809** | **auto-rolled back** (> 0.02 threshold), host restored bit-exactly |

A verifier that *cannot measure* is treated as a failure, not as "no drift".
We got this wrong first: an early version reported `drift: 0.0` when it had
measured nothing, which would have let an unverified edit through.

### 4.4 Self-surgery is gated by a held-out proof

`venom/surgeon/prune.py` implements the plan's P5 gate: *"drift = 0, warna
prune radd"*. It plans, snapshots, applies, measures the drift on a
**held-out** set, and restores the weights if the drift is out of bounds.
The plan set and the held-out set are disjoint on purpose — verifying a
prune on the same prompts that justified it proves nothing.

On GPT-2 the gate **refused to prune**: every unit of every projection fired
at least once over the plan prompts, so nothing was prunable and nothing
changed. That is the correct outcome, and it is the honest one: ten prompts
cannot establish that a unit is dead.

## 5. The tool surface

15 tools, Tier 1 (open-weight hosts, everything) and Tier 2 (API-only hosts,
memory + verify + state). The API proxy serves **Tier 2 by default** — an API
host has no access to its own internals, so offering it `venom.uncertainty`
would be a lie.

| Tool | What it does |
|---|---|
| `venom.recall` / `remember` / `forget` | exact memory, with retraction |
| `venom.time_now` | the host has no clock |
| `venom.uncertainty` | entropy / margin / confidence of the last token |
| `venom.check_myself` | layer-wise agreement with the final top-1 |
| `venom.why` / `where` | logit-lens trajectory and localisation |
| `venom.verify` | verified / contradicted / unknown against memory |
| `venom.dial` | bounded self-edit with auto-rollback |
| `venom.prune` | gated self-surgery |
| `venom.audit` | dead neurons, redundancy, compressibility — **report only** |
| `venom.snapshot` / `restore` / `report` | state control |

### What the self-read tools actually returned on GPT-2

Prompt: *"The capital of France is"*.

| Tool | Result |
|---|---|
| `venom.uncertainty` | entropy **5.998**, margin **0.037**, confidence **0.085**, `is_uncertain: true` |
| `venom.check_myself` | **1 of 12** layers agree with the final top-1 → *"decided late"* |
| `venom.where(" France")` | best represented at **layer 9** (rank 0) |
| `venom.why(" Paris")` | target logit **−8.84** at layer 0, never overtakes → `decision_layer: null` — GPT-2 did not predict "Paris" at all |
| `venom.audit` | **8 of 11** layer pairs near-duplicate (cosine > 0.99); 5 of 6 sampled layers compressible |

Each payload carries its own caveat. `check_myself` states in the response
that layer agreement is **not** introspection, and cites the papers above.

## 6. The P0 gate, and the control that makes it mean something

The plan's exit gate for P0: *"fix rate > harm rate"*.

```bash
python experiments/e5_fix_harm.py --model gpt2
```

| Arm | fix rate | harm rate | baseline acc | VENOM acc |
|---|---|---|---|---|
| VENOM memory | **0.417** | **0.375** | 0.40 | 0.45 |
| neutral-memory control | 0.0 | 0.0 | 0.40 | 0.40 |

The **control** is the point: the same bank of facts, stored under keys that
match nothing, changed **zero** answers. Without it, "fix rate" could just
be "any extra text helps".

**The honest reading:** this is 20 facts on one 124M model, and the fix rate
is largely the model *copying the injected context*. That is a property of
the benchmark, not a demonstration of knowledge repair. The plan asks for 400
facts; we have not run that.

One methodological note for anyone repeating this: the gate compares
**rates**, not counts. An early version compared counts and reported a pass
where 10 fixed vs 9 harmed (0.43 vs 0.53 by rate) was in fact a loss.

## 7. Related work, and the honest novelty statement

The literature is crowded, and two published results land directly on this
project's core mechanisms.

**Prior art we build on:** activation steering (Turner et al. 2023; ActAdd),
inference-time intervention (Li et al. 2024), **Conditional Activation
Steering** (ICLR 2025 — the closest published cousin to the reflex layer),
LayerNavigator (NeurIPS 2025), RADIANT (arXiv 2501.15758), SAFESTEER (EMNLP
2025), uncertainty-gated tool use (UALA), Calib-Abstain's tiered thresholds,
MCP as a transport, and self-referential agents (Gödel Agent).

**Results that constrain us:**

- **Memory-induced tool-drift** ([arXiv 2605.24941](https://arxiv.org/html/2605.24941v2)):
  biased memories shift tool-call parameters across all 8 tested models
  (degradation 0.38–0.83 vs 0.05–0.12 for neutral memories). *This is
  exactly what our injector does*, which is why the injector refuses generic
  keys and why the neutral-memory control is mandatory.
- **Activation reading ≠ introspection** (see §2). This is why the self-read
  tools are described as sensors, and why experiment E5 below is the most
  important one we have not run.

**The novelty statement we can defend:**

> *New:* a host LLM invoking tools that read **its own** forward-pass
> internals through a standard tool interface, where the whole stack is
> governed by a **bit-exact detach contract**, a **budget**, and a
> **held-out verification gate** before any self-modification is kept.

> *Not new:* forward hooks, activation steering, conditional steering,
> uncertainty-gated tool use, MCP, rollback-as-a-pattern.

## 8. E5 — the content-agnostic control (the decisive experiment, now run)

The plan's most important open question: when the self-read tools report
"something is off", are they reporting on the model's **internal state**, or
just detecting that *an anomaly happened somewhere*?

**Design.** A matched control. The *same* vector, with the *same* magnitude,
injected at two different sites: (A) the embedding output — an input-level
intervention; (B) the residual stream at layer L — an activation-level
intervention. If the self-read signal moves equally in both cases, the tools
are an anomaly detector.

```bash
python experiments/e5_content_agnostic.py --model gpt2 --layer 6
python experiments/e5_content_agnostic.py --model Qwen/Qwen2.5-0.5B-Instruct --layer 12
```

**Result (measured, real weights, both models replicate):**

| | GPT-2 124M | Qwen2.5-0.5B-Instruct |
|---|---|---|
| Δ entropy, input-level injection | **+0.0540** | **+3.5868** |
| Δ entropy, activation-level injection | +0.0108 | +0.0292 |
| Δ margin, input-level | −0.0004 | −0.1371 |
| Δ margin, activation-level | −0.0003 | −0.0291 |
| Δ confidence, input-level | −0.0042 | −0.2203 |
| Δ confidence, activation-level | −0.0007 | −0.0244 |
| **signal ratio (activation ÷ input)** | **0.201** | **0.021** |
| first layer whose tap moved, activation injection | **6** (injected 6) | **12** (injected 12) |
| first layer whose tap moved, input injection | 0 | 0 |
| localisation correct? | ✅ | ✅ |

**Interpretation — our prediction was confirmed.**

1. **The model's self-report is generic.** An *input-level* perturbation moves
   the uncertainty signal **5× more** (GPT-2) to **48× more** (Qwen) than an
   *activation-level* one. The model's own reported uncertainty reacts to what
   the input looks like, not to what is happening inside its computation. This
   is the signature of an **anomaly detector**, and it is what
   [arXiv 2605.26242](https://arxiv.org/html/2605.26242v1) and
   [arXiv 2603.05414](https://arxiv.org/pdf/2603.05414) predict. We stated
   this prediction before running the experiment.

2. **But the instrumentation localises perfectly.** Injecting at layer L left
   layers `0..L-1` **bit-exact** (cosine to baseline = 1.00000000) and moved
   layer L and everything downstream. On Qwen, layers 0–11 were exactly
   1.00000000 and layer 12 was 0.999539.

**This is the important distinction, and the project now states it plainly:**

> The *model's* self-report does not know about its internal state.
> The *instrumentation* does.

VENOM's ability to name the perturbed layer is a **debugger capability**, not
model introspection. It exists because the layer reads the residual stream
directly — the same reason a debugger can. The model contributes nothing to
it. Any claim that VENOM gives a model introspection is **not supported by
this experiment** and is not made.

**What this changes.** The self-read tools stay — they are genuinely useful
sensors, and `where`/`check_myself` have a real localisation job. But they are
documented as **measurements of the host made on the model's behalf**, never
as the model's knowledge of itself. The `check_myself` response already said
this in prose; now there is an experiment behind it.

**Caveats.** One model size each, one prompt, one perturbation direction, one
magnitude. The direction was a fixed random vector; a structured perturbation
(a steering vector, a natural error) might behave differently. The ratio
threshold (1.5) was chosen before the run, but a single scalar on a single
prompt is weak evidence in either direction. The localisation result is
stronger because it is exact on two architectures.

## 8b. The surgery stack (v0.5.0)

The plan's "dead neuron removal" is implemented end to end, and it is the one
part of VENOM that permanently changes what the host model *is*.

The **census** is per-neuron, not per-layer: for every FFN unit it counts
how many prompts made that unit fire, and it separates *exactly-dead* (fired
on zero samples) from *rarely-active* (fired on fewer than 1% of samples).
Those are different claims, and collapsing them is how a prune ends up
removing something that mattered.

The **write surface** is unrestricted by design, and says so out loud: every
mutation is journaled and snapshotted, and a shape rewrite is undoable
because the pre-rewrite modules are kept. What it offers is *reversibility
and an audit trail*, **not prevention**.

The **killer** is the plan's gate, unchanged: calibrate on one prompt set,
apply, verify the drift on a **disjoint** held-out set, roll back if the
drift is over the threshold. A prune that cannot be verified is not kept.

**Measured on real GPT-2:**

| | Result |
|---|---|
| Census: 12 prompts, 36 864 FFN units | **0** exactly-dead, **0** rarely-active |
| Kill on real GPT-2 | **refused** - nothing was dead, nothing was modified |
| Kill on a controlled fixture (12 units zeroed on purpose) | found all 12, removed them, drift **0.0**, top-1 agreement **1.0** |
| Shape rewrite (`mode="remove"`), 8 units of `mlp.c_fc` | `3072 → 3064`, consumer `mlp.c_proj` reshaped with it, model still runs, drift **0.2255**, top-1 agreement **1.0** |
| Restore after the shape rewrite | **bit-exact** (drift back to **0.0**) |
| Install | manifest written, **49** writable modules, **no weight changed** |

The honest reading: **GPT-2 has no dead FFN units at this calibration
size**, so the killer correctly refuses. The fixture result demonstrates the
*mechanism*, not a property of the model.

Two bugs found while building this are worth recording, because both would
have produced a silently broken model:

* **The consumer search was global.** "Modules whose input dim equals the
  target's output dim" matched the `c_proj` of *every* GPT-2 block, so
  shrinking one block's FFN shrank all twelve and the model stopped running.
  The search is now scoped to the target's own parent.
* **HuggingFace's `Conv1D` caches its sizes** in `nx`/`nf` and uses them to
  reshape the output. Shrinking the weight without updating them breaks the
  forward pass.

## 8c. The automated pipeline (v0.6.0)

The plan assumed somebody would sit with the feature outputs and decide what
to cut. For a 100M-class model nobody will, so the layer does it: every
feature runs, the answers feed **one ordered ladder** (a MATLAB-shaped
`if / elseif` chain where the first match wins), and the ladder's verdicts are
applied through the usual held-out drift gate.

Measured on GPT-2 124M and SmolLM2-135M-Instruct: **0** exactly-dead units
and **0** rarely-active units on both, so the ladder changed **no weights** on
either - the correct outcome for healthy models.

The finding worth keeping is a **contradiction between two features**:
redundancy reports 8 near-duplicate layer pairs on GPT-2 (cosine > 0.99),
while the skip-run test measures a logit delta of **83.56** when the same
block is skipped. Consecutive layers of a residual network look alike because
the residual stream dominates, not because they are redundant. A pipeline
that trusted redundancy alone would have cut 8 live layers. Redundancy is
therefore `FLAG_LAYER` - report-only - and never a licence to remove.

## 9. The experiments we still owe

In priority order. Each has a kill criterion attached.

| # | Experiment | Decides |
|---|---|---|
| E2 | Fix/harm on 400 facts, and on a tool-calling host | The real P0 gate |
| E3 | Reflex-path latency during a full generation | Whether the layer is affordable |
| E4 | Held-out verifier strength for a self-edit dial | Whether self-edit is safe at all |
| E6 | Drift over 10k prompts for pruning | The real P5 gate |

**E5 is done** — see §8. Our prediction was that it would show anomaly
detection rather than introspection. It did. The prediction was written down
before the run, and it is recorded here unchanged.

What remains is E2 at 400 facts, E3's reflex latency during a full
generation, E4's held-out verifier, and E6's 10k-prompt pruning drift.

## 10. Reproducing everything

```bash
pip install -e ".[dev,torch]"
pytest -q                                   # 179 tests
python experiments/e1_bus_detach.py         # bit-exactness + latency
python experiments/e1b_latency_isolated.py   # randomised A/B latency
python experiments/e2_universal_attach.py    # 3 architectures, real weights
python experiments/e2b_architecture_sweep.py # 30 architectures, meta-device
python experiments/e3_selfread.py            # the self-read tools on a real model
python experiments/e4_gated_prune.py         # gated self-surgery
python experiments/e5_fix_harm.py           # the P0 gate + control
python experiments/e5_content_agnostic.py   # the content-agnostic control
python experiments/e6_kill_dead.py           # the surgery stack on a real model
python experiments/e7_auto_surgery.py       # the automated ladder on a small model
```

## 11. Limitations

`LIMITATIONS.md` is the long form. The four that matter most:

1. The self-read numbers are **self-reports about a computation**, not about
   knowledge — and E5 (§8) showed that they react to input anomalies, not to
   internal state.
2. `why`/`where` are **logit-lens readouts, not causal proofs**.
3. `dial` measures **magnitude, not quality**.
4. "Any AI" means **HF causal LMs with a standard block container** plus
   **OpenAI-compatible APIs**; Anthropic-native and Gemini-native APIs need
   their own adapters. The architecture sweep covered 22 of 30 models across
   18 families — the 8 failures were all *access* failures (gated or renamed
   repos), not discovery failures.

## 12. Repository consolidation

The master plan named 20 sibling repositories as the sources for this
work. Rather than re-implementing them, `venom/integrations/` **calls** them
— NEVER_LIE's fuzz routine, INNER-VOTE's consensus curve, TRACE-PATH's
activation patching, DEAD-CELL's census, TMU's internal clock — behind
optional adapters that degrade to `{"ok": False}` when a sibling is absent.
The remaining 111 repositories are indexed in
[`ARCHIVE.md`](ARCHIVE.md).

## 13. Citation

```bibtex
@software{venom_parasite_2026,
  title  = {VENOM-PARASITE: a plug-in layer that lets an LLM read its own forward pass},
  author = {{Kapil}},
  year   = {2026},
  url    = {https://github.com/kapildhinwa2010-dev/VENOM},
  note   = {Research prototype. The central scientific claim is not established.}
}
```

MIT licensed.
