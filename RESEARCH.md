# RESEARCH.md — VENOM-PARASITE: prior art, novelty boundary, and honest positioning

**Scope.** This document exists to prevent the project from claiming more than the
literature supports. It was compiled from a targeted literature scan (arXiv, Anthropic
Research, transformer-circuits, MCP specification, ICLR/NeurIPS/ACL proceedings).
Every claim below names its source.

---

## 1. The idea, stated precisely

> A training-free middleware that exposes a host LLM's own forward-pass internals to
> that same LLM as **ordinary tools** (MCP `tools/call`), with every intervention
> bounded, measured, and revertible to bit-exact original behaviour.

The project is **not**:
- a trained "parasite" model;
- a method that overwrites host weights;
- a claim that the host becomes "self-aware" or "superintelligent".

The plan's own Reality Check table already says this. This document sharpens it against
the actual literature.

---

## 2. Prior art — the four lines VENOM touches

### 2.1 Activation steering / inference-time intervention — **well established**

Activation steering (adding a direction to the residual stream during inference) is a
well-developed, well-replicated line of work. VENOM's `bus.py` uses the same
mechanism (PyTorch forward hooks) and is **not novel** in itself.

| Work | What it does | Relevance to VENOM |
|---|---|---|
| Turner et al. 2023 (ActAdd) | Adds a steering vector to activations at inference | The canonical form of what `bus.py` + a dial would do |
| Li et al. 2024b, Singh et al. 2024, Yin et al. 2024 | Inference-time activation intervention | Same family |
| **Conditional Activation Steering (CAST)**, ICLR 2025 (IBM, open source) | Applies steering **conditionally** on the input — i.e. a rule/trigger decides whether to intervene | **This is the closest published cousin to VENOM's reflex layer.** CAST's "condition vector" gates whether steering fires. |
| SAFESTEER, EMNLP 2025 | Single forward pass, steering at strategic hidden states | Same mechanism, no training |
| LayerNavigator, NeurIPS 2025 | Finds *which layers* are steerable, cheaply | Directly relevant to VENOM's `hostmap` / future dial-layer selection |
| RADIANT, arXiv 2501.15758 | Layerwise classifier + distributional activation intervention; explicitly "lightweight, **reversible** control over generation" | Same reversibility goal as VENOM's attach/detach contract |
| Circuit-guided weight editing, arXiv 2602.04521 | Argues activation-time methods incur "recurring per-request overhead through forward hooks" and proposes an offline alternative | **A direct counter-argument VENOM must answer**: the hook overhead is real and recurring |

**Consequence for VENOM.** The `bus.py` and any future `dial()` mechanism are *known
technology*. The defensible novelty is not the mechanism — it is the **interface**:
exposing it to the host as a *tool the host itself calls*, with a **verification and
rollback contract attached**.

### 2.2 Introspection — **contested, and the bar has been raised sharply**

This is the literature that most directly threatens the project's framing, and it must
be read carefully before any "self-reading tools" claim is made.

| Work | Finding |
|---|---|
| Lindsey et al., *Emergent introspective awareness in LLMs* (transformer-circuits.pub, arXiv 2601.01828) | Claude Opus 4/4.1 can sometimes notice injected concepts and identify them. Explicitly: "highly unreliable in most of our experiments" |
| *Can LLMs Introspect? A Reality Check* (arXiv 2605.26242) | **The strongest negative result found.** Models cannot reliably distinguish activation-level interventions from *input-level* interventions. Their success reflects detecting anomalies *in general*, not introspective access. Biofeedback results "fully explained by in-context learning" |
| *Detecting the Disturbance* (arXiv 2512.12411) | The binary detection paradigm is a "methodological artifact" — apparent accuracy is entirely explained by global logit shifts biasing toward "yes". On harder differential tasks, real but **layer-dependent** introspection: 88% localisation, 83% strength discrimination, collapsing to chance for late-layer injections |
| *Emergent Introspection in AI is Content-Agnostic* (arXiv 2603.05414) | Replicates the detection paradigm; finds detection is **content-agnostic** — models detect *that* something is off without reliably knowing *what* |
| *A Closer Look at Language Model Introspection* (arXiv 2611.15267) | A supervised probe reaches 98.3% on perturbed-sentence identification while the model's own zero-shot answer stays at chance (11.1%) — **the model's self-report is far weaker than an external probe** |
| Kadavath et al. 2024, *Language models can learn to inspect their own activations*; Karvonen et al. 2025 (Activation Oracles); Patchscopes (Ghandeharioun et al. 2024) | Trained/finetuned models can verbalise internal features — **but this requires training**, which VENOM explicitly excludes |
| Song et al. 2025b | Models predicting their own parameters show **no advantage over predicting a peer's** — behavioural pattern matching, not privileged access |
| *Evidence for Limited Metacognition in LLMs* (arXiv 2509.21545) | Frontier LLMs show *rudimentary* metacognition; token probabilities correlate with delegation decisions. Effects are "relatively small and inconsistent"; models often favour non-introspective difficulty cues |
| *Looking Inward* (ICLR 2025) | Evidence for *privileged access* — but only "in certain domains", and unsuccessful on complex/OOD tasks |

**Consequence for VENOM — this is the most important section of this document.**

1. The project must **not** claim that giving a model read access to its activations
   produces introspection or metacognition. The best current evidence says the opposite:
   a supervised probe beats the model's own self-report by a wide margin
   (98.3% vs 11.1%).
2. The `uncertainty()` / `check_myself()` / `why()` / `where()` tools in the roadmap are,
   at best, **surface-level anomaly detectors**. The "content-agnostic" and
   "global logit shift" findings describe exactly what VENOM's sensors would measure.
3. Therefore the honest framing of VENOM's self-read tools is: **an external probe the
   host can consult about its own computation** — not the host knowing itself.
4. The project's existing **kill criterion** ("P3 ablation gain = 0 → cut the self-read
   tools") is the right instinct and should be treated as *likely*, not unlikely, to fire.

### 2.3 Agents, uncertainty and abstention — **crowded**

| Work | Relevance |
|---|---|
| UALA (Han et al. 2024) | Uncertainty threshold decides direct answer vs tool loop — **the same architecture as VENOM's reflex/agent split** |
| Calib-Abstain (preprints 202610.1224) | Token-level uncertainty + margin ε=0.1 tiers → retrieval / reformulation / abstain. **VENOM's `margin_threshold=0.15` is a re-derivation of this, not a discovery** |
| Dr. No (SAE feature steering for abstention) | Steers abstention behaviour via sparse features, no retraining. Direct precedent for "control abstention at inference time" |
| AgentAbstain (arXiv 2607.10059) | 263 paired tasks in 42 sandboxed MCP environments; requires an **explicit abstention signal** plus **no critical action executed** — a concrete, citable protocol for VENOM's abstain-correctness metric |
| TRUST (arXiv 2606.06976) | Uncertainty-aware reward for tool-calling decisions; +11% on When2Call |
| Uncertainty Quantification for LLM Agents (arXiv 2609.07395) | Taxonomy B1–B8. VENOM's sensors sit in **B3 (token probability)** — the weakest evidence tier. Agent-specific UQ is an open problem |
| Memory-Induced Tool-Drift (arXiv 2605.24941) | **A direct threat to VENOM's memory injection**: biased memories shift tool-call parameters; degradation 0.38–0.83 vs 0.05–0.12 for neutral memories; 6,062 tools across 288 MCP servers audited |
| Martingale analysis of MCP tool agents (arXiv 2602.13320) | Error accumulation is linear in tool-call depth with O(√T) deviation; **re-ground every ~9 steps** |

**Consequence for VENOM.**

- The reflex/agent split is standard practice, not an invention.
- The **memory-induced tool-drift** finding is the single most actionable item in this
  document: VENOM's `injector.py` injects memory into the prompt unconditionally when a
  key matches. That is precisely the mechanism shown to cause drift. **VENOM's injector
  must be relevance-gated** — this is a known, measured failure mode, not a hypothesis.
- The martingale result suggests VENOM needs periodic re-grounding if it ever accumulates
  state across many tool calls.

### 2.4 Self-modification and reversibility

| Work | Relevance |
|---|---|
| Gödel Agent (arXiv 2410.04444, ACL 2025) | Self-referential agent via monkey-patching its own runtime code; actions `self_inspect` / `interact` / `self_update` / `continue_improve` — **the closest published framing to "the agent edits itself"** |
| Darwin Gödel Machine (Zhang et al. 2026), STOP (Zelikman et al. 2024), Promptbreeder | Self-improving scaffolds; all modify **code/prompts**, not model internals |
| *Recoverability and Verifier Strength* (arXiv 2610.27465) | **Directly relevant to VENOM's rollback design**: "keeping the old value makes rollback exact and losing it does not" — and a weak verifier admits worse changes. Measured verifier strength, not assumed |
| *Who Verifies the Verifier?* (arXiv 2610.11464) | "Never validate a self-evolved verifier by the agent's performance: a grader that passes everything passes that test too" — requires a **held-out reference the loop cannot edit** |
| OBLITERATUS (open source) | A large, mature toolkit doing activation probing, steering, layer ablation, KL-drift measurement, and restore — the practical competitor/inspiration for `surgeon/` |
| circuit-guided weight editing (arXiv 2602.04521) | Argues against recurring per-request hook overhead in favour of offline edits |

**Consequence for VENOM.** Two design rules are now **evidence-backed requirements**,
not opinions:

- **R1 — Verifier strength must be measured, not assumed.** Every promotion of a change
  needs a verifier with a *measured* strength; a weak verifier admits worse changes.
- **R2 — The verifier must be independent of the thing it verifies.** VENOM's drift
  checker must be evaluated against a held-out reference set that the intervention
  cannot edit. "Our own drift number went up" is not evidence.

---

## 3. The honest novelty statement

> **New:** a host LLM invoking tools that read and (later) edit **its own** forward-pass
> internals, through a standard tool interface, where the whole stack is governed by a
> **bit-exact detach contract** and a **budget**, and where the host (not a wrapper
> script) decides when to invoke the tools.

> **Not new:** forward hooks, activation steering, conditional steering (CAST),
> uncertainty-gated tool use (UALA), calibration thresholds (Calib-Abstain), MCP as a
> transport, self-referential agents (Gödel Agent), activation-probing toolkits
> (OBLITERATUS), rollback-as-a-pattern.

The defensible contribution, if it survives measurement, is the **composition**:
self-tools + tool-protocol-native access to internals + hard reversibility + honest
abstention. That composition is not yet demonstrated. It is a **hypothesis with an
execution plan**, which is exactly what v0.1 is.

**Explicit non-claims**, to be enforced in every README revision:

1. ❌ "The model becomes self-aware / gains metacognition." The literature says the
   opposite for the mechanism VENOM uses.
2. ❌ "Reading activations tells you what the model understands." It tells you what the
   forward pass computed.
3. ❌ "Self-read tools improve answers." Untested; P3 kill criterion applies.
4. ❌ "The layer costs nothing." Untested; circuit-guided weight editing argues the
   opposite about recurring hook overhead.
5. ❌ "Training-free means effort-free." Calibration thresholds must be tuned — from
   forward-pass statistics, not gradients. This is stated in the plan already.

---

## 4. Threats to the design, ranked by how likely they are to bite

| # | Threat | Evidence | Mitigation (must be in the plan) |
|---|---|---|---|
| T1 | **Memory injection causes tool-drift** — VENOM's own injector is the proven failure mode | arXiv 2605.24941: degradation 0.38–0.83 across 8 models; audited across 288 MCP servers | Relevance-gate every injection; neutral-memory baseline in every bench; measure degradation, not just fix rate |
| T2 | **Self-read tools show no gain** (or the gain is explained by anomaly detection, not introspection) | arXiv 2605.26242, 2603.05414, 2611.15267 | Keep the P3 ablation and its kill criterion. Add a **content-agnostic control**: compare detection of *activation* perturbation vs *input* perturbation. If both are detected equally, the tool is an anomaly detector — say so |
| T3 | **Per-token reflex overhead makes the layer a net loss** | arXiv 2602.04521 (recurring hook overhead); 2609.07395 (agent UQ cost) | Sample the reflex, don't run it per token. Measure latency ratio vs baseline, on a real model, before claiming anything |
| T4 | **Self-edit dials break the host and rollback fails** | arXiv 2610.27465 (rollback exactness needs the old value retained); 2610.11464 (verifier must be independent) | Implement R1 + R2 above. Snapshot before every write; verify against a held-out gold set |
| T5 | **Bit-exact detach claim is false** — the trust contract breaks at the foundation | Not yet tested; `bus.py` keeps tensor references in `self.taps` | **Run E1 first.** If `bit_exact_after_detach` is False, stop and fix the bus before building anything on top of it |
| T6 | **"Parasite" framing attracts the wrong reviewers** — reads as an attack/jailbreak, not research | The name is memorable and will be misread | Keep the README's first line honest: *training-free middleware*, *host = agent*. Consider a neutral name for any paper submission |

---

## 5. What to measure next, in order (and what each result would mean)

| Order | Experiment | Instrument | Decision it drives |
|---|---|---|---|
| 1 | **E1 — bit-exact detach + latency** | `experiments/e1_bus_detach.py` | If detach is not bit-exact → **stop**, fix the bus. If latency ratio > 1.15× → redesign the reflex sampling before anything else |
| 2 | **E2 — injector relevance-gate ablation** | new: inject always vs inject only on relevance-gated key | Confirms or kills T1. Also gives the "neutral memory" baseline the drift paper requires |
| 3 | **E3 — fix rate vs harm rate on a real host** | `venom/bench/correction_bench.py` + GPT-2/Qwen2.5 | The P0 exit gate. fix ≤ harm → **stop**, the memory interface is wrong |
| 4 | **E4 — tool-call recall/precision** | needs a tool-calling host (GPT-2 cannot call tools) | The P1 exit gate. Recall < 30% → thicken reflex, demote agent path |
| 5 | **E5 — content-agnostic introspection control** | new: perturbation at activation layer vs at input | Decides whether the self-read tools are introspection or anomaly detection. **This is the experiment that decides the project's central scientific claim** |
| 6 | **E6 — held-out verifier strength** | new: gold set the intervention cannot edit | Required before any self-edit dial is enabled |

---

## 6. Citation list

- Turner, A. et al. (2023). *Activation Addition: A Lightweight Approach to Steering Language Models.* — canonical activation steering.
- Li, K. et al. (2024). *Inference-Time Intervention: Eliciting High-Quality Control for Language Models.* — inference-time activation intervention.
- Singh, A. et al. (2024); Yin, P. et al. (2024). — same family.
- **Conditional Activation Steering (CAST)**, ICLR 2025. — https://openreview.net/attachment?id=Oi47wc10sm&name=pdf · code: `github.com/IBM/activation-steering` — **closest published cousin to the reflex layer.**
- SAFESTEER, EMNLP 2025. — https://aclanthology.org/2025.emnlp-main.1781.pdf
- LayerNavigator, NeurIPS 2025. — https://proceedings.nips.cc/paper_files/paper/2025/file/9280bfb61f159c768dfdeb729a41ccd0-Paper-Conference.pdf
- RADIANT, arXiv 2501.15758. — https://arxiv.org/html/2501.15758 — reversible inference-time activation control.
- Circuit-guided weight editing, arXiv 2602.04521. — https://arxiv.org/html/2602.04521 — **argues against recurring forward-hook overhead.**
- Lindsey et al. (2025/2026). *Emergent introspective awareness in LLMs.* — https://transformer-circuits.pub/2025/introspection/index.html · https://www.anthropic.com/research/introspection
- *Can LLMs Introspect? A Reality Check*, arXiv 2605.26242. — https://arxiv.org/html/2605.26242v1 — **strongest negative result; model conflates activation- and input-level intervention.**
- *Detecting the Disturbance*, arXiv 2512.12411. — https://arxiv.org/html/2512.12411 — detection paradigm is an artifact; introspection is layer-dependent.
- *Emergent Introspection in AI is Content-Agnostic*, arXiv 2603.05414. — https://arxiv.org/pdf/2603.05414
- *A Closer Look at Language Model Introspection*, arXiv 2611.15267. — https://arxiv.org/pdf/2611.15267v2 — **probe 98.3% vs model self-report 11.1%.**
- Kadavath et al. (2024). *Language models can learn to inspect their own activations.*
- Ghandeharioun et al. (2024). *Patchscopes.*
- Karvonen et al. (2025). *Activation Oracles.*
- *Looking Inward*, ICLR 2025. — https://proceedings.iclr.cc/paper_files/paper/2025/hash/0a6059857ae5c82ea9726ee9282a7145-Abstract-Conference.html
- *Evidence for Limited Metacognition in LLMs*, arXiv 2509.21545. — https://arxiv.org/pdf/2509.21545v2
- *Self-reference in large language models*, arXiv 2607.04277. — https://arxiv.org/html/2607.04277v1 — quasi-introspection; structural bottlenecks.
- UALA — Han et al. (2024), uncertainty-aware language agent.
- Calib-Abstain — https://www.preprints.org/manuscript/202610.1224
- AgentAbstain, arXiv 2607.10059. — https://arxiv.org/pdf/2607.10059 — MCP abstention protocol.
- TRUST, arXiv 2606.06976. — https://arxiv.org/html/2606.06976v1
- Uncertainty Quantification for LLM Agents, arXiv 2609.07395. — https://arxiv.org/html/2609.07395v1
- **Memory-Induced Tool-Drift**, arXiv 2605.24941. — https://arxiv.org/html/2605.24941v2 — **direct threat to the injector.**
- Martingale analysis of MCP tool agents, arXiv 2602.13320. — https://arxiv.org/html/2602.13320v1
- Gödel Agent, arXiv 2410.04444 / ACL 2025. — https://aclanthology.org/2025.acl-long.1354.pdf
- Darwin Gödel Machine (Zhang et al. 2026); STOP (Zelikman et al. 2024); Promptbreeder (Fernando et al. 2023).
- *Recoverability and Verifier Strength*, arXiv 2610.27465. — **rollback exactness + measured verifier strength.**
- *Who Verifies the Verifier?*, arXiv 2610.11464. — **independent held-out verifier.**
- OBLITERATUS — https://github.com/elder-plinius/OBLITERATUS
- Model surgery — arXiv 2407.08770, https://github.com/lucywang720/model-surgery
- Model Context Protocol specification (2026-07-28 revision) — https://modelcontextprotocol.io/specification/2026-07-28/server/tools · client best practices (progressive discovery, programmatic tool calling) — https://modelcontextprotocol.io/docs/2026-07-28/develop/clients/client-best-practices

---

## 7. Bottom line

The idea is **not** absurd and the plan is **not** naive — but the literature is more
crowded than the original framing assumed, and two published findings land directly on
the project's core mechanisms (memory injection → tool drift; activation reading →
content-agnostic anomaly detection, not introspection).

The right move is unchanged and now sharper:

1. **Measure E1 first.** Bit-exact detach is the load-bearing claim. It is currently
   untested, and everything else sits on top of it.
2. **Fix the injector before benching.** Add a relevance gate and a neutral-memory
   baseline; otherwise the bench will measure drift, not capability.
3. **Add the content-agnostic control (E5).** It decides whether the self-read tools are
   introspection or anomaly detection — the difference between a real contribution and
   a repackaged probe.
4. **Enforce R1 and R2** on any future self-edit dial: a measured verifier, and one the
   intervention cannot edit.
5. **Keep the kill criteria.** The project already wrote them down before having
   results. That is the strongest methodological asset here, and the literature says
   two of them are more likely to fire than not.
