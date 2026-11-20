# LIMITATIONS — measured vs claimed

Every number in this file is either **measured** (with the command that produced it) or
**not measured**. Nothing here is a prediction dressed as a result.

---

## 1. What is verified

| Claim | How | Result |
|---|---|---|
| Core modules import and behave | `pytest tests/test_p0.py` | **44/44 passed** |
| Regression tests for the self-review bug batch | `pytest tests/test_regressions.py` | **37/37 passed** |
| Torch-level trust-contract tests | `pytest tests/test_bus_torch.py` | **8/8 passed** |
| **Total suite** | `pytest -q` | **89/89 passed** |
| Exact recall, abstain on unknown, abstain on corruption | unit tests | ✅ |
| Bounded LRU eviction | unit test | ✅ |
| Reflex fires on memory hit and on low margin, stays quiet otherwise | unit tests | ✅ |
| Tool budget caps calls, unknown tools rejected, budget re-armed per turn | unit tests | ✅ |
| MCP stdio server: JSON-RPC 2.0 conformance (notification vs request, id preservation, error codes, tool exceptions) | unit tests | ✅ |
| Per-token entropy / margin / confidence, empty-row safety | unit tests (pure python) | ✅ |
| Drift classification (fixed / harmed / shifted / abstained) is mutually exclusive and exhaustive | unit tests | ✅ |
| Snapshot / restore is isolated from later mutation | unit tests | ✅ |
| Session state survives a dict round-trip **with capacity, budget usage and digests intact** | unit test | ✅ |
| Host fingerprinting honours `config.model_type` for 6 families | unit test (synthetic) | ✅ |
| **Bus attach → detach is bit-exact on a real model** | `python experiments/e1_bus_detach.py` on **GPT-2 124M** | ✅ `bit_exact_during=true`, `bit_exact_after_detach=true`, `hash_before == hash_after == 5ea46f3e53e2012a` |
| Bus taps are snapshots, not live aliases | `tests/test_bus_torch.py` | ✅ |
| Bus hook survives tuple-returning modules and repeated attach/detach cycles | `tests/test_bus_torch.py` | ✅ |
| Bus latency overhead, GPT-2 124M, CPU | `experiments/e1b_latency_isolated.py`, 9 randomised A/B rounds | **≈ +2% median** (bus_retain 1.02×, null_hook 1.02–1.07×) — **noise-dominated, see §5** |
| Bus releases its captured activations | `experiments/e1_bus_detach.py` | ✅ 184,320 bytes retained → 0 after `release()` |

## 2. What is NOT verified

| Claim | Status | Blocked by |
|---|---|---|
| Bit-exact detach on **Llama / Qwen / Mistral / Phi / Gemma** | ❌ not run | needs those model weights; `hostmap` now resolves them correctly but has only been tested on synthetic shapes |
| Reflex path adds ≤ 15% latency on a real generation | ❌ not run | the reflex needs a full sampling loop, not a single forward pass |
| Fix rate ≥ 40% | ❌ not run | needs a real host model; `MockHost` is plumbing only |
| Harm rate ≤ 2% | ❌ not run | same |
| Tool-call recall / precision | ❌ not run | needs a tool-calling host (GPT-2 cannot call tools) |
| Abstain correctness ≥ 95% under corruption | ❌ not run on a real model | needs a real host |
| Dead-neuron report is correct on a real model | ❌ not run | needs activation capture on a real model |
| The content-agnostic introspection control (E5 in `RESEARCH.md`) | ❌ not built | this is the experiment that decides the central scientific claim |
| Latency of the **reflex** path (not just the bus) | ❌ not run | needs the sampling loop |
| `sensors/logitlite.py` runtime | ❌ not run | needs torch |
| Multi-model-family attach | ❌ not run | needs 5+ model downloads |

**`MockHost` in `venom/bench/correction_bench.py` is a plumbing test, not a result.** Its
"fix rate" of 1.0 is the mock answering its own injected string. It proves the metric
pipeline computes the right thing; it proves nothing about any model.

## 3. Bugs found and fixed (by the v0.1.1 self-review)

Each of these was confirmed by a reproducing script, not suspected.

| # | Bug | Impact | Fix |
|---|---|---|---|
| B1 | `hostmap` labelled **every** non-GPT-2 model `"llama"` (Llama/Qwen2/Mistral/Phi/Gemma all expose `model.layers`); Phi's returned `norm_path` did not exist on the model | attaching to 5 of 6 supported families was broken | `config.model_type` is now authoritative; path probing is a fallback that requires both blocks **and** norm |
| B2 | `store.norm()` **deleted** punctuation instead of separating it | `"hello,world"` → `"helloworld"` (never recalled); `"don't"`/`"dont"` and `"U.S.A."`/`"usa"` collapsed into one slot, **silently losing a stored fact** | punctuation → space |
| B3 | `Session` load **recomputed the digest**, so a tampered session file loaded as a valid HIT | the digest guard died exactly at the persistence boundary | `MemoryStore.restore()` keeps the persisted digest |
| B4 | `Session` round-trip lost **memory capacity** and **budget usage** | restoring a session silently re-armed the full tool-call budget and removed the memory cap | both are serialised and restored |
| B5 | `Injector`'s multi-hit branch was **dead code** (`decide()` stopped at the first match); `max_entries` did nothing; the multi-line `CONTEXT:` format was not parseable by `MockHost` | multi-fact prompts injected only one fact | `Reflex.matching_keys()` yields every match; single-line format shared with the bench |
| B6 | `drift_report` checked abstention **before** the gold comparison | replacing a **correct** answer with "abstain" was invisible in `harmed` and inflated `net_gain` | order fixed; correct→abstain now counts as harm |
| B7 | `server.py` detected notifications by **method name**, not by the absence of `id` | answered a notification (JSON-RPC violation) and **silently dropped** a request that had an id | notification = no `id` |
| B8 | A malformed `tools/call` answered with `id: null` and code `-32700`; a raising tool killed the request | the client could never correlate the response | id preserved, `-32603` for internal errors, tool exceptions become `isError` results |
| B9 | `token_uncertainty([])` raised `ValueError` | one padded row killed a whole per-token sweep | empty row returns a safe "uncertain" record |
| B10 | `bench._match` was a **substring** test over a normalised gold | `_match("the cat sat", "C++")` → True; `_match("ukraine", "uk")` → True; a negated answer counted as correct — corrupting both `fix_rate` and `harm_rate` | word-boundary match |
| B11 | The bench called the host twice per fact with **no shared seed** | on any sampled host, fixed/harmed were dominated by sampling noise | `repeats` + `seed` parameters, same seed for both runs |
| B12 | **No adapter** between `Injector` (returns a dict) and `run_bench` (needs `prompt → prompt`) | the end-to-end pipeline could not run at all; the bench test used a hand-written lambda instead of the real injection format | `Injector.as_prompt_fn()` |
| B13 | A bus tap was a **live alias** of the block output (`detach()` shares storage) | any later in-place write to the residual stream silently rewrote the recorded history | `detach().clone()` |
| B14 | `Bus.detach()` kept every layer's full activation on the host device, and nothing called the cleanup | for a 70B model at 4k context this is tens of GB — the "harmless middleware" could OOM the host | `clear_taps()` / `release()`; `offload_to_cpu` option |
| B15 | The "already attached" guard derived from the handle list | `Bus(model, [])` → `attached` was False and a second `attach()` did not raise | explicit `_attached` flag |
| B16 | The tool-call budget was **per process**, never reset per answer | after 3 calls the host was mute for the rest of the session | `initialize` re-arms the budget |
| B17 | `logits_hash` required numpy, which torch does not depend on | the bit-exactness gate crashed in a torch-without-numpy environment | numpy-free hashing |
| B18 | `corrupt_for_test` raised a bare `KeyError` | indistinguishable from a real failure | explicit message |
| B19 | `compressibility_report` returned a **different schema** for an all-zero spectrum | consumers doing `details[name]["eff_rank"]` got a `KeyError` | uniform schema |
| B20 | The "low" bucket bound was hardcoded at 0.05 | raising `threshold` above it silently emptied the bucket | `low_threshold` parameter |
| B21 | CI never ran the torch tests (`importorskip`) | the only automated test of the central claim never executed | separate `trust-contract` CI job |
| B22 | `bus.detach()` **wiped the taps** on exit | destroyed the data the bus was attached to collect | `detach()` keeps taps; `clear_taps()` frees them |
| B23 | The injector treated any stored key as relevant | memory-induced tool drift is a *measured* failure mode (arXiv 2605.24941) | generic-key guard (`min_key_words`, `min_key_chars`) |
| B24 | The E1 latency measurement timed the baseline first and the bus second | reported the bus as **2–12% faster** than no bus — impossible | randomised interleaved A/B + a null-hook control |

## 4. Design limitations (not bugs)

1. **Exact match only.** The memory has no fuzzy retrieval by design — a wrong-but-close
   recall is worse than an abstain. The reflex can only fire on literal n-gram overlap.
2. **`norm()` still collapses some keys.** `"C++"` → `"c"`, and `"don't"` → `"dont"`.
   The store side has no minimum-key-length rule yet (the injector has one).
3. **Case is not preserved.** `norm()` lowercases; case-sensitive facts are unsupported.
4. **The digest guard detects tampering, not wrong writes.** If the caller stores a wrong
   value, the digest matches and the wrong value is returned.
5. **The bus keeps every tap in memory.** Mitigations exist (`clear_taps()`, `release()`,
   `offload_to_cpu=True`) but the default retains all layers.
6. **The reflex is not integrated into generation.** `low_margin` needs top-1/top-2
   probabilities, i.e. a forward pass. Running it on every token is expensive; the real
   integration must sample.
7. **`MemoryStore` is not thread-safe.** The MCP server is single-threaded stdio today.
8. **The reflex cannot match a stored key longer than `max_ngram=6` words.** An 8-word
   key never triggers. Undocumented limit, now noted.
9. **`Reflex` uses `store.has()`, which does not check the digest** — the trigger reports
   `memory_hit` for a corrupted slot. The injector then correctly refuses to inject, so
   nothing wrong is emitted, but the telemetry is misleading.
10. **`surgeon/` is report-only.** It reads counters you feed it; it does not capture
    activations itself.
11. **Negation is not detected by the bench matcher.** "not paris" still counts as
    correct. A real experiment needs a judge model, not a substring test.

## 5. Claims deliberately not made

- **No latency claim beyond "about 2%".** The measurement on this shared CPU is
  noise-dominated: a *null* hook also measured between 0.87× and 1.09× across runs.
  Anything more precise would be false precision.
- **No quality claim for the memory stack.** That the stack costs nothing in quality is
  a hypothesis until a host model runs it.
- **No "self-aware model" claim.** `check_myself` / `why` / `where` are not implemented.
  Reading activations tells you what the forward pass computed, not what the model
  "knows". `RESEARCH.md` §2.2 documents published evidence that this is weaker than it
  sounds — a supervised probe beats the model's own self-report 98.3% to 11.1%.
- **No novelty claim beyond the composition.** Forward hooks, activation steering,
  conditional steering (CAST), uncertainty-gated tool use (UALA) and MCP itself are all
  prior art. See `RESEARCH.md` §3.
- **No speed win claim.** The layer adds work.

## 6. What would make the next claim honest

```bash
pytest -q                              # gate 1: core still green (89 tests)
python experiments/e1_bus_detach.py    # gate 2: bit-exact detach  ✅ already passing
python experiments/e1b_latency_isolated.py   # gate 3: honest latency, randomised
```

Then add, in this order (`RESEARCH.md` §5):

1. **E2** — injector relevance-gate ablation + a neutral-memory baseline
   (required by the tool-drift literature).
2. **E3** — fix rate vs harm rate against a real host.
3. **E4** — tool-call recall/precision (needs a tool-calling host).
4. **E5** — the content-agnostic introspection control. *This is the experiment that
   decides whether the self-read tools are introspection or an anomaly detector.*
5. **E6** — a held-out verifier the intervention cannot edit.

The numbers from those runs — and only those — may appear in a README.
