# LIMITATIONS — measured vs claimed

Every number in this file is either **measured** (with the command that produced it) or
**not measured**. Nothing here is a prediction dressed as a result.

---

## 1. What is verified

| Claim | How | Result |
|---|---|---|
| Core modules import and behave | `pytest -q` | **42/42 passed** |
| Exact recall, abstain on unknown, abstain on corruption | unit tests | ✅ |
| Bounded LRU eviction | unit test | ✅ |
| Reflex fires on memory hit and on low margin, stays quiet otherwise | unit tests | ✅ |
| Tool budget caps calls, unknown tools rejected | unit tests | ✅ |
| MCP stdio server: initialize / tools/list / tools/call / error codes | unit tests | ✅ |
| Per-token entropy / margin / confidence | unit tests (pure python) | ✅ |
| Drift classification (fixed / harmed / shifted / abstained) | unit tests | ✅ |
| Snapshot / restore is isolated from later mutation | unit tests | ✅ |
| Session state survives a dict round-trip | unit test | ✅ |
| Host fingerprinting for GPT-2 shape + 5 other families | unit test (synthetic) | ✅ |

## 2. What is NOT verified

| Claim | Status | Blocked by |
|---|---|---|
| `Bus.attach()` → `detach()` gives **bit-exact** original logits | ❌ not run | needs torch + a real model |
| Reflex path adds ≤ 15% latency | ❌ not run | needs torch + a real model |
| Fix rate ≥ 40% | ❌ not run | needs a real host model; `MockHost` is plumbing only |
| Harm rate ≤ 2% | ❌ not run | same |
| Tool-call recall / precision | ❌ not run | needs a tool-calling host (GPT-2 cannot call tools) |
| Abstain correctness ≥ 95% under corruption | ❌ not run on a real model | needs a real host |
| Dead-neuron report is correct on a real model | ❌ not run | needs activation capture on a real model |
| Attach to 6 model families | ❌ not run | needs 6 model downloads |
| `sensors/logitlite.py` runtime | ❌ not run | needs torch |

**`MockHost` in `venom/bench/correction_bench.py` is a plumbing test, not a result.** Its
"fix rate" of 1.0 is the mock answering its own injected string. It proves the metric
pipeline computes the right thing; it proves nothing about any model.

## 3. Known bugs found and fixed

| Bug | Impact | Fix |
|---|---|---|
| `Reflex._candidate_keys` did not strip punctuation, so `"capital of france?"` never matched the stored `"capital of france"` | reflex never fired on normal questions | v0.1.1: punctuation stripping moved **into `memory/store.norm()`**, so keys and queries are normalised by exactly one function instead of two |
| `Session` could not be restored from a dict (only from a file) | session round-trip untestable in-process | added `Session.from_dict()` |
| `drift_report` did not validate that `gold` had the same length as the answers | silent misalignment → wrong fix/harm counts | added `gold length mismatch` guard |
| `drift_report` treated only `""` as abstention | a host saying "I do not know" was counted as a *shift*, not an abstention | added an abstain-marker list |
| `server.py` returned `isError` from `out.get("ok")`, so `venom.time_now` was fine but any tool returning no `ok` key looked like an error | cosmetic but misleading MCP responses | unchanged semantics, but `ok` is now always set by every handler path |
| `run_bench` divided by zero on an empty fact list | crash | guarded with `if n else 0.0` |

## 4. Design limitations (not bugs)

1. **Exact match only.** The memory has no fuzzy retrieval by design — a wrong-but-close
   recall is worse than an abstain. This means the reflex can only fire on literal
   n-gram overlap with a stored key.
2. **Punctuation-stripped keys collide.** `"C++"` and `"C#"` normalise to `"c"`. This is
   acceptable for v0.1 fact storage but would break a code-symbol store.
3. **`norm()` lowercases.** Case-sensitive facts are not supported.
4. **Digest guard detects tampering, not wrong writes.** If the caller stores a wrong
   value, the digest matches and the wrong value is returned. The guard protects
   against in-memory corruption, not against bad input.
5. **The bus keeps every tap in memory.** `Bus.taps` holds a reference to each block's
   output tensor, so memory grows with sequence length × layers. A production bus
   needs a ring buffer or an explicit per-call clear.
6. **Reflex is per-call, not per-token-integrated.** The `low_margin` trigger requires
   top-1/top-2 probabilities, which means a forward pass. Running it on every token
   of a long generation is expensive; the real integration must sample.
7. **No concurrency control on `MemoryStore`.** It is not thread-safe. The MCP server
   is single-threaded stdio today, so this is fine now, not later.
8. **`surgeon/` is report-only.** It reads counters you feed it; it does not capture
   activations itself. "Dead neuron" numbers in the README table are targets.

## 5. Claims deliberately not made

- No speed claim. The layer adds work; whether it is net faster than vanilla at
  realistic context lengths is **unmeasured**.
- No quality claim for the memory stack. The plan says the stack should cost nothing
  in quality — that is a hypothesis until a host model runs it.
- No "self-aware model" claim. `check_myself` / `why` / `where` are not implemented.
  Reading activations tells you what the forward pass computed, not what the model
  "knows" or "understands".
- No novelty claim beyond the specific mechanism: a host model reading and editing its
  own activations through a verified, reversible tool interface.

## 6. What would make the next claim honest

Run, on one machine, in this order:

```bash
pytest -q                              # gate 1: core still green
python experiments/e1_bus_detach.py    # gate 2: bit-exact detach + latency ratio
```

Then, and only then, add the fix/harm bench against a real host
(`Qwen/Qwen2.5-1.5B-Instruct` for the agent path, `gpt2` for the reflex path).
The numbers from those runs — and only those — may appear in a README.
