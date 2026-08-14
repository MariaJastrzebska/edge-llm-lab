# Optimization methods for the constrained-decoding latency tax (edge, single-stream)

*Author: systems-ML research agent, 2026-08-14. Grounds on
`experiments/results/NIGHT-REPORT-2026-08-13.md` (005: the crossover),
`-2026-08-12.md`, `NIGHT-REPORT.md`, `docs/research/from-science/2026-08-12-G1-go-no-go-agent.md`,
`docs/research/STATUS.md`, and a fresh web pass over XGrammar / XGrammar-2 /
llguidance / SGLang / vLLM / CFGzip (2026). All rig numbers are the repo's own.*

---

## 0. The mechanism, stated precisely (what the method must attack)

llama.cpp's grammar sampler applies the constraint **on the critical path, after
the GPU forward pass returns logits**: for every decode step it walks the
candidate array — the **entire vocabulary** — matching each token against the
grammar stacks and writing `-INFINITY` to every token that would violate the
grammar (confirmed against llama.cpp's `llama-grammar` apply path; the DeepWiki
description matches: *"the sampler iterates through all possible tokens in the
vocabulary … this iteration through the full vocabulary during each sampling step
is the primary source of computational overhead"*). This is an **O(vocab)**
CPU-side, single-threaded, serial cost that runs **between** `llama_decode`
(forward) and `common_sampler_sample`.

When the forward pass is long (≥34 ms/token, the 2–4 B models in rounds 002–004)
this CPU walk is a rounding error and hides in measurement noise (tax ≤ ±1.8 %,
half of it CI-crossing-zero). When the forward pass is **short** (sub-1B models,
11–25 ms/token in 005) the walk stops hiding and shows up as a paired tax:

| model | decode | absolute mask cost | relative tax |
|---|---|---|---|
| qwen2.5:0.5b | 11.65 ms/tok | +0.214 ms | +1.84 % |
| qwen3:0.6b | 13.71 | +0.331 ms | +2.42 % |
| gemma3:1b | 19.77 | **+6.002 ms** | **+30.36 %** |
| llama3.2:1b | 25.08 | +1.543 ms | +6.15 % (~+3.4 % length-corrected) |

The absolute cost varies **28×** and is **not** cleanly explained by vocab size
(STATUS iter-7 desk check: llama3.2:1b has the *smallest* vocab, 128 256, yet the
2nd-highest tax; the two Qwens share a vocab and differ 32 % in tax). **This
matters for method selection:** *whatever* the model-specific driver of the 28×
spread is (vocab, grammar-stack depth, tokenizer BPE structure), it is
**per-token CPU work sitting on the critical path**. A method that removes that
work from the critical path recovers the tax **without needing to resolve the
28× question** — a robustness property no vocab-specific trick has.

---

## 1. Candidate methods, ranked

Feasibility is judged against **this** stack: llama.cpp `030ebb5`, `llama-server`
subprocess, GBNF + `json_schema` grammars, Ollama weights, single-stream
(`--parallel 1`), Apple-Silicon unified memory.

| # | Method | Attacks mechanism by | Expected recovery (of +1.8…+30 %) | Feasibility on this stack | Changes outputs? | Novelty for single-stream edge |
|---|---|---|---|---|---|---|
| **A** | **Shadow-mask pipelining** — compute step-t's grammar mask on a CPU worker *concurrently with the GPU forward pass for step t*; only a cheap masked-write stays on the critical path | Moves the O(vocab) stack-walk **off** the critical path into the forward-pass shadow — restores the hiding that the short forward pass removed | **Most / all** of the tax on every model; gemma3's 6 ms mask fits inside its 19.8 ms forward with room. Conservative floor ~50–70 % on gemma3 after sync/contention | **Medium** — localized fork of the server decode loop + grammar sampler, ~200–400 LOC, no new deps, **bit-exactness-testable** | **No** (bit-identical logits/sampling) | **HERO** — never done in llama.cpp; prior overlap is datacenter-batched only; unvalidated on short-forward-pass edge; no one reports it in Joules |
| B | **State-keyed mask memoization** — LRU cache of the reject-set keyed on a hash of the grammar-stack state | Skips the walk on **repeated** grammar states (JSON free-string / whitespace runs revisit one state many steps) | Partial; strong on free-text-heavy schemas, ~0 on first visit and rare-state runs | **High** — pure CPU-side cache in the grammar sampler, ~100 LOC, no threading | No | Low — this is XGrammar's cache idea ported down; least novel |
| C | **Sparse accept-set (trie-driven)** — enumerate the *allowed* tokens via a byte-trie instead of scanning all vocab; O(accept-set) not O(vocab) | Kills the walk at **structural** positions where only a handful of tokens are legal | Large at structural positions, ~0 in free-string positions (accept-set ≈ vocab there) | **Low/Med** — reimplements the mask engine (this is essentially llguidance's tokenizer-trie); large fork | No | Low — duplicates llguidance/XGrammar's core |
| D | **Jump-forward / compressed FSM** — emit deterministic grammar runs (`{"tool_calls":[{"name":"…` ) with **no forward pass** | Removes whole forward passes on unique-continuation runs (attacks token *count*, not mask cost) | Can go **net-negative tax** (faster than unconstrained) but measures a *different* quantity | **Low** — heavy fork; retokenization hazard; needs KV-cache surgery | **Yes** (retokenization) | Low — SGLang already owns this (2024) |
| E | **SIMD/vectorized walk** | Shrinks the constant factor of the walk | Small; still O(vocab), still on critical path | Med | No | Low |
| F | **Move mask to GPU/Metal** | Offloads the walk to the GPU | Likely **negative** on edge — control-heavy stack-walk is GPU-hostile and competes with the forward pass on the shared memory bus | Low | No | Low |

**Winner: A — shadow-mask pipelining.** Highest impact (recovers the tax on every
model, headline gemma3), correctness-safe (bit-exact outputs), medium feasibility
on the exact rig Maria already builds, and — critically — the only candidate whose
benefit is **independent of the unresolved 28× driver**. B is the natural fallback
(cheaper patch, no threading) and can ride the same harness if A's threading proves
too invasive.

---

## 2. The hero method — shadow-mask pipelining, in detail

### 2.1 The key observation (why it's clean on single-stream edge)

The grammar mask at step *t* forbids tokens that violate the grammar **given
tokens 0…t-1 only**. It does **not** depend on `logits(t)`. The grammar stacks
are fully determined the instant token *t-1* is sampled. Therefore the *entire*
mask for step *t* can be computed **before `logits(t)` exist** — concurrently with
the GPU forward pass for step *t*, which also needs only token *t-1*.

XGrammar exploits exactly this (*"the mask generation only requires CPU, and only
depends on the previously generated tokens … the mask generation time is less than
LLM inference, so it will not become the bottleneck"* — XGrammar §CPU-GPU overlap).
But XGrammar splits tokens into context-independent (precompute) / context-dependent
(runtime) **because it amortizes across a batch** in datacenter serving. In our
**single-stream** case there is no batch to amortize and the whole one-sequence mask
is cheap (0.2–6 ms), so the split is unnecessary: we compute the **full** mask in
the shadow. The single-stream edge case is *simpler and cleaner* than the datacenter
one — a paper point, not a limitation.

### 2.2 The pipeline

```
             ┌────────────── step t ──────────────┐
sample(t-1) ─┤ advance grammar stacks → state S_t  │
             │   ┌ GPU: forward(t)  [11–25 ms]     │  ← run concurrently
             │   └ CPU worker: walk vocab vs S_t → │
             │        forbidden-bitset M_t [0.2–6 ms]
             │ JOIN: logits(t) ← forward           │
             │ apply M_t to logits(t)  [cheap]     │  ← only this stays on crit. path
             │ sample(t) → token[t]                │
             └─────────────────────────────────────┘
```

The expensive part — per-token **stack matching** over the vocab — moves into the
GPU shadow. Only the **masked write** (`for i in M_t: logits[i] = -inf`, a linear
memory pass, cheap) remains serial, and it is unavoidable because logits don't exist
until the forward completes.

### 2.3 Implementation sketch on llama.cpp `030ebb5`

Single binary, one runtime flag `--grammar-pipeline {off|on}` so the *only*
difference between arms is the pipelining (perfect within-run pairing; no binary
drift).

1. **Snapshot at accept.** In the grammar sampler's `accept` (`llama_grammar_accept_impl`,
   `src/llama-grammar.cpp`), after advancing the stacks for token *t-1*, take a
   cheap snapshot/clone of the current stacks (`llama_grammar_copy`-style — the
   stacks are small).
2. **Launch the shadow worker.** In `llama-server`'s decode loop (`server.cpp`),
   immediately before the `llama_decode` that runs forward(t), post a job to a
   **single persistent worker thread**: "compute the forbidden-token bitset for
   snapshot S_t". The worker runs the existing per-token match logic
   (`llama_grammar_apply_impl`'s inner test) against the snapshot, writing a
   `std::vector<uint8_t>` / bitset of length `n_vocab`. **Pin the worker to an
   efficiency core** (M-series `pthread_set_qos` / `THREAD_AFFINITY`) so it does
   not steal a performance core that feeds the GPU.
3. **Consume, don't recompute.** Replace the grammar sampler's `apply` with a
   fast path: when a precomputed bitset for the current step is ready, `join` the
   worker and apply the bitset (`logits[i] = -INFINITY where bitset[i]`) instead
   of walking the vocab. If the bitset is not ready (worker still running / flag
   off) fall back to the original synchronous walk — correctness is never at risk.
4. **Ordering guarantee.** The worker for step *t* is launched after `accept(t-1)`
   and joined before `sample(t)`; it can only ever race the `llama_decode(t)` call,
   which is precisely the overlap we want. Grammar state is read-only in the worker
   (operates on the snapshot), so no locking of the live grammar is needed.

**Scope:** ~200–400 LOC across `server.cpp`, `common/sampling.cpp`,
`src/llama-grammar.cpp`. No new dependencies. Maria already builds this tree
(`030ebb5`); the patch is a fork she can carry and upstream.

### 2.4 Expected impact, grounded in the numbers

Mask cost 0.214–6.002 ms; forward pass 11.65–25.08 ms — **every model's mask fits
inside its forward pass**, so the ceiling is full tax removal:

| model | tax now | mask 6.0 ms fits in fwd? | expected residual tax after pipelining |
|---|---|---|---|
| gemma3:1b | +30.36 % | 6.0 ms in 19.8 ms — **yes, ample** | headline target ~+3–10 % (≥⅔ recovered) |
| llama3.2:1b | +6.15 % | 1.5 ms in 25.1 ms — yes | ~+1–2 % |
| qwen3:0.6b | +2.42 % | 0.33 ms in 13.7 ms — yes | ~0–1 % |
| qwen2.5:0.5b | +1.84 % | 0.21 ms in 11.7 ms — yes | ~0–1 % |

**The one honest unknown — and why this is a validation, not a foregone win:** on a
**unified-memory** SoC the CPU shadow-walk and the GPU forward share one memory bus.
The walk reads small grammar structures and writes an n_vocab bitset (low bandwidth),
so contention *should* be minor — but it is not zero, and the forward pass here is
short, so even a small sl-down partially eats the gain. **That is exactly the
quantity the experiment measures.** The efficiency-core pin is the mitigation.

### 2.5 Risks

- **Outputs:** none — bit-identical mask, bit-identical sampling for a given seed.
  The experiment *asserts* this (per-seed output equality between arms); any
  mismatch is a pipeline bug, not a result.
- **Memory:** one n_vocab bitset (~0.13–0.26 MB) + one thread. Negligible.
- **Contention:** the real risk (§2.4); measured, mitigated by e-core pinning.
- **Deep/degenerate grammars:** if a single step's walk ever exceeds the forward
  pass, the pipeline caps the tax at (mask − forward), never worse than baseline
  (fallback path guarantees it).

---

## 3. Experiment spec — `experiments/queue/007-g1-mask-pipelining.yaml`

Measures before/after paired latency tax on the four sub-1B models, within one run,
with the pipelining flag as the only varying factor.

```yaml
name: g1-mask-pipelining
description: >
  HERO-METHOD VALIDATION. Tests shadow-mask pipelining: compute the grammar token
  mask on a CPU worker concurrently with the GPU forward pass (mask depends only on
  already-sampled tokens), leaving only the cheap masked-write on the critical path.
  Target: recover the constrained-decoding latency tax that 005 found on sub-1B
  models (qwen2.5:0.5b +1.84%, qwen3:0.6b +2.42%, llama3.2:1b +6.15%,
  gemma3:1b +30.36%).

  REQUIRES a patched llama-server exposing --grammar-pipeline {off|on} (single
  binary, one flag, so pipelining is the ONLY difference between arms — perfect
  within-run pairing, no cross-night binary drift). If the patch is not yet built,
  this config is a NO-OP placeholder: the runner records "patched binary absent"
  and skips, exactly like a failed model pull. Do not fake it.

  Four conditions per (model, prompt, seed), condition order rotated per repeat as
  in g1_mask_cost_pilot:
    - unconstrained                : reference forward-pass-only latency
    - gbnf_serial   (pipeline=off) : the current mask-on-critical-path (== 005)
    - gbnf_pipelined(pipeline=on)  : mask overlapped with the forward pass
    - json_schema_pipelined        : same, native converter, to confirm converter
                                     equivalence still holds under pipelining

  Primary readouts, paired within-run by (prompt, seed), n=60/model:
    1. tax_now      = gbnf_serial    − unconstrained         (must reproduce 005)
    2. tax_after    = gbnf_pipelined − unconstrained         (residual tax)
    3. recovered    = gbnf_serial    − gbnf_pipelined        (the speedup itself,
                       cleanest number — same constraint, only pipelining differs)
       report each with 95% CI; headline = recovered / tax_now on gemma3:1b.
    4. Joules per condition (baseline-subtracted) + J-per-valid-output, so the
       energy leg lands in the same run as the latency leg.
    5. CORRECTNESS GUARD: for each (prompt, seed), assert gbnf_pipelined output ==
       gbnf_serial output byte-for-byte. Report the equality rate; anything below
       ~100% means the pipeline is not mask-preserving and the latency win is void.
    6. CONTENTION PROBE: report gbnf_pipelined's forward-pass time (predicted_ms
       with mask cost removed) vs unconstrained's — if the shadow walk steals
       memory bandwidth on the unified-memory SoC, the pipelined forward is slower
       than the unconstrained forward; quantify that leak so the recovery number is
       honest about what the overlap costs.

  Reading the result: if `recovered` is CI-clean-positive and eats most of `tax_now`
  on gemma3:1b (the +30% model) while outputs stay bit-identical, the hero method is
  proven and the paper's optimization leg is done. If contention eats the gain
  (pipelined forward measurably slower than unconstrained forward), that is itself a
  clean, publishable finding about the limits of CPU-GPU overlap on unified-memory
  edge — the short forward pass that creates the tax also starves the fix.

requires: [ollama, git]
repeats: 1
command: >
  caffeinate -i python3 scripts/g1_mask_cost_pilot.py
  --models qwen2.5:0.5b,qwen3:0.6b,llama3.2:1b,gemma3:1b
  --conditions unconstrained,gbnf_serial,gbnf_pipelined,json_schema_pipelined
  --repeats 3
  --n-predict 768
timeout_minutes: 360
```

*Harness notes for the Mac side:* `gbnf_serial` / `gbnf_pipelined` /
`json_schema_pipelined` are new condition labels that map to the `--grammar-pipeline`
flag on the patched `llama-server` — a small addition to `g1_mask_cost_pilot.py`
(`LlamaServer.start()` gains the flag; `run_one` selects it by condition). Keep
`cache_prompt: false`, the warm-up discard, and the per-(repeat,prompt) condition
rotation unchanged — they are load-bearing. gemma3:1b is the headline model; give it
enough budget (8 of its 60 constrained runs in 005 hit the 768 cap — consider
n_predict=1024 for gemma3 only if headroom allows, but keep it fixed across arms so
the pairing holds).

---

## 4. Prior-art / novelty statement (honest)

**The overlap idea is not algorithmically new — its application here is.** Be precise
about what is and is not novel:

- **XGrammar** (2411.15100, CMU/MLC): adaptive token-mask cache (context-independent
  tokens >99 %, mask memory 160 MB→0.46 MB) **and** CPU-GPU overlap
  (*"mask generation only depends on previously generated tokens … less than LLM
  inference time"*). This is the **direct intellectual ancestor** of the hero method
  and must be cited as such. But XGrammar is evaluated on **H100 / RTX 4090** at
  **batch 1/16/32** as datacenter/desktop **throughput**; its edge presence is
  browser/WebAssembly only; it reports **no Joules**; and it is **not in llama.cpp**.
- **XGrammar-2** (2601.04426, May 2026): Earley-based adaptive mask cache, JIT,
  TagDispatch, Cross-Grammar Cache; *"near-zero end-to-end overhead in modern LLM
  serving systems"*; integrated into **SGLang/vLLM/TensorRT-LLM/MLC**. Explicitly
  **serving-system** framed; moved *further* from edge. Cite as the moving frontier;
  does not scoop.
- **llguidance** (2502.05111): ~50 µs/token CPU for a 128k tokenizer via tokenizer
  trie + lazy lexer + Earley parser. This is candidate C's territory (shrink the walk),
  **orthogonal** to pipelining (move the walk). General-purpose, not edge-energy.
- **SGLang compressed FSM + jump-forward** (LMSYS 2024; 2312.07104): candidate D;
  *also* overlaps mask generation with the GPU step — but reported as **datacenter
  throughput** (6.4× throughput / 3.7× latency), batched, no edge, no Joules.
- **vLLM structured outputs** (V1): *"minimal per-token overhead"*, backends
  Outlines / lm-format-enforcer / XGrammar — **datacenter serving**, no single-stream
  edge energy characterization.
- **CFGzip** (2605.29986, Saarland, May 2026): offline token-space compression, up to
  7.5× / 2 orders of magnitude on a SOTA CFG engine — candidate-C-family (shrink the
  search space), **not** overlap, **not** edge, **not** Joules.
- **Correctness-tax cluster** (Constraint Tax 2605.26128; "The Parser Already Knows"
  2608.10137): a different axis (validity ≠ correctness) — adjacent, cite, not a scoop.

**What is genuinely unscooped for single-stream edge:**
1. **The measurement.** No one reports the constrained-decoding latency tax — or its
   crossover at ~25 ms/token — on **single-stream** edge silicon. 005 does.
2. **The mechanism, in llama.cpp.** No one attributes the tax to llama.cpp's
   **un-overlapped, on-critical-path** vocab walk and shows it appears exactly when
   the forward pass shortens. STATUS's own architecture check found no evidence
   llama.cpp overlaps the mask the way SGLang/vLLM do.
3. **The fix, in llama.cpp, on edge, in Joules.** Porting overlap **into llama.cpp**
   (the dominant edge runtime) and validating it on **short-forward-pass** hardware —
   where it is least obvious the overlap fits and where the tax actually bites — in
   **both latency and energy**, is new. The regime inverts the datacenter intuition:
   there the forward pass is long and hiding the mask is trivial; here it is short,
   the mask is a large fraction of it, and whether the unified-memory bus lets the
   overlap pay off is an **open empirical question this experiment answers**.

The paper must not claim to invent overlap. It claims: the first edge single-stream
characterization + mechanism, and the first llama.cpp/edge/energy validation of
shadow-mask pipelining, including the unified-memory contention limit that datacenter
work never had to face.

---

## 5. Resulting paper contribution statement

> **Characterization.** On single-stream, Apple-Silicon-class edge inference, grammar-
> constrained decoding carries a per-token latency tax that is invisible at
> ≥34 ms/token but appears on **every** sub-1B model once the forward pass drops below
> ~25 ms/token — from +1.8 % (qwen2.5:0.5b) to **+30.4 % (gemma3:1b)** — contradicting
> the 3.6–8.2× figures that turn out to be a **batch-512 datacenter serving artifact**.
>
> **Mechanism.** The tax is llama.cpp's O(vocab) grammar mask-walk running
> **single-threaded on the critical path, after the forward pass**; it hides behind a
> long forward pass and stops hiding when the forward pass shortens. Its 28× cross-model
> spread is governed by a model-specific property (vocab size explains only gemma3;
> grammar-stack depth / tokenizer structure remain candidates) — but the *fix does not
> depend on resolving it*, because the cost is on the critical path regardless of size.
>
> **Optimization.** **Shadow-mask pipelining** computes the mask on a CPU worker
> concurrently with the GPU forward pass (the mask depends only on already-sampled
> tokens), leaving only a cheap masked-write on the critical path — restoring the hiding
> the short forward pass removed. Ported into llama.cpp (a ~200–400 LOC fork, bit-exact
> outputs) it recovers the bulk of the tax on edge silicon — headline: gemma3:1b's
> +30 % — measured in **both latency and Joules**, and it surfaces the
> **unified-memory contention limit** that the short-forward-pass edge regime imposes on
> CPU-GPU overlap, a constraint datacenter overlap work never encountered.

Three legs — characterization + mechanism + a validated optimization method — clearing
Maria's DECISIONS.md bar (never measurement alone). Robotics framing (DECISIONS.md
challenge #1): the tail-latency the pipeline removes is a control-loop deadline miss for
an on-robot action-grammar decoder, not just a JSON nicety.
