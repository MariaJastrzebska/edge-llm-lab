# NIGHT REPORT — 2026-08-12 (Mac session): enum-fixed re-measure

**Instrument:** MacBook Air, `MacBookAir10,1`, Apple M1, Darwin 25.3.0 arm64.
**Repo at run time:** enum fix `b64b4ed` present, later rebased onto `c07b2e0`.
**llama.cpp:** `030ebb5` (unchanged from 2026-08-10, so builds are comparable).
**Predecessor:** `NIGHT-REPORT.md` (2026-08-10/11, confounded run).

**Purpose:** re-measure 002 and 003 with the corrected prompt asset so the
validity and energy-per-valid numbers are unconfounded by the uppercase-enum bug
documented in §4 of the previous report.

---

## 1. What ran

| Experiment | Runs | Duration | Status |
|---|---|---|---|
| 002-g1-mask-cost-pilot (re-run) | 300/300 | 6054 s | ok |
| 003-g1-model-axis (re-run) | 360 | 10206 s | ok, `gemma3:4b` skipped |
| 004-g1-enum-fixed-remeasure | — | — | started after 003 |

Results in fresh timestamped dirs; 2026-08-10/11 dirs untouched.
`gemma3:4b` skipped again for the same reason (GGUF will not load in llama.cpp
`030ebb5`, server exits rc=1) — recorded in `models_skipped`, never faked.
No `*_expected.json` fixture was modified.

Energy: 93,752 samples on 003, window coverage 1.00 throughout, nothing
interpolated. Telemetry fix `e0b8f58` is in the tree; I confirmed the
`cpu_freq()` bug did **not** damage the completed 002 — both resource snapshots
are intact with no `error` key.

---

## 2. THE DELTA — validity and energy-per-valid, before vs after

This is what the night was for. Model = `granite3.1-dense:2b`, experiment 002,
n=100 per condition.

| Metric | 08-10 (confounded) | 08-12 (enum fixed) |
|---|---|---|
| schema-valid, unconstrained | **1/100** | **18/100** |
| schema-valid, gbnf | 98/100 | **100/100** |
| schema-valid, json_schema | 98/100 | **100/100** |
| J per valid, unconstrained | **12347.32** | **702.08** |
| J per valid, gbnf | 123.15 | 122.84 |
| J per valid, json_schema | 124.59 | 123.36 |
| truncation, unconstrained | 5 % | 3 % |
| truncation, constrained | 2 % | 0 % |

**The fix worked, and it was not the whole story.** Unconstrained validity rose
1 % → 18 %; it did not approach the 100 % the constrained conditions now reach.
The 100× J-per-valid gap became a ~5.7× gap.

**The residual failures are now genuinely diverse**, which is the point — no
single artifact dominates any more. Experiment 002 residual modes (82 invalid
unconstrained runs): 28 × type error (an object supplied where a string is
required), 9 × no JSON at all, 6 × missing required property, 5 × `anyOf`
failure, 5 × out-of-range, 4 × `None` where a type is required, and a tail.

One of those is worth knowing about because it is *correct behaviour being
punished*: prompt **p15 deliberately says "I'm 310 cm tall"**, the model
faithfully records 310, and validation correctly rejects it against
`maximum: 250`. The grammar cannot emit 310 at all. Whether that should count
as an unconstrained "failure" is a scoring decision for the Linux side, not a
measurement one.

---

## 3. Decode time — FLAGGED, as requested

Decode-time results were expected to be unchanged. **They are not, and I am not
smoothing it.** Same model, same llama.cpp build, same flags, same prompts:

| granite3.1-dense:2b | unconstrained | gbnf |
|---|---|---|
| 08-10 (002) | 41.90 ms/token | 42.18 |
| 08-11 (003) | 36.83 | 36.92 |
| 08-12 (002) | 37.03 | 37.27 |
| 08-12 (003) | **34.64** | **34.75** |

Across all models, tonight is faster than 08-11: phi4-mini 61.56 → 52.71,
qwen3:4b 65.69 → 56.82 ms/token.

**Thermal drift does not explain it.** I checked the idle baselines, and they
point the other way: 08-10 had the *lowest* idle draw (1210 mW, vs 2536 mW on
08-11 and 2767 mW tonight) while producing the *slowest* decode.

**Leading hypothesis: observer contention — i.e. me.** On 08-10 I polled the
running experiment heavily on the same 8-core M1: progress checks, git commits,
file edits, ad-hoc Python analysis, process inspection, all while it measured.
On 08-11 I was somewhat quieter, and during tonight's runs I stayed off the
machine almost entirely. The ordering of the absolute latencies tracks that
behaviour monotonically. This is a hypothesis consistent with the data, not a
proven cause; a deliberate A/B (idle machine vs scripted background load) would
settle it and has not been run.

**Consequence for the analysis, which matters more than the cause:** absolute
per-token latency from this rig is **not comparable across nights**. Only
**within-run paired** comparisons should be used. The paired design was
included for thermal drift; it turns out to have been load-bearing for a
different reason.

---

## 4. Paired effect — the actual G1 quantity

Paired by (prompt, seed) within a single run, `gbnf − unconstrained`:

| model | run | unconstrained | paired diff | 95 % CI | relative |
|---|---|---|---|---|---|
| granite3.1-dense:2b | 08-12 002 | 37.03 ms/tok | +0.243 ms | [+0.002, +0.485] | +0.66 % |
| granite3.1-dense:2b | 08-12 003 | 34.64 | **+0.118 ms** | [+0.026, +0.209] | +0.34 % |
| phi4-mini | 08-12 003 | 52.71 | **+0.794 ms** | [+0.573, +1.015] | **+1.51 %** |
| qwen3:4b | 08-12 003 | 56.82 | **−1.000 ms** | [−1.630, −0.371] | −1.76 % |

Compare 08-11: granite +0.088 [−0.094, +0.269], phi4-mini −0.939 [−2.227,
+0.349], qwen3 −0.879 [−1.279, −0.479].

**What changed and what did not.** The magnitudes are still ~1 % — the picture
of "no large tax" is unchanged, and the 3.6–8.2× end of the literature remains
contradicted. What *did* change is resolution: with a quieter machine the
intervals tightened enough that all three now exclude zero, where two of three
did not on 08-11.

**The signs disagree across models**, and that is the most interesting line in
this report:
- granite **+0.34 %** and phi4-mini **+1.51 %** — small positive taxes, now
  resolved rather than lost in noise.
- qwen3 **−1.76 %** — constrained decode measured *faster* per token.

**Do not read qwen3's negative as a rebate.** Its unconstrained condition hit
the 768-token cap in **60/60** runs, so the two conditions are not decoding
comparable material: unconstrained qwen3 is being measured mid-reasoning-prose
at a truncation boundary, constrained qwen3 is producing a finished JSON object.
Per-token cost across those two regimes is not a like-for-like comparison. Fixing
this is exactly what queued config 006 is for; it was **not** run tonight, per
instruction.

Note also that phi4-mini's energy inverts (gbnf 150.9 J vs unconstrained
135.5 J) while granite's does not (117.4 vs 120.4). With n=60 and the token
counts differing between conditions, I am reporting that, not explaining it.

---

## 5. Per-model validity, experiment 003 (re-run)

| model | valid unconstrained | valid gbnf | J unc | J gbnf | tokens unc | truncated unc |
|---|---|---|---|---|---|---|
| qwen3:4b | 0/60 | 60/60 | 312.7 | 177.6 | 768.0 | **60/60** |
| phi4-mini | 4/60 | 59/60 | 135.5 | 150.9 | 265.1 | 1/60 |
| granite3.1-dense:2b | 8/60 | 60/60 | 120.4 | 117.4 | 355.4 | 2/60 |

Pooled residual unconstrained failure modes: 80 × no JSON parse (dominated by
qwen3's truncation), 21 × type error, 20 × `anyOf` failure, 8 × missing required,
6 × unexpected additional property, 5 × out-of-range.

qwen3's 0/60 is **entirely the truncation artifact** and has nothing to do with
the enum, exactly as flagged in advance. Its 312.7 J remains "cost of burning
the whole 768-token budget", not the cost of an answer.

---

## 6. Grader case-sensitivity — checked before running, as asked

- The grader **is** case-insensitive where it matters:
  `unreferenced_evaluator.py:127-131` lowercases model output and substring-
  matches into an internal canonical value. The lowercase prompt fix does not
  break scored evals.
- Uppercase literals at lines 381/470/489/799 compare against that internal
  canonical value, never raw model output. **No fixture was rewritten.**
- **`expected_status` is dead code** — zero Python references anywhere in the
  repo. The hazard does not exist because no comparison happens. Wiring it up or
  deleting it is a decision, not a bug fix; I did neither.
- Pre-existing quirk: the normalizer is two-valued. `'skipped'` does not contain
  `'complete'`, so it collapses to `INCOMPLETE`. `MedScreeningInputStatus` has
  three states; the evaluator models two.
- `validation.txt:19` uses a different field (`validation_status`) with no
  pydantic model, and is not on the G1 pilot's path.

---

## 7. Standing caveats

Unchanged from the previous report and still binding:

- This contradicts the 3.6–8.2× literature but **does not test** XGrammar's
  "<40 µs/token" claim: 40 µs against a 34–57 ms forward pass is ~0.1 %, and the
  best detection floor reached was ~0.3 %.
- **G1's actual mechanism was never stressed.** Nothing measured decodes faster
  than 34.6 ms/token, so the CPU mask always had ample room to hide behind the
  forward pass. Queued config 005 (short forward pass) targets this and was
  deferred by Maria, not rejected.
- `powermetrics` measures whole-SoC package power, not per-process. Figures are
  gross; baseline-subtracted columns are provided.

**No go/no-go conclusion is drawn here.** That decision is Maria's and the Linux
side's.
