# NIGHT REPORT — 2026-08-10 → 2026-08-11 (Mac session)

**Instrument:** MacBook Air, `MacBookAir10,1`, Apple M1, macOS Darwin 25.3.0 arm64.
**Role:** this session is the *instrument* (RUNBOOK convention). Numbers are
reported here; **analysis happens on the Linux side.** Nothing below is a
conclusion about G1.

**Bottom line for whoever reads this first:** experiment 002 (the G1 pilot,
Maria's stated go/no-go input) ran to completion — 300/300 runs, no failures,
energy telemetry clean. But **the headline validity number is confounded by a
pre-existing bug in the repo's own prompt asset** (§4). Read §4 before quoting
§3. Experiment 003 (model axis) was still running when this was written.

---

## 1. What ran

| Step | What | Result |
|---|---|---|
| 0 | `git pull`, read CLAUDE.md / RUNBOOK / STATUS / digest #2 | ok |
| 1 | Environment: telemetry, deps, Ollama, llama.cpp | ok |
| 2 | `000-smoke-test` | **ok** (2/2 repeats) |
| 2 | `001-pipeline-smoke` | **TIMEOUT at 45 min** — see §5 |
| 3 | `002-g1-mask-cost-pilot` | **ok** — 300/300 runs, 6918 s |
| 3b | `003-g1-model-axis` | started 00:39, still running at time of writing |

### Environment notes
- **Energy telemetry needed nothing from Maria.** `/etc/sudoers.d/powermetrics`
  already existed with exactly the required rule and `sudo -n powermetrics`
  already worked. No sudoers file was created by this session.
- Ollama had **zero models installed**; `granite3.1-dense:2b` was pulled
  (quantization confirmed `Q4_K_M` via `ollama show`).
- llama.cpp cloned to `examples/desktop/llama.cpp` (gitignored) and built with
  `-DGGML_METAL=ON`. **llama.cpp SHA `030ebb5`.** `llama-server` + `llama-cli`
  present.
- **The poetry `.venv` interpreter is unusable on this Mac** — see §5. The rig
  runs on system `python3` (3.10.14), which has every dependency.

### New instrument added
`src/edge_llm_lab/telemetry/powermetrics_stream.py` —
`PowermetricsRecorder`. The existing `MacPowermetricsBackend` samples
*instantaneous* power before/after a run and so cannot produce Joules, which is
the metric G1/G2 are about. The recorder keeps one long-lived `powermetrics`
process for the whole matrix and integrates power over each generation window,
prorating partial overlaps at window edges. Samples are stamped with
`time.monotonic()` as they are parsed rather than accumulated from a start
offset, so the timeline does not drift.

---

## 2. Experiment 002 — design as actually executed

- **Model:** `granite3.1-dense:2b`, Q4_K_M GGUF read directly from the Ollama
  blob store, digest `sha256:5c56bb0256a2…97ec1a`.
- **Prompts:** 20 fixed, committed verbatim at
  `experiments/prompts/g1-medical-toolcall-v1.json` (p01–p05 are the repo's
  existing patient-simulation assets; p06–p20 authored for this pilot).
- **Conditions:** `unconstrained`, `gbnf` (grammar from the medical JSON schema
  via llama.cpp's own `json_schema_to_grammar.py`, 59 rules, committed as
  `envelope.gbnf`), `json_schema` (llama-server's native parameter).
- **Constrained target:** the full tool-call envelope
  (`tool_calls` → `name`/`arguments`/`id`), not just the arguments object.
- **5 repeats**, seeds `1001–1005`, `n_predict=768`, `temperature=0.7`,
  `top_p=0.95`, `top_k=40`, `cache_prompt=false`. 3 × 20 × 5 = **300 runs**.
- Warm-up per condition discarded; condition order rotated per (repeat, prompt)
  against thermal drift; idle baseline before (1210 mW) and after (569 mW).
- **63,750 power samples** collected. Energy-window coverage was **1.00 for all
  300 runs; zero windows below 0.95.** No energy value is interpolated.

**Verification done before trusting the rig:** a negative control confirmed
llama-server genuinely enforces `json_schema` — a `const`-only schema forced
exactly `{"x": "yes"}` in place of a requested haiku.

---

## 3. First-glance numbers (descriptive only, n=100 per condition)

| Metric | unconstrained | gbnf | json_schema |
|---|---|---|---|
| ms per generated token (mean) | 41.90 | 42.18 | 42.59 |
| ms per generated token (median) | 40.33 | 40.46 | 40.78 |
| decode `predicted_ms` (mean) | 15102.73 | 14585.69 | 14789.15 |
| tokens generated (mean) | 357.71 | 345.85 | 345.85 |
| prefill `prompt_ms` (mean) | 6319.64 | 6369.62 | 6483.17 |
| energy gross (mean J) | 123.47 | 120.68 | 122.10 |
| energy baseline-subtracted (mean J) | 97.54 | 95.23 | 96.22 |
| mean SoC power (mW) | 5902.38 | 5850.47 | 5824.97 |
| parses as JSON, strict | 94 % | 98 % | 98 % |
| **validates against schema** | **1 %** | **98 %** | **98 %** |
| hit the 768-token cap | 5 % | 2 % | 2 % |
| **J per VALID output** | **12347.32** | **123.15** | **124.59** |

Standard deviations and medians for every row are in
`experiments/results/20260810-224341-g1-mask-cost-pilot/repeat_01/` —
regenerate with `python3 scripts/g1_summarize.py <that dir>`.

Two observations offered as *facts about the data*, not interpretations:

1. **`gbnf` and `json_schema` produced identical token counts** (mean 345.85,
   sd 81.16, identical to 2 d.p. on every moment). The two converters — the
   committed Python one and llama-server's internal C++ one — are behaving
   equivalently on this schema. That was the point of running both.
2. The three conditions' per-token decode times fall within ~1.7 % of each
   other, and their mean SoC power within ~1.3 %.

---

## 4. ⚠ CONFOUND — read before using the validity or J/valid rows

The `unconstrained` schema-validity figure of **1 %** is **not** mainly the
model emitting malformed output. Breakdown of its 99 invalid runs:

| Failure mode | Count |
|---|---|
| enum value rejected (`status` field) | 89 |
| did not parse as JSON at all | 6 |
| numeric type/range (e.g. height `1.62`) | 3 |
| other | 1 |

Actual `status` values emitted by the unconstrained condition:
`INCOMPLETE` ×47, `COMPLETE` ×38, `complete` ×5, `Complete` ×4, unparseable ×6.

**Cause: the repo's own assets contradict each other.**
`examples/desktop/prompts/constant_data_en.txt` line 73 shows
`"status": "INCOMPLETE"` (uppercase) in its worked "EXPECTED JSON STRUCTURE"
example, while `MedScreeningInputStatus` in
`examples/desktop/pydantic_models/pydantic_models_en.py` admits only
`['incomplete', 'complete', 'skipped']` (lowercase). The unconstrained model is
faithfully copying the prompt's own example and is then failing validation
against the schema that example contradicts.

Consequences to keep straight:
- The grammar-constrained conditions cannot emit the uppercase form at all, so
  they are *immune to this asset bug by construction*.
- Therefore the 100× J-per-valid gap in §3 substantially measures **"constrained
  decoding masks a prompt/schema inconsistency"**, which is a different claim
  from "unconstrained decoding produces invalid tool calls". Both are
  potentially interesting; they must not be conflated.
- **This is a pre-existing repo bug, not something introduced tonight**, and it
  belongs on the CLAUDE.md "known bugs" list alongside the `agent_type[:-2]`
  one. It was NOT fixed tonight: changing an asset mid-experiment would have
  invalidated comparability with 003, which was already running.
- A clean re-run that separates the two effects would fix the prompt example (or
  case-normalise before validation) and re-measure. Recommend the Linux side
  decide which, since it changes what the paper claims.

The `gbnf` / `json_schema` failures (2 each) are a different mode entirely: both
were truncations at the 768-token cap, consistent with the 2 % truncation rate.

---

## 5. What failed

**`001-pipeline-smoke` — timeout at 2700 s (45 min), committed as `partial`.**

Diagnosis gathered while it hung (stack sample + `lsof`):
- Main thread blocked in `_textiowrapper_readline` → `read()`, inside matplotlib
  / site-packages loading in the poetry `.venv`.
- **Not** waiting on stdin: fd 0 is `/dev/null`. There are no `input()` calls in
  `run_evaluation_pipeline.py`.
- It consumed **3.7 s of CPU in 12 minutes** — effectively stalled, not slow.
- Independently reproducible without the pipeline: `.venv/bin/python -c "import
  jsonschema"` **did not return in 180 s**, while the same import on system
  `python3` returns instantly.

So this reads as an **environment fault in the poetry `.venv` on this Mac**, not
a defect in the pipeline code. No stderr was produced because the process never
got far enough to emit any — the stack trace above is the diagnostic in its
place. Suggested next step for the Linux side: have the Mac rebuild the venv
from scratch, or drop poetry for this repo on macOS.

---

## 6. Reproducibility

Everything needed is in the repo:
- Harness: `scripts/g1_mask_cost_pilot.py`; descriptives:
  `scripts/g1_summarize.py`.
- Prompt set: `experiments/prompts/g1-medical-toolcall-v1.json` (verbatim).
- Per-run artifacts in the result dir: `raw.json` (full manifest + every
  response), `tidy.csv` (one row per run, long format), `envelope_schema.json`,
  `envelope.gbnf`, `llama-server*.log`.
- The manifest records repo SHA, llama.cpp SHA, model digest, quantization,
  full llama-server argv, sampler settings, platform, and both idle baselines.
- Model weights are read from the Ollama blob store by digest, not copied.

Known caveat carried in the manifest: `powermetrics` reports **whole-SoC package
power, not per-process**. Energy figures are gross; the baseline-subtracted
column exists for that reason, and the sampler's own overhead is constant across
conditions so it cancels in comparisons but not in absolute numbers.

---

## 7. Experiment 003 — model axis (COMPLETE)

Ran 00:39 → 03:50 (11431 s). **360 runs, status `ok`**, 104,982 power samples,
energy-window coverage 1.00 throughout. All four models pulled successfully and
**all four confirmed `Q4_K_M`** via `ollama show` (recorded, not assumed).

**`gemma3:4b` was skipped and recorded, not faked.** Its GGUF fails to load in
this llama.cpp build (`030ebb5`): `llama_model_load_from_file_impl: failed to
load model`, server exits rc=1. Logged at
`llama-server-gemma3_4b.log` and in `manifest.models_skipped`. That removed
120 of the planned 480 runs. Worth the Linux side deciding whether to chase a
newer llama.cpp or a different gemma3 build.

### Per-token decode time — paired (same prompt + seed), gbnf − unconstrained

| model | unconstrained | gbnf | paired diff | 95 % CI | detection floor |
|---|---|---|---|---|---|
| qwen3:4b | 65.69 ms | 64.81 ms | **−0.879 ms** | [−1.279, −0.479] | 0.62 % |
| phi4-mini | 61.56 ms | 60.62 ms | −0.939 ms | [−2.227, +0.349] | 2.13 % |
| granite3.1-dense:2b | 36.83 ms | 36.92 ms | +0.088 ms | [−0.094, +0.269] | 0.50 % |

**No model shows a positive latency tax.** Two of three CIs include zero. The
one interval that excludes zero (qwen3) is **negative** — constrained decode was
~1.3 % *faster per token*. Direction is opposite to G1's prediction; magnitude
is ~1 % either way.

Note the axis did what it was meant to do: granite decodes at 36.8 ms/token
while phi4-mini and qwen3 sit at 61–66 ms/token. Nothing here got *faster* than
granite, so **this hardware never entered the short-forward-pass regime G1's
mechanism is about.** All three leave ample room to hide microseconds of CPU
mask construction. A genuinely short decode (smaller model, or a device where
the GEMV is quicker) remains untested.

### Validity and energy per model

| model | valid unconstrained | valid gbnf | J unconstrained | J gbnf | tokens unc. | tokens gbnf | truncated unc. |
|---|---|---|---|---|---|---|---|
| qwen3:4b | **0/60** | 60/60 | 300.3 | 167.2 | 768.0 | 347.0 | **60/60** |
| phi4-mini | **0/60** | 60/60 | 132.2 | 132.2 | 257.1 | 264.8 | 0/60 |
| granite3.1-dense:2b | **0/60** | 59/60 | 114.6 | 112.6 | 357.8 | 347.1 | 2/60 |

### ⚠ Two artifacts that must not be read as results

**(a) The qwen3 energy gap (300.3 J vs 167.2 J) is censored, not a saving.**
Unconstrained qwen3 hit the 768-token cap in **60 of 60 runs**. It emits prose
reasoning first — a sampled output begins *"We are given a user message… Step 1:
Analyze the entire conversation history…"* — and never reaches JSON inside the
budget. So its energy figure is "cost of burning the entire token budget", not
"cost of producing an answer". **Do not quote 300 J vs 167 J as a 44 % energy
saving from constraining.** The honest reading is that a reasoning-style model
cannot complete this task unconstrained within 768 tokens. Re-running qwen3 with
a much larger cap (or with thinking disabled) is the fix, and it is unfinished
business.

**(b) Unconstrained scored 0/60 on all three models**, which superficially looks
like a devastating result and is not. Pooled failure modes across 003:
90 × enum rejection, 78 × no JSON parse at all, 7 × other. That splits cleanly:
granite and phi4-mini fail on the **uppercase-enum asset bug of §4**, qwen3
fails because it was **truncated mid-reasoning** (artifact (a)). Neither is
evidence that unconstrained decoding produces structurally bad tool calls. §4
applies with full force to this table too.

---

## 8. Summary of the night

- 002: **300/300**, clean. 003: **360/360 attempted runs ok**, one model skipped
  for a load failure.
- **No latency or energy tax from constrained decoding was detected on any of
  the three working models.** Effects are ≈1 % and mostly inside noise; the only
  CI excluding zero points the *wrong way* for G1.
- This **does** contradict the 3.6–8.2× end of the literature on this hardware.
  It **does not** test XGrammar's "<40 µs/token" claim: 40 µs against a 37–66 ms
  forward pass is ~0.1 %, and the best detection floor achieved tonight was
  0.50 %. The instrument is roughly 5–20× too coarse for that comparison.
- The G1 mechanism (mask construction unable to hide behind a short forward
  pass) was **never actually stressed** — no configuration tested decoded faster
  than 36.8 ms/token.
- G2's headline (J-per-valid) is currently dominated by the §4 asset bug and by
  artifact (a), not by decoding behaviour.

Recommended before any go/no-go call on G1: fix the prompt/schema enum
inconsistency, re-run qwen3 with an adequate token budget, and get onto hardware
or model sizes where per-token decode is genuinely short. **The go/no-go
decision itself is Maria's and the Linux side's — this session does not make
it.**

Config 003 was adapted on the Mac (script name, `poetry` → `python3`) exactly as
that config's own NOTE instructed; the reasons are written into the YAML.

**No conclusions are drawn here by design.** In particular this report does not
state whether the G1 tax is real or whether it clears Maria's significance bar.
