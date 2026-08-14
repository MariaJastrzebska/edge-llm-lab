# Mechanism & measurement gaps for the constrained-decoding latency tax (2026-08-14)

*Author: systems-ML research agent. Grounds: `experiments/results/NIGHT-REPORT-2026-08-13.md`
(005/006), `docs/research/digests/2026-08-13.md` + `2026-08-14.md`, `NEEDS_INPUT.md`
(iter 6–7), `scripts/g1_mask_cost_pilot.py`, the 005 `repeat_01/raw.json` (480 runs,
full model outputs preserved), the `constant_data_en` assets and their ground-truth
fixtures. Incorporates Maria's controlled-matrix refinement (2026-08-14).*

---

## TL;DR

1. **Mechanism.** 005's four models (`qwen2.5:0.5b`, `qwen3:0.6b`, `llama3.2:1b`,
   `gemma3:1b`) differ on **vendor, parameter count, vocab, and architecture all at
   once**, so the 28× tax spread is attributable to nothing. The fix is a **controlled
   matrix that varies one factor at a time**: (A) same-family size sweeps (vocab &
   arch fixed, only forward-pass length changes), (B) a vocab-isolation cell (decode
   speed roughly matched, vocab swept 32k→262k), (C) cross-vendor replication of both
   effects, plus (D) a **trivial-grammar control condition** that separates the pure
   `O(vocab)` mask *scan* from grammar-*stack* depth. The harness must record
   `n_vocab`, `n_layer`, `n_embd`, `n_params`, and measured decode speed per model so
   the analysis can regress relative tax on the *mask* predictors vs the *forward-pass*
   predictor and attribute the variance.

2. **Correctness gap.** The harness grades schema-validity only. `raw.json` for 005 and
   006 **preserves the full model `output` string for every one of the 480/40 runs**, so
   correctness can be **re-scored 100% offline — no re-run needed**. Only p06–p20 need a
   gold-answer asset authored (p01–p05 already have `_expected.json` fixtures).

3. **Biggest reviewer risk:** the central mechanism claim ("tax ∝ vocab") is
   **contradicted by the authors' own four-point table** (llama3.2 has the *smallest*
   vocab yet the 2nd-largest tax; the two Qwens share one vocab but differ 32% in tax).
   Rebuttal: stop selling vocab as *the* driver. Report a **two-factor mask-cost model**
   — per-token cost `≈ c·n_vocab` (isolated by the trivial-grammar control) plus a
   fragmentation/steps term — fit on the *controlled* matrix, not the confounded four.

---

## TASK 1 — MECHANISM: the controlled matrix that makes the 28× attributable

### 1.1 Reconciling iteration 7 (the vocab hypothesis "weakened")

`digests/2026-08-14.md` Part 1 ranked the 005 four by public `n_vocab`:

| model | n_vocab | 005 raw tax | length-corrected (NIGHT-REPORT §4) |
|---|---|---|---|
| llama3.2:1b | 128,256 | +6.15% | **~+3.4%** |
| qwen2.5:0.5b | 151,936 | +1.84% | ~+1.8% |
| qwen3:0.6b | 151,936 | +2.42% | ≳+2.4% |
| gemma3:1b | 262,144 | +30.36% | ~+29% |

Vocab does not rank-order the *raw* tax (llama3.2 smallest vocab, 2nd tax; Qwens tie on
vocab, differ 32%). **But this table cannot refute the vocab hypothesis, because it
also cannot confirm anything** — the four models differ on every axis simultaneously,
and two of the discrepancies partly dissolve once known confounds are removed: llama3.2's
tax nearly halves under length correction (→ ~+3.4%, much closer to the Qwens), and the
Qwen 0.5b/0.6b pair differ in *size and architecture* (Qwen2.5 vs Qwen3), not just seed
noise. The honest reading is **"vocab is neither confirmed nor refuted by 005; 005 is the
wrong instrument to ask."** The controlled matrix below is what settles it.

Note also: even a *pure* `O(vocab)` model does not predict gemma3's magnitude — 262k is
2× llama3.2's 128k but the tax is ~8×. That super-linearity is itself a clue that a
second factor (tokenizer fragmentation / per-state acceptance-set size / SentencePiece
large-vocab scan) co-determines the cost. The experiment is designed to expose it, not
to assume vocab wins.

### 1.2 Why 005 "tests nothing" and what one-factor-at-a-time buys

The mechanism under test is: **per generated token, llama.cpp walks the vocabulary on the
CPU testing each candidate token against the current grammar stack, inside the sampler
chain, on the critical path — a cost that cannot hide behind the GPU forward pass once the
forward pass is short** (llama.cpp does *not* overlap mask construction with inference the
way vLLM/SGLang/XGrammar-precompute do — see §Scoop). Decomposed:

```
relative_tax  ≈  mask_ms_per_token / forward_ms_per_token
mask_ms_per_token  ≈  c(model) · n_vocab · steps_factor
```

- `forward_ms_per_token` is set by **parameter count / arch** (the denominator).
- `n_vocab` sets the **scan width** (the numerator's leading term).
- `c(model)` and `steps_factor` absorb **tokenizer fragmentation** (how many tokens a
  model spends to emit the same JSON bytes) and **per-state acceptance-set size**.

005 moved the numerator *and* the denominator *and* `c(model)` all at once. To attribute,
hold all-but-one fixed:

### 1.3 The matrix (Ollama-available; two vendors per effect where feasible)

**Cell A — same-family SIZE sweeps (vocab, tokenizer, arch FIXED; only forward-pass grows).**
This is the cleanest test and runs first. Prediction: within a family, absolute mask
cost (ms/token) is **~constant** (same vocab, same grammar), so **relative tax falls
~1/decode_time** as the model gets bigger.

| cell | Ollama ref | ~params | n_vocab (expected) | role |
|---|---|---|---|---|
| A-qwen | `qwen2.5:0.5b` | 0.49B | 151,936 | fast decode |
| A-qwen | `qwen2.5:1.5b` | 1.54B | 151,936 | mid |
| A-qwen | `qwen2.5:3b`   | 3.09B | 151,936 | slow decode |
| A-llama | `llama3.2:1b` | 1.24B | 128,256 | fast |
| A-llama | `llama3.2:3b` | 3.21B | 128,256 | slow |
| A-gemma | `gemma3:1b`   | 1.0B  | 262,144 | fast, huge vocab |
| A-gemma | `gemma3:4b`   | 4.3B  | 262,144 | slow, huge vocab |

The gemma sub-sweep is the strongest single prediction in the whole design: if gemma3:1b's
+30% is an `O(vocab)` mask cost hiding poorly behind a short forward pass, then **gemma3:4b
should pay the same ~6 ms absolute mask cost but a much smaller relative tax** (longer
forward pass). If instead gemma3:4b still pays ~30%, the cost is *not* a fixed mask hiding
behind decode — it scales with something that also scales with the model, and the
"CPU-mask-stops-hiding" story is wrong for gemma.

**Cell B — VOCAB isolation (params/decode roughly matched; vocab swept ~4×).** ~1–2B tier
so forward-pass times are close; residual decode-speed differences are regressed out
(they are also directly measured per run).

| Ollama ref | ~params | n_vocab (expected) | tokenizer family |
|---|---|---|---|
| `tinyllama:1.1b` | 1.1B | **32,000** | Llama-2 SentencePiece (small) |
| `smollm2:1.7b` | 1.7B | 49,152 | GPT-2/SmolLM BPE |
| `stablelm2:1.6b` | 1.6B | 100,352 | GPT-NeoX BPE |
| `llama3.2:1b` | 1.24B | 128,256 | tiktoken BPE |
| `qwen2.5:1.5b` | 1.54B | 151,936 | Qwen BPE |
| `gemma3:1b` | 1.0B | **262,144** | Gemma SentencePiece (large) |

Prediction if vocab is a driver: with forward-pass held roughly constant, absolute mask
cost rises monotonically with `n_vocab` (tinyllama ≈ floor, gemma ≈ ceiling). tinyllama at
32k is the key point — if it pays a near-zero mask cost while gemma at 262k pays ~6 ms and
the middle interpolates, `O(vocab)` is confirmed as *a* driver.

**Cell C — cross-vendor replication.** Cells A and B already span Qwen (Alibaba), Llama
(Meta), Gemma (Google). Add one more vendor at the ~1–2B tier if a clean Ollama build
exists (`phi3.5` is 3.8B — too big/slow for the fast band; `internlm2:1.8b`, vocab
92,544, is a better matched cross-vendor point). The claim we can make from C: whether the
size-effect (A) and vocab-effect (B) **replicate across architecture families** or are
family-specific. Two models per cell where the pull succeeds; a model that will not load is
recorded and skipped (existing harness behaviour), never faked.

**Cell D — trivial-grammar CONTROL condition (separates scan width from stack depth).**
Add a third condition alongside `unconstrained` and `gbnf` (envelope): **`gbnf_trivial`**,
a grammar that constrains essentially nothing per step, e.g.

```gbnf
root ::= char+
char ::= [^\x00]
```

llama.cpp still walks the **full vocabulary** against this grammar every step, so
`gbnf_trivial` pays the pure `O(vocab)` scan **with almost no grammar-stack work**. Then:

- `tax(gbnf_trivial)` ≈ the pure scan-width cost (should track `n_vocab`).
- `tax(gbnf_envelope) − tax(gbnf_trivial)` ≈ the grammar-stack / acceptance-set /
  fragmentation cost.

If envelope ≈ trivial → the tax is dominated by the vocab scan (vocab wins). If envelope
≫ trivial → grammar-stack depth dominates and vocab is a red herring. **This single control
condition is what turns "we regressed on public vocab numbers" into "we measured which
term dominates."** It costs one extra condition per run.

### 1.4 What the harness must record per model (currently it records NONE of this)

**Gap found:** `g1_mask_cost_pilot.py` writes a per-model `llama-server-<ref>.log`, but
`.gitignore` line 59 (`*.log`) means those logs — which contain llama.cpp's full metadata
dump including `n_vocab` — are **never committed**, and the script never parses them. So
`n_vocab` was *available on the Mac and thrown away*. Required change:

1. After `server.start()`, read the server log and extract, via regex over the
   `print_info:` / `llm_load_print_meta:` block:
   - `n_vocab` (`r'n_vocab\s*=\s*(\d+)'`, fallback `r'vocab_size\s*=\s*(\d+)'`)
   - `n_layer` (`r'n_layer\s*=\s*(\d+)'`), `n_embd`, `n_head`, `n_params`,
     `n_ctx_train`, and the printed model size / architecture string.
2. **Belt-and-braces (log-independent):** read the GGUF metadata directly with the
   `gguf` Python reader on `model_info["gguf_path"]` — keys `<arch>.vocab_size`,
   `<arch>.block_count`, `<arch>.embedding_length`, and `len(tokenizer.ggml.tokens)`.
   This does not depend on log format drift across llama.cpp versions.
3. Write these into `manifest["models_resolved"][ref]` **and** as columns on every tidy
   row (`n_vocab`, `n_layer`, `n_embd`, `n_params`) so the regression frame is
   self-contained.
4. Stop discarding the server log: either commit it (drop it from `.gitignore` for the
   results tree) or persist the parsed metadata block as `model_meta.json` in the run dir.

Also record, to rule out the batch/threading alternative: `--parallel` (already 1),
`-t`/`--threads` (currently unset → llama.cpp default; **set it explicitly and record
it**), `-ngl` (99), and the quantization (already captured via `ollama show`).

### 1.5 Analysis that confirms or refutes "tax ∝ vocab"

Per (model, condition) compute the paired within-run `constrained − unconstrained`
per-token latency, matched on (prompt, seed) — as 005 already does — but now **length-
controlled**: fit per-token decode time as a function of output-token position (or include
`predicted_n` as a covariate) so the KV-growth confound of NIGHT-REPORT §4 is removed with
a fitted model, not a two-bucket slope.

Then two regressions on the **controlled** frame:

- **Within-family (Cell A):** regress absolute mask cost (ms/token) on `1/n_vocab`-free
  terms — i.e. test that within a family it is ~constant in size and that relative tax ∝
  `1/decode_time`. A clean `1/decode_time` fit *with vocab constant* is direct confirmation
  of the forward-pass-hiding mechanism, independent of vocab.
- **Across the pooled matrix:** regress absolute mask cost on `n_vocab`,
  `decode_ms/token`, `predicted_n` (fragmentation proxy), and vendor dummy. Report
  standardized coefficients and partial-R². The mechanism claim is supported iff
  **`n_vocab` carries significant variance after `decode_ms` and `predicted_n` are
  partialled out**, and the trivial-grammar decomposition (Cell D) shows the scan term is
  the larger of the two. If `n_vocab`'s partial contribution vanishes once fragmentation is
  included, report *that* — it is still a clean mechanism result ("cost is dominated by
  number of constrained steps, not raw vocab width").

### 1.6 Alternative drivers and how each is ruled in/out

| candidate driver | how it is isolated / measured |
|---|---|
| **forward-pass length** (decode speed) | Cell A holds vocab/arch fixed and varies only size → any tax change is pure forward-pass. Recorded as `decode_ms/token` per run. |
| **vocab / mask-scan width** | Cell B holds size≈fixed, sweeps vocab 32k→262k; `n_vocab` recorded per run; trivial-grammar (D) isolates the scan term. |
| **grammar-stack depth / acceptance-set size** | `tax(envelope) − tax(trivial)` from Cell D; same grammar across all models so this term varies only via tokenizer. |
| **tokenizer fragmentation** (steps to emit same JSON) | tokenize one fixed gold envelope with each model's tokenizer; record `tokens_per_gold_envelope`; also use `predicted_n` on the constrained arm as covariate. |
| **allowed-tokens-per-state** | (optional deep probe) instrument the llama.cpp sampler to log mean accepted-set size per step; otherwise argued constant since grammar is identical and bounded by tokenizer. |
| **batch / threading / SIMD** | held constant: `--parallel 1`, `--threads` set explicitly & recorded, same machine; noted as a scope boundary, not a variable. |
| **quantization** | all pulled at Ollama default (record actual via `ollama show`); affects forward-pass not mask cost — recorded, not swept. |
| **length / KV-growth confound** | length-controlled regression (§1.5), not two-bucket slopes. |

### 1.7 Queue-YAML spec

```yaml
# experiments/queue/007-g1-controlled-mechanism-matrix.yaml
name: g1-controlled-mechanism-matrix
description: >
  Attribute the 28x mask-cost spread from 005 by varying ONE factor at a time.
  005's four models differ on vendor/size/vocab/arch simultaneously, so the spread
  is unattributable. This runs three controlled cells plus a trivial-grammar control:
    A) same-family SIZE sweeps  -> vocab & arch fixed, forward-pass varies
         qwen2.5:0.5b/1.5b/3b (vocab 151936), llama3.2:1b/3b (128256),
         gemma3:1b/4b (262144)   -- prediction: relative tax ~ mask_ms/decode_ms,
         absolute mask cost ~constant within family.
    B) VOCAB isolation  -> params/decode ~matched, vocab swept 32k..262k
         tinyllama:1.1b(32000), smollm2:1.7b(49152), stablelm2:1.6b(100352),
         llama3.2:1b(128256), qwen2.5:1.5b(151936), gemma3:1b(262144)
    C) cross-vendor replication of A and B (Qwen/Llama/Gemma already spanned;
         add internlm2:1.8b if it loads cleanly).
    D) CONTROL condition gbnf_trivial (root ::= char+) isolates the pure O(vocab)
         scan from grammar-stack depth:
         tax(envelope) - tax(trivial) = the grammar/fragmentation term.
  The harness MUST record per model: n_vocab, n_layer, n_embd, n_params, quantization,
  and measured decode ms/token, so the analysis can regress relative tax on the mask
  predictors (vocab, allowed-tokens-per-state) vs the forward-pass predictor (decode
  time) and attribute variance within-family and across-family. Models that fail to
  pull/load are recorded and skipped, never faked.
requires: [ollama, git]
repeats: 1
command: >
  caffeinate -i python3 scripts/g1_mask_cost_pilot.py
  --models qwen2.5:0.5b,qwen2.5:1.5b,qwen2.5:3b,llama3.2:1b,llama3.2:3b,gemma3:1b,gemma3:4b,tinyllama:1.1b,smollm2:1.7b,stablelm2:1.6b,internlm2:1.8b
  --conditions unconstrained,gbnf,gbnf_trivial
  --repeats 3
  --record-model-meta
timeout_minutes: 600
# NOTE for the Mac side: --record-model-meta and the gbnf_trivial condition are NEW.
# If unimplemented at run time, (a) n_vocab/n_layer/n_params can still be recovered
# offline from the committed server logs / GGUF headers, so DO NOT block the run on it,
# and (b) drop gbnf_trivial and run unconstrained,gbnf -- the size/vocab cells still
# execute; only the scan-vs-stack decomposition is deferred. Prefer splitting into two
# nights (Cell A first: qwen2.5:0.5b/1.5b/3b + llama3.2:1b/3b + gemma3:1b/4b) if 600 min
# is tight -- Cell A alone answers the mechanism question.
```

*(Directory note: `experiments/queue/` does not exist in the working tree — the six run
configs live in `experiments/done/`. Create `experiments/queue/` and drop this YAML there,
matching the RUNBOOK flow; the runner pulls from the queue and moves to `done/` on
completion.)*

---

## TASK 2 — THE CORRECTNESS-TAX GAP

### 2.1 The gap, concretely

Per the Constraint Tax paper (arXiv 2605.26128): on a near-identical task shape (sub-2B
model, hard tool-call JSON schema) a model can go 91.5%→48.0% **answer correctness** while
schema-validity stays 100%→100%. 005 grades **schema-validity only** (60/60 constrained vs
0/60 unconstrained). So 005's constrained arm's "60/60 valid" **cannot currently distinguish
a correctly-filled medical envelope from schema-valid nonsense the model would never have
produced unconstrained.** No paper claim of the form "constrained decoding is good/free/
harmless here" is defensible until a correctness axis exists.

### 2.2 What "correct" means for the medical-envelope task

The `constant_data_en` task is **single-turn field extraction**: given one patient
utterance, emit a `send_medical_data_en` tool call whose `arguments.current_info` holds the
five fields extracted from that utterance, a `status`, and (if incomplete) a `missing_info`
question. Correctness is graded on three axes, most-objective first:

1. **Field extraction (primary, deterministic).** Per prompt, `current_info` must match the
   gold fields after normalization:
   - `name`: exact/case-insensitive string, or `null`.
   - `gender`: `male|female|other`, incl. name-inference cases (p11 "Katarzyna" → female;
     the system prompt *instructs* this inference).
   - `height`: normalize to integer cm — `5'5"`→165, `six foot two`→188, `1.75 m`→175.
   - `date_of_birth`: normalize to ISO — `12/06/1987`, `8 Aug 1992`, `1995-04-02` → `YYYY-MM-DD`.
   - `blood_type`: normalize synonyms — `zero positive`→`O+`, `B positive`→`B+`.
   Score per-field accuracy (5 fields) **and** an all-fields-exact flag.
2. **Hallucination / refusal handling (primary for the hard prompts).** For refusal (p04),
   confused (p05), off-topic (p17), injection (p18), out-of-range (p15) prompts, "correct"
   = **does NOT fabricate values** (fields the user never gave must be `null`) and does not
   break format under injection. p15 is a trap: height 310 / year 2049 / blood type "Q+"
   all violate the schema's ranges/enums — a correct model emits `null` for the invalid
   fields, it does not launder them into the envelope.
3. **Status (secondary).** `status ∈ {incomplete, complete, skipped}` must be in the gold's
   acceptable set. Graded loosely: single-turn "complete vs incomplete" is genuinely
   ambiguous under the system prompt's "optional asked once" rule, so status is a secondary
   signal, not a pass/fail gate. (Also: the existing fixtures use **uppercase**
   `COMPLETE`/`INCOMPLETE` while the schema enum is lowercase — see CLAUDE.md's enum-case
   note — so the status comparison MUST be case-insensitive.)

**Headline correctness metric:** *fraction of runs whose `current_info` field-extraction is
exactly correct* (axis 1) **and** *not-hallucinated on the hard prompts* (axis 2). This is
what feeds "Joules per valid-AND-correct output" (the G1′ denominator from the go/no-go doc).

### 2.3 Where ground truth comes from

- **p01–p05:** already exist as
  `examples/desktop/input/agents/constant_data_en/conversation_stimulation/{successful,unsuccessful}/*_expected.json`
  (`expected_data` = the five fields, `expected_status`). Caveats to encode in the grader:
  these were authored as **multi-turn** ground truth (`max_rounds`, `acceptable_partial_data`)
  and use uppercase status; the single-turn grader consumes only `expected_data` +
  case-normalized `expected_status`. One fixture is internally loose (Minimal_patient lists
  `height:"160"` though p02 says 5'5"≈165) — the grader should treat height with a ±1 cm
  tolerance and flag such fixtures for Maria rather than silently trusting them.
- **p06–p20:** authored for the pilot, **no fixtures yet.** Author a single gold asset
  `experiments/prompts/g1-medical-toolcall-v1-gold.json` — one entry per prompt id with
  `expected_current_info` (5 fields, normalized), `acceptable_status` (a set), a
  `must_be_null` list (for hallucination checks), and a free-text `note` (e.g. p15
  out-of-range, p18 injection, p14 contradiction — accept last-stated `female`). This is
  ~1 hour of authoring and makes the set fully reproducible from the repo.

### 2.4 Re-score offline, or re-run?

**Re-score offline — no re-run needed for 005 or 006.** Verified: the 005
`repeat_01/raw.json` stores the full `output` string for **all 480 runs** (60 per
model×condition, every one present), and 006 does the same. A standalone grader reads
`raw.json`, extracts `tool_calls[0].arguments.current_info` from each output (reusing the
harness's balanced-brace / fence-stripping extractor in `assess_validity`), normalizes,
compares to gold, and writes correctness columns alongside the existing tidy rows.

Practical scope of what offline re-scoring can answer *today*:
- **Constrained arm:** fully answerable — of gemma3's 52/60 and llama3.2's 58/60
  schema-valid constrained outputs, how many are also field-correct? This is the exact
  Constraint-Tax question and it needs **zero new hardware**.
- **Unconstrained arm:** 005 unconstrained was 0/60 schema-valid on all four models, so
  most unconstrained outputs have no parseable envelope → correctness ≈ 0 by construction.
  That is itself the point (unconstrained sub-1B models don't produce the envelope), but it
  means the *interesting* correctness contrast is **valid-vs-correct within the constrained
  arm**, plus lenient extraction from unconstrained prose where possible.

The correctness axis only needs a **re-run** if Maria wants it wired into the live harness
for future runs (007+) — desirable, but not a blocker for scoring what already exists.

### 2.5 Grader design (offline, deterministic + optional judge)

```
scripts/grade_correctness.py  <run_dir>  [--gold experiments/prompts/g1-...-gold.json]
  -> reads raw.json, for each run:
       parse output -> tool_calls[0].arguments.current_info   (reuse _extract_first_object)
       normalize {height->cm int, dob->ISO, blood_type synonyms, gender lower}
       compare to gold; emit per-field bools + all_fields_exact + no_hallucination
       status_ok = current.status.lower() in gold.acceptable_status
  -> writes correctness.csv (joins to tidy.csv on run_id) with columns:
       run_id, field_name_ok, field_gender_ok, field_height_ok, field_dob_ok,
       field_blood_ok, all_fields_exact, no_hallucination, status_ok, correct_overall
  -> prints per (model,condition): schema_valid_rate vs correct_rate  (the tax table)
```

Deterministic normalization is preferred over an LLM judge (reproducible, no judge
confound); an LLM-judge pass is a *secondary* cross-check on the free-text `thoughts`
field only, never the primary gate. Output feeds "J per valid-correct output" by joining
correctness to the existing `combined_j` / `combined_j_baseline_subtracted` columns.

---

## TASK 3 — REVIEWER DEFENSE & SCOOP

### 3.1 Reviewer-attack table

| # | Attack | Rebuttal / preemption |
|---|---|---|
| 1 | **n=1 hardware** (single M1 Air) — effect is a quirk of one SoC. | Mechanism is architecture-general (shared CPU/GPU memory bus + on-critical-path `O(vocab)` CPU scan). Report the **crossover as a forward-pass threshold (~25 ms/token)**, not an absolute ms, so it transfers. Commit to replicating on a 2nd device class — the repo's telemetry already has Linux-RAPL and Jetson-tegrastats backends (CLAUDE.md); one x86 + one Jetson run closes this. |
| 2 | **Single task** (one medical schema). | The `O(vocab)` scan is **task-independent by construction** — the trivial-grammar control (Cell D) shows the scan cost with essentially no schema. Add one 2nd grammar family (generic JSON + a regex/enum grammar) to show grammar-generality. |
| 3 | **llama.cpp-specific** — vLLM/SGLang/XGrammar already hide this. | *That is the scope, stated as the contribution boundary.* vLLM/SGLang/XGrammar **overlap or precompute** the mask; llama.cpp masks on the critical path and is what **on-device** ships (fllama in the fed-mobile app is llama.cpp). Claim is "for the dominant on-device engine, single-stream," and we cite XGrammar/XGrammar-2 as the datacenter contrast. |
| 4 | **Single-stream only** — real serving is batched. | Single-stream **is** the edge regime and is exactly the corner the literature's big taxes do *not* cover (those are batch-serving: ~37.5% at batch 512). We characterize the opposite corner; batch=1 is declared as the contribution boundary, not hidden. |
| 5 | **Validity ≠ correctness** (Constraint Tax 2605.26128; "When Correct Isn't Usable" 2605.02363). | Task 2 adds correctness grading; report **Joules-per-valid-AND-correct**. Existing 005/006 re-scored offline so the correctness axis is present at submission, not promised. |
| 6 | **Length confound** — grammar changes how much the model writes. | Length-controlled regression (§1.5); every tax reported on matched output length; per-run token counts already in `tidy.csv`. |
| 7 | **Energy is whole-SoC, not per-process.** | powermetrics caveat documented in the manifest; report gross **and** baseline-subtracted; idle baseline measured before & after each run. |
| 8 | **"Constrained decoding is *faster*, not slower"** (JSONSchemaBench/llguidance report speedups via speculative masking). | True *for engines that speculate/overlap the mask*. llama.cpp does neither — this is precisely why the tax appears here and not there. We measure the engine that doesn't hide it; the contrast is the paper's point. |
| 9 | **★ The vocab mechanism is contradicted by your own data** (llama3.2 smallest vocab, 2nd tax; two Qwens same vocab, 32% apart). | **The single biggest risk.** Do NOT sell "tax ∝ vocab." Reframe to a **two-factor mask-cost model** — per-token cost `≈ c·n_vocab` (scan, isolated by the trivial-grammar control) **plus** a tokenizer-fragmentation/steps term — and fit it on the **controlled matrix**, where the confounded four-point table is replaced by within-family sweeps that hold vocab/arch fixed. The contradiction *is* the motivation for the controlled experiment, presented as such. |
| 10 | **Cherry-picked models.** | Models are pre-registered as a controlled matrix (one factor per cell, ≥2 vendors); pull failures recorded and skipped, never dropped silently. |

### 3.2 Scoop verdict

**The specific claim is UNSCOOPED as of 2026-08-14.** No indexed 2024–2026 work measures
grammar-mask overhead **in milliseconds and Joules, on single-stream edge hardware, as a
function of when the CPU mask stops hiding behind a short forward pass**. The near-misses
partition cleanly and none occupy this square:

- **Mask-cost *algorithm* work** — XGrammar (2411.15100) & XGrammar-2 (2601.04426), CFGzIP
  / Token-Space-Compression (2605.29986, "10–20× latency reduction"), PSC, GRID (2607.11951),
  Pre³ (2506.03887), Attention-Meets-Reachability (2603.05540): all **lower mask cost
  algorithmically**, all **datacenter/throughput**, **none in edge Joules**, none touch the
  CPU/GPU sync-stall. They lower the ceiling we push against; they do not measure our corner.
  XGrammar-2 is the one thing that moved since the last pass and it moved *away* from edge.
- **Correctness-vs-validity** — Constraint Tax (2605.26128), "When Correct Isn't Usable"
  (2605.02363, **new this sweep** — small-model structured-output reliability), "The Parser
  Already Knows" (2608.10137): all **accuracy/quality**, **no latency/energy**. These are the
  citations for Task 2, not scoops.
- **Edge energy without a constraint axis** — SLM energy on Pi/Jetson (2511.11624): Wh /
  tokens-per-Wh / EDP but **no grammar/constraint axis** at all. Confirms our square is empty.
- **Speed benchmarks** — JSONSchemaBench (2501.10868): reports constrained *speedups* via
  speculation, i.e. the engines that hide the cost — the contrast case (attack #8), not a scoop.

**One genuine internal threat, not an external scoop:** the mechanism claim must survive
attack #9. The controlled matrix + trivial-grammar control is what converts the vocab story
from "contradicted by our own four points" into a fitted, attributable two-factor model. If
that experiment lands, the square is ours and unscooped.

*Caveats on the sweep:* `mlsys.org`, `arxiv.org` direct fetch, and `huggingface.co` are
egress-blocked from this session (per digests) — the above rests on WebSearch snippets +
prior-iteration `seen.json`, not PDF reads. Re-verify 2605.02363 and 2605.29986 from an
unblocked session before the related-work section is finalized.

Sources consulted this pass: [XGrammar](https://arxiv.org/pdf/2411.15100),
[Token Space Compression / CFGzIP](https://arxiv.org/pdf/2605.29986),
[When Correct Isn't Usable](https://arxiv.org/pdf/2605.02363),
[JSONSchemaBench](https://arxiv.org/pdf/2501.10868),
[Schema Key Wording as an Instruction Channel](https://arxiv.org/pdf/2604.14862),
[Lookahead-then-Verify (diffusion CFG)](https://arxiv.org/pdf/2602.00612).
```
