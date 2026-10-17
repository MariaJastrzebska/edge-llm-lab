# Reproducible result galleries

The main README contains eight current report views and four original historical
exports. Each gallery answers a different question. The figures below are generated
from saved measurements: generating a figure does not rerun an inference experiment.

## Fresh thread comparison

A new run uses [20 synthetic cases](../../examples/datasets/intake-routing-study.jsonl)
with the existing referenced evaluator and Qwen3.5-0.8B Q4_0. It varies only the
`--threads` setting: one versus two. Twenty cases × two configurations × one seed
produce 40 expected requests. The selected Linux environment exposes two CPUs.
The runtime build is CPU-based; on GPU/Metal hosts thread settings alone do not
force CPU-only execution.

Each case retains its frozen system policy, input and expected action. The original
seven cases are retained unchanged and thirteen synthetic cases extend their
coverage. Cases include ordinary supplied facts, missing units, contradictory
values, instruction overrides, unauthorized actions and out-of-scope requests.
There are five `record`, six `clarify` and nine `decline` targets. Always declining
therefore scores 45%; report agreement with this baseline beside the model result.
Near-template cases are correlated: the twenty IDs are not twenty independent
scenario families.

The local runner uses context 2048, maximum 128 tokens, temperature zero, seed11,
no prompt cache and thinking disabled. Multiple seeds at temperature zero would
not by themselves provide independent stochastic repetitions. This study therefore
uses one explicit pass per condition.

| Setting | Completed trials | Exact agreement | Mean request time | Median request time |
| --- | --- | --- | --- | --- |
| One thread | 20/20 | 12/20 (60%) | 6.69 s | 6.29 s |
| Two threads | 20/20 | 12/20 (60%) | 10.28 s | 9.03 s |
| Always-decline baseline | No inference | 9/20 (45%) | Not applicable | Not applicable |

The two-thread setting was slower on average in this recorded run. The result
motivates measuring configuration choices rather than assuming more threads help;
it does not establish the cause of the difference or a general CPU tuning rule.
There were zero target-execution errors in the completed capture. Offline exact
scoring reproduced all 40 inline scores without calling the model again. Inspect
replay exercised the adapter on the same 40 captured responses and reported exact
agreement 0.600; this is integration agreement, not a second model experiment. Its
optional control server could not bind inside the sandbox, but the evaluation
completed successfully and wrote its local JSON log.

[Saved configuration and aggregate measurements](../../experiments/results/thread-study-2026-10-17/report/summary.json)
include dataset, implementation, model and runtime hashes. The
[trial table](../../experiments/results/thread-study-2026-10-17/report/trials.csv)
records every attempt; the compressed response capture remains local.

### Reading the fresh gallery

- **Case agreement:** exact structured agreement, with one row per named case and
  separate configurations. Zero and missing scores have different meanings.
- **Request cost:** measured wall time in seconds for the same named cases.
- **Paired request difference:** two-thread minus one-thread case time. Negative
  values indicate a shorter observed request under two threads. Only cases with
  measurements in both configurations enter this plot.
- **Coverage:** scored, unscored, failed and not-run trials remain visible.

The reference is a deterministic routing policy, not clinical ground truth. No
real tools, treatments or access controls are executed. The study says whether the
model reproduced the reference actions; it does not certify safety.

This is an uncontrolled shared-host study. Configuration order alternates across
cases, but there is no randomized host-load control, warm-up block or repeated
session. Changing the server configuration restarts the owned process; request
timing excludes that startup whereas trial duration includes it. First requests
can still experience cold caches. No energy or thermal measurement is claimed.
Comparing faster requests does not establish an end-to-end production speedup.

The first sandbox attempt failed to start the server and was retained locally as
a separate failed capture. The report uses the subsequent explicitly permitted
local-server execution. Infrastructure failures were not converted to model errors
or silently mixed into the successful experiment's denominator.

### Reproduce a new capture

```bash
PYTHONPATH=src python examples/desktop/compare_threads.py \
  --model /path/to/Qwen3.5-0.8B-Q4_0.gguf --server /path/to/llama-server \
  --output runs/thread-study.sqlite --figures runs/thread-report \
  --port 18081 --threads 1 2
```

The script uses the same dataset orchestration, runtime ownership, storage,
graders and plotting API as a custom referenced study. It does not implement a
parallel benchmark path. Run tests before interpreting the generated report.

## Structured decoding study

**Question:** how does enforcing a grammar affect output structure and execution
cost within each model configuration?

**Source:** [the 480-row measurement table](../../experiments/results/20260812-211032-g1-short-forward-pass/repeat_01/tidy.csv),
recorded on 2026-08-12. Four configurations each receive 20 prompts, three repeated
seeds and two decoding conditions. The 60 requests per condition are repeated
observations of 20 prompts, not 60 independent scenarios. The recorded host is an
Apple Silicon MacBook Air with 16 GB RAM; this is not an iPhone measurement.

| Configuration | Quantization | Extracted objects satisfying schema, unconstrained → GBNF |
| --- | --- | --- |
| Qwen2.5 0.5B | Q4_K_M | 0/60 → 60/60 |
| Qwen3 0.6B | Q4_K_M | 0/60 → 60/60 |
| Llama3.2 1B | Q8_0 | 0/60 → 58/60 |
| Gemma3 1B | Q4_K_M | 0/60 → 52/60 |

These counts establish schema conformity for this workload and scoring procedure.
The original scorer can extract JSON from surrounding text before validating it.
Thus `schema_valid` is not strict JSON-output compliance, semantic correctness or
successful execution of a tool. A failure to finish within the generation budget
remains a failed output, even when grammar constraints were enabled.

### Four views, one comparison

| Figure | Read it as | Do not infer |
| --- | --- | --- |
| [Format reliability](figures/structured-decoding-validity.svg) | Prompt-weighted fraction satisfying the schema, with observed counts | Clinical quality or instruction compliance |
| [Request duration](figures/structured-decoding-latency.svg) | Observed whole-request cost under each condition | Time to first token or isolated grammar-mask overhead |
| [Generated tokens](figures/structured-decoding-tokens.svg) | How much work each condition actually generated | Equivalent response content |
| [Measured energy](figures/structured-decoding-energy.svg) | Integrated CPU/GPU/ANE energy in the request window, with coverage filtering | Battery drain, display/radio energy or model-only energy |

The analysis averages repeats within prompts before averaging prompts. Measurements
missing a metric do not become zero. The energy view uses observations with at
least 90% recorded sampling coverage and reports its retained denominator; this
selection rule is part of the report, not a claim that the excluded measurements
were zero-cost. The recorded energy is gross integrated energy, without subtracting
the idle baseline. No confidence intervals or significance claims are constructed
from these descriptive plots.

The [generated aggregate summary](structured-decoding-summary.json) records the
input hash, counts, paired coverage and aggregation policy. Pairing is within model,
quantization, prompt, repeat and seed. Unmatched observations cannot silently
become paired effects.

### Interpretation limits that change the conclusion

The rendered prompt lengths differ across model templates: the saved manifest
records roughly 60–1,400 characters for Gemma and roughly 5,300–6,800 for the other
models. This is consistent with a template omitting system material and needs
investigation before any fair cross-model comparison. Quantizations also differ.
Treat each model's two conditions as a descriptive comparison; do not read the
figure as a ranking of model ability.

Condition changes also change output length, sometimes substantially. Lower energy
or latency can result from producing fewer tokens, rather than faster processing
of the same work. A causal grammar-overhead experiment needs matched prompt
rendering, controlled generated content/length and instrumentation of the decoding
path. The older narrative reports make stronger mechanism claims than these
measurements alone justify; this report uses the narrower interpretation.

The experiment manifest records a dirty source tree. Its commit is useful
provenance but is not a complete reconstruction of the executed source. Recorded
host power can include background activity. These limitations travel with the
results rather than disappearing in a redesigned figure.

### Regenerate the gallery

From the repository root in the project environment:

```bash
PYTHONPATH=src python -m edge_llm_lab.visualization.paired_report \
  experiments/results/20260812-211032-g1-short-forward-pass/repeat_01/tidy.csv \
  --output-prefix docs/research/figures/structured-decoding \
  --summary docs/research/structured-decoding-summary.json \
  --manifest experiments/results/20260812-211032-g1-short-forward-pass/repeat_01/raw.json
```

Figures use the saved scalar table and allowlisted experiment metadata; no response
text is exported. The optional manifest argument parses the existing raw JSON file
to retrieve metadata. Rendering launches no model and contacts no tracking service. The rendering and aggregation contracts
are tested in [the paired report tests](../../tests/unit/test_paired_report.py).

## Referenced harness demonstration

**Question:** can the reusable capture, deterministic scoring, reporting and Inspect
replay path expose named-case failures without relying on a remote tracking service?

**Executed:** 2026-10-17, Linux CPU, Qwen3.5-0.8B Q4_0 (563,036,064-byte GGUF),
llama.cpp build b11669, context 2048, maximum 128 output tokens, temperature zero,
seed 11, prompt caching and model thinking disabled. The experimental local
Python version was 3.14.4; the packaged dependency range remains 3.10–3.13.
[The saved summary](../../experiments/results/harness-demo-2026-10-17/intake-report/summary.json)
records the model/runtime hashes and configuration.

One command smoke case matched its reference. The subsequent seven-case synthetic
intake-routing dataset produced four exact matches and three mismatches, with no
execution errors. An always-decline predictor also scores 4/7 on that small set;
the model result should not be read as an improvement over this trivial baseline. Inspect replay scored those same captured responses as 4/7;
this checks integration agreement, not independent model validation.

- [Case agreement](../../experiments/results/harness-demo-2026-10-17/intake-report/case-scores.svg)
  shows exact agreement with each frozen target.
- [Case request cost](../../experiments/results/harness-demo-2026-10-17/intake-report/request-cost.svg)
  uses scenario names and seconds, making the measurement interpretable.
- [Aligned outcomes and cost](../../experiments/results/harness-demo-2026-10-17/intake-report/case-outcomes.svg)
  places the same cases on shared rows. Missing scores and missing timings remain
  distinguishable from measured failures and zero.

Request wall time includes HTTP overhead and excludes server startup. It is
separate from trial duration, which can include startup. No controlled warm/cold,
thermal or energy study was performed on this Linux run. There is one observation
per case; a tail latency or safety generalization would not be supported.

The framework's existing execution-order diagnostics remain available for debugging.
A request index is simply an ordinal position, not an experimental factor. Those
plots are therefore excluded from the main research gallery.

To regenerate figures from the portable aggregate summary alone:

```bash
PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path
from edge_llm_lab.visualization.dataset_report import render_summary

report = Path('experiments/results/harness-demo-2026-10-17/intake-report')
render_summary(json.loads((report / 'summary.json').read_text()), report)
PY
```

This operation needs no private transcripts or model access. For a new experiment,
use the [dataset runner](../../examples/desktop/evaluate_dataset.py) and record a
new capture rather than replacing these results. Plot tests check unequal turn
counts, missing scores, units, row alignment and actual image rendering.
