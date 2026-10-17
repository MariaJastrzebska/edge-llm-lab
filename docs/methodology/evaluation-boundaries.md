# Device execution and offline assessment

Edge-LLM-Lab separates three research questions. A result in one does not answer the others.

| Question | Execution location | Evidence |
| --- | --- | --- |
| Can this configuration run within the device budget? | Target device | Captured responses, latency, memory, failures; energy and temperature only when measured |
| Does it complete the intended task? | Target-device generation; grading may run elsewhere | Frozen references, task outcomes, rubric scores, case-level coverage |
| How does it behave under adversarial inputs? | A controlled test environment, with deployment checks on the device | Threat-specific attack success, benign task success, refusals and failures |

## Capture first, score later

For frozen referenced replay, pass `--defer-scoring` to
[the dataset runner](../../examples/desktop/evaluate_dataset.py). This skips all
per-response graders. Copy the completed SQLite database and its adjacent
`.artifacts` directory together after the run has closed. The compressed artifacts
contain the captured context, reference and model response. Treat them as sensitive
if the input contains sensitive information; the example datasets are synthetic.

Run deterministic grading on a workstation without loading the target model:

```bash
PYTHONPATH=src python -m edge_llm_lab.evaluation.posthoc capture.sqlite --output scores.sqlite
```

The scorer verifies trace hashes, preserves capture and grading failures, and
writes a separate database with source manifest and grader implementation hashes.
It currently grades JSON validity and exact structured equality, not medical safety.
It does not create native Inspect logs. Scoring changes must not overwrite capture
measurements. Report score coverage alongside accuracy and aggregate repeated
trials within cases before calculating dataset summaries.

Lightweight validation can remain inline when it affects control flow. In an
interactive unreferenced trajectory, simulator replies and tool execution are part
of the experiment: they cannot simply be moved after generation. Final outcome
assessment can be moved offline, but this runner does not yet provide deferred
scoring for the unreferenced mode.

Grading outside the inference timer is insufficient for clean device measurements:
a heavy grader between requests can change memory pressure, caches and thermals.
Run the capture phase without heavy graders, then score it. Distinguish startup,
warmup, steady-state requests and sustained operation in the measurement protocol.
Do not infer energy use or time to first token from total request duration.

## Optional Inspect AI track

[Inspect supports deferred scoring and rescoring](https://inspect.aisi.org.uk/scoring-workflow.html).
For evaluations run natively through Inspect, capture with `inspect eval ... --no-score`
and later use `inspect score` on the resulting Inspect log. These commands do not
accept Edge-LLM-Lab's SQLite capture directly. Use the implemented `captured_task` adapter described below for Edge-LLM-Lab
captures. No standardized security benchmark result is claimed here.

Use Inspect as an optional evaluation controller. The controller may run on a
server while the target model runs through a device endpoint. If generation is
moved to a server, record it as a separate experiment. Match model artifact,
quantization, tokenizer, prompt template, runtime revision, context limit, tool
schema and decoding settings; even matching these is not proof of identical
behaviour across hardware backends. Server/network timings are not device timings.

For a diabetes intake assistant, start with a scoped threat model:

- Untrusted patient text attempts to override instructions or invoke an unavailable tool.
- Synthetic records contain instructions attempting to reveal another synthetic user's data.
- The assistant changes supplied values, invents missing information, or silently resolves contradictory units.
- Requests outside the permitted intake workflow elicit unsupported treatment actions.

Use isolated mock tools and synthetic records. Pair adversarial cases with benign
controls so blanket refusal cannot look like successful protection. Report each
failure category separately from task completion; a high mean usefulness score
must not hide a critical boundary violation. Clinician-reviewed criteria are
needed before interpreting medical safety scores. This is research evaluation,
not clinical validation.

[AgentDojo](https://ukgovernmentbeis.github.io/inspect_evals/) can inform tests of
indirect prompt injection in tool-using systems. Its environment and workload are
not diabetes intake. A selected subset or adapted task must be labelled with its
selection rule and version; it is not a comparable full-benchmark result. General
cybersecurity capability tests are a separate research question and are not a
substitute for these application-specific boundaries.

## Interpreting historical runs

Historical runs are exploratory measurements on the selected scenarios, models
and hardware recorded at the time. Referenced replay evaluates responses against
frozen conversation prefixes; it does not measure recovery from the model's own
earlier mistakes. Interactive unreferenced runs assess trajectories but depend on
the simulator and outcome specification. Conversation turns and repeated seeds
are dependent observations, not additional independent patients. Historical
figures should retain their original protocol and dates and should not be pooled
with new dataset evaluations or presented as evidence of clinical safety.

## Implemented Inspect adapter

Install the optional integration with `python -m pip install -r requirements-inspect.txt`.
The tested adapter targets Inspect AI 0.3.281. It shares dataset validation,
reference-prefix construction, capture integrity checks and grading functions
with the local runner. Inspect is not imported by the core package.

```bash
# Native Inspect execution against your chosen provider/endpoint.
PYTHONPATH=src inspect eval src/edge_llm_lab/integrations/inspect_ai.py@referenced_task \
  -T dataset="$PWD/examples/datasets/intake-boundaries.jsonl" \
  --model YOUR_PROVIDER/YOUR_MODEL --no-score

# Native Inspect scoring of an existing Edge-LLM-Lab capture; no target generation.
PYTHONPATH=src inspect eval src/edge_llm_lab/integrations/inspect_ai.py@captured_task \
  -T capture="$PWD/capture.sqlite" --model mockllm/model --log-format json
```

Use `inspect score` to grade a native Inspect log generated with `--no-score`.
For captured-task imports, the target model identifier is a placeholder: the replay
solver supplies the captured response and never calls `generate`. The adapter
refuses captures containing failed trials, so failures cannot silently disappear
from the imported sample denominator. The original SQLite coverage report remains
the source for capture failures. Request timings in Inspect replay are scoring
execution times, not target-device inference times.

The intake-boundaries dataset is a seven-case synthetic router demonstration with
benign controls. It tests constrained action selection under the stated policy.
It does not execute tools, demonstrate access-control enforcement, establish
clinical safety, or constitute a standardized security benchmark. Tool execution,
retrieved-record injection and cross-user isolation require separate integration
tests against a concrete application. Exact-match scores must not be described as
a general safety score. Inspect sample means weight turns; use case/family metadata
for grouped research analysis when conversations have unequal lengths.

## Reusable study design

1. Define the intended task, allowed actions, threat model and device budgets before
   choosing a model. Keep the agent configuration separate from the evaluator.
2. Version the cases and their provenance. Split by patient/scenario family, not by
   paraphrase or conversation turn. Tune on dev; freeze test before comparison.
3. Run a small smoke subset, followed by a predeclared stratified test sample. Large
   datasets can be streamed or sharded; device runtime, heat and battery constrain
   the evaluation budget more than the size of a text dataset.
4. Record model and runtime hashes, quantization, prompt, tools, sampling seed and
   limits. Fix the scoring protocol. Record incomplete runs and missing measures.
5. Measure cold startup separately, warm up explicitly, and collect repeated target
   requests without heavy graders. Control or record thermal/power conditions.
6. Grade captured responses offline. Use deterministic checks for machine-readable
   contracts and reviewed rubrics for semantic correctness and safety. Calibrate
   judge-model decisions against a human-labelled subset; preserve disagreements.
7. Report case-level task performance, critical violation counts, benign refusals,
   coverage, and request cost separately. Cluster uncertainty estimates by case
   family; repeated seeds do not increase the independent sample size.
8. Publish sanitized examples, aggregate tables, protocol and limitations. Retain
   sensitive transcripts and proprietary prompts privately. A public report must
   distinguish reproducible examples from restricted historical evidence.

The historical specialized agents remain examples. A modern consolidated intake
agent is a plausible baseline, not an assumed improvement: compare it with the
specialized agents on identical held-out cases, permitted tools and budgets.
Measure end-to-end calls and latency as well as correctness. Keep specialization
only where it produces measurable benefits or enforces useful boundaries.

## Figures that answer research questions

The implemented report shows per-case exact agreement, individual request wall
measurements, and coverage (scored, unscored, failed, not run). It reads scalar SQL
metrics, never invokes models, and does not load transcripts to draw figures.
Missing measurements remain missing; trial time is not substituted for request
latency. Exact text match and JSON match have distinct labels.

For larger comparative studies, add quality versus latency/memory scatterplots
with a Pareto frontier only over comparable configurations, latency distributions
with cold/warm phases separated, and category-level failure rates with sample
counts. Report medians/p95 with sample counts; do not imply a stable tail estimate
from a handful of requests. Energy and thermal plots require actual instrumentation.
