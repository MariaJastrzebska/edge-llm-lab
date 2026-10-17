# Historical research figures

These aggregate exports were recovered from the `framework_moved` Git branch.
The selected figures belong to the same referenced run, `run_130`, dated
2026-01-18 in their filenames. [The manifest](provenance.json) records the exact
source commit, original path and SHA-256 for each unchanged image. They can be
viewed directly from this repository without a Neptune account.

## Questions represented in the historical work

| Question | Available evidence | Interpretation boundary |
| --- | --- | --- |
| How does output quality differ across configurations? | Judge-score exports | A judge rating is not a calibrated probability of correctness. |
| What is the latency cost of a configuration? | Latency and throughput exports | Generated length, failures, startup and workload can confound comparisons. |
| How do quality and latency interact? | Joint scatter plot | Descriptive configuration comparison, not an isolated optimization effect. |
| What resources does execution consume? | Power, RAM and swap exports | Power is not energy; host memory is not necessarily model memory. |
| Does performance change during execution? | Historical resource timelines | Requires verified timestamps, sensor coverage and controlled workloads. |
| How do replay and simulated interaction differ? | Separate evaluator implementations | Only referenced results are included in this recovered selection. |

The historical branch contains 1,188 PNG paths, including repeated exports,
per-model timelines and composite figures. This count is not the number of
independent experiments. This selection is not a reconstruction of the full study.
The underlying per-example measurements for these four figures have not been
recovered, so their means, weighting, uncertainty and failure handling have not
been independently recomputed. The original labels remain unchanged.

## Output quality

![Historical judge scores](figures/scores.png)

The vertical axis is the original GPT-judge rating on a 0–10 scale. The figure
suggests substantial differences between configurations on the selected workload.
It does not establish clinical adequacy, inter-rater agreement or performance
across a representative population. References, judge version and rubric must be
recovered before treating this as a reproducible quality comparison.

## Latency

![Historical latency by model configuration](figures/latency_bars.png)

The horizontal axis is mean latency in milliseconds; bar annotations use seconds.
Compare this with quality before choosing a configuration. The plot alone cannot
distinguish a genuinely efficient response from a short or unsuccessful response.
It does not identify the number of independent cases or controlled warm-up policy.

## Power

![Historical CPU and GPU power](figures/resources.png)

These are reported CPU/GPU power values in milliwatts, not energy per answer.
Energy requires integrating power over time with a defined measurement boundary
and an idle baseline. The export does not establish sensor sampling coverage or
whether host background work was isolated.

## Joint view

![Historical quality versus latency](figures/scatter.png)

Each point represents one configuration, with the original model-size bubble
encoding. The fastest configurations in this export received lower judge scores.
This motivates reporting quality and cost together; it does not demonstrate that
quantization caused the difference. Different model families also differ in
training, architecture, prompts and response lengths.

## Legacy figures that need correction before reuse

- The category-winner export in the same run displays a judge score of 9.08/10
  as 908%. It also combines dimensions into an insufficiently explained mobile
  score. It is excluded from this gallery.
- The mobile-performance export draws symmetric latency error bars below zero.
  The exported image does not identify the statistical unit or interval meaning.
  Recompute distributional summaries from raw observations before reuse.
- The resource-health export overlays an 8 GB iPhone reference line on RAM
  measurements. This is a hypothetical capacity comparison, not an iPhone test;
  device memory capacity is not an application's available memory budget.
- A flat CPU-frequency percentage is insufficient evidence of no throttling.
  Missing or substituted telemetry must be distinguished from observed stability.

## Reporting new experiments

Use a configuration table linked to model/runtime hashes, device, workload and
measurement protocol. Keep case-level rows, failures and missing telemetry.
Prefer the following views when the required observations exist:

1. Quality versus median and tail latency, with uncertainty clustered by case.
2. Paired baseline/optimization differences on identical held-out cases, reporting
   quality changes alongside speed changes.
3. Latency distributions split by cold/warm execution and successful/failed cases.
4. Measured peak memory and joules per completed task, with telemetry coverage.
5. Per-scenario boundary violations and completion rates, showing denominators.
6. Time-aligned throughput, temperature and power for sustained device workloads.

Do not derive these statistics from screenshots or manufacture intervals from
single observations. The new seven-case CPU demonstration tests the harness and
does not replace this historical research or supply its missing raw measurements.

## Neptune availability

OpenAI [announced the acquisition of Neptune](https://openai.com/index/openai-to-acquire-neptune/)
and integration into its training infrastructure. The official
[shutdown notice](https://support.neptune.ai/en/articles/13925165-service-shutdown-overview)
states that Neptune SaaS closed on 2026-03-06. These notices do not identify a
renamed public successor where these runs can be uploaded. Local exports remain
usable independently; this repository does not require remote tracking to read
the figures or run an evaluation.
