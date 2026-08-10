# edge-llm-lab

Maria's laboratory for evaluating LLMs on edge devices — latency/memory/energy
telemetry + structured tool-calling accuracy metrics + LLM-judge scoring. Target:
a platform rigorous enough for a publishable paper (edge LLM engineering, possibly
medical-robotics venues).

## The one rule that governs this repo

**edge-llm-lab = laboratory, fed-mobile = app.** Every model test, benchmark, and
evaluation — on any device — happens HERE. The fed-mobile Flutter app only ships
models that won here. Never add app features here; never add evaluation code to
fed-mobile.

## Current state (2026-08-10) — read before touching anything

`main` is a hand-copied fork of `fed-mobile/thesis_generators/` and **does not run**:
`src/edge_llm_lab/visualization/__init__.py` imports a nonexistent `plots` module,
and the default agent's judge prompt was deleted. The Feb 2026 "academic polish"
commits also deleted the statistical analysis script (ANOVA/Tukey/Mann-Whitney),
the evaluation datasets, and CI — **all recoverable from branch
`origin/framework_moved` / commit `07ffb90`**. Recovery is step 1 of the rebuild.

A restructure is planned (see fed-mobile's `docs/plans/2026-08-10-revival-plan.md`,
Part D): clean package with `runtime/` (llama-server lifecycle, `_parse_llama_timings`,
Ollama metadata normalizer), `telemetry/` (pluggable backends: macOS powermetrics /
Linux RAPL / Jetson tegrastats), `metrics/` (tool-call fidelity + BERTScore, judge as
pluggable scorer), `tracking/` (`RunTracker` Protocol — Neptune impl exists),
`analysis/` (recovered statistics). ~1,500 LOC of instruments are ported verbatim;
the 55 matplotlib methods in the 6.8k-line evaluator god-class are dropped — figures
regenerate from tidy result frames.

## Known bugs to fix during porting

- `base_eval.py:100` — `agent_type[:-2] == 'en'` should be `[-2:]` (English agents
  silently load Polish schemas). Same bug exists in fed-mobile's copy.
- `save_evaluation_log` (base_eval.py:364) — two wrong kwarg names, dead code.
- `inference_params.yaml` — duplicate agent keys; first silently wins.
- `run_evaluation_pipeline.py` — "golden model" tracking is last-model-wins.

## Paper-grade requirements (why the rebuild exists)

Every run must record a manifest (git SHA, llama.cpp SHA, hardware, thermal
precondition); measurements repeated n≥3 with warm-up discards; seeds swept, not
hardcoded once; results in tidy long format (one row per model × config × repeat ×
round × metric); external benchmark calibration (e.g. BFCL subset); results tagged
with device identity so laptop/Pi/phone runs land in one comparable table.

## Engine note

The lab drives **llama.cpp** (llama-server subprocess + Ollama for model registry).
The app uses fllama (also llama.cpp) — same core family, so lab and app numbers are
comparable. Keep it that way.
