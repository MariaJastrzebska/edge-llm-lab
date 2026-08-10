# RUNBOOK — running experiments on your Mac (one command)

The collaboration protocol between Maria's MacBook (the instrument) and Claude's
Linux session (design + analysis):

```
Claude designs experiments → commits YAML configs to experiments/queue/ → pushes
YOU on the Mac:   git pull && python3 scripts/run_experiments.py
   → runner executes every queued config, captures results + hardware manifest,
     moves configs to experiments/done/, commits and pushes results
Claude pulls results → analyzes → designs next round → repeat
```

You never need to understand an individual experiment to run the queue. If the
queue is empty, the runner says so and exits.

## One-time Mac setup
```bash
cd edge-llm-lab
poetry install            # or: pip install -e . (deps from pyproject)
ollama --version          # Ollama must be installed and running
```

## Every time
```bash
git pull
python3 scripts/run_experiments.py        # runs everything queued
git push                                   # if the runner didn't auto-push
```

Options:
- `--dry-run` — show what would run, run nothing
- `--only NAME` — run a single queued config
- `--no-commit` — leave results uncommitted (inspect first)

## What a queued config looks like (`experiments/queue/*.yaml`)
```yaml
name: q4-vs-q8-latency-pilot
description: Granite 2B Q4_K_M vs Q8_0, default runtime flags, 5 repeats
requires: [ollama]          # runner checks these exist before starting
repeats: 5
command: >                  # executed repeats× with RUN_DIR + REPEAT env vars
  poetry run python examples/desktop/run_evaluation_pipeline.py
  --agent constant_data_en --mode logs_and_viz
timeout_minutes: 60
```
The runner wraps every repeat with: hardware manifest (git SHA, machine, RAM,
CPU), resource snapshots before/after (edge_llm_lab.telemetry), full
stdout/stderr capture — into `experiments/results/<timestamp>-<name>/`.

## Rules
- Never edit files in `experiments/done/` or `experiments/results/` by hand.
- If a run crashes, the runner records the failure and CONTINUES the queue —
  a broken experiment must never cost you the whole night. Failures land in the
  result dir with `status: failed` for Claude to debug.
