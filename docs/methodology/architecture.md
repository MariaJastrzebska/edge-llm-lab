# Architecture and migration

The reusable boundary is a case, an execution protocol, captured evidence and a
versioned grader. Diabetes-specific prompts and schemas are example inputs, not
requirements of the capture store or scoring API.

```text
Frozen JSONL cases ── validation / reference_turns ──┬─ referenced evaluator ─ llama-server
                                                   └─ Inspect task ─ provider endpoint
                                                                │
                                                     captured responses
                                                                │
                          SQLite scalar measurements + compressed trace artifacts
                                         │                      │
                                    SQL reports             CaptureReader
                                                                │
                                             offline scorer / Inspect replay
                                                                │
                                                  shared grade_response
```

- `evaluation/dataset.py`: validates splits and families, sequences trials and
  resumes compatible runs. It calls `evaluate_case`; it does not own model serving.
- `runtime/server.py`: owns only its subprocess and reuses matching live configurations.
- `storage/trials.py`: transactions, indexed measurements, compressed trace artifacts.
  `storage/capture.py`: one read-only integrity-checked reader for offline consumers.
- `metrics/grading.py`: deterministic structured or literal-text scoring. No model
  loading, credentials, plotting or network access.
- `integrations/inspect_ai.py`: optional adapters; no copied scoring or reference slicing.
- `visualization/dataset_report.py`: offline SQL aggregation and figure rendering.
- Existing referenced/unreferenced classes preserve historical entry points. Their
  plotting methods were moved to compatibility mixins and referenced metrics to a
  separate module. These legacy mixins are not the new extension API.

A custom referenced task needs a JSONL dataset with `id`, `family_id`, `split` and
`conversation`. Assistant messages are targets. Use `structured` for JSON contracts
or `text` for exact text equality. Neither profile measures semantic safety.
Arbitrary semantic graders should implement a separate, versioned scoring policy;
do not silently substitute a paid judge for deterministic scoring.

Unreferenced evaluation retains its existing simulator-driven protocol. It is
stateful and domain-specific; the local factory is intended for referenced replay.
A general interactive protocol needs explicit simulator, tool environment,
termination and outcome interfaces. Do not present the legacy patient simulator
as a domain-independent agent runner or use a missing expected-outcome file as a
successful test.

## Deliberate limits

This refactor isolates new reusable boundaries while preserving historical plot
and pipeline entry points. It does not prove every historical plotting combination
works, migrate every old JSON file, provide an iOS runtime, or validate a medical
application. A broader rewrite of historical adapters should follow contract tests
for the historical inputs that must remain supported.

The current loader reads a JSONL split into memory, and execution is sequential.
This is suitable for bounded device studies; it is not an unbounded dataset engine.
Keep inference concurrency at one for isolated latency experiments. For throughput
studies, define concurrency as an experimental factor, not an invisible speedup.
