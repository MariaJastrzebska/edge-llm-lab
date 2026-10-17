# Validation record — 2026-10-17

Changes developed in test-first cycles included:

| Boundary | Observed red test | Implemented correction |
| --- | --- | --- |
| Text grading | No shared grading module/profile | Pure text/structured grading shared across consumers |
| Dataset split | No family leakage guard | Reject a family spanning dev and test |
| Trajectory termination | `not complete` and `almost_complete` ended a trajectory | Require exact normalized `complete` |
| Runtime telemetry | Unknown load time reported as zero | Preserve unknown load time and add request wall time |
| New report | Text scores labelled JSON | Profile-specific labels |
| Historical patient plot | Missing latency raised TypeError | Preserve missing data as NaN |
| Inspect integration | Adapter absent | Native task and captured-response replay |

Broader unit validation: 90 passed, one pre-existing failure in the GCP secret-name
registry test. The source at baseline commit `0e49df9` maps TELEGRAM_CHAT_ID to USER_ID;
the test expects EDGE_LAB_CHAT_ID. No credential values were read to resolve this
unrelated discrepancy. The two optional Inspect integration tests passed separately.
Undefined-name lint and Python compilation passed. This is not full branch coverage
of the historical harness, nor a claim that every legacy plotting option was tested.

Actual local inference: Qwen3.5-0.8B Q4_0, one command case then seven synthetic
intake cases, seed 11. Zero execution errors; exact agreement 1/1 and 4/7 respectively.
Inspect subsequently scored the seven captured responses with the same 4/7 result
without generating new target responses. Native Inspect JSON logs were used; raw
logs and capture artifacts remain local. SVG/PNG figures were rendered and visually
reviewed. No Mac/iPhone measurement or clinical validation was performed.

Remaining migration work includes reducing legacy constructor/import coupling,
translating remaining historical comments/docstrings, testing further historical
plot combinations, packaging a lightweight install, and providing a generic injected
simulator/tool environment for interactive evaluations. These are not claimed complete
by the new dataset path or the module extraction.
