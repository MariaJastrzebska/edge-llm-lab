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

## Report-gallery revision and fresh thread study

The report revision adds contract tests for named-case alignment, grouped
configurations, paired differences, missing-versus-zero values, unequal turn/repeat
counts, energy coverage, strict CSV booleans and complete PNG/SVG rendering.
Regression tests prevent a rejected rerender from mixing old figures with new
JSON/CSV data, including through the public database-rendering entry point.
Incomplete CSVs report observed counts rather than assuming a Cartesian grid.

A fresh Qwen3.5-0.8B Q4_0 run completed 40/40 attempts across 20 synthetic cases
and one/two CPU threads with no execution errors. Each configuration matched 12/20
references; the always-decline baseline matches 9/20. Offline scoring reproduced
all 40 inline exact scores. Inspect replay also completed 40 samples at 0.600
accuracy without calling the target. No energy measurement was available on Linux.
The separate sandbox attempt failed to start the local server and remains a local
diagnostic capture; it is not presented as model behavior.

The full suite initially reported 137 passes and the previously documented registry
expectation failure. Historical commit 706e745 and the registry's existing comment
confirm USER_ID as the intended recipient mapping. The outdated test expected the
bot's own identifier; its expectation was corrected without changing credential
resolution, accessing secret values or sending a message. All 16 registry tests then
passed. The final full run passed all 138 tests (including optional Inspect tests);
eight dependency deprecation warnings remain. Undefined-name lint and editor
diagnostics were clean.

Local validation used Python 3.14.4 in the existing development environment;
this is outside the package's declared Python range and does not replace the
supported-version CI matrix. A wheel build also passed without resolving or
installing the full dependency graph. CI now includes the posthoc tests, names
each OS/Python combination separately, and runs the real Inspect replay tests
with explicit optional dependencies instead of an empty integration directory.

The eight current README figure views and four original historical thumbnails were
reviewed visually. New plots are regenerated from recorded observations; no values
were reconstructed from screenshots. Historical originals retain their provenance.
