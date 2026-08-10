# Standing auto-research brief — the PAPER MACHINE (edge-llm-lab)

Mission: get Maria published at a strong conference in **robotics / edge systems**,
preferably (not necessarily) around a **medical problem** — her thesis use case.
This loop follows a staged research process; every iteration advances the current
stage and reports to files (no external pings).

## Hard constraints from Maria (2026-08-10)
- The contribution must be **super-technical and LOW-LEVEL** — systems /
  implementation work that "pushes the needle": runtimes, kernels, schedulers,
  memory & energy mechanisms, decoding machinery. NOT high-level evaluation or
  benchmark-assembly framing (she considers her thesis-level work too simple for
  the venues she wants).
- **Reuse her assets where sensible** but do not be limited by them: edge-llm-lab
  harness (llama-server orchestration, timing/energy telemetry, tool-call fidelity
  metrics, pilot data), fed-mobile iOS app (on-device llama.cpp/fllama), Apple
  Silicon Macs + iPhone 16 Pro, possible Pi 5 / Jetson purchase, llama.cpp/fllama
  codebases she intends to contribute to.
- Affiliation: independent preferred; **Tooploox** (the company Maria currently
  works at) may be used if a venue requires an affiliation.

## The staged process

### Stage 1 — VENUES (find the target list first)
Build a table of realistic venues: robotics (ICRA/IROS/RA-L/RO-MAN/ICORR/CoRL),
edge/embedded systems (MLSys, SenSys/IPSN merged, EuroSys+workshops, EdgeSys,
tinyML, IPDPS), medical-technical (EMBC, JBHI, ML4H). For each: deadline, page
format, acceptance character (what gets in — check recent accepted papers),
whether "low-level systems + LLM + edge" papers appear there.
Output: `VENUES.md`, ranked by (deadline feasibility × prestige × fit with
low-level systems work). Keep deadlines current in every iteration.

### Stage 2 — GAPS (per shortlisted venue)
For the top venues: sweep recent proceedings + arXiv for what's being done in
low-level on-device LLM systems (KV-cache management, quantization kernels,
grammar-constrained/speculative decoding, heterogeneous-core & NPU scheduling,
unified-memory management, energy-aware inference, real-time guarantees for
robot control loops). Identify what is NOT done — especially where Maria's assets
give unfair advantage. Output: gap list in the digest, each gap annotated with
"why still open" and "what asset we'd reuse".

### Stage 3 — HYPOTHESES (rank candidate papers)
Turn the best gaps into concrete testable hypotheses with an experiment sketch,
target venue, and effort estimate. Score each: technical depth (Maria's bar),
feasibility on her hardware, deadline fit, medical-applicability bonus.
Output: top-3 candidates in `STATUS.md`, full list in the digest.

### Stage 4 — VALIDATION (before committing months)
For the leading hypothesis: deeper novelty check (has anyone done it), pilot
experiment design (1-week falsification test via experiments/queue), and a
one-page paper skeleton. Only after this does Maria pick.

## Reporting contract (files only, Maria's choice)
Every iteration: dated digest in `docs/research/digests/`, `seen.json` updated,
`STATUS.md` overwritten (current stage, top-3 candidates, nearest deadlines,
single next action). Commit.

## Venue facts already gathered (2026-08-10 — verify each iteration)
- ICRA 2027: submissions ~**Sep 17–22, 2026** (5 weeks!)
- **MLSys 2027: deadline Oct 30, 2026** — strong low-level systems fit
- SenSys 2027 (merged SenSys/IPSN/IoTDI — embedded AI flagship): 2026 round had a
  May deadline; check next cycle
- RA-L: rolling, anytime
- IROS 2027 ~Mar 2027; RO-MAN/ICORR ~Feb–Mar 2027 (paper #2 / medical-HRI angle)
