# Paper machine — STATUS (updated 2026-08-10, iteration 2)

**Scope decisions so far (newest first):**
1. 2026-08-10 (iter 2, worker narrowing): axes **federated SLM** and **edge
   VLM/VLA for robots** DEMOTED — both scooped by mid-2026 work (see digest #2).
   Surviving axis: **energy-aware edge inference, narrowed to the decode loop's
   structured-generation machinery**.
2. 2026-08-10: direction = **edge × robotics × privacy and/or medical**; must be
   low-level/systems-strong; should ideally (not necessarily) feed bigos later.
3. 2026-08-10: "multi-turn agentic eval" framing DEMOTED (not low-level enough).

**Current stage: 3 — HYPOTHESES.** Stage 2 closed this iteration.

<details><summary>Stage-2 exit justification (honest accounting)</summary>

~40 sources swept. The surviving axis (energy/decoding machinery) clears the
≥15-source bar comfortably (25+ direct: energy-aware serving corpus + the
constrained-decoding corpus). Axes 1 and 2 were closed at 8–9 sources each by the
**kill criterion** ("direct prior work found — scooped"), not by exhaustive sweep
— reviving either would need a top-up sweep first. Each surviving gap has a
falsifiable statement (G1/G2/G3 in digest #2).
</details>

## Nearest deadlines (verified 2026-08-10, iter 2)
| Venue | Deadline | Note |
|---|---|---|
| ~~ICRA 2027~~ | **Sep 15, 2026** (verified, was mis-recorded as ~Sep 17–22) | **5 wks 1 day — OUT OF REACH**, past the T-4wk governor with no validated idea |
| **MLSys 2027** | **~Oct 30, 2026 — UNCONFIRMED** | mlsys.org still shows only 2026 (its deadline was Oct 30 '25). Ours is an extrapolation; aggregators call it *predicted*. Re-check every iteration; announcement expected ~mid-Sep. |
| RA-L | rolling | backup — absorbs any MLSys slip with no loss |
| IROS 2027 / RO-MAN / ICORR | ~Feb–Mar 2027 | paper #2 / medical-HRI angle |

## Top-3 candidates (post-gap-sweep — to be turned into hypotheses next iteration)

1. **The structured-generation energy tax on edge (G1)** — *leader.*
   Grammar/schema-constrained decoding is claimed "near-zero overhead"
   (XGrammar: <40 µs/token) — but that claim rests on **overlapping CPU mask
   construction with a long GPU forward pass**. On a 0.5–3B model on Apple
   Silicon / mobile / Pi-class hardware, decode is short and memory-bound and CPU
   and GPU share one memory bus, so the overlap structurally fails. The
   literature already contradicts itself (near-zero vs. 3.6–8.2× reported
   elsewhere) and llama.cpp's own GBNF docs admit unquantified "performance
   gotchas". Nobody has measured it on edge, and nobody has measured it in
   **Joules**. Motivating evidence: on edge, decode is **86–97% of total
   inference energy** (arXiv 2607.09520). Contribution is a mechanism, not a
   benchmark: get the mask off the critical path (build step *t+1*'s mask during
   step *t*'s GEMV), kill the vocab-sized CPU↔GPU sync, persist the grammar
   stack — upstreamable llama.cpp work. Venue: MLSys. Assets: entire harness.
2. **Energy-per-*valid*-output (G2)** — every energy paper reports J/token or
   tokens/watt; in an agentic/robot setting a structurally invalid tool call
   burns full energy for zero work, and retries are invisible in J/token.
   Strongest as contribution #2 inside candidate 1 rather than a paper alone.
3. **Tail latency under constraint for robot control loops (G3)** — a control
   loop is a deadline system; constrained decoding perturbs the *distribution*,
   not just the mean. SafeDec (2509.01728) asserts "minimal overhead" with no
   distribution reported. This is the robotics hook that makes candidate 1
   legible at ICRA/IROS/RA-L as well as MLSys.

*(Demoted: federated SLM — scooped by EdgeFlowerTune 2605.08636 + peak-memory
LoRA 2606.19528. Edge VLM/VLA — scooped by vla.cpp 2606.08094, ROSA 2607.01088,
and 2607.09520; also needs robot hardware Maria lacks.)*

## Known competitive pressure
MLPerf Edge Agentic owns generic tool-calling benchmarks; ~3.5 BPW threshold
(2505.15030) is prior art; generic energy-aware edge scheduling is crowded
(EnerInfer, HeteroMosaic, Agent.xpu, KAIROS, BiScale). **All of it optimizes
*unconstrained* generation — none touches the constrained decode loop.**

## Open debt
- `VENUES.md` (Stage-1 artifact) was never written; the venue table lives here
  and in BRIEF. Only worth writing if MLSys 2027 fails to materialize and we
  need to re-rank.

## Next action (iteration 3)
Stage 3 — write up G1 as 2–3 concrete, ranked hypotheses: each with an
experiment sketch (models × grammar complexity × device, unconstrained vs.
GBNF baseline, energy + p99 latency + tool-call validity), feasibility on
Maria's actual hardware (Macs + iPhone 16 Pro; note whether a Pi 5 purchase is
load-bearing), effort estimate, and venue mapping. Then hit the **[NEEDS-INPUT]
gate** — Maria's commitment to the topic is Stage-3's exit criterion.
