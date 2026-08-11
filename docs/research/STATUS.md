# Paper machine — STATUS (updated 2026-08-11, iteration 4)

**Scope decisions so far (newest first):**
1. 2026-08-11 (iter 4): no new scope decision — light tracking pass only,
   per iter 3's own instruction not to spend Mac-hardware time or draft
   further G1 hypotheses while `NEEDS_INPUT.md` is unanswered. Still well
   under the 48h pace-governor reminder threshold (gate opened 06:26 UTC
   today).
2. 2026-08-11 (iter 3): discovered that Stage 4 had already effectively
   started outside this loop's own tracking — an overnight Mac-side pilot
   (300 + 360 runs, see `experiments/results/NIGHT-REPORT.md`) tested G1
   before this file was updated to say so. This iteration's real work was
   analyzing that pilot (Part 1 of digest 2026-08-11.md), not drafting
   fresh Stage-3 hypotheses as previously planned. **Stage corrected: 3 → 4.**
3. 2026-08-10 (iter 2, worker narrowing): axes **federated SLM** and **edge
   VLM/VLA for robots** DEMOTED — both scooped by mid-2026 work (see digest #2).
   Surviving axis: **energy-aware edge inference, narrowed to the decode loop's
   structured-generation machinery**.
4. 2026-08-10: direction = **edge × robotics × privacy and/or medical**; must be
   low-level/systems-strong; should ideally (not necessarily) feed bigos later.
5. 2026-08-10: "multi-turn agentic eval" framing DEMOTED (not low-level enough).

**Current stage: 4 — VALIDATION, blocked on Maria at a [NEEDS-INPUT] gate.**
The overnight pilot (see below) is a real falsification attempt on G1: no
positive latency/energy tax detected on 2–4B models on an M1 Air, one model
significantly in the *wrong* direction. But the regime G1's mechanism
actually predicts a tax in (short GPU forward pass) was never reached —
nothing decoded faster than 36.8 ms/token. This is exactly the kind of
kill-adjacent-but-not-clean result BRIEF's constitution reserves for Maria:
continuing costs a specific new bet of her Mac-hardware time (or a hardware
purchase). See `NEEDS_INPUT.md` at repo root. **The loop will not queue new
experiments or draft further hypotheses on G1 until she answers.**

<details><summary>Stage-2 exit justification (honest accounting, unchanged from iter 2)</summary>

~40 sources swept. The surviving axis (energy/decoding machinery) clears the
≥15-source bar comfortably (25+ direct: energy-aware serving corpus + the
constrained-decoding corpus). Axes 1 and 2 were closed at 8–9 sources each by the
**kill criterion** ("direct prior work found — scooped"), not by exhaustive sweep
— reviving either would need a top-up sweep first. Each surviving gap has a
falsifiable statement (G1/G2/G3 in digest #2).
</details>

## Nearest deadlines (re-checked 2026-08-11, iter 4)
| Venue | Deadline | Note |
|---|---|---|
| ~~ICRA 2027~~ | **Sep 15, 2026** (verified, was mis-recorded as ~Sep 17–22) | **5 wks — OUT OF REACH**, past the T-4wk governor with no validated idea |
| **MLSys 2027** | **~Oct 30, 2026 — still UNCONFIRMED** | mlsys.org and both deadline-tracker aggregators (mlsys-deadlines.github.io, casys-kaist.github.io) still show 2027 as TBD/predicted off the 2026 Oct 30 date; official mlsys.org 2027 CFP page unreachable this iteration too. One low-trust secondary source (clocate.com) puts the conference itself at May 17–22, 2027 with conflicting venue city — no deadline signal. No official CFP yet. Re-check every iteration. |
| RA-L | rolling | backup — absorbs any MLSys slip with no loss |
| IROS 2027 / RO-MAN / ICORR | ~Feb–Mar 2027 | paper #2 / medical-HRI angle |

## G1 pilot result (Stage 4, 2026-08-10 night → analyzed 2026-08-11)

**002-g1-mask-cost-pilot** (granite3.1-dense:2b, 300 runs) and
**003-g1-model-axis** (4 models, 360 runs) ran on Maria's MacBook Air M1 per
the Linux/Mac protocol in `experiments/RUNBOOK.md`. Full manifest discipline
(git SHA, llama.cpp SHA, model digest, idle baselines, energy-window coverage
1.00 throughout). Paired per-token decode time, gbnf − unconstrained:

| model | paired diff | 95% CI | detection floor |
|---|---|---|---|
| qwen3:4b | **−0.879 ms** | [−1.279, −0.479] | 0.62% |
| phi4-mini | −0.939 ms | [−2.227, +0.349] | 2.13% |
| granite3.1-dense:2b | +0.088 ms | [−0.094, +0.269] | 0.50% |

**No model shows a positive tax; the one significant CI (qwen3) points the
wrong way.** But nothing tested decoded faster than 36.8 ms/token, and G1's
mechanism needs a short forward pass to fail to hide the mask cost — that
regime is untested, not falsified. A pre-existing repo bug (uppercase vs.
lowercase status enum, `constant_data_en.txt` vs. `pydantic_models_en.py`)
confounds the validity/energy-per-valid numbers; now in CLAUDE.md's
known-bugs list. Full analysis: digest 2026-08-11.md Part 1; raw report:
`experiments/results/NIGHT-REPORT.md`.

**This is the [NEEDS-INPUT] gate** — see `NEEDS_INPUT.md` for the options
(chase the short-decode regime with a smaller model / different hardware;
fix the enum bug first; kill G1 as scoped; or narrow to the smaller claim
the data does support).

## Scoop-check status (iter 3: 11 sources; iter 4 top-up: 4 more, none a kill)
**PSC (parser-stack-classification grammar masking, OpenReview SEjxNfQTHN)**
is a moderate erosion risk — it drives mask-*computation* cost toward zero
algorithmically, which shrinks (but doesn't eliminate) the sync-stall
mechanism G1 targets, since even a near-zero-cost mask still blocks the next
token if unpipelined. Iter 4 re-ran the specific query BRIEF flagged as the
kill-upgrade trigger ("PSC-style masking + pipelining + Joules, on edge
hardware") — still nothing found combining all three; XGrammar's own
CPU/GPU overlap is documented only on datacenter GPUs (H100), not in the
short-forward-pass edge regime. **Still erosion-risk, not a kill.** Full
list and verdicts: digest 2026-08-11.md Part 2 (iter 3) and Iteration 4
section (iter 4 top-up, incl. two edge-SLM energy-characterization papers
relevant to the Pi 5/Jetson hardware-purchase question in NEEDS_INPUT.md).

## Known competitive pressure
MLPerf Edge Agentic owns generic tool-calling benchmarks; ~3.5 BPW threshold
(2505.15030) is prior art; generic energy-aware edge scheduling is crowded
(EnerInfer, HeteroMosaic, Agent.xpu, KAIROS, BiScale). **All of it optimizes
*unconstrained* generation — none touches the constrained decode loop.**

## Open debt
- `VENUES.md` (Stage-1 artifact) was never written; the venue table lives here
  and in BRIEF. Only worth writing if MLSys 2027 fails to materialize and we
  need to re-rank.
- The three draft hypotheses from this iteration's original plan (pipeline
  the mask across the CPU/GPU sync boundary; energy-per-valid-output as
  metric; tail-latency framing for robotics) are still reasonable *shapes*
  for G1's mechanism/contribution — but writing them up as commitment-ready
  was premature given the pilot data above. Revisit once Maria answers
  NEEDS_INPUT.md; don't re-draft from scratch, the shapes are still sound.

## Next action (iteration 5)
**Still blocked on Maria at the [NEEDS-INPUT] gate** (`NEEDS_INPUT.md`,
unanswered since 06:26 UTC 2026-08-11). Iteration 4 completed the light
tracking pass iteration 3 called for — MLSys 2027 CFP: no change, still
unconfirmed; PSC scoop re-check: no kill found — and turned up nothing that
changes the decision Maria needs to make. Next iteration should repeat the
**same light tracking pass** (re-check MLSys 2027 CFP, re-run the PSC
kill-upgrade query) and continue to hold off on new Mac-hardware experiments
or further G1 hypothesis drafts. Per BRIEF's pace governor: the gate opened
2026-08-11 06:26 UTC, so the 48h reminder threshold is **~2026-08-13 06:26
UTC** — if still unanswered by the iteration that runs at or after that
time, send exactly one reminder ping, then go quiet on this thread until
Maria answers.
