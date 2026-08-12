# Paper machine — STATUS (updated 2026-08-12, iteration 5)

**Scope decisions so far (newest first):**
1. 2026-08-12 (iter 5): no new scope-narrowing decision. Caught STATUS.md up
   to reality — the `NEEDS_INPUT.md` gate from iteration 3 was **answered by
   Maria on 2026-08-11 (option 2: fix the enum bug, then re-run)** and is
   closed; this file had not been updated to reflect that. Since the answer,
   outside this loop's own tracking (same pattern as iteration 3): the enum
   fix landed (`b64b4ed`), 002/003 were re-run clean, and a new **experiment
   004 (enum-fixed remeasure) completed** overnight 2026-08-12 with a
   genuinely new result — see below. Two follow-up experiments (005
   short-forward-pass, 006 qwen3-budget-fix) are already queued in
   `experiments/queue/` for Maria's next Mac session; they do not require a
   new gate to exist, only her hardware time to run.
2. 2026-08-11 (iter 4): light tracking pass only, per the (then-open) gate.
3. 2026-08-11 (iter 3): Stage corrected 3 → 4 — a Stage-4 pilot had already
   run outside this loop's tracking.
4. 2026-08-10 (iter 2): axes **federated SLM** and **edge VLM/VLA for
   robots** DEMOTED — both scooped by mid-2026 work. Surviving axis:
   **energy-aware edge inference, narrowed to the decode loop's
   structured-generation machinery.**
5. 2026-08-10: direction = **edge × robotics × privacy and/or medical**;
   must be low-level/systems-strong; should ideally (not necessarily) feed
   bigos later.
6. 2026-08-10: "multi-turn agentic eval" framing DEMOTED (not low-level
   enough).

**Current stage: 4 — VALIDATION. Gate closed, no new blocking gate open.**
The G1 pilot now has four measurement rounds across two nights. The headline
change since iteration 4: **phi4-mini's latency tax replicated cleanly three
times running (+1.51%, +1.65%, +1.78%, all CIs excluding zero)**; granite's
earlier apparent effect (+0.34%–+0.66%, CIs excluding zero in 002/003) **did
not survive** a larger-n rerun in 004 (CI spans zero, one arm flips sign) —
the source report itself declines to keep the earlier granite number. This
is the first genuinely informative (not null, not wrong-direction) data
point in the whole G1 investigation. Full analysis:
`docs/research/digests/2026-08-12.md` Part 1; raw report:
`experiments/results/NIGHT-REPORT-2026-08-12.md`.

No new `[NEEDS-INPUT]` gate opened this iteration — the natural next
experiments (005, 006) are already queued and don't need permission to
exist, only Mac time. But the significance question Maria reserved for
herself in `DECISIONS.md` (2026-08-10, point 2: "if the tax is below
meaningful, G1 dies, no attachment to the candidate") now has real data to
apply to on at least one model, independent of whatever 005/006 show. Flagged
in the digest as worth her direct read, not gated on a worker decision.

<details><summary>Stage-2 exit justification (honest accounting, unchanged since iter 2)</summary>

~40 sources swept pre-Stage-4; the surviving axis (energy/decoding machinery)
cleared the ≥15-source bar comfortably. Axes 1 and 2 were closed at 8–9
sources each by the kill criterion, not exhaustive sweep.

</details>

## Nearest deadlines (re-checked 2026-08-12, iter 5)
| Venue | Deadline | Note |
|---|---|---|
| ~~ICRA 2027~~ | Sep 15, 2026 | out of reach, unchanged |
| **MLSys 2027** | **~Oct 30, 2026 — still UNCONFIRMED** | Aggregators (mlsys-deadlines.github.io, casys-kaist.github.io) still show 2027 as TBD/predicted off the 2026 date. **New this iteration: `mlsys.org` is blocked by this session's network egress policy** — could not attempt direct verification against the primary source this pass (prior iterations reported it as merely unreachable/no-CFP-yet; this iteration it's an explicit access block). Re-check every iteration; try again from a session without the block if one is available. |
| RA-L | rolling | backup — absorbs any MLSys slip with no loss |
| IROS 2027 / RO-MAN / ICORR | ~Feb–Mar 2027 | paper #2 / medical-HRI angle |

## G1 status — four measurement rounds, mixed but informative (2026-08-10 → 2026-08-12)

| model | rounds w/ CI excluding zero | direction | best estimate |
|---|---|---|---|
| **phi4-mini** | 3 of 3 (003, 004-gbnf, 004-json_schema) | positive, consistent | **+1.5–1.8%**, replicated |
| granite3.1-dense:2b | 2 of 4 (002-rerun, 003-rerun); NOT in 004 (larger n) | inconsistent, effect not established | treat as null |
| qwen3:4b | 2 of 2 | negative | **artifact** — 60/60 unconstrained runs hit the 768-token cap mid-reasoning-prose; not a like-for-like comparison, queued experiment 006 targets this directly |

**Reading:** magnitude on the one model with a real signal is ~1.5–1.8%, far
below the 3.6–8.2× literature high end — consistent with what iterations 3–4
already suspected, now on firmer footing (replicated, not just "not
detected"). **G1's actual predicted mechanism (short forward pass) is still
untested** — nothing measured decodes faster than ~34.6ms/token. Queued
experiment 005 targets that regime directly (sub-1B models via Ollama,
should reach <20ms/token). Absolute decode-time is **not comparable across
nights** (suspected observer-contention confound, not thermal — see
NIGHT-REPORT-2026-08-12.md §3); only within-run paired comparisons are valid,
and all numbers above already are.

## Scoop-check status (iter 5 top-up: 5 new sources, none a kill)
Re-ran the PSC kill-upgrade query ("algorithmic mask speedup + CPU/GPU
pipelining + Joules + edge hardware") — still nothing found. Two new
algorithmic mask-speedup papers (GRID, arXiv 2607.11951; "Attention Meets
Reachability," arXiv 2603.05540) join PSC in the same erosion-risk bucket —
neither measures edge hardware or energy, neither touches the sync-stall
mechanism. Two new validity-axis "tax" papers found (Constraint Tax, arXiv
2605.26128; Format Tax, arXiv 2604.03616) — different mechanism from G1
(correctness/prompt-framing, not latency/energy), cited as adjacent
background; Constraint Tax specifically worth reading before 005 runs, since
its sub-1B/1.5B results suggest constrained small models may trade validity
for correctness in exactly the size class 005 targets — a confound the
current pilot's grading may not currently distinguish. Full list and
verdicts: digest 2026-08-12.md Part 2.

## Known competitive pressure
MLPerf Edge Agentic owns generic tool-calling benchmarks; ~3.5 BPW threshold
(2505.15030) is prior art; generic energy-aware edge scheduling is crowded
(EnerInfer, HeteroMosaic, Agent.xpu, KAIROS, BiScale). Algorithmic mask-speedup
work (PSC, GRID, Attention-Meets-Reachability) is a growing but still
non-overlapping erosion-risk cluster — none on edge hardware, none in Joules,
none with CPU/GPU pipelining measured against a short forward pass. **All of
it still optimizes *unconstrained* generation or mask *computation* cost —
none touches the constrained-decode sync-stall mechanism G1 is about.**

## Open debt
- `VENUES.md` (Stage-1 artifact) never written; only worth writing if MLSys
  2027 fails to materialize and a re-rank is needed.
- The three draft hypothesis *shapes* from iteration 3 (pipeline the mask
  across the CPU/GPU sync boundary; energy-per-valid-output as metric;
  tail-latency framing for robotics) are still sound and still not written
  up as commitment-ready. With phi4-mini's replicated effect now in hand,
  the "pipeline the mask" shape is the one most directly motivated by data —
  worth drafting once 005/006 land, not before (their result changes what
  the hypothesis should claim).
- Constraint Tax (arXiv 2605.26128) not yet read in full — flagged as a
  pre-read for 005, not done this iteration (time budget).

## Next action (iteration 6)
1. **Literature tracking (repeat, ~5 min):** re-check MLSys 2027 CFP status
   (retry `mlsys.org` directly in case the egress block was session-specific
   rather than durable; otherwise stick with aggregator TBD reading);
   re-run the PSC/GRID/Attention-Meets-Reachability kill-upgrade query.
2. **Read Constraint Tax (arXiv 2605.26128) in full** and note in the next
   digest whether its validity/correctness tradeoff on sub-1B/1.5B models is
   likely to confound queued experiment 005's grading (i.e. does 005 need a
   correctness check added to its scope, not just schema-validity, before it
   runs) — a research-literature question this loop can answer without
   Mac hardware, and worth resolving *before* 005 runs rather than after.
3. Continue watching for 005/006 results landing in `experiments/results/`
   (same "check the working tree before trusting STATUS.md" discipline that
   caught iterations 3 and 5's stale gate state) — if they've run, analyzing
   them is Stage-4 priority work over any of the above.
4. No new Mac-hardware experiments or new `[NEEDS-INPUT]` gate needed from
   this loop; 005/006 are Maria's to run on her own schedule.
