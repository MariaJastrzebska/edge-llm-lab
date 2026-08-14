# Paper machine — STATUS (updated 2026-08-14, iteration 7)

**Scope decisions so far (newest first):**
1. 2026-08-14 (iter 7): no new scope-narrowing decision. `NEEDS_INPUT.md` (iter 6)
   still unanswered — stayed in reduced-pace literature-tracking mode per the pace
   governor, no new hardware queued, no drafting. Checked the working tree first:
   no new files in `experiments/results/` since 005/006. Did the research-only
   vocab-size rank-order check STATUS asked for last iteration: **it does not
   cleanly explain the 28× spread** (llama3.2:1b has the smallest vocab of the
   four tested models but the second-highest tax — contradicts a clean
   vocab-size-as-sole-driver story). Appended an update note to `NEEDS_INPUT.md`
   since it weakens (doesn't kill) that file's option 2 premise. Two new
   literature items logged (arXiv 2605.07698, 2608.10137) — both adjacent, neither
   a scoop. MLSys 2027 CFP still unconfirmed, still blocked at the network level,
   no change from iteration 6. Full detail: `docs/research/digests/2026-08-14.md`.
2. 2026-08-13 (iter 6): no new scope-narrowing decision, but a new
   `[NEEDS-INPUT]` gate opened — see below. Analyzed the overnight 005/006
   results (already committed by the time this iteration ran, same
   check-the-working-tree-before-trusting-STATUS discipline as iterations 3
   and 5) and one new literature confound found this iteration (Constraint
   Tax's tool-call correctness collapse, arXiv 2605.26128 — read in full
   this iteration, previously only skimmed on 2026-08-12).
2. 2026-08-12 (iter 5): no new scope-narrowing decision. Caught STATUS.md up
   to reality — the `NEEDS_INPUT.md` gate from iteration 3 was **answered by
   Maria on 2026-08-11 (option 2: fix the enum bug, then re-run)** and is
   closed; this file had not been updated to reflect that. Since the answer,
   outside this loop's own tracking (same pattern as iteration 3): the enum
   fix landed (`b64b4ed`), 002/003 were re-run clean, and a new **experiment
   004 (enum-fixed remeasure) completed** overnight 2026-08-12.
3. 2026-08-11 (iter 4): light tracking pass only, per the (then-open) gate.
4. 2026-08-11 (iter 3): Stage corrected 3 → 4 — a Stage-4 pilot had already
   run outside this loop's tracking.
5. 2026-08-10 (iter 2): axes **federated SLM** and **edge VLM/VLA for
   robots** DEMOTED — both scooped by mid-2026 work. Surviving axis:
   **energy-aware edge inference, narrowed to the decode loop's
   structured-generation machinery.**
6. 2026-08-10: direction = **edge × robotics × privacy and/or medical**;
   must be low-level/systems-strong; should ideally (not necessarily) feed
   bigos later.
7. 2026-08-10: "multi-turn agentic eval" framing DEMOTED (not low-level
   enough).

**Current stage: 4 — VALIDATION. NEW GATE OPEN — see `NEEDS_INPUT.md` at
repo root.**

Stage 4's own exit criterion just fired: "a 1-week pilot on Maria's Mac
supports or kills the hypothesis." Experiment 005 (480/480 runs, all four
sub-1B models, overnight 2026-08-12/13) is the first pilot round where **all
four tested models show a positive, CI-clean latency tax** — the crossover
G1 predicted, at the sub-25ms/token regime the mechanism actually needs.
Numbers: qwen2.5:0.5b +1.84%, qwen3:0.6b +2.42%, llama3.2:1b +6.15%,
gemma3:1b **+30.36%**. Full analysis: `docs/research/digests/2026-08-13.md`
Part 1; raw report: `experiments/results/NIGHT-REPORT-2026-08-13.md`.

**This is genuinely new footing, not just "more of the same":** in four
prior pilot rounds (002–004) nothing ever showed a clean positive effect on
more than one model at a time, and the strongest replicated signal was
phi4-mini's flat +1.5–1.8%. 005 is the first time the *mechanism itself*
(short forward pass can't hide the mask cost) shows up cleanly and
everywhere it was predicted to.

**But it doesn't resolve to one number.** Absolute mask cost varies 28×
across the four models in the same 11–25ms decode-speed band, and the
relative tax is not monotone in decode speed — gemma3:1b pays 30% while
decoding *faster* than qwen3:0.6b's 2.4%. Model identity dominates. The
report's candidate driver (vocabulary size — mask cost is O(vocab),
supported as mechanistically plausible by the general grammar-decoding
literature, see digest Part 2) is untested on this data. **A new
literature finding this iteration adds a second open thread**: arXiv
2605.26128 ("The Constraint Tax," read in full this iteration) shows that
on a near-identical tool-call-schema task, schema-valid outputs can hide a
collapsed answer-correctness rate (91.5%→48.0% in one worked example,
while validity holds at 100%→100%). 005/006 grade only schema validity —
nothing in the harness currently checks whether the 60/60-valid constrained
runs in 005 are semantically correct medical envelopes, as opposed to
schema-valid noise.

Because Stage 4's exit condition fired with a result that's real but not
yet a single clean claim, and because the correctness-tax gap needs harness
code (outside this loop's `docs/research/`-only scope) to close, this
iteration opened `NEEDS_INPUT.md` — the question is how Maria wants to
spend the runway to MLSys 2027: draft now on the honest-but-messy claim, or
spend one more validation round (vocab-size isolation + correctness
grading) before committing to draft.

<details><summary>Stage-2 exit justification (honest accounting, unchanged since iter 2)</summary>

~40 sources swept pre-Stage-4; the surviving axis (energy/decoding machinery)
cleared the ≥15-source bar comfortably. Axes 1 and 2 were closed at 8–9
sources each by the kill criterion, not exhaustive sweep.

</details>

## Nearest deadlines (re-checked 2026-08-14, iter 7)
| Venue | Deadline | Note |
|---|---|---|
| ~~ICRA 2027~~ | Sep 15, 2026 | out of reach, unchanged |
| **MLSys 2027** | **~Oct 30, 2026 — still UNCONFIRMED** | `mlsys.org` and `mlsys-deadlines.github.io` both still blocked by this session's network egress policy (same block as iterations 5–6, no change). WebSearch snippets still show only a *predicted* date carried over from 2026, no official 2027 CFP page found. Re-check every iteration; try again from a session without the block if one is available. |
| RA-L | rolling | backup — absorbs any MLSys slip with no loss |
| IROS 2027 / RO-MAN / ICORR | ~Feb–Mar 2027 | paper #2 / medical-HRI angle |

## G1 status — five measurement rounds, first clean crossover (2026-08-10 → 2026-08-13)

| model | rounds w/ CI excluding zero | direction | best estimate |
|---|---|---|---|
| **qwen2.5:0.5b** (005) | 1 of 1, cleanest number (near-zero length confound) | positive | **+1.84%** |
| **qwen3:0.6b** (005) | 1 of 1 | positive | **+2.42%** (likely understated, see length-drift note) |
| **llama3.2:1b** (005) | 1 of 1 | positive | **+6.15%** (~+3.4% length-corrected) |
| **gemma3:1b** (005) | 1 of 1, largest effect seen to date | positive | **+30.36%** (length-correction negligible) |
| phi4-mini | 3 of 3 (003, 004-gbnf, 004-json_schema) | positive, consistent | +1.5–1.8%, replicated, but at 52ms/tok — outside the regime that now matters most |
| granite3.1-dense:2b | 2 of 4; NOT in 004 (larger n) | inconsistent | treat as null |
| qwen3:4b | 006 budget fix did NOT resolve truncation; length-confounded, unusable for speed comparison | — | not comparable, needs thinking disabled or length-matched design |

**Reading:** the crossover G1 predicted (tax appears once the forward pass gets
short enough to stop hiding the mask cost) is now directly observed, at
sub-25ms/token, on every model tested there. What's not yet established is
*why* the size of that tax varies 28× by model rather than tracking decode
speed. **Vocabulary size, checked against public model specs this iteration
(iter 7, no new hardware), does NOT cleanly rank-order with the spread** —
llama3.2:1b has the smallest vocab of the four (128,256) but the
second-highest tax (+6.15%), higher than both Qwen models despite their
~19% larger vocab (151,936) and lower tax (+1.84%, +2.42%); the two Qwen
models even share the same vocab_size and still differ 32% relative in tax.
Only gemma3:1b (largest vocab, largest tax) fits. Full table:
`docs/research/digests/2026-08-14.md` Part 1. Downgraded from "leading
untested candidate" to "explains at most 1 of 4 data points" — see
`NEEDS_INPUT.md` update note. Absolute decode-time remains not comparable
across nights (observer-contention confound, unchanged); paired within-run
comparisons (all numbers above) are unaffected.

## Correctness-tax gap (new this iteration — see digest Part 2 for the full read)
005/006 measure schema validity, not answer correctness. arXiv 2605.26128
("The Constraint Tax") shows the two can diverge sharply on this exact task
shape (tool-call JSON schema, small models): one worked example goes
91.5%→48.0% correctness while staying 100%→100% valid. Until the harness
scores constrained outputs for semantic correctness (not just schema
validity) — a code change outside this loop's scope — G1's validity numbers
cannot rule out that some fraction of "valid" constrained runs are
schema-valid junk. Flagged in `NEEDS_INPUT.md`, not fixed by this loop.

## Scoop-check status (iter 7 top-up: two new adjacent papers, neither a scoop)
PSC / GRID / Attention-Meets-Reachability query re-run plus a general
grammar-constrained-decoding sweep. Two new URLs entered `seen.json`:
arXiv 2605.07698 ("Future Validity is the Missing Statistic" — grammar-faithful
*speculative* decoding sampling-correctness, different mechanism from G1,
differentiate-from) and arXiv 2608.10137 ("The Parser Already Knows" —
lightweight bias correction for GCD's valid-but-suboptimal distortion,
reinforces the correctness-tax thread already open in `NEEDS_INPUT.md`, cite).
Full read: `docs/research/digests/2026-08-14.md` Part 2. Everything else the
sweep surfaced was already in `seen.json` from prior iterations, verdicts
unchanged — no direct scoop of G1's specific claim (edge-hardware,
latency/Joules, mask-cost-hidden-by-forward-pass mechanism) found.

## Known competitive pressure
MLPerf Edge Agentic owns generic tool-calling benchmarks; ~3.5 BPW threshold
(2505.15030) is prior art; generic energy-aware edge scheduling is crowded
(EnerInfer, HeteroMosaic, Agent.xpu, KAIROS, BiScale). Algorithmic mask-speedup
work (PSC, GRID, Attention-Meets-Reachability) is a growing but still
non-overlapping erosion-risk cluster — none on edge hardware, none in Joules,
none with CPU/GPU pipelining measured against a short forward pass. A quick
architecture check this iteration found no evidence llama.cpp currently
overlaps mask computation with the GPU forward pass the way SGLang/vLLM are
reported to — supports "pipeline the mask" as still novel *for llama.cpp
specifically*, even though the general idea has prior art on other engines.

## Open debt
- `VENUES.md` (Stage-1 artifact) never written; only worth writing if MLSys
  2027 fails to materialize and a re-rank is needed.
- The three draft hypothesis *shapes* from iteration 3 (pipeline the mask
  across the CPU/GPU sync boundary; energy-per-valid-output as metric;
  tail-latency framing for robotics) are still sound and still not written
  up as commitment-ready. "Pipeline the mask" is now the best-motivated of
  the three, but writing it up is explicitly gated on `NEEDS_INPUT.md`'s
  answer, not free to do this iteration.
- Correctness-tax grading gap (new this iteration, see above) — needs a
  harness change outside `docs/research/`; not this loop's to fix.
- Vocab-size as the *sole* driver of the 28× spread in mask cost — now
  **tested against public specs (iter 7) and only partially consistent**
  (explains gemma3, contradicted by llama3.2 vs. the two Qwen models). A
  co-driver or different explanation (grammar-stack-depth,
  tokenizer-fragmentation, something llama3.2-architecture-specific) is
  still open. See `NEEDS_INPUT.md` update note before choosing option 2's
  experiment design.

## Next action (iteration 8)
1. **Check `NEEDS_INPUT.md` first.** If Maria has answered, close the gate
   per her choice (same pattern as iteration 5 closing iteration 3's gate)
   and resume normal pacing. If not yet answered, this loop stays in
   reduced-pace literature-tracking mode per the pace governor — do not
   queue new Mac-hardware experiments or draft the paper skeleton without
   an answer. Note for the pace governor: iter 7 is not itself an "empty"
   iteration under BRIEF's "3 consecutive empty iterations → 1/day" rule —
   it produced the vocab-size test and two new literature items — so the
   empty-iteration counter is still at 0. It would start counting from the
   first iteration that finds genuinely nothing new while the gate stays
   open.
2. **Literature tracking (repeat, ~5 min):** re-check MLSys 2027 CFP status
   (retry `mlsys.org` directly in case the block is session-specific);
   re-run the PSC/GRID/Attention-Meets-Reachability kill-upgrade query.
3. If time remains and the gate is still open: research-only follow-up on
   *why* llama3.2:1b breaks the vocab-size ranking — candidates worth a
   literature check (not new hardware) are tokenizer/BPE-merge structure
   (llama3.2 uses a different BPE than the Qwen/Gemma tokenizers), grammar-
   stack depth for the specific JSON schema used in 005, and whether
   llama.cpp's grammar implementation has any known per-architecture cost
   asymmetry. Goal: find a candidate that's at least directionally
   consistent across all four models, or confirm no single public-spec
   variable explains it and say so plainly in the next digest.
4. Continue the check-the-working-tree-before-trusting-STATUS discipline
   (caught stale gate state in iterations 3, 5, and unreported results
   landing before tracking in iteration 6) — if new experiment results
   have landed in `experiments/results/`, analyzing them is Stage-4
   priority work over any of the above.
