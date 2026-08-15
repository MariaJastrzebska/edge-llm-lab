# Paper machine — STATUS (updated 2026-08-15, iteration 8)

**Scope decisions so far (newest first):**
1. 2026-08-15 (iter 8): reconciliation iteration, not a new scope-narrowing
   decision by the loop itself — but it caught STATUS.md up to five real
   decisions/artifacts that landed between iter 7 and now without ever being
   folded in (see "What changed since iter 7" below). Opened a **new
   `NEEDS_INPUT.md` gate**: has Maria actually confirmed Route A (the
   from-science strategic review's recommended direction — edge VLM medical
   structured-report generation), or do her hardware decisions only read as
   consistent with it without being an explicit pick? Did one piece of
   literature-only feasibility prep for Route A regardless of the answer
   (mtmd+grammar support check — de-risked, no llama.cpp fork needed for the
   smoke test) plus a 2-item literature scoop-check (neither a scoop). Full
   detail: `docs/research/digests/2026-08-15.md`.
2. 2026-08-14 (via chat, outside loop tracking — answered the iter-6 gate):
   experiment 005 CONFIRMED G1's crossover (all four sub-1B models, positive
   paired latency tax, 28× spread). Maria's decision on how to spend the
   runway: (a) NOT a characterization-only paper — must be characterization +
   MECHANISM + an OPTIMIZATION METHOD that reduces the tax, that's the
   headline contribution; (b) mechanism must be nailed via a vocab-controlled
   experiment before drafting; (c) a correctness-tax gate (task-correctness
   grading, not just schema validity) must exist before any "constrained
   decoding is harmless" claim. See `docs/research/DECISIONS.md`.
3. 2026-08-14 (via chat): controlled-experimental-design correction — 005's
   four models differ on vendor/size/vocab/architecture simultaneously, so the
   28× spread is unattributable as designed. Future driver experiments MUST
   vary one factor at a time (same-family size sweep, vocab isolation at
   matched decode speed, cross-vendor replication at matched size+vocab).
4. 2026-08-14 (via chat — STRATEGIC DIRECTION REVIEW, pause G1 build): before
   committing to G1's optimization machinery, Maria asked for a from-scratch
   direction review — survey previous rounds of the target venues, return to
   her actual stated interests (medical/surgical robotics, privacy, edge
   energy), produce a scored slate of candidate hypotheses (G1 included),
   benchmark G1 against the alternatives, **keep the repo/harness regardless
   of outcome**. Until she picks from the slate: **literature-only tracking —
   no new Mac experiments, no G1-optimization-fork build, no drafting.** This
   restriction is still in force (see gate below).
5. 2026-08-14 (via chat — extend scope to VLMs): consider vision-language
   models, not just text LLMs. Her real area (medical/surgical robotics) is
   vision-centric (endoscopy, surgical-scene understanding, VLA); VLMs also
   have a different compute profile (heavy vision-encoder prefill + short LM
   decode) that changes the constrained-decoding-tax mechanism and opens a
   less-explored angle. Harness extends via llama.cpp/Ollama (llava,
   moondream, qwen2-VL, etc. — mtmd tool).
6. 2026-08-14 (via chat — hardware framing, Maria's reviewer-grade
   challenge): "Apple isn't a natural deployment environment" accepted as a
   valid reviewer risk. Reframed every route's scientific claim around the
   **unified-memory edge-SoC class**, not "Apple" specifically: M1 = dev
   instrument + clinician's-laptop use case (on-device privacy), Jetson =
   deployment environment. Approved as **load-bearing**: buy a Jetson Orin
   Nano for a cross-device replication leg (closes the "n=1 / unnatural
   environment" reviewer objection).
7. 2026-08-14 (via chat — hardware, stay on M1 for now): Jetson purchase
   (Nano ~1.5k / NX 16GB ~4k PLN) **postponed** — return to it after first
   results on M1. The unified-memory-SoC framing and planned replication leg
   stand unchanged; just not buying yet.
8. 2026-08-11 (iter 3, closed by iter 5): Stage corrected 3 → 4 — a Stage-4
   pilot had already run outside this loop's tracking.
9. 2026-08-10 (iter 2): axes **federated SLM** and **edge VLA for robots**
   DEMOTED — both scooped by mid-2026 work at the time. (Note: "edge VLM" —
   vision *models*, not the VLA/robot-action framing — was independently
   revived by decision #5 above under the medical/vision-centric lens; not a
   contradiction, different axis.)
10. 2026-08-10: direction = **edge × robotics × privacy and/or medical**;
    must be low-level/systems-strong; should ideally (not necessarily) feed
    bigos later.
11. 2026-08-10: "multi-turn agentic eval" framing DEMOTED (not low-level
    enough).

## What changed since iter 7 (why this reconciliation was needed)

STATUS.md had not been touched since iteration 7 (2026-08-14), but real,
committed work kept happening via direct chat with Maria and a separate
Claude Science project (`docs/research/from-science/`, per
`BRIEF-FOR-SCIENCE-G1.md`). In order: the iter-6 gate got answered (item 2
above) → Maria asked for a full strategic direction review before building
G1's optimization machinery (item 4) → she separately flagged VLMs in scope
(item 5) → the review landed and synthesized into
`docs/research/from-science/2026-08-14-FINAL-strategy-routes.md`,
recommending **Route A** (edge VLM medical structured-report generation) over
three alternatives → Maria made two hardware decisions (items 6–7) that read
as consistent with Route A's medical/unified-memory framing without an
explicit "yes, Route A" statement. This iteration is the first to fold all of
that into STATUS.md; the discipline of checking the working tree/repo state
before trusting a stale STATUS.md (established in iterations 3, 5, 6) applied
here at a larger scale than before.

## Current stage: **3/4 boundary — blocked on route confirmation.**
**NEW GATE OPEN — see `NEEDS_INPUT.md` at repo root.**

Experiment 005 already satisfied Stage 4's exit criterion ("a 1-week pilot
supports or kills the hypothesis") for **G1 in its original text-only
framing**: all four tested sub-1B models show a positive, CI-clean latency
tax below ~25ms/token (qwen2.5:0.5b +1.84%, qwen3:0.6b +2.42%, llama3.2:1b
+6.15%, gemma3:1b +30.36%). That result stands and isn't in question.

What's NOT resolved is which *paper* that result feeds into. The from-science
strategic review scored G1's original text framing as the **weakest** of four
routes (relegated to "Trasa D — fallback… a generalization cell within A")
and recommended **Route A**, the VLM/medical generalization, instead. Per
Stage 3's own exit criterion in `BRIEF.md` — "top candidate passes novelty
check + feasibility + **MARIA CONFIRMS it** ([NEEDS-INPUT] gate — genuinely
hers)" — that confirmation hasn't happened in so many words yet, even though
her subsequent hardware decisions read as compatible with it. See
`NEEDS_INPUT.md` for the exact question and the four options (A/B/C/D) with
their tradeoffs.

<details><summary>Stage-2 exit justification (honest accounting, unchanged since iter 2)</summary>

~40 sources swept pre-Stage-4 on the original text-LLM framing; the surviving
axis (energy/decoding machinery) cleared the ≥15-source bar comfortably. Axes
1 and 2 were closed at 8–9 sources each by the kill criterion, not exhaustive
sweep. The from-science strategic review (2026-08-14) constitutes a second,
much larger sweep specifically for the route-selection question — see
`docs/research/from-science/2026-08-14-landscape-systems.md` and
`2026-08-14-landscape-robotics-medical.md` for their own source counts.

</details>

## Nearest deadlines (re-checked 2026-08-15, iter 8)
| Venue | Deadline | Note |
|---|---|---|
| ~~ICRA 2027~~ | ~Sep 15, 2026 | ~4 weeks out — too tight for a primary target regardless of route; from-science review already scored Route B (robot-grammar) as losing to Route A partly for this reason. Stays open as a paper #2 possibility in the spring robotics season. |
| **MLSys 2027** | **~Oct 30, 2026 — still UNCONFIRMED** | `mlsys.org` and `mlsys-deadlines.github.io` both still blocked by this session's network egress policy — unchanged from iterations 5–7. WebSearch snippets still show only a *predicted* date carried over from the 2025→2026 cycle pattern, no official 2027 CFP page found. Re-check every iteration; try again from a session without the block if one is available. |
| RA-L | rolling | backup — absorbs any MLSys slip with no loss; also Route C's (zero-egress) named venue if that route is picked. |
| IROS 2027 / RO-MAN / ICORR | ~Feb–Mar 2027 | paper #2 / medical-HRI angle; also where Route B would land if revived. |
| SenSys 2027 | 2026 round had a May deadline; next cycle unchecked this iteration | embedded-AI flagship; Route C's named alternate venue. |

## G1 status — text-only framing, five measurement rounds, first clean crossover (2026-08-10 → 2026-08-13)

Unchanged since iter 7 — no new hardware runs this iteration (literature-only
mode, per the still-open direction-review restriction).

| model | rounds w/ CI excluding zero | direction | best estimate |
|---|---|---|---|
| **qwen2.5:0.5b** (005) | 1 of 1, cleanest number (near-zero length confound) | positive | **+1.84%** |
| **qwen3:0.6b** (005) | 1 of 1 | positive | **+2.42%** (likely understated, see length-drift note) |
| **llama3.2:1b** (005) | 1 of 1 | positive | **+6.15%** (~+3.4% length-corrected) |
| **gemma3:1b** (005) | 1 of 1, largest effect seen to date | positive | **+30.36%** (length-correction negligible) |
| phi4-mini | 3 of 3 (003, 004-gbnf, 004-json_schema) | positive, consistent | +1.5–1.8%, replicated, but at 52ms/tok — outside the regime that now matters most |
| granite3.1-dense:2b | 2 of 4; NOT in 004 (larger n) | inconsistent | treat as null |
| qwen3:4b | 006 budget fix did NOT resolve truncation; length-confounded | — | not comparable, needs thinking disabled or length-matched design |

**Vocab-size as sole driver of the 28× spread:** downgraded in iter 7 to
"explains at most 1 of 4 data points" (llama3.2:1b has the smallest vocab but
second-highest tax). Per decision #3 above, the fix isn't more literature —
it's a controlled matched-factor experiment, which is hardware work and thus
on hold behind the same gate as everything else. If Route A is confirmed,
this becomes the "generalization cell" question inside Route A's controlled
matrix rather than a separate text-only experiment.

## Route A feasibility prep (this iteration, literature-only)

Checked whether Route A's named first step — an `mtmd` (multimodal) smoke
test on the M1, image input + grammar constraint together — is blocked by
anything in the current llama.cpp build. It isn't: `llama-mtmd-cli` already
supports `--grammar-file` / `-j/--json-schema` alongside `-mm/--mmproj`, and
the supported vision architectures include all four models the from-science
report named (LLaVA, Qwen2-VL, MiniCPM-V; moondream separately). No fork
needed just to run the smoke test. Full detail:
`docs/research/digests/2026-08-15.md` Part 1.

## Correctness-tax gap (open since iter 7, unchanged)
005/006 measure schema validity, not answer correctness. arXiv 2605.26128
("The Constraint Tax") shows the two can diverge sharply on this exact task
shape. Still needs a harness change (task-correctness grading) outside this
loop's `docs/research/`-only scope — flagged, not fixed, applies to whichever
route is confirmed.

## Scoop-check status (iter 8: two new items, neither a scoop)
An ACM 2026 paper on edge-deployed VLMs for medical-imaging report generation
(dl.acm.org/doi/10.1145/3796263.3796284) is the closest-yet paper to Route A's
exact use case, but its mechanism is model compression (distillation/pruning/
quantization), not constrained-decoding cost — differentiate-from, worth a
related-work sentence if Route A proceeds. A general (non-edge) constrained-
decoding speedup via token-space compression (arXiv 2605.29986) is a citable
alternative-mitigation reference, not competitive with the shadow-mask
pipelining method. Full detail: `docs/research/digests/2026-08-15.md` Part 2.

## Known competitive pressure (unchanged since iter 7)
MLPerf Edge Agentic owns generic tool-calling benchmarks; ~3.5 BPW threshold
(2505.15030) is prior art; generic energy-aware edge scheduling is crowded
(EnerInfer, HeteroMosaic, Agent.xpu, KAIROS, BiScale). Algorithmic
mask-speedup work (PSC, GRID, Attention-Meets-Reachability, and now
Token-Space-Compression 2605.29986) is a growing but still non-overlapping
erosion-risk cluster — none on edge hardware, none in Joules, none with
CPU/GPU pipelining measured against a short forward pass, and (as of this
iteration) none on VLMs either.

## Open debt
- `VENUES.md` (Stage-1 artifact) never written; only worth writing if MLSys
  2027 fails to materialize and a re-rank is needed.
- Correctness-tax grading gap — needs a harness change outside
  `docs/research/`; not this loop's to fix.
- Vocab-size as sole driver of the 28× spread — still open, folded into
  Route A's planned controlled matrix if that route is confirmed (see above).
- The three draft hypothesis shapes from iteration 3 (pipeline the mask;
  energy-per-valid-output; tail-latency framing) are subsumed by whichever
  route Maria confirms — Route A explicitly inherits the pipelining method as
  its "hero optimization."

## Next action (iteration 9)
1. **Check `NEEDS_INPUT.md` first.** If Maria has answered the route
   question: close the gate per her choice, update the "Current stage" line
   to reflect the confirmed route, and — only if she also said the
   literature-only restriction is lifted — resume normal pacing (queue the
   route's first hardware step, e.g. Route A's mtmd smoke test). If not yet
   answered: stay in reduced-pace literature-only mode. Pace-governor note:
   this iteration (8) is not "empty" (it produced a feasibility de-risk
   finding and two new literature items), so the empty-iteration counter
   stays at 0.
2. **Literature tracking (repeat, ~5 min):** re-check MLSys 2027 CFP status;
   if Route A still looks live, a light follow-up sweep on "edge VLM energy
   profiling" and "medical report generation grammar/schema constraints"
   would sharpen the blue-ocean claim further before drafting starts.
3. Continue the check-the-working-tree-before-trusting-STATUS discipline —
   this iteration found the biggest gap yet (five real decisions
   unreflected); check `docs/research/DECISIONS.md` and
   `docs/research/from-science/` for new files every iteration, not just
   `experiments/results/`.
