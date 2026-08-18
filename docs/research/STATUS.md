# Paper machine — STATUS (updated 2026-08-18, iteration 11)

**Scope decisions so far (newest first):**
1. 2026-08-18 (iter 11): no scope-narrowing decision — `NEEDS_INPUT.md` (route
   confirmation, opened iter 8) is still open, gate age now ~72h. The
   constitution's one-reminder-ping already went out at iter 10; **not
   re-pinged this iteration** (no new blocking event opened). Stayed in
   reduced-pace literature-only mode: no new Mac experiments, no Route-A
   build, no drafting. MLSys 2027 re-check: still unconfirmed — `mlsys.org`,
   `mlsys-deadlines.github.io`, and `mlciv.com` (both its general page and,
   new this iteration, its per-conference `?id=mlsys27` URL) all still
   blocked by network egress; WebSearch shows only the same extrapolated
   prediction seen since iter 5. Widened the sweep per iter 10's own
   suggestion (moving past the DCCD/schema-key-wording-specific checks,
   which had come up empty two iterations running) to edge-VLM/medical-
   structured-generation and Apple-Silicon-inference-engine angles — six new
   items entered `seen.json`, none competitive with G1/Route A's hardware-
   cost claim (none touch grammar/schema-constrained decoding), but two
   (BaseRT, arXiv 2607.00501 and its M5 follow-up 2607.19438 — a native-Metal
   runtime beating llama.cpp decode throughput 1.15–1.56× on the exact model
   families G1 tested) are flagged differentiate-from as useful "Apple
   Silicon is an active optimization target" reviewer-defense ammunition and
   a future baseline-drift risk to watch. **Non-empty iteration** — empty-
   iteration counter resets to 0 of 3 (was 1 of 3 after iter 10). Full
   detail: `docs/research/digests/2026-08-18.md`.
2. 2026-08-17 (iter 10): no scope-narrowing decision — `NEEDS_INPUT.md` (route
   confirmation, opened iter 8, 2026-08-15 06:15:57 UTC) crossed the
   constitution's 48h blocked-on-Maria threshold during this iteration (gate
   age at iter 10: ~48h11m). Per the pace governor ("Blocked on Maria >48h →
   exactly ONE reminder ping, then silence"), sent that one reminder via
   push notification this iteration — full context, both questions from
   `NEEDS_INPUT.md` (route A/B/C/D pick, and whether the literature-only
   restriction lifts with it), written in Maria's required layered-Polish
   report style. **Do not re-ping** on this gate in future iterations unless
   she responds and a *new* blocking event opens later — the constitution
   says exactly one, then silence. Stayed in reduced-pace literature-only
   mode per the gate's own instructions: no new Mac experiments, no Route-A
   build, no drafting. Literature check (~5 min, per iter 9's next-action):
   MLSys 2027 re-check — still unconfirmed/blocked, no change; targeted
   citing-work check on DCCD (2603.03305) and the schema-key-wording paper
   (2604.14862) — no edge-hardware or latency follow-up found yet; DCCD's
   only update is a confirmed ICML 2026 poster listing (peer-review
   confirmation, not a new competitive signal). **Empty iteration for
   pace-governor purposes** (zero new competitive-landscape items) — empty-
   iteration counter now at 1 of 3 (iter 9 was non-empty, so counter reset
   there; needs 2 more consecutive empty iterations to trigger the 1/day
   pace drop). Full detail: `docs/research/digests/2026-08-17.md`.
3. 2026-08-16 (iter 9): no scope-narrowing decision — `NEEDS_INPUT.md` (route
   confirmation, opened iter 8) is still open and unanswered; no chat
   decision or `DECISIONS.md` entry landed since commit `cd62ba8`. Stayed in
   reduced-pace literature-only mode per the gate's own instructions: no new
   Mac experiments, no Route-A build, no drafting. Did a light follow-up
   literature sweep (MLSys 2027 recheck — still unconfirmed/blocked, no
   change; edge-VLM-energy and medical-structured-generation sweep — 7 new
   items, none competitive with G1/Route A, two reinforce the open
   correctness-tax gap). Full detail: `docs/research/digests/2026-08-16.md`.
4. 2026-08-15 (iter 8): reconciliation iteration, not a new scope-narrowing
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
5. 2026-08-14 (via chat, outside loop tracking — answered the iter-6 gate):
   experiment 005 CONFIRMED G1's crossover (all four sub-1B models, positive
   paired latency tax, 28× spread). Maria's decision on how to spend the
   runway: (a) NOT a characterization-only paper — must be characterization +
   MECHANISM + an OPTIMIZATION METHOD that reduces the tax, that's the
   headline contribution; (b) mechanism must be nailed via a vocab-controlled
   experiment before drafting; (c) a correctness-tax gate (task-correctness
   grading, not just schema validity) must exist before any "constrained
   decoding is harmless" claim. See `docs/research/DECISIONS.md`.
6. 2026-08-14 (via chat): controlled-experimental-design correction — 005's
   four models differ on vendor/size/vocab/architecture simultaneously, so the
   28× spread is unattributable as designed. Future driver experiments MUST
   vary one factor at a time (same-family size sweep, vocab isolation at
   matched decode speed, cross-vendor replication at matched size+vocab).
7. 2026-08-14 (via chat — STRATEGIC DIRECTION REVIEW, pause G1 build): before
   committing to G1's optimization machinery, Maria asked for a from-scratch
   direction review — survey previous rounds of the target venues, return to
   her actual stated interests (medical/surgical robotics, privacy, edge
   energy), produce a scored slate of candidate hypotheses (G1 included),
   benchmark G1 against the alternatives, **keep the repo/harness regardless
   of outcome**. Until she picks from the slate: **literature-only tracking —
   no new Mac experiments, no G1-optimization-fork build, no drafting.** This
   restriction is still in force (see gate below).
8. 2026-08-14 (via chat — extend scope to VLMs): consider vision-language
   models, not just text LLMs. Her real area (medical/surgical robotics) is
   vision-centric (endoscopy, surgical-scene understanding, VLA); VLMs also
   have a different compute profile (heavy vision-encoder prefill + short LM
   decode) that changes the constrained-decoding-tax mechanism and opens a
   less-explored angle. Harness extends via llama.cpp/Ollama (llava,
   moondream, qwen2-VL, etc. — mtmd tool).
9. 2026-08-14 (via chat — hardware framing, Maria's reviewer-grade
   challenge): "Apple isn't a natural deployment environment" accepted as a
   valid reviewer risk. Reframed every route's scientific claim around the
   **unified-memory edge-SoC class**, not "Apple" specifically: M1 = dev
   instrument + clinician's-laptop use case (on-device privacy), Jetson =
   deployment environment. Approved as **load-bearing**: buy a Jetson Orin
   Nano for a cross-device replication leg (closes the "n=1 / unnatural
   environment" reviewer objection).
10. 2026-08-14 (via chat — hardware, stay on M1 for now): Jetson purchase
   (Nano ~1.5k / NX 16GB ~4k PLN) **postponed** — return to it after first
   results on M1. The unified-memory-SoC framing and planned replication leg
   stand unchanged; just not buying yet.
11. 2026-08-11 (iter 3, closed by iter 5): Stage corrected 3 → 4 — a Stage-4
   pilot had already run outside this loop's tracking.
12. 2026-08-10 (iter 2): axes **federated SLM** and **edge VLA for robots**
    DEMOTED — both scooped by mid-2026 work at the time. (Note: "edge VLM" —
    vision *models*, not the VLA/robot-action framing — was independently
    revived by decision #6 above under the medical/vision-centric lens; not a
    contradiction, different axis.)
13. 2026-08-10: direction = **edge × robotics × privacy and/or medical**;
    must be low-level/systems-strong; should ideally (not necessarily) feed
    bigos later.
14. 2026-08-10: "multi-turn agentic eval" framing DEMOTED (not low-level
    enough).

## What changed since iter 7 (why this reconciliation was needed)

STATUS.md had not been touched since iteration 7 (2026-08-14), but real,
committed work kept happening via direct chat with Maria and a separate
Claude Science project (`docs/research/from-science/`, per
`BRIEF-FOR-SCIENCE-G1.md`). In order: the iter-6 gate got answered (item 3
above) → Maria asked for a full strategic direction review before building
G1's optimization machinery (item 5) → she separately flagged VLMs in scope
(item 6) → the review landed and synthesized into
`docs/research/from-science/2026-08-14-FINAL-strategy-routes.md`,
recommending **Route A** (edge VLM medical structured-report generation) over
three alternatives → Maria made two hardware decisions (items 7–8) that read
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

## Nearest deadlines (re-checked 2026-08-18, iter 11)
| Venue | Deadline | Note |
|---|---|---|
| ~~ICRA 2027~~ | ~Sep 15, 2026 | ~3.5 weeks out — too tight for a primary target regardless of route; from-science review already scored Route B (robot-grammar) as losing to Route A partly for this reason. Stays open as a paper #2 possibility in the spring robotics season. |
| **MLSys 2027** | **~Oct 30, 2026 — still UNCONFIRMED** | `mlsys.org`, `mlsys-deadlines.github.io`, and `mlciv.com` (both its general deadlines page and, new this iteration, its per-conference `?id=mlsys27` URL) are all still blocked by this session's network egress policy — unchanged from iterations 5–10. WebSearch snippets still show only a *predicted* date carried over from the 2025→2026 cycle pattern, no official 2027 CFP page found. Re-check every iteration; try again from a session without the block if one is available. |
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
second-highest tax). Per decision #4 above, the fix isn't more literature —
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

## Correctness-tax gap (open since iter 7, strengthened by iter 9's sweep)
005/006 measure schema validity, not answer correctness. arXiv 2605.26128
("The Constraint Tax") shows the two can diverge sharply on this exact task
shape — now joined by two independent iter-9 finds (DCCD 2603.03305,
schema-key-wording 2604.14862) making the same point by different routes.
Still needs a harness change (task-correctness grading) outside this loop's
`docs/research/`-only scope — flagged, not fixed, applies to whichever route
is confirmed.

## Scoop-check status (iter 11: six new items, none a scoop; iter 10: zero new items; iter 9: seven new items, none a scoop)
Iter 11 widened the sweep beyond the DCCD/schema-key-wording-specific checks
(empty two iterations running) to edge-VLM/medical-structured-generation and
Apple-Silicon-inference-engine angles. Six new items, none touching grammar/
schema-constrained decoding cost, so none competitive with G1/Route A. Two
flagged **differentiate-from**: BaseRT (arXiv 2607.00501) and its M5
follow-up (arXiv 2607.19438) — a native-Metal LLM runtime beating llama.cpp
decode throughput 1.15–1.56× and MLX up to 1.35× on M3/M4 Pro, tested on the
Qwen3, Llama 3.2, and Gemma families (the same families G1 measured). Not a
grammar/constrained-decoding paper, so not a scoop, but two things worth
carrying forward: (a) it's ammunition for the "Apple Silicon is an active,
well-funded optimization target, not an unnatural environment" reviewer
defense (ties to decision #8 above); (b) it's a **baseline-drift risk** —
if a BaseRT-class runtime ever lands under llama.cpp/Ollama, decode-speed
baselines shift and could move models across G1's crossover threshold; worth
a rerun sanity-check whenever the harness's underlying engine is upgraded,
not an action item now. Also newly seen but scored **ignore** (general
engine/benchmark methodology, no constrained-decoding or edge-cost angle):
a Comparative Study of MLX/MLC-LLM/Ollama/llama.cpp (2511.05502), ConsumerBench
(2506.17538), LLM-Inference-Bench (2411.00136). And **differentiate-from**
on the application-space axis: CWCD (arXiv 2604.10410, U. Buffalo) generates
structured radiology reports region-by-region with contrastive decoding to
suppress hallucinated pathology co-occurrences — same *application* as Route
A (structured medical report generation) but a different *mechanism*
(accuracy/hallucination-focused, cloud-scale, no grammar masking, no edge
angle) — worth one related-work sentence distinguishing the two if Route A
proceeds. Full detail: `docs/research/digests/2026-08-18.md`.

Iter 10 targeted-checked whether DCCD or the schema-key-wording paper had
picked up citing/follow-up work touching edge hardware or latency — none
found. DCCD's only update is a confirmed ICML 2026 poster listing
(icml.cc/virtual/2026/poster/62339): peer-review confirmation, not a new
competitive signal. Full detail: `docs/research/digests/2026-08-17.md`.
An ACM 2026 paper on edge-deployed VLMs for medical-imaging report generation
(dl.acm.org/doi/10.1145/3796263.3796284) is the closest-yet paper to Route A's
exact use case, but its mechanism is model compression (distillation/pruning/
quantization), not constrained-decoding cost — differentiate-from, worth a
related-work sentence if Route A proceeds. A general (non-edge) constrained-
decoding speedup via token-space compression (arXiv 2605.29986) is a citable
alternative-mitigation reference, not competitive with the shadow-mask
pipelining method.

Iter 9 added two papers to the **correctness-tax cluster** (cite, both
strengthen the already-open gap rather than compete with it): "The Hidden
Cost of Structured Generation in LLMs: Draft-Conditioned Constrained
Decoding" (arXiv 2603.03305) shows grammar masking can push generation toward
locally-valid-but-semantically-wrong output and proposes a draft-then-
constrain fix (+24pp strict accuracy on GSM8K); "Schema Key Wording as an
Instruction Channel..." (arXiv 2604.14862) shows schema-key wording alone
shifts accuracy under constrained decoding, model-dependent. Neither touches
edge hardware or latency/energy — accuracy-only, cloud-scale studies — so
neither competes with G1/Route A's hardware-cost claim, but both are now
independent literature evidence that the correctness-tax gate (below) is a
real risk, not a hypothetical one. Also swept and differentiated-from: three
energy/carbon-on-edge papers (DynaSplit 2410.23881 — split-computing, not
decode-time; CO2-Meter 2511.08575 — phase-aware carbon *estimator*, citable
methodology only; CarbonCall 2504.20348 — tool-selection + carbon-intensity
scheduling, different layer) and two structure/workflow papers scored
low-relevance (SGDe/2604.17450 — DAG-level workflow compilation, not
token-level grammar; CL4Health 2026 CRF-filling / LLM StructCore 2604.20560 —
accuracy-focused clinical NLP, no systems angle). Full detail:
`docs/research/digests/2026-08-15.md` Part 2 and
`docs/research/digests/2026-08-16.md`.

## Known competitive pressure (updated iter 11)
MLPerf Edge Agentic owns generic tool-calling benchmarks; ~3.5 BPW threshold
(2505.15030) is prior art; generic energy-aware edge scheduling is crowded
(EnerInfer, HeteroMosaic, Agent.xpu, KAIROS, BiScale, and now also DynaSplit,
CarbonCall — split-computing and tool-selection layers, still non-overlapping
with per-token mask cost). Algorithmic mask-speedup work (PSC, GRID,
Attention-Meets-Reachability, and now Token-Space-Compression 2605.29986) is
a growing but still non-overlapping erosion-risk cluster — none on edge
hardware, none in Joules, none with CPU/GPU pipelining measured against a
short forward pass, and none on VLMs either. A **correctness-tax literature
cluster** (2605.26128, 2603.03305, 2604.14862) is forming independently of
this loop — not a threat to G1/Route A's hardware-cost claim, but raises the
bar for the correctness-grading gate below: any final paper claiming
"constrained decoding is cheap" now has to reckon with three separate 2026
papers showing it isn't free on the accuracy axis. New this iteration: a
**general-purpose inference-engine speedup cluster** (BaseRT 2607.00501 /
2607.19438) is improving raw decode throughput on the exact unified-memory
hardware class and model families this work uses — still non-overlapping
(no grammar/constrained-decoding angle) but a baseline-drift risk to watch,
not act on.

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

## Next action (iteration 12)
1. **Check `NEEDS_INPUT.md` first.** Still open as of iter 11 — no chat
   decision or `DECISIONS.md` entry answering the route question landed
   between iter 10 and iter 11 (gate age ~72h at iter 11). **The one
   constitution-mandated reminder ping was already sent at iter 10** — do
   NOT send another one for this same gate; only re-ping if a genuinely new
   blocking event opens later (the current long silence is not itself a new
   event). If Maria has answered by iter 12: close the gate per her choice,
   update the "Current stage" line to reflect the confirmed route, and —
   only if she also said the literature-only restriction is lifted — resume
   normal pacing (queue the route's first hardware step, e.g. Route A's
   mtmd smoke test). If not yet answered: stay in reduced-pace
   literature-only mode. Pace-governor note: iteration 11 was **non-empty**
   (six new items in `seen.json`); empty-iteration counter is back to 0 of 3.
2. **Literature tracking (repeat, ~5 min):** re-check MLSys 2027 CFP status
   (still blocked/unconfirmed as of iter 11 across `mlsys.org`,
   `mlsys-deadlines.github.io`, `mlciv.com` — try again in case the egress
   block or the venue's own CFP page changes). For the general sweep: either
   continue the Apple-Silicon-inference-engine angle opened this iteration
   (check if BaseRT, 2607.00501/2607.19438, or any successor starts touching
   grammar/schema-constrained decoding specifically — currently it doesn't),
   or pick a fresh angle (e.g. Jetson Orin VLM inference benchmarks, given
   Jetson is still the planned replication leg per decision #9 even though
   postponed) — avoid repeating the same specific query two iterations
   running, as iter 10/11 both found that unproductive.
3. Continue the check-the-working-tree-before-trusting-STATUS discipline —
   check `docs/research/DECISIONS.md` and `docs/research/from-science/` for
   new files every iteration, not just `experiments/results/`.
