# Paper machine — STATUS (updated 2026-08-21, iteration 14)

**Scope decisions so far (newest first):**
1. 2026-08-21 (iter 14): no scope-narrowing decision — `NEEDS_INPUT.md` (route
   confirmation, opened iter 8) is still open, gate age now ~6 days. Local
   `main` was 4 commits behind `origin/main` at the start of this iteration
   (a leftover detached-HEAD checkout, not new remote work); fast-forwarded
   via `git pull` — the 4 commits were all iteration 13's own already-known
   content (STATUS.md, digests 18–20, Maria's collab-protocol files), nothing
   new since iter 13. No `DECISIONS.md` entry or chat/docs decision has
   answered the route question. **No re-ping** — same reasoning as iter
   11–13 (one constitution-mandated ping already fired at iter 10, no new
   blocking event this iteration). Did the two literature tasks iter 13
   queued: (1) **verified the Ollama/XGrammar claim directly** — fetched
   `ollama/ollama#7900` (the actual structured-outputs PR) and found the
   team explicitly declined to adopt XGrammar, keeping grammar generation
   and sampling coupled in their own implementation; **resolves iter 13's
   flagged baseline-drift risk as a non-issue**, no action needed. (2)
   **MLSys 2027 CFP still unconfirmed** — `mlsys.org` etc. blocked by egress
   again this iteration, but WebSearch surfaced MLSys 2026's actual dates
   (May 17–22, 2026, Bellevue WA, deadline Oct 30, 2025), which firms up
   (doesn't change) the ~Oct 30, 2026 extrapolation for MLSys 2027 already
   carried since iter 5. General sweep: two new items, neither competitive
   — ATLAS-RTC (arXiv 2603.27905, closed-loop token-level runtime control
   for structured generation, correctness/reliability angle with its own
   latency claims, differentiate-from) and Weave of Formal Thought (arXiv
   2606.25987, sound/complete constrained decoder for *code* generation,
   different domain, ignore). Format Tax citing-work check inconclusive
   (one citing paper found, arXiv 2604.13006, but `arxiv.org` was blocked
   this iteration so its actual argument couldn't be confirmed — carried
   forward, not resolved). **Non-empty iteration** — empty-iteration counter
   stays at 0 of 3. Full detail: `docs/research/digests/2026-08-21.md`.
2. 2026-08-20 (iter 13): no scope-narrowing decision — `NEEDS_INPUT.md` (route
   confirmation, opened iter 8) is still open, gate age now ~5 days (~120h).
   **Found something important while doing the standing repo-state check**:
   local `main` was 3 commits stale (fetch showed `origin/main` already had
   iter 11, iter 12, and — new — a commit authored by Maria directly on
   2026-08-19, `docs/WSPOLPRACA-BADAWCZA.md` + `docs/MAPA-WIEDZY.md` +
   `docs/lekcje/LEKCJA-01-maska-i-3-procent.md`, a human-in-the-loop
   collaboration protocol for the whole repo, not just the science track).
   Nothing was unpushed/at risk — fast-forwarded the local ref, no content
   lost. The protocol's hard rule ("night/cron sessions execute only
   hypotheses Maria selects in a Monday block") is already this loop's own
   practice since iteration 8, so **no operational change for this loop** —
   but it explains the gate's 5-day silence: Maria's own words are that a
   digest-level summary isn't enough for her to decide from. Wrote a
   synthesis into `NEEDS_INPUT.md` connecting LEKCJA-01 §6's two bundled
   questions (what to do with G1's "mask hides in the shadow" result: retry
   on weaker hardware / close G1 and chase residual errors / VLM bridge) to
   this loop's existing Route A–D framing — option 3 there **is** Route A,
   option 2 **is** Route D, option 1 doesn't map cleanly and is worth naming
   as a fifth option ("D′") so it isn't dropped when Maria and the science
   track work through it. **No re-ping** — the one constitution-mandated
   ping already fired at iter 10, and Maria is visibly not silent (she's
   actively working the problem, just via a different artifact). Literature
   sweep (~10 min): MLSys 2027 still unconfirmed (no change since iter 5);
   one new citable item — "The Format Tax" (arXiv 2604.03616) — strengthens
   the correctness-tax cluster with a mechanism claim (the tax is mostly
   prompt-level, not decoder-level) and a candidate mitigation orthogonal to
   G1/Route A's own optimization target; NVIDIA's TensorRT Edge-LLM SDK
   (JetPack 7.1) flagged as a differentiate-from / baseline-drift-risk for
   the still-postponed Jetson leg, not an action item now. One unverified
   search-snippet claim ("Ollama uses XGrammar under the hood") explicitly
   **not** asserted as fact — could not be corroborated on a follow-up
   search; flagged for direct verification next time, since if true it would
   be a real baseline-drift risk for G1's own measurements. **Non-empty
   iteration** — empty-iteration counter stays at 0 of 3. Full detail:
   `docs/research/digests/2026-08-20.md`.
3. 2026-08-19 (iter 12): no scope-narrowing decision — `NEEDS_INPUT.md` (route
   confirmation, opened iter 8) is still open, gate age now ~96h (~4 days).
   Checked `docs/research/DECISIONS.md`, `docs/research/from-science/`, and
   `experiments/results/` for anything new since iter 11's commit — nothing
   found. The constitution's one-reminder-ping already went out at iter 10;
   **not re-pinged this iteration** (no new blocking event opened — a longer
   silence on an already-reported gate is not itself a new event). Stayed in
   reduced-pace literature-only mode: no new Mac experiments, no Route-A
   build, no drafting. MLSys 2027 re-check: still unconfirmed — same
   TBD/predicted-date state as iterations 5–11, no official CFP page found.
   Re-anchored the general sweep on the "algorithmic mask/grammar-speedup
   cluster" named in the Known-competitive-pressure section (avoiding a
   third iteration on the Apple-Silicon-engine angle, per iter 11's own
   instruction not to repeat a query two iterations running) and found three
   new items, none competitive with G1/Route A: Pre³ (arXiv 2506.03887, ACL
   2025) and XGrammar-2 (arXiv 2601.04426) both attack constrained-decoding
   mask overhead but in large-batch/multi-request cloud-serving contexts
   (SGLang/vLLM) — differentiate-from, both now explicit cluster members
   alongside PSC/GRID/Attention-Meets-Reachability/Token-Space-Compression;
   "The Speedup Paradox" (arXiv 2606.28529, embodied-task lossy-optimization
   framework TISED) is a different optimization class (not grammar masking)
   but independently shows optimization effects are hardware-configuration-
   dependent — cite as framing support, useful if Route B is ever revived.
   **Non-empty iteration** — empty-iteration counter stays at 0 of 3. Full
   detail: `docs/research/digests/2026-08-19.md`.
4. 2026-08-18 (iter 11): no scope-narrowing decision — `NEEDS_INPUT.md` (route
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
5. 2026-08-17 (iter 10): no scope-narrowing decision — `NEEDS_INPUT.md` (route
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
6. 2026-08-16 (iter 9): no scope-narrowing decision — `NEEDS_INPUT.md` (route
   confirmation, opened iter 8) is still open and unanswered; no chat
   decision or `DECISIONS.md` entry landed since commit `cd62ba8`. Stayed in
   reduced-pace literature-only mode per the gate's own instructions: no new
   Mac experiments, no Route-A build, no drafting. Did a light follow-up
   literature sweep (MLSys 2027 recheck — still unconfirmed/blocked, no
   change; edge-VLM-energy and medical-structured-generation sweep — 7 new
   items, none competitive with G1/Route A, two reinforce the open
   correctness-tax gap). Full detail: `docs/research/digests/2026-08-16.md`.
7. 2026-08-15 (iter 8): reconciliation iteration, not a new scope-narrowing
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
8. 2026-08-14 (via chat, outside loop tracking — answered the iter-6 gate):
   experiment 005 CONFIRMED G1's crossover (all four sub-1B models, positive
   paired latency tax, 28× spread). Maria's decision on how to spend the
   runway: (a) NOT a characterization-only paper — must be characterization +
   MECHANISM + an OPTIMIZATION METHOD that reduces the tax, that's the
   headline contribution; (b) mechanism must be nailed via a vocab-controlled
   experiment before drafting; (c) a correctness-tax gate (task-correctness
   grading, not just schema validity) must exist before any "constrained
   decoding is harmless" claim. See `docs/research/DECISIONS.md`.
9. 2026-08-14 (via chat): controlled-experimental-design correction — 005's
   four models differ on vendor/size/vocab/architecture simultaneously, so the
   28× spread is unattributable as designed. Future driver experiments MUST
   vary one factor at a time (same-family size sweep, vocab isolation at
   matched decode speed, cross-vendor replication at matched size+vocab).
10. 2026-08-14 (via chat — STRATEGIC DIRECTION REVIEW, pause G1 build): before
   committing to G1's optimization machinery, Maria asked for a from-scratch
   direction review — survey previous rounds of the target venues, return to
   her actual stated interests (medical/surgical robotics, privacy, edge
   energy), produce a scored slate of candidate hypotheses (G1 included),
   benchmark G1 against the alternatives, **keep the repo/harness regardless
   of outcome**. Until she picks from the slate: **literature-only tracking —
   no new Mac experiments, no G1-optimization-fork build, no drafting.** This
   restriction is still in force (see gate below).
11. 2026-08-14 (via chat — extend scope to VLMs): consider vision-language
   models, not just text LLMs. Her real area (medical/surgical robotics) is
   vision-centric (endoscopy, surgical-scene understanding, VLA); VLMs also
   have a different compute profile (heavy vision-encoder prefill + short LM
   decode) that changes the constrained-decoding-tax mechanism and opens a
   less-explored angle. Harness extends via llama.cpp/Ollama (llava,
   moondream, qwen2-VL, etc. — mtmd tool).
12. 2026-08-14 (via chat — hardware framing, Maria's reviewer-grade
   challenge): "Apple isn't a natural deployment environment" accepted as a
   valid reviewer risk. Reframed every route's scientific claim around the
   **unified-memory edge-SoC class**, not "Apple" specifically: M1 = dev
   instrument + clinician's-laptop use case (on-device privacy), Jetson =
   deployment environment. Approved as **load-bearing**: buy a Jetson Orin
   Nano for a cross-device replication leg (closes the "n=1 / unnatural
   environment" reviewer objection).
13. 2026-08-14 (via chat — hardware, stay on M1 for now): Jetson purchase
   (Nano ~1.5k / NX 16GB ~4k PLN) **postponed** — return to it after first
   results on M1. The unified-memory-SoC framing and planned replication leg
   stand unchanged; just not buying yet.
14. 2026-08-11 (iter 3, closed by iter 5): Stage corrected 3 → 4 — a Stage-4
   pilot had already run outside this loop's tracking.
15. 2026-08-10 (iter 2): axes **federated SLM** and **edge VLA for robots**
    DEMOTED — both scooped by mid-2026 work at the time. (Note: "edge VLM" —
    vision *models*, not the VLA/robot-action framing — was independently
    revived by decision #6 above under the medical/vision-centric lens; not a
    contradiction, different axis.)
16. 2026-08-10: direction = **edge × robotics × privacy and/or medical**;
    must be low-level/systems-strong; should ideally (not necessarily) feed
    bigos later.
17. 2026-08-10: "multi-turn agentic eval" framing DEMOTED (not low-level
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

## Nearest deadlines (re-checked 2026-08-21, iter 14)
| Venue | Deadline | Note |
|---|---|---|
| ~~ICRA 2027~~ | ~Sep 15, 2026 | ~3.3 weeks out — too tight for a primary target regardless of route; from-science review already scored Route B (robot-grammar) as losing to Route A partly for this reason. Stays open as a paper #2 possibility in the spring robotics season. |
| **MLSys 2027** | **~Oct 30, 2026 — still UNCONFIRMED** | `mlsys.org`, `mlsys-deadlines.github.io`, and `mlciv.com` all blocked by network egress again this iteration (WebFetch); no official 2027 CFP page found. WebSearch (unblocked) did surface MLSys 2026's own confirmed dates — May 17–22, 2026, Bellevue WA, deadline Oct 30, 2025 — which is the concrete data point the "~Oct 30, 2026" extrapolation for 2027 has been resting on since iter 5; this iteration is the first to see it stated directly rather than inferred. Firms up the estimate, doesn't change it. Re-check every iteration. |
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

## Scoop-check status (iter 14: two new items, none a scoop; iter 13: one new item, not a scoop; iter 12: three new items, none a scoop; iter 11: six new items, none a scoop; iter 10: zero new items; iter 9: seven new items, none a scoop)
Iter 14 found two new items, neither competitive with G1/Route A's hardware-
cost claim. **ATLAS-RTC** (arXiv 2603.27905) is a closed-loop token-level
runtime-control system for structured generation — monitors decoding for
drift and intervenes with biasing/masking/rollback, claiming 20–37.8pp
first-attempt-success gains and up to 88% latency reduction "in
failure-dominated settings." Differentiate-from: it's a correctness/
reliability mechanism (fewer retries), not a hardware-cost-of-masking
mechanism — no edge, Joules, or single-stream angle in the abstract, but
worth carrying as a correctness-tax-cluster-adjacent citation (same spirit
as DCCD's draft-then-constrain fix). **Weave of Formal Thought** (arXiv
2606.25987) is a sound-and-complete constrained decoder for *code*
generation against the full Tree-sitter grammar — different domain, no
latency/energy/edge content — ignore. Iter 14 also resolved iter 13's
flagged Ollama/XGrammar risk (see the top scope-decision entry above):
verified directly against `ollama/ollama#7900` that Ollama's team explicitly
declined to adopt XGrammar, so G1's baseline (llama.cpp/Ollama's own GBNF
engine) stands as previously assumed — not a scoop or a baseline-drift risk.
Full detail: `docs/research/digests/2026-08-21.md`.

Iter 13 found one new citable item, **"The Format Tax"** (arXiv 2604.03616,
Apr 2026) — joins the correctness-tax cluster (2605.26128, 2603.03305,
2604.14862), not competitive with G1/Route A's hardware-cost claim. It tests
six open-weight + four API models across four structured formats and finds
the accuracy loss is dominated by the *prompt-level* instruction to format
(suppressing step-by-step reasoning) rather than the *decoder-level* sampling
constraint — a mechanism claim, not just a fourth confirmation that the tax
exists. Also checked NVIDIA's TensorRT Edge-LLM SDK (JetPack 7.1, native C++
LLM+VLM runtime for Jetson) — differentiate-from / baseline-drift-risk for
the still-postponed Jetson leg, not competitive (no grammar/constrained-
decoding angle). Full detail: `docs/research/digests/2026-08-20.md`.

Iter 12 re-anchored the sweep on the "algorithmic mask/grammar-speedup
cluster" already named in Known-competitive-pressure below (avoiding a third
iteration on the Apple-Silicon-engine angle, per iter 11's own instruction).
Three new items, none competitive with G1/Route A's hardware-cost claim.
**Pre³** (arXiv 2506.03887, ACL 2025) precomputes prefix-conditioned edges to
turn LR(1) grammar parsing into a deterministic pushdown automaton, removing
runtime path-exploration overhead from constrained decoding — but explicitly
targets large-batch cloud serving ("especially inefficient under large
inference batches"), not edge/single-stream. **XGrammar-2** (arXiv
2601.04426, AI & Agentic Systems 2026) adds tag-triggered structure switching
and cross-request grammar caching for agentic tool-calling, claims >6× faster
compilation and near-zero overhead — but benchmarked inside SGLang/vLLM
multi-request serving, not on-device. Both **differentiate-from**: they now
join PSC/GRID/Attention-Meets-Reachability/Token-Space-Compression as named
members of the mask-speedup cluster (see below) — the pattern holds that
every entrant in this cluster optimizes batched cloud serving, none touch
edge hardware, Joules, or a short single-stream forward pass. **The Speedup
Paradox** (arXiv 2606.28529, embodied-task lossy-optimization framework
TISED) is a different optimization class (quantization/pruning-style, not
grammar masking) but independently shows task-level optimization effects are
non-monotonic and hardware-configuration-dependent — **cite** as framing
support (same shape of claim G1 is building toward), useful ammunition if
Route B is ever revived. Full detail: `docs/research/digests/2026-08-19.md`.

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

## Known competitive pressure (updated iter 14)
MLPerf Edge Agentic owns generic tool-calling benchmarks; ~3.5 BPW threshold
(2505.15030) is prior art; generic energy-aware edge scheduling is crowded
(EnerInfer, HeteroMosaic, Agent.xpu, KAIROS, BiScale, and now also DynaSplit,
CarbonCall — split-computing and tool-selection layers, still non-overlapping
with per-token mask cost). Algorithmic mask-speedup work (PSC, GRID,
Attention-Meets-Reachability, Token-Space-Compression 2605.29986, and now
also Pre³ 2506.03887 and XGrammar-2 2601.04426) is a growing but still
non-overlapping erosion-risk cluster — none on edge hardware, none in Joules,
none with CPU/GPU pipelining measured against a short forward pass, none on
VLMs either, and the two newest entrants are explicitly batched/multi-request
cloud-serving optimizations (large-batch and SGLang/vLLM respectively) —
the cluster's defining trait (batch-serving-side, not edge single-stream) is
now four-for-four across its most recent members. A **correctness-tax
literature cluster** (2605.26128, 2603.03305, 2604.14862) is forming
independently of this loop — not a threat to G1/Route A's hardware-cost
claim, but raises the bar for the correctness-grading gate below: any final
paper claiming "constrained decoding is cheap" now has to reckon with three
separate 2026 papers showing it isn't free on the accuracy axis. A
**general-purpose inference-engine speedup cluster** (BaseRT 2607.00501 /
2607.19438, iter 11) is improving raw decode throughput on the exact
unified-memory hardware class and model families this work uses — still
non-overlapping (no grammar/constrained-decoding angle) but a baseline-drift
risk to watch, not act on. New this iteration: an independent **embodied
hardware-dependent-optimization framing** (TISED / "The Speedup Paradox"
2606.28529) is not competitive pressure at all — it's citable support for
the general shape of G1's claim (device class changes an optimization's net
effect), from a different optimization class (lossy quantization/pruning on
robot tasks, not grammar masking). New this iteration: **ATLAS-RTC**
(2603.27905) joins the correctness-tax cluster's orbit as a *fix*-side
paper (closed-loop biasing/masking/rollback to cut structured-generation
failures) rather than a *problem*-side one — still not competitive with
G1/Route A's hardware-cost claim (no edge/Joules/single-stream angle), but
another sign that "make constrained decoding not silently wrong" is an
active, multi-approach research line the eventual paper's framing should
acknowledge.

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

## Next action (iteration 15)
1. **Check `NEEDS_INPUT.md` first, including its iter-13 "Update" section**
   (still the latest — iter 14 added no new update to the file itself, since
   nothing new arrived to fold in). Still open as of iter 14 (gate age ~6
   days). No `DECISIONS.md` entry or chat decision answering the route
   question yet, but Maria is actively engaged via the collab protocol
   (`docs/WSPOLPRACA-BADAWCZA.md` et al., committed 2026-08-19) — **do not
   treat continued gate silence as grounds to re-ping**; the one
   constitution-mandated ping already fired at iter 10, and this isn't the
   "blocked and silent" case that rule exists for. If Maria has answered by
   iter 15 (via chat, `DECISIONS.md`, or a Monday-block artifact under
   `docs/`): close the gate per her choice, update "Current stage," and —
   only if she also said the literature-only restriction is lifted — resume
   normal pacing. If not yet answered: stay in reduced-pace literature-only
   mode. Pace-governor note: iteration 14 was **non-empty** (two new items in
   `seen.json` plus a resolved risk-flag); empty-iteration counter stays at
   0 of 3.
2. **Format Tax citing-work check, unfinished.** Iteration 14 found one
   citing paper (arXiv 2604.13006, "One Token Away from Collapse") but
   `arxiv.org` was blocked by egress this iteration, so it's unconfirmed
   whether that paper actually connects Format Tax's prompt-level-tax finding
   back to decoder-level/grammar-masking cost. Try `arxiv.org` directly next
   time (it cycles between blocked/unblocked iteration to iteration); if
   still blocked, try Semantic Scholar or a direct WebSearch quote-check
   instead of another arXiv fetch attempt.
3. **Literature tracking (~10 min):** re-check MLSys 2027 CFP status directly
   at `mlsys.org` (blocked again this iteration; try again — the domain block
   has been inconsistent run to run) in case the 2027 CFP page has gone live.
   For the general sweep: pick a fresh angle not yet covered two iterations
   running — iter 14 covered the mask/grammar-speedup and correctness-tax
   clusters again, so iter 15 should widen back out (e.g. re-check the
   edge-VLM/medical-structured-generation angle last swept at iter 11, or the
   Apple-Silicon-engine baseline-drift watch last touched at iter 11/12) per
   the standing instruction not to repeat the same narrow query twice running.
4. Continue the check-the-working-tree-before-trusting-STATUS discipline —
   check `docs/research/DECISIONS.md`, `docs/research/from-science/`, and
   `docs/` more broadly (real decisions can land as direct commits outside
   `docs/research/` too, even though this loop only ever writes inside
   `docs/research/`) for anything new every iteration, not just
   `experiments/results/`. Also re-verify local `main` is not behind
   `origin/main` before trusting any local file state (iteration 14 found it
   4 commits stale from a leftover detached-HEAD checkout — harmless this
   time, but worth checking early every iteration rather than assuming).
