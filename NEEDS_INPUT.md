# NEEDS INPUT — route confirmation (opened iteration 8, 2026-08-15)

## What happened since the loop last updated STATUS.md

STATUS.md was frozen at iteration 7 (2026-08-14), sitting behind a
`NEEDS_INPUT.md` gate about how to spend the runway after experiment 005. A
lot happened outside this loop's own tracking since then — all of it is real,
committed, and readable in the repo, but STATUS.md never caught up:

1. You answered the iteration-6 gate (`docs/research/DECISIONS.md`,
   2026-08-14 "answering the iteration-6 NEEDS_INPUT gate"): 005 confirmed
   G1's crossover; the paper must be characterization + mechanism +
   **optimization method**, not measurement alone; mechanism needs the
   vocab-controlled experiment; a correctness-grading gate is required before
   any "constrained decoding is harmless" claim.
2. You then asked for a full **strategic direction review** before building
   G1's optimization machinery on the assumption it was automatically the
   right paper (DECISIONS.md, "STRATEGIC DIRECTION REVIEW, pause G1 build") —
   survey the venues fresh, score G1 against alternatives, keep the repo
   either way.
3. You separately flagged that **vision models** should be in scope, not just
   text (DECISIONS.md, "extend scope to VLMs") — medical/surgical robotics is
   vision-centric, and VLMs have a different compute profile (heavy encoder
   prefill + short decode) that changes the mechanism G1 is testing.
4. A large review landed in `docs/research/from-science/` (landscape sweeps,
   a hypothesis slate, mechanism/optimization deep-dives) and synthesized into
   `2026-08-14-FINAL-strategy-routes.md`, recommending **Route A**: the VLM/
   medical version of G1 — an edge vision-language model that fills a
   structured clinical report (endoscopy/surgical-camera use case) under
   grammar constraints, on-device for patient privacy. The report's own
   closing line is explicit: *"Decyzja należy do Marii — worker czeka w trybie
   literature-only"* (the decision is Maria's — the worker waits in
   literature-only mode).
5. You then made two hardware decisions (DECISIONS.md) that read as *consistent
   with* Route A but don't say so in as many words: reframe the science claim
   around the unified-memory edge-SoC class rather than "Apple" specifically,
   with M1 as instrument + clinician's-laptop use case and Jetson as the
   deployment leg (approved as load-bearing) — then, most recently
   (`55050ff`, 2026-08-14), decided to **stay on M1 for now and postpone the
   Jetson purchase** until after first results.

## The actual question

**Is Route A (edge VLM, medical structured-report generation, constrained-
decoding tax + shadow-mask pipelining) the confirmed direction, or is that
still open?**

The hardware decisions (clinician's-laptop framing, Jetson-as-deployment)
line up with Route A's medical-VLM framing, but they're generic enough
("dotyczy każdej trasy" — applies to every route) that they could just as
easily sit under Route C (zero-egress medical energy premium) or even D
(classic G1, medical JSON). I don't want to silently treat "sounds
consistent with A" as a real confirmation and then build a paper skeleton
around a route you didn't actually pick — that's exactly the kind of
decision the loop's constitution reserves for you.

**Options, from `2026-08-14-FINAL-strategy-routes.md`:**
- **A (recommended by the from-science review):** edge VLM medical
  structured report, constrained-decoding tax + mechanism + shadow-mask
  pipelining, "Seeing is Free, Speaking is Not" (2607.09520) as the
  generic-profile prior work to replicate/extend on unified memory.
  First step if chosen: an `mtmd` smoke test (image + grammar) on the M1 —
  feasibility already checked this iteration, see digest: `llama-mtmd-cli`
  already supports `--grammar-file` / `-j/--json-schema` alongside
  `--mmproj`, so no llama.cpp fork is needed just to run the smoke test.
- **B:** robot action-grammar under a control-loop deadline (ICRA/CoRL
  framing). Caveat: ICRA 2027's ~Sep 15, 2026 deadline is now about 4 weeks
  out — the from-science report already scored this as too tight for a
  primary target; still viable as a paper #2 in the spring robotics season.
- **C:** zero-egress privacy energy premium (on-device vs. sanitize-then-
  cloud). Cleanest privacy story, most schedule slack (SenSys/RA-L).
- **D (fallback):** classic G1, text-only, medical JSON. Already has 005's
  clean pilot result and is the most "shovel-ready," but it's the framing you
  yourself flagged as weakest (arbitrary schema, narrow).
- **Something else / more review needed.**

**A second, smaller question bundled in:** if you confirm a route, should the
loop resume queuing Mac-hardware experiments and drafting (per BRIEF's normal
stages), or does the "literature-only, no experiments, no drafting" hold from
the direction-review decision stay in force until you say otherwise? I'm
treating it as still in force unless you say so here, so the loop doesn't
jump ahead of you on a decision that's supposed to be yours.

## What the loop is doing while this is open

Reduced-pace literature-only tracking per the pace governor: re-checking
MLSys 2027's CFP (still unconfirmed, still network-blocked from this
session), and this iteration additionally did feasibility/landscape
groundwork for Route A specifically (mtmd+grammar support check, a scoop
check for adjacent edge-medical-VLM papers) since that prep is safe to do
regardless of which way you land — see `docs/research/digests/2026-08-15.md`.
No new hardware runs queued, no paper skeleton drafted, no route assumed.
