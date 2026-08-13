# NEEDS INPUT — what to do with a confirmed-but-messy G1 result (2026-08-13, iteration 6)

**This is a [NEEDS-INPUT] event under BRIEF's constitution.** Stage 4's exit
criterion just fired ("a 1-week pilot on Maria's Mac supports or kills the
hypothesis" — it supports), which per BRIEF is exactly the point where "Maria
picks." Full analysis: `docs/research/digests/2026-08-13.md`. Raw report:
`experiments/results/NIGHT-REPORT-2026-08-13.md`.

## What happened, in short

Overnight (2026-08-12/13, outside this loop's tracking — already committed by the
time this iteration ran), your Mac ran experiment 005: four sub-1B models at
11–25 ms/token, 480/480 runs. **For the first time in five pilot rounds, all four
models show a positive, statistically clean (CI excludes zero) latency tax under
grammar-constrained decoding** — the crossover G1 predicted. That's the good news,
and it's real: qwen2.5:0.5b +1.8%, qwen3:0.6b +2.4%, llama3.2:1b +6.2%, gemma3:1b
+30%.

**The complication:** the tax is not one number. It spans 28× in absolute terms
across four models decoding at nearly the same speed, and it isn't even monotone
in decode speed — the slowest of the four pays the smallest tax, the second-fastest
pays the largest by an order of magnitude. Model identity, not decode speed, seems
to drive it. The leading candidate explanation (vocabulary size — mask cost is
`O(vocab)`, gemma3 has an unusually large one) is plausible from the general
literature but **untested on this data** — nobody extracted `n_vocab` from the
server logs.

**Also new this iteration:** literature reading (arXiv 2605.26128, "The Constraint
Tax") found that on a near-identical task shape — small models under a hard
tool-call JSON schema — schema-valid outputs can hide a collapse in actual
answer correctness (one example: 91.5%→48.0% correctness while validity stays
100%→100%). 005/006 currently grade only schema validity. Nothing in the pilot
harness currently checks whether the 60/60-valid constrained runs are also
*correct* medical envelopes, as opposed to schema-valid noise. This is a gap in
the measurement, not a result — it could go either way once measured.

## The decision

1. **Commit to drafting now.** Write the one-page paper skeleton around the honest
   claim the data supports: *"the constrained-decoding latency tax is real on
   sub-25ms/token edge models, can be large (up to 30%), and its magnitude is
   governed by an unidentified model-specific property — not decode speed alone."*
   Queue the vocab-size and correctness-tax follow-ups in parallel as strengthening
   experiments, not gates to drafting. Fastest path to MLSys (~11 weeks out), but
   commits to a paper whose central mechanism claim ("what actually drives the
   28× spread") is still an open question at submission time unless the parallel
   experiments land first.
2. **One more validation round before committing.** Queue two more Mac experiments
   first: (a) a `n_vocab`-controlled test (isolate vocabulary size as the driver —
   need at least one large-vocab small model and one small-vocab small model
   matched on other properties), (b) add semantic/task-correctness grading (not
   just schema validity) to 005-style runs, re-score the existing runs where
   possible before spending new hardware time. Only after both land does the
   paper skeleton get written. Slower, but the central claim would be load-bearing
   at draft time instead of retrofitted.
3. **Reframe down to what's already solid.** Skip chasing the model-specific driver
   entirely; write the smaller, fully-supported claim only — "grammar-constrained
   decoding's latency cost becomes significant once decode speed drops below
   ~25ms/token on edge hardware, magnitude model-dependent" — and leave the 28×
   spread as an explicitly open question in the paper's limitations, not something
   promised as explained. Lowest effort, weakest single contribution, still
   truthful.
4. **Something else** — e.g. treat the correctness-tax gap as disqualifying until
   fixed (no paper claim about constrained decoding being "good" here without a
   correctness axis), independent of the vocab-size question.

## Why a worker can't pick this

Every option trades scarce resources only you can allocate: Mac-hardware nights
against an ~11-week runway to MLSys, versus writing time on a claim that may need
walking back if the driver experiment lands a surprise. This is also exactly the
significance call `DECISIONS.md` (2026-08-10, point 2) reserved for you — the
pilot no longer returns "below meaningful," it returns "meaningful but not yet
explained," which the constitution didn't anticipate as a distinct case and this
loop is not positioned to resolve by fiat. Fixing the correctness-tax gap also
means adding grading code to the harness — outside `docs/research/`, outside what
this worker touches.

## What happens on no answer

Per BRIEF's pace governor: this loop will not draft a paper skeleton, queue new
Mac-hardware experiments, or advance past Stage 4 without an answer here. Next
iterations do literature-only tracking (MLSys 2027 CFP re-check, scoop-query
top-up) at reduced pace until you respond.
