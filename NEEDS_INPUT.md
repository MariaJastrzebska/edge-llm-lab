# ANSWERED 2026-08-11 — G1 go/no-go after the overnight pilot

> **Maria's decision: option 2 — fix the enum-case bug first, then re-run
> 002/003 with the corrected prompt asset, before spending a night on the
> short-decode regime.**
>
> Status of that decision as of this commit:
> - Enum bug **fixed** in both `constant_data_en.txt` and the Polish
>   `constant_data.txt` (lines 70 and 73 in each; the PL copy had the identical
>   bug and was not previously logged). CLAUDE.md known-bugs updated.
> - Two adjacent enum-case suspects found and **deliberately left alone** —
>   the `*_expected.json` ground-truth fixtures and `validation.txt:19`. See
>   CLAUDE.md "Still-open enum-case suspects". Resolve before the next
>   *scored* eval; they do not block a decode-time/energy re-run.
> - Re-run of 002/003 queued for Maria's Mac.
>
> **The gate is closed. The pace governor no longer applies** — the loop may
> queue Mac-hardware time again. The short-decode regime (option 1) is
> explicitly *deferred*, not rejected: revisit once 002/003 come back clean.

---

<details><summary>Original gate text (2026-08-11, iteration 3)</summary>

# NEEDS INPUT — G1 go/no-go after the overnight pilot (2026-08-11, iteration 3)

**This is a [NEEDS-INPUT] event under BRIEF's constitution: the overnight
pilot's effect is at or below statistical resolution in the regime tested
(a kill-adjacent signal), but the regime G1 actually predicts a tax in was
never reached — so this is a real decision, not a research question a
worker can resolve alone.** Full analysis: `docs/research/digests/2026-08-11.md`
Part 1. Raw report: `experiments/results/NIGHT-REPORT.md`.

## What happened, in short

Your Mac ran two pilots overnight (300 + 360 runs, clean telemetry, real
Joules) testing G1 — the claim that grammar-constrained decoding has a real
energy/latency cost on edge because the CPU mask-walk can't hide behind a
short GPU forward pass. **Result: no model showed a positive tax. One
(qwen3) was significantly *faster* under constraint — the wrong direction
for G1.** But nothing tested decoded faster than 36.8 ms/token, and G1's
mechanism specifically needs a *short* forward pass to fail to hide the
mask cost — that regime was never reached. So this isn't "G1 is dead," it's
"G1 is untested where it matters, and the one place it was tested, it
didn't show up."

Also found and logged (not fixed): a real pre-existing bug in
`constant_data_en.txt` (uppercase `"INCOMPLETE"` vs. the schema's lowercase
enum) that confounds the validity/energy-per-valid numbers in the pilot —
now in CLAUDE.md's known-bugs list.

## The decision

1. **Keep going — target the short-decode regime.** Design a follow-up
   pilot with a genuinely small model (sub-1B, e.g. Qwen2.5-0.5B) and/or
   faster hardware, so a config actually decodes in the low single-digit
   ms/token range where a mask-walk could plausibly matter. This is the
   real falsification test BRIEF's Stage 4 asks for. Costs: another
   Mac-hardware night (or Pi 5/Jetson if you want that hardware in the
   loop — tell us whether to treat that purchase as load-bearing now).
2. **Keep going, but only after fixing the enum-case bug** — re-run 002/003
   with a corrected prompt asset so the validity/energy-per-valid metric
   stops being confounded, before spending a night on the short-decode
   regime. Slower, cleaner.
3. **Kill G1 as scoped** — the null result on 2–4B models is enough for
   you; not worth more Mac-hardware nights chasing a shrinking effect
   given MLSys is ~11 weeks out. Go back to Stage 2's demoted axes
   (federated SLM, edge VLM) for a top-up sweep, or open a new axis.
4. **Something else** — e.g. narrow G1 to a claim the data *does* support:
   "on 2–4B-class edge models, constrained decoding's energy cost is a
   wash, contradicting the 3.6–8.2× literature claims for this hardware
   class" is itself a small, defensible, low-effort finding, just a much
   smaller paper than the original G1 pitch.

## Why a worker can't pick this

Every option spends a scarce resource that's genuinely yours to allocate —
either your Mac's overnight hardware time on a specific new bet, or the
decision to stop chasing G1 with ~11 weeks left before MLSys. BRIEF's
constitution reserves exactly this call for you ("pilot effect smaller than
our statistical resolution" is a named kill-criterion trigger).

## What happens on no answer

Per BRIEF's pace governor: this loop will not queue new Mac-hardware
experiments or draft further G1 hypotheses without an answer here. Next
iterations will do literature-only tracking (re-check MLSys 2027 CFP
status, re-run the PSC scoop query) at reduced pace until you respond.

</details>
