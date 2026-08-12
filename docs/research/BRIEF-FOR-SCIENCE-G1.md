# Brief for Claude Science — G1 go / no-go / pivot

Paste this into a Claude Science project (suggested name: **edge-lab-G1**, so
`research_sync.py` routes its output back here). Science works on this same
disk — point the project at `/root/repos/edge-llm-lab/docs/research/` and
`/root/repos/edge-llm-lab/experiments/results/` as attached directories so it
reads the real data, not a summary.

---

## Your task

Decide whether hypothesis **G1** is worth continuing, and either (a) sharpen it
into a paper-ready claim, or (b) declare it dead and propose the strongest
replacement hypothesis in the same niche. Then write a plan the MacBook can
execute tonight/next.

## What G1 is

**G1:** grammar-constrained decoding (GBNF / json_schema) carries a real
energy/latency tax on edge inference, because CPU-side grammar-mask construction
cannot hide behind the GPU forward pass when that forward pass is short. The
literature's high end claims 3.6–8.2×; we are testing whether that holds on
Apple-Silicon-class edge hardware.

## The evidence so far (read the raw reports; summary here)

Four measurement rounds over two nights on a MacBook Air M1. Sources:
`experiments/results/NIGHT-REPORT.md` (08-10/11, confounded) and
`experiments/results/NIGHT-REPORT-2026-08-12.md` (08-12, clean), plus
`docs/research/digests/2026-08-12.md`.

1. **No large tax on any model.** Paired within-run gbnf−unconstrained is ~1 %,
   never the 3.6–8.2× literature figure. That contradiction is robust.
2. **phi4-mini shows a small, REAL, replicated tax:** +1.51 %, +1.65 %, +1.78 %
   across three independent runs, all 95 % CIs excluding zero.
3. **granite3.1-dense:2b does NOT:** an apparent +0.34 % in 003 collapsed to
   −0.04 % [−0.24, +0.16] in the larger-n 004. Treat as below the noise floor.
4. **The mechanism was never actually stressed.** Everything decoded at
   36.8–65.7 ms/token; a ~40 µs mask (XGrammar's claim) is ~0.1 % of a 37 ms
   forward pass, below the ~0.5 % detection floor. Experiment **005**
   (sub-1B models, sub-20 ms/token) runs tonight to reach the regime where
   G1's mechanism predicts the tax grows.
5. **Rig caveat:** absolute latency is not comparable across nights (observer
   contention); only within-run paired diffs are valid.
6. **Converter control holds:** the repo's Python GBNF converter and
   llama-server's C++ one produce identical token counts across 3 models.
7. **Energy is not monotone:** phi4-mini's constrained conditions cost *more*
   energy (155 J vs 135 J); granite's do not. Unexplained, reported honestly.

## The questions to answer

1. **Is G1 alive?** Given a real ~1.5–1.8 % tax on one model but null on
   another, and the mechanism-stressing 005 result (which will exist by the
   time you run — read it if present), is there a defensible paper here?
2. **If alive, what is the exact claim?** e.g. "constrained decoding's latency
   tax on 2–4B edge models is 0–2 %, model-dependent, and contradicts the
   3.6–8.2× literature" is small but defensible. Is there a bigger one the data
   supports — e.g. a crossover model where the tax appears below some
   forward-pass duration?
3. **If dead, what replaces it?** Propose the strongest hypothesis in the
   surviving niche (energy-aware edge inference, structured-generation
   machinery) that (a) is not scooped — check the scoop list in
   `docs/research/digests/` and search current literature, (b) is measurable on
   the same rig, (c) ideally feeds the bigos startup thesis (agents on edge).
4. **Scoop re-check:** is anyone else measuring constrained-decode energy on
   edge hardware (not datacenter GPUs)? PSC (OpenReview SEjxNfQTHN) and XGrammar
   are the known near-misses; confirm nothing new closes the gap.
5. **Venue reality:** MLSys 2027 CFP is still unconfirmed (~Oct 30). Is there a
   confirmed venue whose deadline this work can actually hit?

## Deliverable

A dated markdown file (Science will save it as an artifact) with: the go/no-go
call with reasoning, the exact paper claim if go, the replacement hypothesis if
no-go, and a concrete next-experiment spec the MacBook can run (in the same
shape as `experiments/queue/005-g1-short-forward-pass.yaml`). Keep it grounded
in the numbers above and whatever 005/006 produced.

When done, tell Maria; on the Linux box `scripts/research_sync.py --project
edge-lab-G1 --commit` pulls your artifact into this repo.
