
## 2026-08-10 ~22:30 (via chat with Claude — Maria's challenge to G1)
Maria questions G1 on two fronts; the Stage-3 hypothesis write-up MUST answer both:
1. ROBOTICS RELEVANCE: "Pydantic agents" is not how robotics builds systems.
   Response required: frame constraints as the field does (action grammars,
   SafeDec-style STL safety specs, VLA action-token masking) — Pydantic is only
   one authoring layer. At least one hypothesis must use a ROBOTICS-NATIVE
   constrained workload (robot command/action grammar with control-loop
   deadlines), not only medical JSON.
2. SIGNIFICANCE: Maria suspects the contribution may be too small. Response
   required: (a) the paper shape must be characterization + MECHANISM +
   improvement, never measurement alone; (b) pilot 002's measured effect size
   is the go/no-go — if the tax is below meaningful (per constitution kill
   criteria), G1 dies and we move on. No attachment to the candidate.

## 2026-08-14 (via chat with Claude — answering the iteration-6 NEEDS_INPUT gate)
005 CONFIRMED G1's crossover: at <25 ms/token all four sub-1B models show a
positive paired latency tax (qwen2.5:0.5b +1.84%, qwen3:0.6b +2.42%,
llama3.2:1b +6.15%, gemma3:1b +30.36%), 28x spread, non-monotone in decode
speed. Maria's decision on how to spend the runway:

1. NOT drafting a characterization-only paper. Reaffirming the 2026-08-10
   decision: the contribution MUST be characterization + MECHANISM +
   OPTIMIZATION METHOD. Proving/finding the tax rule is necessary but NOT
   sufficient for a top venue — the paper must PROPOSE AND VALIDATE a
   mitigation technique that reduces the tax (e.g. pipeline the CPU mask-walk
   across the CPU/GPU sync boundary, mask caching keyed on grammar-stack state,
   vocab-aware/sparse mask computation). This optimization is the headline
   contribution; the measurement is its motivation.
2. MECHANISM must be nailed before drafting: run the n_vocab-controlled
   experiment to test "tax ∝ vocab" as the driver of the 28x spread. Extract
   n_vocab into the results (nobody has yet).
3. CORRECTNESS-TAX gate: add task-correctness grading (not just schema
   validity) to the pilots; re-score existing runs if possible. No claim that
   constrained decoding is "good" or the optimization "harmless" until a
   correctness axis exists (arXiv 2606.26128 shows validity can stay 100%
   while correctness collapses).

Next action for the loop: fold the two research artifacts landing in
docs/research/from-science/ (2026-08-14-optimization-methods.md and
2026-08-14-mechanism-and-gaps.md) into STATUS; queue, in order: the
vocab-isolation experiment, correctness-grading (code + re-score), then the
hero optimization method's before/after experiment. Advance to paper drafting
only once the optimization method has a measured effect.

## 2026-08-14 (via chat with Claude — controlled experimental design)
Maria: 005's four models (qwen2.5:0.5b, qwen3:0.6b, llama3.2:1b, gemma3:1b)
differ on EVERYTHING at once — vendor, size, vocab, architecture — so the 28×
tax spread is unattributable; as designed it "tests nothing." All future
driver experiments MUST use a CONTROLLED matrix varying ONE factor at a time:
1. Same-family size sweep (tokenizer/vocab/arch fixed, only param count varies):
   Qwen2.5 0.5B/1.5B/3B (~152k vocab), Llama-3.2 1B/3B (128k vocab). Tests
   tax ≈ mask_cost / forward_pass_time — should scale ~1/decode_time.
2. Vocab isolation: models matched on decode speed but very different vocab
   (~32k vs ~150k–256k).
3. Cross-vendor replication at matched size+vocab (≥2 companies, 2 models/cell
   where feasible) — is the effect universal or architecture-family-specific?
Record per model: n_vocab, n_layers/params, hidden size, measured decode speed.
Analysis regresses relative tax on mask predictors (vocab, allowed-tokens/state)
vs forward-pass predictor (decode time), within- and across-family, to ATTRIBUTE
variance. No single-factor claim about the driver without this control.

## 2026-08-14 (via chat with Claude — STRATEGIC DIRECTION REVIEW, pause G1 build)
Maria: G1 was chosen from what was at hand (the replay + available data), not
from a systematic survey of what is important AND unexplored in her actual
interest area. Before committing the lab to G1's optimization machinery, run a
proper direction review:
- Survey PREVIOUS ROUNDS of the target venues (MLSys + robotics/medical:
  ICRA/IROS/RA-L/CoRL, medical AI) — map what is saturated vs open.
- Return to her stated interests: medical/surgical robotics, PRIVACY, edge
  medical-AI inference optimization, edge ENERGY on edge AI; systems-strong;
  should ideally feed bigos (OS for agents on edge).
- Produce a SCORED SLATE of candidate top-paper hypotheses (G1 included),
  ranked on impact × novelty("blue ocean", unexplored, not taken) × fit-to-
  interests × feasibility-on-EXISTING-edge-llm-lab-code × top-venue-fit.
  BENCHMARK G1 against the best alternatives; keep G1 only if it still wins.
- KEEP THE REPO. The lab (energy/latency/memory telemetry on edge, structured
  generation, tool-calling eval, LLM-judge, llama.cpp harness) is an asset;
  the goal is its highest-impact use, not discarding it.

Until Maria picks a direction from the slate: the paper-worker does
LITERATURE-ONLY tracking — do NOT queue new Mac experiments, do NOT build the
G1 optimization fork, do NOT draft. G1 experiment specs (vocab matrix,
correctness re-score, 007 mask-pipelining) remain designed and on hold, not
cancelled.
