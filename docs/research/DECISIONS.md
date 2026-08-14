
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
