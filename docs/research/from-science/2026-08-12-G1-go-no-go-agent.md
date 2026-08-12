# G1 go / no-go — agent recommendation (2026-08-12)

*Author: research-analyst agent. Grounded in `experiments/results/NIGHT-REPORT-2026-08-12.md`
(clean 4-round run), `NIGHT-REPORT.md` (confounded predecessor), `docs/research/digests/2026-08-12.md`,
and a fresh web scoop/venue pass. All numbers quoted are from the repo's own reports.
Experiments 005 and 006 had **not** run at write time (only 8 result dirs present, none 005/006).*

---

## TL;DR

**Qualified GO — but kill the literal hypothesis and reframe.** The strong form of G1
("grammar-constrained decoding carries a real energy/latency tax; literature's high end
3.6–8.2×") is **dead on Apple-Silicon-class edge hardware**: the largest paired within-run
`gbnf−unconstrained` diff across four rounds and three models is **+1.78%**, and two of three
models are null or negative — nothing within an order of magnitude of 3.6–8.2×. What survives
is small but real: **phi4-mini shows +1.51% / +1.65% / +1.78% across three independent runs,
all 95% CIs excluding zero**; granite collapses to **−0.04% [−0.24, +0.16]** at larger n;
qwen3's −1.76% is a truncation artifact (60/60 unconstrained runs hit the 768-token cap).
The scoop gap is **genuinely still open** — nobody measures grammar-mask overhead in Joules
on single-stream edge hardware — so a reframed paper (*"where does the structured-decoding tax
turn on, and does it ever matter on the battery?"*) is defensible. **The GO is contingent on
experiment 005**: it is the only run that stresses G1's actual mechanism (short forward pass).
If 005 is also null at <20 ms/token, flip to NO-GO and take the **energy-per-valid-output**
replacement below — which the existing data already half-supports and which inverts the sign
of the whole story.

---

## Is G1 alive?

**The literal claim is falsified; a narrower one survives; the mechanism is still untested.**
Three separate questions, three separate answers:

**1. Does the 3.6–8.2× literature tax reproduce on this hardware? No — robustly.**
Across four rounds (002-rerun, 003-rerun, 004, plus the confounded 08-11 baseline) the paired
per-token tax is:

| model | best estimate | replication status |
|---|---|---|
| **phi4-mini** | **+1.5 to +1.8%** | 3/3 CIs exclude zero (003, 004-gbnf, 004-json_schema) — **real** |
| granite3.1-dense:2b | ~0% | +0.34% in 003 collapsed to −0.04% [−0.24,+0.16] in 004 — **not established** |
| qwen3:4b | −1.76% | artifact — 60/60 unconstrained runs truncated mid-reasoning; not like-for-like |

This is the paper's most robust result and it is a **refutation**: the 3.6–8.2× high end does
not hold at batch=1 on an M1. That contradiction is worth stating plainly, and the scoop pass
below shows *why* the literature is high — those numbers come from **batched datacenter serving**
(a search result reports Meta-Llama-3-8B at "up to 37.5% higher latency **at batch size 512**…
due to non-parallelizable validation steps"). The tax scales with batch; single-stream edge is
exactly the regime where it should be smallest — and it is. That mechanistic reading turns a
null result into a *characterization* result.

**2. Is there any real effect at all? Yes, on one model, small.**
phi4-mini's replicated +1.5–1.8% is "the closest thing to a real, reproducible
structured-generation latency tax measured so far" (NIGHT-REPORT §7). It is model-dependent —
same rig, same schema, same converter, and granite shows nothing. Model-dependence is itself
a finding, not a nuisance, *if* the paper can attribute it (candidate: phi4-mini's grammar
makes it emit **more** tokens — 310.4 vs 265.1 unconstrained — so part of "tax" is the grammar
changing *how much* the model writes, not per-token cost; this must be disentangled).

**3. Is G1's stated mechanism supported? Unknown — never stressed.**
Every configuration decoded at **34.6–65.7 ms/token**. A ~40 µs XGrammar-class mask against a
37 ms forward pass is ~0.1%, below the ~0.5% detection floor. G1 predicts the tax **grows as
the forward pass shortens** (the CPU mask can no longer hide). That regime (<20 ms/token,
sub-1B models) has not been entered. **Experiment 005 is the whole ballgame** — it either finds
the crossover (strong GO, mechanism confirmed) or shows the tax still absent at 10–15 ms/token
(mechanism dead on this hardware family; take the replacement).

**Verdict:** alive enough to run 005, not alive enough to commit months on today's data. The
strong hypothesis is dead; the reframed one is a coin whose flip is 005.

---

## The exact paper claim (if GO)

> **On single-stream, Apple-Silicon-class edge inference, the latency and energy cost of
> grammar-constrained decoding (GBNF / json_schema) is 0–2%, model-dependent, and does not
> reproduce the 3.6–8.2× penalties reported for batched datacenter serving. The tax becomes
> resolvable only as the forward pass shortens: it is undetectable at 35–66 ms/token and
> [emerges at / remains absent below] ~15 ms/token, locating the crossover where CPU-side mask
> construction stops hiding behind the GPU forward pass.**

The bracket is filled by 005. This claim clears Maria's bar (DECISIONS.md 2026-08-10 pt 2:
characterization + **mechanism** + improvement, never measurement alone) *only if* 005 supplies
the crossover — that is the mechanism half. The "improvement" half is then near-free: if the
tax appears below a known forward-pass duration, the mitigation is to **pipeline mask
construction across the CPU/GPU sync boundary** (STATUS.md open-debt hypothesis-shape #1), and
a prototype speedup on 005's fastest model is the paper's third leg.

**Smallest experiment set that nails it:**
1. **005** (queued) — sub-1B models to <20 ms/token; find or exclude the crossover. *Decisive.*
2. **006** (queued) — qwen3 at n_predict=3072 to convert the censored 300 J vs 167 J into a
   quotable number (or kill it). Needed so no reviewer can call the energy axis contaminated.
3. **One mechanism probe** — on 005's fastest model, an A/B of mask-on-critical-path vs
   mask-precomputed/pipelined, to show the crossover is the *sync stall*, not token-count drift.
   This is what converts "we measured a tax" into "we explain and remove it."
4. **Token-count control** — re-report every tax as per-token time on **matched output length**
   (phi4-mini writes more under grammar); otherwise the +1.8% is partly a length artifact.

If 005 shows a crossover and step 3 removes it, the paper is real and un-scooped. If 005 is null,
stop here — do not chase it onto a Pi/Jetson hoping the number appears; take the replacement.

---

## Replacement hypothesis (NO-GO path, and the recommended hedge)

**G1′ — Constrained decoding is a net energy *win* per valid output on edge agents, and the
per-token latency tax is dwarfed by the retry/runaway-generation energy it eliminates.**

This **inverts G1's sign** and is the stronger paper, because the existing data already points
at it and it is *not* the number everyone else chases. Evidence already in hand:

- **qwen3:4b unconstrained burned 300–312 J** hitting the 768-token cap in 60/60 runs and
  produced **0/60 valid** JSON; constrained produced valid JSON for ~167–178 J. Even after
  de-censoring (006), the task-level story is "unconstrained agents burn the battery emitting
  prose that never becomes a valid tool call."
- **J-per-valid-output** was 702 J (unconstrained) vs ~123 J (constrained) on granite even
  *after* the enum fix — a ~5.7× task-level energy gap that has nothing to do with per-token cost.
- The metric the field measures (per-token tax) is the wrong denominator for an **agent on a
  battery**, which cares about **Joules per completed, valid action**, including retries.

**Why it's the right pivot for this project:**
- **Un-scooped.** The scoop pass found energy-on-edge papers (SLM energy on Pi5/Jetson, arXiv
  2511.11624, in Wh / tokens-per-Wh / EDP) that **explicitly do not distinguish output by grammar
  or constraint**, and structured-generation papers that measure latency/throughput but not
  edge Joules. Nobody owns "energy-per-valid-agent-action under constrained vs free decoding on
  edge." (STATUS.md hypothesis-shape #2.)
- **Measurable on the identical M1 rig** — the harness already logs Joules, validity, truncation,
  and token counts per run. No new hardware, no new instrument.
- **Feeds the bigos startup thesis directly** (agents at home, on battery): "constraining is a
  power-management decision, not just a correctness one" is a founder-grade claim.
- **Robotics hook** (Maria's DECISIONS.md challenge #1): reframe as Joules-per-valid-**action**
  for a mobile robot on a battery mission — action grammars, not Pydantic — where a runaway
  unconstrained generation is both an energy leak and a control-loop deadline miss.

**Threat to control:** the Constraint Tax paper (arXiv 2605.26128) shows hard-schema decoding on
sub-2B models raises validity 61.5%→100% but *drops* answer accuracy 19.7%→11.0%. So "valid" ≠
"correct." G1′ must measure **Joules per valid-AND-correct output**, or it measures the wrong win.
Add an LLM-judge / gold-answer correctness gate to the denominator before claiming the energy win.

---

## Scoop check

**The core gap is still open: no one measures grammar-mask overhead in Joules on single-stream
edge hardware.** Detail, with the cluster sorted by threat:

**Near-misses that do NOT close the gap (cite / differentiate-from):**
- **XGrammar** ([2411.15100](https://arxiv.org/pdf/2411.15100)) — <40 µs/token, near-zero
  overhead, supports Apple Silicon — but measured as **datacenter serving throughput**, no edge
  energy. This is the source of G1's "the mask is cheap" tension.
- **XGrammar-2** ([2601.04426](https://arxiv.org/abs/2601.04426), CAIS 2026, released May 2026)
  — *new since G1's literature framing.* 6× faster compilation, "near-zero end-to-end overhead
  in modern LLM serving systems." **Still datacenter/agentic serving, still not edge Joules.**
  Worth citing as the moving frontier; does not scoop.
- **The batching mechanism**: a search result reports constrained Meta-Llama-3-8B at **"up to
  37.5% higher latency at batch size 512… due to non-parallelizable validation steps"** — direct
  support that the literature's big taxes are a **batch-serving** phenomenon, explaining the
  single-stream edge null. Track down and cite the primary source; it is load-bearing for the
  reframe.

**Algorithmic mask-speedup cluster (erosion-risk, non-overlapping — same bucket as PSC):**
- **PSC** (parser-stack-classification, OpenReview SEjxNfQTHN), **GRID**
  ([2607.11951](https://arxiv.org/abs/2607.11951)), **Attention-Meets-Reachability**
  ([2603.05540](https://arxiv.org/pdf/2603.05540)), **Pre³** deterministic pushdown automata
  ([2506.03887](https://arxiv.org/pdf/2506.03887)). All shrink mask-computation cost; **none**
  measure edge hardware, **none** report Joules, **none** touch the CPU/GPU sync-stall. They
  lower the ceiling G1 pushes against but do not occupy the edge-energy square.
- **"Hidden Cost of Structured Generation: Draft-Conditioned Constrained Decoding"**
  ([2603.03305](https://arxiv.org/pdf/2603.03305)) — closest by *title/framing* ("hidden cost").
  Could not confirm from the PDF whether it measures edge or energy; **read in full before
  committing** — it is the single most important paper to rule out as a scoop.

**Different-axis (validity/correctness, not latency/energy — cite as adjacent):**
- **Constraint Tax** ([2605.26128](https://arxiv.org/abs/2605.26128)) — validity↑/accuracy↓ on
  sub-2B models; the correctness caveat above. **Format Tax**
  ([2604.03616](https://arxiv.org/html/2604.03616)) — prompt-framing, not decoder constraints.

**Energy-on-edge, no-constraint-axis (confirms the replacement gap is open):**
- **SLM energy footprint on edges** ([2511.11624](https://arxiv.org/html/2511.11624)) —
  Pi5 / Jetson Nano / Orin Nano, Wh / tokens-per-Wh / EDP, **but no grammar/constraint axis.**
- **LLM Inference at the Edge under sustained load** ([2603.23640](https://arxiv.org/html/2603.23640)),
  **single-board-computer LLM benchmark** ([2604.24785](https://arxiv.org/html/2604.24785)) —
  general edge perf/energy, no structured-decoding axis.

**Bottom line:** nothing new closes "constrained-decode + energy + edge single-stream." Both the
reframed G1 and the G1′ replacement sit in open ground. XGrammar-2 is the one thing that moved
and it moved *away* from edge, not toward it.

---

## Venue reality

- **MLSys 2027 — still UNCONFIRMED.** Primary source `mlsys.org` shows only MLSys **2026**
  (Bellevue, deadline was Oct 30 '25). No 2027 CFP page exists yet. Aggregators predict a
  **~Oct 30, 2026** deadline off the historical pattern; a secondary listing puts the conference
  at **May 17–22, 2027**. Treat Oct 30 2026 as a planning date, not a commitment. Re-check
  every iteration.
- **RA-L — rolling, the real target.** Given MLSys uncertainty and Maria's robotics framing,
  RA-L absorbs any MLSys slip with zero loss and accepts the "Joules-per-valid-action for edge
  robot agents" framing directly. **Recommend building toward RA-L as primary, MLSys 2027 as an
  opportunistic secondary if its CFP lands with the reframed result in hand.**
- Backups already logged: IROS 2027 / RO-MAN / ICORR (~Feb–Mar 2027) for a paper-#2 medical-HRI
  angle. ICRA 2027 (~Sep 15–22, 2026) is out of reach for a mechanism paper needing 005+006+probe.

**Deadline math:** 005+006 are one Mac-night each; the mechanism probe + correctness gate + write
is ~3–4 weeks. RA-L rolling makes that comfortable; MLSys Oct 30 is feasible only if 005 confirms
the crossover on the first try.

---

## Next experiment spec

**005 is already the correct next run — do not replace it, run it first.** It is the decision
gate. Below: (a) run 005/006 as queued, then (b) the new **007** that operationalizes the
recommended reframe/replacement so the same Mac night produces the paper's second leg regardless
of which way 005 breaks.

```yaml
name: g1-energy-per-valid-action
description: >
  Operationalizes the reframe/replacement (G1'): the paper metric is JOULES PER
  VALID-AND-CORRECT OUTPUT, not per-token latency tax. Runs AFTER 005 (mechanism
  gate) and 006 (qwen3 de-censor), and is designed to stand whichever way 005 breaks:
  if 005 finds the crossover, this supplies the energy/task leg; if 005 is null,
  this becomes the primary result and G1 is retired in its favor.

  Three conditions per model (unconstrained, gbnf, json_schema) x the sub-1B/2-4B
  models already characterized, but scored on a DIFFERENT denominator than 002-005:
    - schema-valid (already have)
    - valid-AND-correct: add a correctness gate (gold tool-call match or LLM-judge)
      so "valid-but-wrong" JSON does not count as a completed action. This directly
      answers the Constraint Tax (arXiv 2605.26128) confound: on sub-2B models,
      hard-schema decoding can raise validity while dropping correctness.
    - Joules per valid-correct output, WITH unconstrained retries counted: an
      unconstrained agent that fails validity would retry; charge it the retry energy
      (model up to k=3 retries) so the comparison reflects real agent behavior, not
      a single shot. This is the number bigos/robotics actually pays on a battery.

  Report, per model, per condition:
    - J per valid-correct output (primary), single-shot AND retry-adjusted
    - truncation rate and tokens emitted (qwen3-style runaway is an energy leak)
    - the per-token paired tax from 004/005 carried alongside, so the paper can show
      the per-token tax (~0-2%) is dwarfed by the task-level energy delta (~5x+).

  Reading the result: if constrained decoding wins on J-per-valid-correct across
  models once retries are counted, the honest headline flips from "constraining costs
  energy" to "constraining SAVES energy per completed action on edge" - un-scooped,
  bigos-relevant, robotics-framable as Joules-per-valid-action under a control-loop
  deadline. If constrained LOSES on correctness (Constraint Tax confound fires), that
  is itself a clean, publishable caution for edge agent builders.

requires: [ollama, git]
repeats: 1
command: >
  caffeinate -i python3 scripts/g1_mask_cost_pilot.py
  --models qwen3:0.6b,llama3.2:1b,granite3.1-dense:2b,phi4-mini
  --conditions unconstrained,gbnf,json_schema
  --repeats 3
  --score-correctness
  --model-retries 3
  --n-predict 1536
timeout_minutes: 360
```

*Notes for the Mac side:* `--score-correctness` and `--model-retries` are new flags the harness
does not yet expose — if unimplemented, run 007 with the existing flags for the energy/validity/
truncation columns and add correctness scoring on the Linux/analysis side from the saved `raw.json`
responses (same split-of-labor as the grader check in NIGHT-REPORT §6). n_predict=1536 is a
middle ground so sub-1B models finish without qwen3-style censoring but the run stays inside one
night. **Order: 005 → 006 → 007.** 005 alone can turn this whole recommendation from GO to NO-GO;
007 is written so that either outcome still yields a paper.
```
