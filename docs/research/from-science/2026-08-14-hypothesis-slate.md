# Scored hypothesis slate — keep G1 or pivot? (2026-08-14)

*Author: research strategist agent, answering the 2026-08-14 DECISIONS "STRATEGIC
DIRECTION REVIEW" gate. Grounds on: `STATUS.md` (iter 7), `DECISIONS.md`,
`NIGHT-REPORT-2026-08-13.md` (005/006), the two 2026-08-14 from-science docs
(optimization-methods, mechanism-and-gaps), `BRIEF.md`, and a fresh web novelty
pass (2026-08-14). Purpose: give Maria a scored, benchmarked keep-or-pivot call.
The lab is an ASSET and is kept in every option below — the question is its
highest-impact use.*

---

## What the lab can actually measure (the reuse envelope every candidate is scored against)

Per-token **latency + memory + ENERGY (Joules)** on edge (powermetrics / RAPL /
tegrastats); **constrained generation** (GBNF / `json_schema` / llama.cpp mask);
**tool-call fidelity** + **LLM-judge**; **paired within-run** measurement;
manifests (git/llama.cpp SHA, hardware, thermal). Rig: M1 Air, Ollama,
sub-1B–4B, single-stream (`--parallel 1`), can target Pi5/Jetson. Pilot assets:
005/006 (480+40 runs, full outputs preserved), the shadow-mask-pipelining design
(007), the g1_mask_cost_pilot harness.

Maria's non-negotiables (DECISIONS): (1) **low-level/systems**, never
measurement-alone — paper = characterization + **mechanism** + **validated
optimization**; (2) a **robotics-native** workload must exist, not only medical
JSON; (3) blue-ocean / unscooped; (4) reuse the lab, keep the repo.

---

## The eight candidates (score 1–5 each; killer experiment named)

### G1 — Constrained-decoding energy/latency tax + shadow-mask pipelining (llama.cpp, medical JSON) — *incumbent*
The measured tax (gemma3:1b **+30.4 %** at <25 ms/tok), attributed to llama.cpp's
on-critical-path O(vocab) mask walk, fixed by computing the mask in the GPU
forward-pass shadow. Bit-exact, ~200–400 LOC fork, measured in Joules.

- **Impact 4** — real, mechanistic, but framed around JSON tool-calls; "too small / niche" is Maria's own stated worry.
- **Novelty 4** — edge single-stream Joules + the llama.cpp-specific un-overlapped mask is unscooped; but the overlap *idea* has datacenter ancestry (XGrammar, SGLang jump-forward). Not blue-ocean, just an empty corner of a mapped ocean.
- **Fit 3** — energy: yes. Medical: one JSON schema. Robotics: bolted on via a tail-latency sentence. Privacy: absent.
- **Feasibility 5** — code + confirmed pilot (005/006) + designed optimization (007) already exist. Nothing else is this shovel-ready.
- **Venue 4** — strong MLSys fit; robotics/medical venues weaker.
- **Killer experiment:** run 007 — shadow-mask pipelining before/after on gemma3:1b's +30 % tax, within-run, bit-exact output guard, reported in ms **and** Joules; headline = recovered/tax.
- **Sum 20/25.**

### G2 — The privacy energy tax: on-device vs cloud medical inference, in Joules
Quantify what "keep the data local" costs in Joules/latency/battery for a medical
extraction workload, and optimize it down.

- **Impact 3** — important question, but the headline answer already exists publicly.
- **Novelty 2** — occupied: EnerInfer (on-device 138 % of cloud power), "Medicine on the Edge" (clinical on-device comparison), John Snow Labs tokens-per-Joule clinical, hybrid cloud-edge decision guides. The framing is taken and reads as measurement.
- **Fit 5** — privacy + medical + energy bullseye of her stated interests.
- **Feasibility 4** — energy telemetry exists; needs cloud-side accounting + a task suite; hard to make *low-level*.
- **Venue 2** — reads as benchmark/characterization → hits Maria's own anti-benchmark bar.
- **Killer experiment:** Joules + latency for medical field-extraction on-device vs cloud round-trip across model sizes — but the mechanism/optimization leg is thin, which is why it scores low.
- **Sum 16/25.**

### G3 — Deadline-driven constrained (action-grammar) decoding for robot control loops: the tax under a deadline, and shadow-mask pipelining as the deadline-recovery method ⚑
G1's exact machinery, **re-aimed** from medical JSON to a **robotics-native**
workload: an action / STL-style command grammar decoded inside a control loop with
a per-cycle deadline (e.g. 10–50 Hz). Measure the constrained-decoding tax as a
**deadline-miss rate** and in **Joules (= robot mission time)**; show shadow-mask
pipelining converts missed cycles into met ones without changing outputs.

- **Impact 5** — a missed control-cycle is a first-order robotics failure (jitter, instability), not a JSON nicety; energy = autonomy/mission-time. This is the "pushes the needle" framing Maria wants.
- **Novelty 4** — SafeDec (ICML 2026) does STL-constrained decoding for robot policies via logit=−∞ masking, but reports **zero** energy/latency cost of that masking and offers **no edge deadline-recovery optimization**. The tax-under-deadline-on-edge square is empty; SafeDec proves the venue cares.
- **Fit 5** — robotics-native (finally answers DECISIONS challenge #1 head-on), energy, and feeds bigos (on-device action decoding).
- **Feasibility 4** — reuses the whole G1 harness; an action grammar is just another GBNF; a "deadline" is latency-vs-budget, which the lab already measures per token. 005/006 medical data demote cleanly to a generalization cell. Modest extension, not a rebuild.
- **Venue 5** — dual-home: ICRA/CoRL/RA-L **and** MLSys/SenSys both accept it.
- **Killer experiment:** on an action/STL grammar at a fixed control rate, measure deadline-miss rate + Joules with mask-on-critical-path vs shadow-mask-pipelined on a sub-1B model; headline = "% of cycles rescued from deadline-miss, bit-exact, at lower Joules/cycle."
- **Sum 23/25.**

### G4 — Multi-model agent-orchestration energy on edge: the cold-start / context-switch energy tax (feeds bigos)
Characterize and minimize the Joules + seconds paid every time an on-device agent
OS swaps between N small models on **unified memory** (cold-load vs KV-persist vs
co-resident), and propose an eviction/residency policy.

- **Impact 4** — every model swap on a shared-memory SoC costs real energy and latency; this is the tax an agent-OS lives or dies on.
- **Novelty 4** — datacenter/multi-GPU analogues exist (C2CServe, persistent-Q4-KV multi-agent, model-switch 1.7 s→50 ms) but **single-device unified-memory orchestration *energy*** is thin.
- **Fit 4** — bigos bullseye + energy; medical/robotics only indirect.
- **Feasibility 3** — needs a new orchestration harness (swap/persist/KV residency); the most new code of the strong candidates, and it drifts *away* from the constrained-decoding asset.
- **Venue 4** — MLSys / SenSys.
- **Killer experiment:** per-switch Joules + latency across a fleet of sub-1B models on unified memory under cold-load vs KV-persist vs resident; show a residency policy that cuts the orchestration energy tax by X %.
- **Sum 19/25.**

### G5 — The correctness tax in Joules: validity vs correctness vs energy for on-device medical structured output
Add answer-correctness grading to 005/006 and report **Joules-per-valid-AND-correct**.

- **Impact 3** — sharpens a known effect by adding energy; important but incremental.
- **Novelty 2** — validity-vs-correctness is taken (Constraint Tax 2605.26128, When-Correct-Isn't-Usable 2605.02363, Hidden Cost of Structured Generation 2603.03305, Draft-Conditioned Decoding). The energy axis is the only new thing.
- **Fit 4** — medical + energy; not very low-level.
- **Feasibility 5** — offline re-score of already-preserved 005/006 outputs against a gold asset + existing energy columns; near-zero hardware cost.
- **Venue 3** — excellent as a **leg** of G1/G3, thin as a standalone top paper.
- **Killer experiment:** offline re-score 005/006 field-correctness; show constrained decoding "wins" on validity while paying in correctness *and* Joules-per-correct.
- **Sum 17/25.** → **Fold into G1/G3 as their correctness leg, not a standalone paper.**

### G6 — On-device VLA / action-model energy characterization
- **Impact 4 / Novelty 1 / Fit 4 / Feasibility 1 / Venue 3.** **Checked and closed:** heavily scooped in the last 6 months (Characterizing-VLA-across-XPUs 2604.24447, "Seeing is Free, Speaking is Not" 2607.09520, EdgeVLA, LiteVLA, Cross-Platform-Scaling) **and** the lab has no vision/VLA stack — a major build for a taken square. **Sum 13/25.** Killer experiment moot.

### G7 — KV-cache-quantization energy on unified memory (memory-bound decode energy)
- **Impact 3 / Novelty 2 / Fit 2 / Feasibility 4 / Venue 3.** Apple-Silicon KV work is crowding fast (int4-KV-free 2605.05699, Non-Monotonic-MPS-latency 2605.08913, Open-TQ-Metal 2604.16957) and it drifts off her medical/privacy/robotics interests. Feasible via llama.cpp KV-quant flags + energy telemetry, but weak fit + weak novelty. **Sum 14/25.**

### G8 — Privacy-partitioned agent offloading: the energy-minimal on-device medical agent
For a multi-step medical agent (extract → tool-call → verify), find the
energy-minimal split of steps between device and cloud **subject to "raw PII never
leaves the device."** Combines G2's privacy framing with G4's orchestration.

- **Impact 4** — turns the abstract privacy/energy tradeoff into an actionable partition; useful to bigos.
- **Novelty 3** — local-cloud offloading exists (2502.11007, DVFO) but *privacy-constrained* partitioning that minimizes the on-device energy tax for medical agents is thinner.
- **Fit 5** — privacy + medical + energy + bigos bullseye.
- **Feasibility 3** — needs an offload harness + per-step privacy labels; moderate build.
- **Venue 3** — risks reading as systems-design rather than low-level mechanism.
- **Killer experiment:** measure Joules + leakage per partition point of a medical agent loop; report the energy-minimal partition under the no-PII-egress constraint.
- **Sum 18/25.**

---

## Comparison table

| # | Candidate | Impact | Novelty/blue-ocean | Fit | Feasibility | Venue | **Sum** |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|
| **G3** | **Deadline-driven action-grammar decoding for robot loops (+ pipelining)** | 5 | 4 | 5 | 4 | 5 | **23** |
| **G1** | Constrained-decoding tax + shadow-mask pipelining (medical JSON) | 4 | 4 | 3 | 5 | 4 | **20** |
| G4 | Multi-model orchestration energy on edge (bigos) | 4 | 4 | 4 | 3 | 4 | **19** |
| G8 | Privacy-partitioned agent offloading (energy-minimal medical agent) | 4 | 3 | 5 | 3 | 3 | **18** |
| G5 | Correctness tax in Joules (validity vs correctness vs energy) | 3 | 2 | 4 | 5 | 3 | **17** |
| G2 | Privacy energy tax: on-device vs cloud medical | 3 | 2 | 5 | 4 | 2 | **16** |
| G7 | KV-cache-quant energy on unified memory | 3 | 2 | 2 | 4 | 3 | **14** |
| G6 | On-device VLA/action-model energy (scooped + infeasible) | 4 | 1 | 4 | 1 | 3 | **13** |

---

## RECOMMENDATION: keep the engine, re-aim the paper — G1 → G3

**G1 survives as an *engine* but is *beaten as a paper framing*.** It clears
Maria's bar (mechanism + optimization, low-level, unscooped-for-its-corner) and is
by far the most shovel-ready — but its impact/fit/venue ceiling is capped by the
medical-JSON workload, which is exactly the "too small / not robotics-native"
worry Maria raised in DECISIONS on 2026-08-10 and again on 2026-08-14.

**The strictly more promising direction is G3, and it is a re-framing, not a
pivot** — it reuses essentially 100 % of G1:

- **Reused as-is:** the g1_mask_cost_pilot harness, constrained-decoding path
  (GBNF/json_schema/llama.cpp mask), the O(vocab) mask-cost **mechanism** and its
  crossover, the **shadow-mask pipelining** optimization (007) verbatim, the
  energy/latency/paired-within-run telemetry, manifests, and the vocab-controlled
  matrix.
- **Reused as a generalization cell:** 005/006 and the medical envelope become the
  "text/JSON" point that shows the tax is workload-general; the **headline moves**
  to a robot action / STL-style command grammar decoded under a control-loop
  deadline.
- **What changes (modest):** author one action-grammar workload (another GBNF) and
  express the tax as a **deadline-miss rate** at a fixed control rate — the lab
  already measures the per-token latency this needs. Pull G5's correctness
  re-score in as a leg. No rebuild; the repo is kept intact.

Why G3 dominates: it lifts Impact (4→5, control-loop deadline misses > JSON
validity), Fit (3→5, robotics-native at last), and Venue (4→5, dual-home
ICRA/CoRL/RA-L **and** MLSys) while giving up only one point of Feasibility (5→4),
and SafeDec (ICML 2026) confirms the topic is live at top venues yet has left the
**energy/latency cost of the constraint and its edge deadline-recovery** completely
unmeasured — a genuine blue-ocean corner adjacent to a hot area.

**Runner-up / paper #2:** **G4** (orchestration energy on edge) is the best
bigos-native second paper once G3's engine is proven — it inherits G3's energy
telemetry and targets the agent-OS directly, but needs the most new harness so it
should follow, not lead.

**Do not pursue standalone:** G6 (scooped + infeasible), G7 (crowded + off-interest),
G2 (taken + measurement-flavored). **G5 is not a paper — it is G1/G3's correctness
leg** and should be executed as the cheap offline re-score it already is.

**Net action for Maria:** keep the repo and the G1 code; keep building the 007
shadow-mask pipeline; but **change the workload and the story from medical JSON to
robot action grammars under control-loop deadlines (G3)**, with the medical/energy
result as the generalization cell and correctness (G5) as a leg. That is the
highest-impact use of the existing lab.
