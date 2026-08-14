# Landscape: edge/on-device inference × surgical-medical robotics × privacy — where the blue ocean is (2026-08-14)

*Author: research-strategist agent. Method: WebSearch sweep of ICRA/IROS/RA-L/CoRL/RSS,
MICCAI, ICML, MobiSys/SenSys, and arXiv 2024–2026, cross-checked against Maria's rig
(edge-llm-lab: per-token latency + memory + **Joules**, paired within-run, GBNF/json_schema,
tool-call fidelity, LLM-judge, hardware manifests; MacBook Air M1 / Ollama / llama.cpp,
sub-1B→4B, single-stream; Pi 5 / Jetson reachable). Grounded against G1 history in
`docs/reports/2026-08-11-go-no-go-G1.md`, `from-science/2026-08-14-mechanism-and-gaps.md`.*

---

## TL;DR (read this first)

1. **G1's medical-schema framing is the wrong headline.** Surgical/medical robot **control
   loops do NOT use LLM grammar-constrained JSON decoding** — they run diffusion/flow-matching/
   imitation policies at 10–50 Hz (SRT-H, Diffusion Policy). Language appears only at the
   **high-level planning layer** (SRT-H plans in language space), which is a *slow semantic*
   layer, not a hard-real-time loop. So "constrained-decoding latency tax for surgical control"
   is a false premise. The generic `constant_data_en` medical form is an arbitrary schema, not
   a load-bearing medical-robotics contribution.

2. **But the instrument and the mechanism survive** — repointed at two constraints that DO
   matter: (a) the **energy/latency cost of privacy** for on-device medical LLMs (zero-egress),
   and (b) the **energy/latency tax of *safety*-constrained decoding** (STL/LTL/CBF) for robot
   commands. Both are exactly Maria's rig, both are unmeasured in edge Joules, both matter.

3. **VLA-on-edge energy is now CROWDED** — do not enter it head-on. Multiple 2026 papers already
   characterize VLA latency/energy across accelerators (Char-VLA-XPU, Green-VLA, "Seeing is Free,
   Speaking is Not", the Embodied-FM-at-the-Edge survey). Maria's differentiator is not "VLA
   energy" generally; it is the **constraint/safety/privacy axis** none of them measure.

4. **Top pick:** GAP 1 (privacy "zero-egress tax") or GAP 2 (safety-constrained-decode tax),
   with G1's pipelining idea living inside GAP 3 (deadline-aware on-edge planning). **Venue:**
   **RA-L (rolling)** is the safest real fit; **SenSys 2027 second cycle** (late 2026) is the
   strongest topical home; **ICRA 2027 (Sept 15, 2026)** only if she commits *now*.

---

## Reality-check: does medical/surgical robotics actually need LLM structured decoding?

**Mostly no — and being honest about this is the whole value of the survey.**

| Layer of a surgical/medical robot | What actually runs there | Rate / deadline | LLM structured decode? |
|---|---|---|---|
| Low-level visuomotor control | Diffusion Policy / Flow-Matching / imitation (SRT-H low policy) | 10–50 Hz, tens of ms | **No.** Continuous actions, not JSON. Flow-matching hits 50 Hz single-step. |
| Action tokenization (VLA) | 256-bin discretized action tokens (RT-2/OpenVLA), VQ tokens | ~kHz-aspirational; today 3–10 Hz | Discrete tokens, but the "grammar" is a fixed bin vocabulary, not GBNF. Constraint = control-barrier/STL, not schema. |
| **High-level planning / self-correction** | **SRT-H plans in language space**; task + corrective instructions | seconds; **not** hard real-time | **Yes-ish.** This is where language + structured output live. Slow semantic layer. |
| Clinical decision support / triage / documentation | Medical LLM/SLM, structured extraction | seconds; privacy-gated | **Yes.** Structured output matters; **privacy** (HIPAA/GDPR), not deadline, is the driver. |
| Untethered / capsule / assistive devices | microcontroller + RF; sub-8h battery, 42.5 °C thermal cap | — | No LLM on-board today; **energy/thermal** is the hard wall. |

**Consequences for G1:**
- The claim "grammar-constrained JSON decode has a latency tax that threatens a surgical
  control deadline" is **not defensible** — no surgical control loop consumes JSON tool-calls,
  and the measured tax is ~1.5% anyway (`go-no-go-G1.md`). Kill this framing.
- What is real: at the **planning / clinical / extraction** layers, structured output is used,
  and there the binding constraint is **privacy and energy/battery/thermal**, not a µs control
  deadline. That is the pivot.

---

## Gap map — what's explored vs open, per thread

### Thread A — VLA / policy models on edge (energy, latency)
**Status: CROWDED, closing fast. Avoid head-on.**
- *Characterizing VLA across XPUs* (arXiv 2604.24447): latency + energy + cost leaderboard across
  accelerators — exactly the "which model-hardware hits the control frequency under energy" study.
- *Green-VLA* (2602.00919): staged VLA, energy-framed.
- *"Seeing is Free, Speaking is Not"* (2607.09520): edge-VLM energy bottleneck; a query = 50–500 J,
  output tokens 11–39× the energy of input/visual tokens. **This is Maria's exact style of result,
  already published for generic VLM.**
- *Embodied Foundation Models at the Edge* survey (2603.16952): frames deployment as a **systems**
  problem ("Deployment Gauntlet", 8 barriers); AR-VLA is memory-bandwidth-bound, diffusion is
  compute-bound. Good citation / framing, but it *is* the survey — the measurement is the gap it
  points at, and others are already filling the generic version.
- **Open sub-gap Maria can still own:** none of these put a **constraint/safety-spec axis** on the
  energy. See GAP 2.

### Thread B — Safety-constrained decoding for robot commands (STL / LTL / CBF)
**Status: SEMANTICS solved, COST unmeasured. OPEN and squarely Maria's rig.**
- *SafeDec* (2509.01728, ICML 2026): STL-constrained decoding for navigation policies; masks/reweights
  action logits by STL robustness. Claims "**minimal overhead**" via STLCG++ — but **no edge Joules,
  no per-token latency budget, no battery**. The overhead is asserted, not characterized.
- *Plug in the Safety Chip* (2309.09919), *SELP* (2409.19471): LTL constraint modules for LLM robot
  agents; correctness-framed, **no energy/latency**.
- *VLSA* (2512.11891): plug-and-play CBF safety layer for VLA; safety-framed, cost not the focus.
- *Enforcing Temporal Constraints for LLM Agents* (2512.23738).
- **The gap:** the *tax* of enforcing a safety spec at decode time — in ms and Joules, on edge,
  single-stream — is unmeasured. This is **G1's mechanism with a constraint that actually matters.**

### Thread C — Deadline-aware LLM serving for robotics
**Status: STARTED at datacenter, OPEN on edge.**
- *TimelyLLM* (2412.18695, MobiSys 2026): segmented generation — emit executable commands early,
  pause, overlap with robot execution; time-utility scheduling. **But built on vLLM, datacenter,
  no energy, batched.** This is precisely G1's "pipelining optimization" idea — **already done at
  the datacenter, never on single-stream edge with Joules.**
- **The gap:** segmented/early-emit structured generation measured on edge with energy + deadline
  utility. G1's pipelining optimization lives here.

### Thread D — Privacy-preserving on-device medical inference
**Status: WHY + FEASIBILITY established, ENERGY/BATTERY explicitly DEFERRED. Blue ocean.**
- *Toward Zero-Egress Psychiatric AI* (2604.18302): first fully on-device, zero-egress psychiatric
  LLM app. **Explicitly states energy/battery/latency benchmarking is future work** ("controlled
  device-class benchmark study to establish publication-quality latency, memory, and battery
  metrics"). **That is a written invitation to Maria's exact instrument.**
- *Self-Prompting SLMs for Privacy-Sensitive Clinical Extraction* (2605.04221); *Privacy-Preserving
  RAG on Local Devices for Regenerative Medicine* (medRxiv 2025.10.20): structured clinical
  extraction on-device, privacy-motivated, **no energy characterization**.
- *EnerInfer: Energy-Aware On-Device LLM Inference* (2606.23001): NPU/memory-freq scaling for
  energy — generic, not medical, not structured/privacy-framed. Shows the generic edge-LLM-energy
  square is filling; the **medical-privacy + structured-output** square is still empty.
- **The gap:** the **energy/latency/thermal cost of keeping medical inference on-device (the
  "zero-egress tax")** — J-per-consult, battery-hours, thermal envelope, on a real structured
  clinical task — is uncharacterized. GAP 1.

### Thread E — Confidential / TEE medical inference
**Status: measured at datacenter, OPEN on edge; lower feasibility for Maria.**
- *Confidential LLM Inference across CPU/GPU TEEs* (2509.18886): CPU TEE <10% throughput / <20%
  latency overhead; GPU TEE 4–8%. **Datacenter, not edge, not energy, not medical.**
- **The gap** (TEE energy tax on edge medical inference) is real but needs TEE-capable hardware the
  M1/Ollama stack doesn't expose cleanly. **Flag as reach / later.**

### Thread F — Untethered surgical / assistive device energy & thermal
**Status: hardware-physics literature exists; no on-board LLM. Not Maria's rig today.**
- Capsule robots: silver-oxide coin cell, 1 min–8 h; **42.5 °C intestinal thermal cap**; compute
  (image sensor + MCU + RF) is the hottest component. No LLM inference on-board — the interesting
  version ("could a sub-1B model ever run in this thermal/energy budget?") is a *framing hook* for
  GAP 1's thermal axis, not a standalone study.

### Thread G — Correctness vs validity vs energy for small medical models
**Status: correctness axis published, ENERGY axis missing. Cheap add-on.**
- *When Correct Isn't Usable* (2605.02363), *Constraint Tax* (2605.26128): validity ≠ correctness
  for small structured-output models. **No Joules axis.** Maria already has the offline re-scoring
  path (`from-science/2026-08-14-mechanism-and-gaps.md` Task 2). Incremental on its own; strong as
  a sub-result inside GAP 1.

---

## Ranked OPEN gaps — important, measurable on Maria's rig, not taken

### ★ GAP 1 — The "zero-egress tax": energy/latency/thermal cost of keeping medical inference on-device
**Why it matters:** privacy (HIPAA/GDPR) makes on-device the *requirement*, not an optimization, for
clinical LLM use. The literature proves why cloud is unacceptable and that on-device is feasible, then
**explicitly defers the energy/battery/thermal characterization** (2604.18302). That deferral is the
opening.
**The claim:** *"For privacy-mandated on-device medical LLM/SLM inference, here is the Joules-per-
clinical-query, battery-hours-per-shift, and thermal envelope across sub-1B→4B models and device
classes — and here is the energy cost the privacy requirement imposes vs a (hypothetical) cloud call."*
**Measurable on her rig:** yes, directly. Structured clinical extraction/triage task (reuse the
tool-call harness), paired J + latency + memory, hardware manifests, M1 → Pi 5 / Jetson for
device-class generalization. Structured decoding (G1's machinery) enters as a **component** ("the
envelope must be schema-valid *and* correct — at what J?"), not the headline. Add thermal via
tegrastats/RAPL.
**Blue-ocean confidence:** HIGH. The medical-privacy + on-device-energy square is empty; generic
edge-LLM-energy is filling (EnerInfer) but not this framing.
**Feeds bigos:** directly — an OS for agents on edge needs an energy/privacy budget model for regulated
data.

### ★ GAP 2 — The energy/latency tax of *safety*-constrained decoding for robot/agent commands
**Why it matters:** this is **G1's exact mechanism, re-aimed at a constraint that is load-bearing**
(safety compliance) instead of an arbitrary medical form. SafeDec/VLSA/Safety-Chip enforce STL/LTL/CBF
specs at decode time and *assert* "minimal overhead" **without measuring edge Joules or a per-token
latency budget.**
**The claim:** *"Enforcing an STL/LTL safety spec at decode time costs X ms/token and Y J on
single-stream edge — negligible above Z ms/token forward pass, non-negligible below it — so the
safety guarantee is/ isn't free for untethered robots."* This converts G1's null-ish generic-JSON
result into a **safety-relevant** measurement and directly stress-tests SafeDec's "minimal overhead."
**Measurable on her rig:** yes. Implement STL/enum/regex acceptance masks over a small action/command
grammar; reuse the paired constrained−unconstrained + Joules harness and the trivial-grammar control
already designed in `from-science/2026-08-14-mechanism-and-gaps.md` (Cell D). Robot-command envelopes
are short (20–40 tokens), fixing G1's "medical form is 350 tokens" problem.
**Blue-ocean confidence:** HIGH. Nobody has put edge ms+Joules on safety-constrained decoding.
**Feeds bigos:** directly — agents-on-edge need enforced safety specs; their cost is a budget line.

### GAP 3 — Deadline-aware, energy-measured structured generation on edge (TimelyLLM-on-edge)
**Why it matters:** TimelyLLM proves segmented/early-emit generation helps robotic deadlines, but
**only at datacenter (vLLM, batched, no energy)**. G1's pipelining optimization is exactly this, and
the **single-stream edge + Joules** corner is untaken.
**The claim:** *"Segmented/early-emit structured generation on edge improves deadline utility by N%
at M% energy cost, single-stream."* This is where G1's pipelining idea has a home.
**Measurable:** yes, but requires implementing segmentation in the llama.cpp harness — more build than
GAP 1/2. Best as the *optimization* half of a GAP-2 paper (measure the safety-decode tax, then pipeline
it away), not a standalone.
**Blue-ocean confidence:** MEDIUM (TimelyLLM is close; differentiate on edge + energy + safety-spec).

### GAP 4 — Joules-per-valid-AND-correct for small on-device medical models
**Why it matters:** validity ≠ correctness is published; the **energy** axis on it is not. Cheap
because 005/006 outputs are re-scorable offline (no re-run).
**The claim:** *"Schema-valid ≠ correct, and here is the J wasted on valid-but-wrong medical envelopes
across small models."* Strong as a **sub-result of GAP 1**, thin as a standalone paper.
**Measurable:** yes, nearly free (offline re-score + existing J columns).
**Blue-ocean confidence:** MEDIUM (correctness papers exist; energy axis is the novelty).

### GAP 5 (reach) — Edge TEE / confidential medical-inference energy overhead
**Why it matters:** confidential computing is the other privacy route; its **edge energy cost** is
unmeasured (only datacenter TEEs characterized).
**Blocker:** needs TEE-capable edge hardware the current M1/Ollama stack doesn't expose. **Defer** until
hardware exists; note as a future extension of GAP 1.

---

## Does G1 (grammar-constrained-decoding latency/energy tax) actually matter here?

- **As a surgical/medical-robotics claim: NO.** Control loops don't consume JSON; the tax is ~1.5%;
  the medical form is arbitrary. Kill the "surgical deadline" framing.
- **As an instrument + mechanism: YES, if repointed.** The paired constrained−unconstrained Joules
  harness, the trivial-grammar control, and the pipelining optimization are genuine assets. They
  become **components of GAP 1 (privacy tax), GAP 2 (safety-decode tax), and GAP 3 (deadline
  pipelining)** — where the *constraint* is load-bearing (privacy, safety) rather than a generic
  medical schema. Recommendation: **retire G1-as-medical-JSON; keep G1-as-method inside GAP 1 or GAP 2.**

## Venue + deadline fit

| Venue | Deadline | Fit | Verdict |
|---|---|---|---|
| **RA-L** | **Rolling (monthly)** | Robotics + systems-measured; ideal for GAP 2/GAP 3 (safety-decode tax on edge). ICRA-transfer option (by Dec 31, 2026). | **Safest realistic home.** No cliff; polish then submit. |
| **SenSys 2027** | Cycle-1 **passed** (Jun 5 2026); **Cycle-2 TBD (~late 2026)** | Explicitly lists "LLMs for edge and embedded systems" + energy/embedded. Ideal for GAP 1 (zero-egress energy). | **Strongest topical fit** if Cycle-2 lands in her window. Watch the CFP. |
| **ICRA 2027** | **Sept 15, 2026** (~1 month) | Robotics; GAP 2 fits. Tight for new experiments, but the instrument exists and command envelopes are short. | **Only if she commits now** to GAP 2 with the existing harness. |
| **MLSys 2027** | Unconfirmed (~Oct 30 2026 per prior notes) | Edge-LLM energy/systems; GAP 1/2/3 all fit. | Backup; confirm CFP. |
| MICCAI 2026 | ~early 2026 (passed) | Medical; GAP 1's clinical framing. | Miss this cycle; hold for MICCAI 2027 or an embedded-health workshop. |

**Recommended play:** commit to **GAP 2 (safety-constrained-decode energy tax)** for **ICRA 2027
(Sept 15)** *if* Maria wants a hard target and will start now — it reuses the existing harness + the
already-specified trivial-grammar control, and short robot-command envelopes sidestep G1's length
problem. Otherwise, **GAP 1 (zero-egress medical energy tax) → SenSys 2027 Cycle-2 / RA-L**, which has
more runway, a cleaner privacy story, and the strongest blue-ocean confidence. Either way the M1 →
Pi 5 / Jetson device-class replication closes the "n=1 hardware" reviewer attack.

---

## Key sources
- SRT-H (surgical, language-planning layer): Science Robotics 2025, arXiv 2505.10251
- SafeDec (STL-constrained decoding, "minimal overhead" unmeasured): arXiv 2509.01728 (ICML 2026)
- VLSA (CBF safety layer for VLA): arXiv 2512.11891 · Safety Chip (LTL): 2309.09919 · SELP: 2409.19471
- TimelyLLM (segmented gen, datacenter): arXiv 2412.18695 (MobiSys 2026)
- Characterizing VLA across XPUs (latency+energy): arXiv 2604.24447 · Green-VLA: 2602.00919
- "Seeing is Free, Speaking is Not" (edge-VLM energy): arXiv 2607.09520
- Embodied Foundation Models at the Edge (survey / Deployment Gauntlet): arXiv 2603.16952
- Zero-Egress Psychiatric AI (privacy on-device, energy deferred): arXiv 2604.18302
- Privacy-sensitive clinical extraction SLM: 2605.04221 · Local RAG regenerative medicine: medRxiv 2025.10.20.25337146
- EnerInfer (energy-aware on-device LLM): arXiv 2606.23001
- Confidential LLM inference CPU/GPU TEEs: arXiv 2509.18886
- When Correct Isn't Usable: 2605.02363 · Constraint Tax: 2605.26128
- Capsule-robot thermal cap 42.5 °C / battery: ScienceDirect S0924424725007113; PMC5063027
- Diffusion/Flow control-rate limits (10–50 Hz): arXiv 2303.04137, 2606.20135, 2511.00983
