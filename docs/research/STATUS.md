# Paper machine — STATUS (updated 2026-10-01, iteration 53)

**Scope decisions so far (newest first):**
1. 2026-10-01 (iter 53): no scope-narrowing decision — the five sub-questions
   reserved for Maria are still unanswered; `DECISIONS.md`, `NEEDS_INPUT.md`,
   and `docs/research/from-science/` unchanged since 2026-09-07 (re-checked
   via `git log` on each file). Local `main` matched `origin/main` exactly at
   the start of this iteration (`fc7b0fc`, iter 52's own commit) — no drift,
   no stale checkout. Did iteration 52's queued next action: swept the
   **correctness-tax cluster** (last touched iter 49 — was the most stale of
   the four rotating angles). Four new-to-`seen.json` items, two data-hygiene
   fixes. **Headline find: VAREX** (arXiv 2603.15118, IBM Research) —
   differentiate-from/cite, a 1,777-document/1,771-JSON-schema multi-modal
   structured-extraction benchmark finding that **below 4B parameters,
   structured-output compliance — not extraction capability — is the
   dominant bottleneck**, with "schema echo" (the model emitting
   schema-conforming structure instead of extracted values) depressing
   scores by 45–65 percentage points — the second correctness-tax finding
   scoped specifically to Maria's own sub-4B regime (joins "Repair, Not
   Improvement," iter 49) and the sharpest statement yet of *why* the gap
   is worst at small scale; cloud-model eval, no edge/grammar-mask angle,
   not a scoop. **ExtractBench** (arXiv 2602.12247) — differentiate-from/
   cite, a second independent schema-valid-≠-correct domain (90% schema-valid
   but 12.5% pass rate on one domain; 0% valid output on a 369-field schema).
   **A second, unrelated paper also named "ExtractBench"** (arXiv 2607.29677,
   LlamaIndex) — differentiate-from/cite, flagged as a name collision between
   the two (distinct authors/datasets/metrics, both genuinely on-cluster,
   both kept). **"Prompt Complexity Dilutes Structured Reasoning"** (arXiv
   2603.13351) — ignore, a keyword-overlap false lead (tests a reasoning
   framework's prompt-context fragility, not schema-validity-vs-correctness).
   **Two data-hygiene fixes** (same class as iters 22/29/31/32/33/36/38/41/
   42/43/44/45/46/49): **"The Format Tax"** (arXiv 2604.03616, bare since
   iter 1, two separate bare keys — a second small URL-format-duplicate
   instance) — now annotated: decomposes the structured-output cost into a
   dominant prompt-level cost (~-3.9pts) and a smaller decoder-level cost
   (~-1.6pts further), directly reusable methodology for the harness-OS
   paper's own correctness-grading gate. **Diffusion-LM constrained decoding**
   (arXiv 2607.07026, bare since iter 1) — now annotated, differentiate-from/
   ignore-for-mechanism (wrong generation paradigm, same class as LAVE).
   **MLSys 2027:** direct `WebFetch` on `mlsys.org/Conferences/2027/Dates`
   still `EGRESS_BLOCKED` (domain-level, unchanged). WebSearch surfaced a
   fuller, internally-consistent CFP schedule than prior iterations
   (submissions open Oct 10 2026 20:00 UTC, deadline Oct 30 2026 20:00 UTC
   unchanged, reviews Jan 18 2027, author responses Jan 21 2027,
   notifications Feb 28 2027, **conference June 21–25, 2027, Bellevue, WA**)
   — not a primary-source fetch, but resolves the standing Montréal-vs-
   Bellevue conference-location discrepancy (flagged since iter 45/47/49) in
   favor of **Bellevue/June 2027**, now adopted in the deadlines table below.
   **No change to the Oct 30, 2026 submission-deadline estimate.** Now
   **~4.1 weeks out (29 days), twenty-fourth consecutive iteration** running
   inside the brief's "<8 weeks away" threshold; the T-4-week WRITING-mode
   governor (~Oct 2, 2026) is **still not crossed** as of today (2026-10-01)
   but is now **~1 day away** — correcting iter 52's prediction that it would
   "almost certainly" be crossed by this iteration; it wasn't, but will very
   likely be crossed before iteration 54, still before Maria answers
   sub-question #2 (venue/timing). **Not a tier-1 event** — no scoop, no
   confirmed deadline change (only the conference date/location figure newly
   converged, not the submission deadline), no decision only Maria can make;
   no `NEEDS_INPUT.md` change. **Non-empty iteration** (four new citations —
   one headline sub-4B-regime finding, two independent schema-valid-≠-correct
   domain confirmations with a name-collision flag, one filed ignore — plus
   two data-hygiene fixes) — empty-iteration counter stays at 0 of 3. Full
   detail: `docs/research/digests/2026-10-01.md`.
1. 2026-09-30 (iter 52): no scope-narrowing decision — the five sub-questions
   reserved for Maria are still unanswered; `DECISIONS.md`, `NEEDS_INPUT.md`,
   and `docs/research/from-science/` unchanged since 2026-09-07/2026-09-17/
   2026-09-24 (re-checked via `find -newer`, only `STATUS.md`/`seen.json`/
   `docs/research/digests/` have moved). Local checkout: `main` was stale
   (pinned at `e689e19`, iter 45's commit) while HEAD sat detached exactly on
   `origin/main`'s tip (`e95fc14`, iter 51's own commit) — same harmless
   pattern noted many times before, not drift; ran
   `git checkout -B main origin/main`, nothing lost. Did iteration 51's
   queued next action: swept the **edge-VLM/medical-structured-generation
   angle** (last touched iter 48 — was the most stale of the four rotating
   angles). Six new-to-`seen.json` items, three genuinely useful, one filed
   ignore, two filed background/no-discussion. **Headline find:
   "Efficient Deployment of Vision-Language Models on Mobile Devices: A Case
   Study on OnePlus 13R"** (arXiv 2507.08505, EPFL) — differentiate-from/cite,
   a real-phone deployment study evaluating **llama.cpp** itself (plus
   MLC-Imp, mllm) running LLaVA-1.5-7B/MobileVLM-3B/Imp-v1.5-3B, with CPU/GPU/
   NPU utilization profiling — finds CPU overutilization during decode, GPU/
   NPU underutilization, and GPU-offload-induced device instability — the
   closest "our own engine, a VLM, real edge device, telemetry" precedent
   found yet, directly useful groundwork for the harness-OS thesis's planned
   VLM extension (Route A) regardless of how sub-question #3 lands; no
   grammar/mask angle, not a scoop. **StepX-Edge** (arXiv 2607.22708, ICMI
   2026 CC-MMD) — differentiate-from/cite, a second real-phone VLM-deployment
   telemetry precedent (Snapdragon 8 Gen5, quantized, TTFT/decode-speed/
   memory numbers), UI-grounding domain not medical/robotic. **"From
   Generalist to Specialist: A Context-Fusion Framework for Endoscopic Polyp
   Reporting with a Frozen VLM"** (arXiv 2608.15580) — differentiate-from/
   cite, a frozen VLM prompted to emit a single structured JSON clinical
   report (diameter, Paris classification, morphology) — the closest
   medical-structured-VLM-JSON-report precedent found in a while, converges
   with Route A's exact use-case shape, but no edge deployment and no
   grammar-constrained decoding (prompt-based structuring only). **EndoVLM**
   (arXiv 2608.04472) — ignore, pretraining/representation-learning paper,
   not structured-report-via-decoding, continues the standing
   zero-systems-cost-competitor pattern in medical-VLM literature. Two
   background surveys filed no-discussion (arXiv 2504.09724, pre-2026;
   arXiv 2412.03772, Dec 2024, general mobile-LLM not VLM-specific).
   **Clean-negative checks:** no citing/follow-up work found yet for "Seeing
   is Free, Speaking is Not" (2607.09520, Route A's named generic-profile
   prior work — still too recent for a citation thread); a targeted search
   for the actual scoop shape (edge/on-device + medical-or-robotic + grammar/
   schema-constrained decoding + real energy/latency telemetry) surfaced only
   the already-tracked VLA/Jetson-robot cluster, no new numbered member.
   **MLSys 2027:** `mlsys.org/Conferences/2027/Dates` direct `WebFetch` still
   `EGRESS_BLOCKED` (domain-level, unchanged). WebSearch reconfirmed the
   standing **Oct 30, 2026, 12:00 PM PDT (≈19:00 UTC)** submission-deadline
   figure, portal opens Oct 10 2026, author reviews Jan 18 2027 — all
   matching prior iterations, no change. The queued "OpenReview portal
   opening" angle isn't independently checkable yet (Oct 10 hasn't happened).
   **No change to the Oct 30, 2026 estimate.** Now **~4.3 weeks out (30
   days), twenty-second consecutive iteration** running inside the brief's
   "<8 weeks away" threshold; the T-4-week WRITING-mode governor (~Oct 2,
   2026) is **~2 days away — still not crossed** as of today (2026-09-30),
   but will almost certainly be crossed before iteration 53, before Maria
   answers sub-question #2 (venue/timing). **Not a tier-1 event** — no
   scoop (the two real-device VLM-deployment finds lack the constrained-
   decoding angle; the structured-medical-JSON find lacks the edge-
   deployment and constrained-decoding angle), no confirmed deadline change,
   no decision only Maria can make; no `NEEDS_INPUT.md` change. **Non-empty
   iteration** (six new citations, three genuinely useful — two real-device
   VLM-deployment telemetry precedents including one using the lab's own
   llama.cpp runtime, one structured-medical-VLM-JSON precedent, one filed
   ignore, two filed background) — empty-iteration counter stays at 0 of 3.
   Full detail: `docs/research/digests/2026-09-30.md`.
1. 2026-09-29 (iter 51): no scope-narrowing decision — the five sub-questions
   reserved for Maria are still unanswered; `DECISIONS.md`, `NEEDS_INPUT.md`,
   and `docs/research/from-science/` unchanged since 2026-09-07/2026-09-17/
   2026-09-24 (re-checked via `find -newer`, only `STATUS.md`/`seen.json`
   have moved). Local checkout was detached HEAD pinned exactly on
   `origin/main`'s tip (`2959b5d`, iter 50's own commit) at the start of
   this iteration — same harmless pattern noted many times before, not
   drift; checked out `main` and fast-forwarded, nothing lost. Did
   iteration 50's queued next action: swept the **harness-OS-thesis angle**
   (last touched iter 47 — was the most stale of the four rotating angles).
   Eight new-to-`seen.json` items, six discussed substantively, two filed
   for hygiene only. **Headline find: "Transition-Aware Backend Dispatch
   for Edge LLM Inference"** (arXiv 2607.17415) — differentiate-from/cite,
   a runtime dispatch policy for edge transformer inference (backend
   selection under shape/transition-cost awareness) validated on real
   **Jetson hardware** with real latency/energy/EDP telemetry (17.4%/
   14.4%/28.5% average reductions over the best static policy) — closest
   "runtime-adaptive decision backed by real edge energy telemetry"
   precedent since DVFSLM (iter 39); same *contribution shape* as the
   harness-OS thesis (a runtime layer that adapts to device/workload
   state) at the backend-selection layer rather than G1's constrained-
   decoding-mask layer — not a scoop, no grammar/mask angle. **"The Agent
   Operating System (AOS)"** (arXiv 2608.03214, Open Compute Project) —
   differentiate-from/cite, naming-relevant: a reference architecture
   literally named "Operating System" for agentic systems — 4th
   independent "OS/harness for agents" naming precedent found (joins
   2605.12129, 2602.00887, 2607.08938), but application/orchestration-level
   only, no edge-hardware or telemetry claim — sharpens rather than
   threatens the requirement that bigos's contribution be the edge-
   hardware-mechanism layer these all lack. **"Toward a Modular
   Architecture for Embedded AI Agent Systems at the Edge"**
   (arXiv 2606.02862) — differentiate-from/cite, a tiered on-device/
   cloud-augmented-SLM architecture with a governance layer, targeting
   embedded-microcontroller constraints — closer to BRIEF.md's own
   fleet-of-home-robots framing than most harness-taxonomy finds, still
   architecture-proposal-level (no telemetry, no mechanism). **"AgenTEE:
   Confidential LLM Agent Execution on Edge Devices"** (arXiv 2604.18231,
   EuroMLSys 2026) — differentiate-from/cite, real edge TEE-based
   (Arm CCA) confidential agent execution, <5.15% overhead — a distinct
   privacy mechanism (hardware isolation) complementary to bigos's
   nothing-leaves-the-device framing, not competing. **SWARM-LLM**
   (arXiv 2606.14711) — cite, an edge-SLM-swarm + optional-cloud routing/
   collaboration layer with explicit on-premises-trust-boundary privacy
   framing, resonates with BRIEF.md's federated/fleet candidate axis.
   **SparseDVFS** (arXiv 2603.21908) — differentiate-from/cite, DVFS
   sub-thread 5th member (joins DVFSLM, 2507.02135, 2604.15357, 2606.16106)
   — generic-DNN not LLM-specific, but independently converges with iter
   47's memory-clock-state finding that EMC is a governance factor
   DVFSLM's own governor doesn't model. Two filed for hygiene only:
   GPU-DVFS-for-fine-tuning (arXiv 2607.05933, wrong task — training not
   inference) and "Toward Super Agent System with Hybrid AI Routers"
   (arXiv 2504.10519, same shape as the standing 5-member escalation/
   cascade cluster, not counted as a 6th member since that cluster wasn't
   this iteration's target). Three more cascade/routing papers (PRISM
   2511.22788, Privacy-Preserving LLM Routing 2604.15728, Reliable
   Edge-Cloud-Expert Cascades 2512.20012) recognized as the same cluster
   shape on sight and not individually logged, per digest-size discipline.
   "Last Harness" (named BRIEF.md candidate) searched again — still no
   matching paper, clean-negative status unchanged. **MLSys 2027:**
   `mlsys.org/Conferences/2027/Dates` direct `WebFetch` still
   `EGRESS_BLOCKED` (domain-level, unchanged). Tried the queued new angle
   — MLSys 2027 program-committee/reviewer-invitation announcements dated
   Sept/Oct 2026 — nothing dated surfaced. WebSearch reconfirmed the
   **Oct 30, 2026, 20:00 UTC** submission-deadline figure — no change.
   **No change to the Oct 30, 2026 estimate.** Now **~4.4 weeks out,
   twentieth iteration running inside the brief's "<8 weeks away"
   threshold**; the T-4-week WRITING-mode governor (~Oct 2–3, 2026) is
   **still not crossed** (today is 2026-09-29, governor date ~3–4 days
   out) — this corrects iter 50's prediction that it would "almost
   certainly" be crossed by this iteration; it wasn't, but will very
   likely be crossed before iteration 52, still before Maria answers
   sub-question #2 (venue/timing). **Not a tier-1 event** — AOS and the
   modular-embedded-agent architecture are the closest naming/framing
   overlaps found yet, but neither claims an edge-hardware mechanism or
   telemetry result, so no scoop; no confirmed deadline change; no
   decision only Maria can make; no `NEEDS_INPUT.md` change. **Non-empty
   iteration** (eight new citations, six substantively discussed — one
   headline runtime/telemetry precedent, one naming-relevant "Agent OS"
   framing, one fleet/on-device-tier architecture precedent, one
   confidential-edge-execution mechanism, one federated/swarm-SLM
   citation, one DVFS-thread 5th member) — empty-iteration counter stays
   at 0 of 3. Full detail: `docs/research/digests/2026-09-29.md`.
1. 2026-09-28 (iter 50): no scope-narrowing decision — the five sub-questions
   reserved for Maria are still unanswered; `DECISIONS.md`, `NEEDS_INPUT.md`,
   and `docs/research/from-science/` unchanged since 2026-09-07/2026-09-17/
   2026-09-24 (re-checked this iteration). Local `main` matched `origin/main`
   exactly at the start of this iteration (`413af09`, iter 49's own commit)
   — no drift, no stale checkout, fetch confirmed before trusting local
   state. Did iteration 49's queued next action: swept the **algorithmic
   mask-speedup cluster** (last touched iter 46 — DOMINO precursor found,
   cluster stayed at 11 members — was the most stale of the four rotating
   angles). **No new numbered cluster member** — ran multiple targeted
   searches (GPU/CPU mask-computation speedup, FSM/DFA precomputation,
   vocabulary-mask kernels, edge-device angle, general 2026 sweep); every
   hit was an already-tracked member or adjacent citation. This is the
   **fifth consecutive nothing-new sweep** (iters 33/38/42/46/50) —
   strengthens the possible-saturation read (not a hard conclusion; the loop
   has no formal "declare a cluster closed" mechanism). Two new-to-
   `seen.json` items, both **ignore**: **"Guided Decoding and Its Critical
   Role in Retrieval-Augmented Generation"** (arXiv 2509.06631, Sept 2025) —
   compares Outlines/XGrammar/LM Format Enforcer across multi-turn RAG
   setups, application/evaluation-level, predates the cluster by ~1 year, no
   edge angle. **Control-DAG** (arXiv 2404.06854, 2024) — non-autoregressive
   Directed-Acyclic-T5 decoding, same wrong-generation-paradigm shape as the
   standing LAVE/diffusion-LLM filing (iter 25). One OSS tool logged for
   data hygiene only, same class as GLRMask (iter 42): **`anishesg/
   dfa-sample`** — fuses GPU-resident DFA token masking with sampling into a
   single kernel, eliminating the logit GPU→CPU round-trip; cloud/
   datacenter-GPU-shaped, no accompanying paper, no edge/Joules angle; also
   surfaced open PRs by the same author against `guidance-ai/llguidance` and
   `guoqingbao/attention.rs` proposing GPU/PDA-resident masking upstream in
   production libraries — a further practitioner-ecosystem data point that
   mask-cost optimization stays an active OSS concern. **MLSys 2027:**
   direct `WebFetch` on `mlsys.org/Conferences/2027/Dates` and
   `.../CallForResearchPapers` both still `EGRESS_BLOCKED`. **Closed out an
   open question from iter 49's next-action list:** confirmed the block is
   at the **domain level** (`mlsys.org` itself, not just `/Conferences/2027/*`
   subpaths) — no different subpath is worth trying going forward. Tried the
   queued workshop-proposal-deadline angle — nothing dated found, only the
   standing main-track figure. A further venue/date search reconfirmed the
   unadopted "May 17–22, 2027, location TBA" secondary figure (same family as
   iter 45's Bellevue claim) — not newly confirmed, not a new contradiction.
   **No change to the Oct 30, 2026 estimate.** Now **~4.3 weeks out,
   nineteenth iteration running inside the brief's "<8 weeks away"
   threshold**; the T-4-week WRITING-mode governor (~Oct 2–3, 2026) is
   **not yet crossed** (today is 2026-09-28, governor date ~4 days out) but
   will almost certainly be crossed before the next iteration, continuing
   the pattern flagged since iter 48 — before Maria answers sub-question #2
   (venue/timing). **Not a tier-1 event** — no scoop, no confirmed deadline
   change, no decision only Maria can make; no `NEEDS_INPUT.md` change.
   **Non-empty iteration** (the fifth-consecutive-nothing-new cluster result
   is itself the substantive finding, plus two items filed to avoid
   re-discovery, one OSS tool logged for completeness, one standing
   MLSys-egress question closed out) — empty-iteration counter stays at 0 of
   3. Full detail: `docs/research/digests/2026-09-28.md`.
1. 2026-09-27 (iter 49): no scope-narrowing decision — the five sub-questions
   reserved for Maria are still unanswered; `DECISIONS.md`, `NEEDS_INPUT.md`,
   and `docs/research/from-science/` unchanged since 2026-09-07/2026-09-17/
   2026-09-24 (re-checked this iteration). Local `main` was 3 commits behind
   `origin/main` at the start of this iteration (this session's own stale
   starting checkout — iters 46/47/48's own commits — not new remote work);
   fast-forwarded via `git checkout main && git merge --ff-only`, nothing
   lost. Did iteration 48's queued next action: swept the **correctness-tax
   cluster** (last touched iter 45 — was the most stale of the four rotating
   angles). Seven new-to-`seen.json` items. **"Repair, Not Improvement:
   Decomposing Constrained Decoding in Tool-Call Abstention"** (arXiv
   2608.13959) — differentiate-from/cite, tests open-weight **0.6B–4B
   models — Maria's own regime** — on tool-call *abstention* specifically
   (declining to call anything when no tool applies) and argues grammar-
   constrained decoding's accuracy win for pick-a-tool doesn't extend to the
   abstain case, since an enum over tool names deletes the "no tool applies"
   answer along with the invalid ones; sharpens the cluster's core finding
   into a concrete sub-1B–4B failure mode relevant to any future tool-calling
   harness eval with a legitimate no-tool case. **"When JSON Is Not Enough:
   Semantic Reliability of Schema-Constrained LLM Ordering Agents"** (arXiv
   2607.18261) — differentiate-from/cite, OrderBench (agentic-ordering
   domain), 100% schema validity but only ~80% semantic success on the
   strongest model, double-digit schema-valid-but-unsafe acceptances on
   weaker ones — another independent domain confirming schema-valid ≠
   correct/safe. **PA-Tool** / "Don't Adapt Small Language Models for Tools;
   Adapt Tool Schemas to the Models" (arXiv 2510.07248, ACL 2026) —
   differentiate-from/cite, a genuinely different fix-side mechanism
   (training-free schema-*renaming* to match pretraining familiarity, not
   decode-time masking) — 17% improvement on MetaTool/RoTBench, 80% fewer
   schema-misalignment errors, joins schema-key-wording/schema-descriptions
   as a third schema-phrasing-matters fix family. **TOOLDEC** (arXiv
   2310.07075, 2023) — differentiate-from, a fifth "foundational-precursor-
   found-late" lineage citation (joins CRANE/GAD/AprAD/DOMINO), not an active
   competitor. Three items filed **ignore** to avoid re-discovery: Projectional
   Decoding (arXiv 2605.30054, SE-artifact domain, wrong task), ATLAS (arXiv
   2510.25890, AUTOSAR/MDE domain, wrong field — **name collision with the
   already-tracked ATLAS-RTC correctness-tax paper, distinct papers**), and
   Semantic Probabilistic Control of Language Models (arXiv 2505.01954,
   attribute/toxicity steering, a different sense of "semantic control"
   entirely). **Second instance of the `seen.json` URL-format-duplicate
   hygiene issue** (first flagged iter 45 for 2606.25605): a bare-date-only
   `.../pdf/2605.02363` entry duplicates the properly-annotated
   `.../abs/2605.02363` entry — not fixed unilaterally, flagged as a second
   data point. **MLSys 2027:** standing CFP/Dates/2027-group/OpenReview
   targets all still `EGRESS_BLOCKED`; **new angle tried — Wayback Machine /
   archive.org — also `EGRESS_BLOCKED`** (archive.org itself is blocked by
   the egress proxy, not just the specific snapshot); an area-chair/program-
   chair search surfaced nothing dated for 2027. WebSearch reconfirmed the
   **Oct 30, 2026, 20:00 UTC** submission-deadline figure — no change,
   seventeenth iteration running. **New conference-date data point:** a
   WebSearch synthesis attributes **"June 21–25, 2027, Bellevue, WA"** to
   "the official MLSys website" (unverifiable directly, still
   `EGRESS_BLOCKED`) — a **third distinct conference-date figure** (after
   Feb-Montréal iter 28/36 and the unadopted May-Bellevue iter 45), diverging
   on exact dates but **agreeing with the May-Bellevue figure on location**
   — now 2-of-3 secondary sources point to Bellevue rather than Montréal;
   still no primary-source confirmation of either. **No change to the Oct 30,
   2026 estimate.** Now **~4.4 weeks out**; the T-4-week WRITING-mode
   governor (~Oct 2–3, 2026) is **effectively here** — at the current pace it
   is being crossed this iteration or next, before Maria answers
   sub-question #2 (venue/timing). **Not a tier-1 event** — no scoop, no
   confirmed deadline change (only further softening of a conference-
   location/date discrepancy, not the deadline), no decision only Maria can
   make; no `NEEDS_INPUT.md` change. **Non-empty iteration** (seven new
   citations, three genuinely useful — one directly on Maria's own model
   class, one new fix mechanism, one new independent domain — one lineage
   precursor, three filed to avoid re-discovery) — empty-iteration counter
   stays at 0 of 3. Full detail: `docs/research/digests/2026-09-27.md`.
1. 2026-09-26 (iter 48): no scope-narrowing decision — the five sub-questions
   reserved for Maria are still unanswered; `DECISIONS.md`, `NEEDS_INPUT.md`,
   and `docs/research/from-science/` unchanged since 2026-09-07/2026-09-17/
   2026-09-24 (re-checked this iteration). Local `main` was in a detached-HEAD
   state pinned exactly on `origin/main`'s tip (`455e137`, iter 47's own
   commit) at the start of this iteration — same harmless pattern noted many
   times before, not drift; checked out `main` and fast-forwarded, nothing
   lost. Did iteration 47's queued next action: swept the
   **edge-VLM/medical-structured-generation angle** (last touched iter 44 —
   was the most stale of the four rotating angles). Seven new-to-`seen.json`
   items. Three genuinely useful finds, all from the VLA/embodied-robotics
   side rather than medical-VLM: **"Embodied Foundation Models at the Edge: A
   Survey of Deployment Constraints and Mitigation Strategies"** (arXiv
   2603.16952) — differentiate-from/cite, its "Deployment Gauntlet" framing
   (deployment is a systems co-design problem, not model compression) echoes
   the harness-OS thesis's own motivation almost verbatim, scoped to
   VLA/action models rather than LLM/VLM structured-text decoding. **"Embodied
   .cpp: A Portable Inference Runtime of Embodied AI Models on Heterogeneous
   Robots"** (arXiv 2607.02501) — differentiate-from/cite, **headline find**:
   a llama.cpp-shaped portable C++ runtime purpose-built for VLA/world-
   action-model deployment across heterogeneous robot hardware — the closest
   "harness/runtime-OS for embodied edge AI" analog found in 48 iterations of
   sweeping, structurally parallel to bigos's own ambition but for the
   action-model family, not language models; not a scoop (no structured text
   to grammar-mask in this family). **"Jetson-PI: Towards Onboard Real-Time
   Robot Control via Foresight-Aligned Asynchronous Inference"** (arXiv
   2607.12659) — differentiate-from/cite, real Jetson Orin onboard VLA
   deployment with no edge-cloud split, real-time-control-loop-deadline
   framing directly relevant to BRIEF.md's Stage-2 theme; VLA not structured
   text, no grammar/mask angle. Four more items scored **ignore**: one
   accuracy/bias-focused medical-VLM paper (**"Medical Context Distorts
   Decisions in Clinical Vision Language Models,"** arXiv 2605.17436,
   continuing the standing pattern that this literature has zero systems-cost
   competitors) and three more edge-cloud-split VLA papers (**EcoVLA** arXiv
   2608.15502, **RAPID** arXiv 2603.07949, **RoboECC** arXiv 2603.20711),
   all the same DynaSplit-shaped weakens-not-serves-on-device-privacy pattern
   as CoVSpec/INAR-VL/LAST — that cluster is now six members on file.
   **MLSys 2027:** the standing CFP/Dates/OpenReview targets are all still
   `EGRESS_BLOCKED`; **new tooling note: direct `arxiv.org` WebFetch is now
   also `EGRESS_BLOCKED` this session** (abstracts were still obtainable via
   WebSearch snippets, so this didn't block the sweep). Tried the queued
   program-committee/reviewer-invitation search angle — nothing dated for
   2027 specifically surfaced. WebSearch reconfirmed the **Oct 30, 2026,
   20:00 UTC** submission-deadline figure via secondary aggregator (no
   change) — **but a new aggregator explicitly states MLSys 2027 conference
   dates/venue "have not yet been announced,"** which doesn't resolve iter
   47's Montréal/Feb-vs-Bellevue/May discrepancy but does newly suggest
   neither specific figure should be treated as confirmed — softens rather
   than sharpens that discrepancy. **No change to the Oct 30, 2026 estimate.**
   Now **~4.7 weeks out, sixteenth iteration running inside the brief's "<8
   weeks away" threshold**; the T-4-week WRITING-mode governor (~Oct 2–3,
   2026) is now **under a week away** — at the current pace this governor
   will almost certainly be crossed before Maria answers sub-question #2
   (venue/timing). **Not a tier-1 event** — no scoop, no confirmed deadline
   change (only a conference-location non-confirmation), no decision only
   Maria can make; no `NEEDS_INPUT.md` change. **Non-empty iteration** (seven
   new citations, three genuinely useful — one survey directly echoing the
   harness-OS thesis's own framing, one runtime/harness analog for the
   embodied-AI family, one on-device real-time-control-loop precedent, four
   items filed to avoid re-discovery) — empty-iteration counter stays at 0 of
   3. Full detail: `docs/research/digests/2026-09-26.md`.
1. 2026-09-25 (iter 47): no scope-narrowing decision — the five sub-questions
   reserved for Maria are still unanswered; `DECISIONS.md`, `NEEDS_INPUT.md`,
   and `docs/research/from-science/` unchanged since 2026-09-07/2026-09-17
   (re-checked this iteration). Local `main` matched `origin/main` exactly
   at the start of this iteration (`284346d`, iter 46's own commit) — no
   drift, no stale checkout. Did iteration 46's queued next action: swept
   the **harness-OS-thesis angle** (last touched iter 43 — was the most
   stale of the four rotating angles). Four new-to-`seen.json` items, one a
   genuine headline find: **"Edge-Inference Governors Need Memory-Clock
   State"** (arXiv 2606.16106) — differentiate-from/cite, a deadline-aware
   DVFS-governor paper showing that ignoring the memory clock (EMC) causes
   an edge governor to miss 25–28% of cycles at tight deadlines on real
   **Jetson Orin** hardware (vs. ≤0.9% misses with an EMC-aware refit);
   strongest harness-OS-thesis citation since DVFSLM (iter 39), extends the
   DVFS sub-thread (now 4 papers deep: DVFSLM 2609.13153, 2507.02135,
   2604.15357, this one) with a governance factor DVFSLM's own mechanism
   doesn't model — not a scoop of G1 (different mechanism layer). Three
   more differentiate-from/cite telemetry-methodology precedents:
   **"Transferable Latency Prediction for Fast LLM Screening on
   Heterogeneous Edge Devices"** (arXiv 2607.21602, real Pixel/Jetson
   Nano/Orange Pi 5 Pro/RTX 3090 hardware, DVFS+thermal-aware latency
   prediction for model selection); **"Triple-Bottom-Line Sustainability of
   Language Models for Edge AI"** (arXiv 2609.00665, Holistic Sustainability
   Score comparing SLMs vs. quantized LLMs, real GPU energy measurement, no
   edge-CPU hardware); **"From Retrieved Context to Runtime Control:
   Adaptive Compression for Edge-based RAG"** (arXiv 2608.19535, real
   Jetson AGX Thor, a runtime-control mechanism adapting to live device
   telemetry — different task (RAG) but the same contribution *shape* as
   G1's own runtime-adaptive-to-device-state mechanism). Three items filed
   **ignore** to avoid re-discovery, continuing the standing pattern that
   cloud-scale/orchestration-level and distributed-multi-node papers aren't
   competitors: a cloud-native distributed-systems research-agenda survey
   (2604.17227), a geo-distributed multi-edge-node context-management
   system (DisCEdge, 2511.22599), and a multi-LLM edge-orchestration survey
   (2507.00672). **MLSys 2027:** tried the OpenReview 2027 group page
   directly again — still `EGRESS_BLOCKED`; new angle this iteration —
   searched for the named organizers (Zhihao Jia, Tianqi Chen, Ameet
   Talwalkar) — surfaced real primary-source tweets (not secondary-
   aggregator synthesis) from Zhihao Jia and Jeff Dean, but about the
   **MLSys 2026** cycle specifically (deadline Oct 30, 2025; conference May
   2026, Seattle), not 2027. **No change to the Oct 30, 2026 submission-
   deadline estimate** (the pattern is now corroborated by a quasi-primary
   source for the first time, not contradicted), but this **newly flags the
   standing "Feb 16–23, 2027, Montréal" conference-*date*/location figure as
   a live discrepancy** — if MLSys runs in May as a matter of course (as
   these tweets show for 2024 and 2026), the previously-not-adopted
   secondary-source "May 17–22, Bellevue, WA" claim (iter 45) now looks more
   plausible than the Montréal/February figure; neither confirmed by a
   primary 2027 source. Now **~5.0 weeks out, fourteenth iteration running
   inside the brief's "<8 weeks away" threshold**; the T-4-week WRITING-mode
   governor (~Oct 2–3, 2026) is now **~7–8 days away** — at the current pace
   this governor will almost certainly be crossed before Maria answers
   sub-question #2 (venue/timing). **Not a tier-1 event** — no scoop, no
   confirmed deadline change (only a conference-location discrepancy
   newly flagged, not the deadline), no decision only Maria can make; no
   `NEEDS_INPUT.md` change. **Non-empty iteration** (four new citations, one
   a genuine headline find extending the DVFS sub-thread, three items filed
   to avoid re-discovery, one conference-date discrepancy newly flagged) —
   empty-iteration counter stays at 0 of 3. Full detail:
   `docs/research/digests/2026-09-25.md`.
1. 2026-09-24 (iter 46): no scope-narrowing decision — the five sub-questions
   reserved for Maria are still unanswered; `DECISIONS.md`, `NEEDS_INPUT.md`,
   and `docs/research/from-science/` unchanged since 2026-09-07/2026-09-17
   (re-checked this iteration, `DECISIONS.md` tail re-read, nothing dated
   after 2026-09-07). Local `main` matched `origin/main` exactly at the start
   of this iteration (`e689e19`, iter 45's own commit) — no drift, no stale
   checkout. Did iteration 45's queued next action: swept the **algorithmic
   mask-speedup cluster** (last touched iter 42 — no new member, third
   consecutive nothing-new sweep after iters 33/38 — was the most stale of
   the four rotating angles). One new-to-`seen.json` item: **DOMINO** / "Guiding
   LLMs The Right Way: Fast, Non-Invasive Constrained Generation" (arXiv
   2403.06988, ETH Zurich SRI Lab, Feb 2024) — differentiate-from, a
   lineage/precursor citation for the cluster (same non-numbered-member role
   as 2502.05111/SynCode): subword-aligned constrained decoding via
   precomputation + speculative decoding, near-zero overhead and up to 2x
   speedup, also fixing subword-vocabulary misalignment that biases naive
   masking toward valid-but-suboptimal outputs; predates the 2026 cluster by
   over two years, a fourth "foundational-precursor-found-late" case for this
   cluster. Not counted as a 12th member — **cluster stays at 11, now a
   fourth consecutive sweep (iters 33/38/42/46) with no new numbered
   entrant** — flagged plainly as a strengthening possible-saturation signal,
   not a conclusion (see digest for the full reasoning: more likely this loop
   has found the cluster's core members than the area itself going quiet).
   DOMINO is also flagged as dual-relevant to the correctness-tax cluster's
   fix-side (a second distributional-bias mechanism alongside GAD's
   sampling-bias account) but not filed there this iteration since that
   cluster was just swept iter 45. **One data-hygiene fix** (same class as
   iters 22/29/31/32/33/36/38/41/42/43/44/45): **arXiv 2608.10137** ("The
   Parser Already Knows: Lightweight Bias Correction in Constrained
   Decoding," UCLA) — bare since iter 1-2, never annotated — now annotated:
   a lightweight offline logit correction using parser/lexer state to
   restore probability mass rigid masking removes, correctness-tax
   cluster's fix-side, a fifth independent mechanistic account/fix (joins
   CRANE, GAD, AprAD, DCCD). No cluster-count change. **MLSys 2027:** tried
   `mlsys.org/Conferences/2027/CallForResearchPapers` (standing target) and
   `x.com/mlsysconf` (queued new target, the organizer's own account) —
   both still `EGRESS_BLOCKED`; a `site:x.com OR site:twitter.com` WebSearch
   surfaced only MLSys 2024/2025 organizer posts, nothing dated 2027; a
   dblp/accepted-papers search confirmed (via secondary summary only) the
   2027 cycle has no accepted-papers list yet, consistent with the standing
   estimate but not a primary-source confirmation. WebSearch's own synthesis
   continues to state Oct 30, 2026, 20:00 UTC, attributing it to "the
   official MLSys website" — the search engine's own claim, not something
   this session could verify directly (CFP page still `EGRESS_BLOCKED`) —
   **no change to the Oct 30, 2026 estimate.** Now **~5.1 weeks out,
   thirteenth iteration running inside the brief's "<8 weeks away"
   threshold**; the T-4-week WRITING-mode governor (~Oct 2–3, 2026) is now
   **under 9 days away** — at the current pace this governor will very
   likely be crossed before Maria answers sub-question #2 (venue/timing),
   flagged again, still not a loop-authored gate. **Not a tier-1 event** —
   no scoop (DOMINO is a 2024 precursor, not an active competitor), no
   confirmed deadline change, no decision only Maria can make; no
   `NEEDS_INPUT.md` change. **Non-empty iteration** (one new citation, one
   data-hygiene fix, one cluster-saturation observation) — empty-iteration
   counter stays at 0 of 3. Full detail: `docs/research/digests/
   2026-09-24.md`.
1. 2026-09-23 (iter 45): no scope-narrowing decision — the five sub-questions
   reserved for Maria are still unanswered; `DECISIONS.md`, `NEEDS_INPUT.md`,
   and `docs/research/from-science/` unchanged since 2026-09-07/2026-09-17.
   Local `main` matched `origin/main` exactly at the start of this iteration
   (`6c9ece1`, iter 44's own commit) — no drift, no stale checkout; also
   confirmed no commits landed outside `docs/research/` since 2026-09-07
   other than `NEEDS_INPUT.md` itself. Did iteration 44's queued next action:
   swept the **correctness-tax cluster** (last touched iter 41 — AprAD found
   — was the most stale of the four rotating angles). Three new-to-
   `seen.json` items: **headline find "Constrained Decoding Eliminates
   Structural Failures in Small LLMs but Reveals a Scale-Dependent Semantic
   Gap"** (arXiv 2609.23742, submitted 2026-09-20) — differentiate-from/cite,
   the most directly on-point correctness-tax paper yet for Maria's own
   sub-1B–4B regime: benchmarks 0.6B–4B models with a two-axis (structural
   vs. semantic) evaluation close to what the harness-OS paper's own
   correctness-grading gate needs to do; constrained decoding eliminates
   structural failures entirely but leaves a scale-dependent semantic gap
   (instruction-semantic failures CD-resistant, type-coercion failures
   CD-rescuable). **"Capacity, Not Format: Rethinking Structured Reasoning
   Failures"** (arXiv 2606.09410) — differentiate-from/cite, reframes format
   cost as capacity-dependent rather than inherent, converges with "Lost in
   Space" (iter 42) and the headline find above on "weaker/smaller models pay
   more," cloud-model comparison (Sonnet/Haiku), no edge angle. **"Beyond
   Fluent Generation: A CPU Reliability Benchmark for MCP-Style Tool Calling
   in Sub-2B Small Language Models for Edge Deployment"** (arXiv 2609.07370)
   — differentiate-from/cite, a genuine sub-2B + CPU-edge + tool-calling
   reliability data point (includes Qwen2.5-0.5B/1.5B, overlapping Maria's
   own 005 model set) but tests **native decoding + a recovery parser, not
   constrained decoding** — evidence *for* the constrained-decoding problem
   statement (0.5% raw JSON validity lab-wide without it) rather than a
   competitor. **Two data-hygiene fixes** (same class as iters 22/29/31/32/
   33/36/38/41/42/43/44): **arXiv 2605.26128** ("The Constraint Tax," bare
   since iter 1 despite already being discussed in STATUS.md's own
   "Correctness-tax gap" section since iter 7/9) — now annotated, an
   independent naming-convergence with the cluster's own "correctness tax"
   term (not a citation relationship, same phenomenon named twice). **arXiv
   2603.03305** (DCCD itself, bare since iter 9 despite being referenced by
   name elsewhere since iter 9) — now annotated. **One false lead caught
   before duplication:** arXiv 2606.25605 ("Constraint Tax in Open-Weight
   LLMs" / Tool Suppression) kept resurfacing in searches as if new — it is
   already tracked (annotated iter 31) but filed under the
   `https://arxiv.org/pdf/2606.25605` key rather than the `.../abs/...` form
   used almost everywhere else in `seen.json`; checked, confirmed not new,
   nothing added — flagged as a standing URL-format-inconsistency risk for
   future sweeps rather than "fixed" (fixing it would mean either duplicating
   the entry under both keys or renaming the existing key, neither clearly
   safer than leaving a note in the digest). **MLSys 2027:**
   `mlsys.org/Conferences/2027/Dates` (new direct target) still
   `EGRESS_BLOCKED`; tried the queued alternate angles (organizer social
   posts on x.com/twitter.com/linkedin.com, MLSys 2026 closing-remarks page)
   — both surfaced only MLSys **2026** figures, nothing dated for 2027. New
   unadopted secondary-source data point: an aggregator states MLSys 2027
   conference dates as "May 17–22, Bellevue, WA," conflicting with the
   standing "Feb 16–23, 2027, Montréal" figure (iter 28/36) — likely the same
   kind of year-mix-up already seen in secondary sources, **not adopted**,
   filed as a new entry in the conference-dates spread (distinct from the
   submission-deadline figure, which stayed stable). **No change to the
   Oct 30, 2026 estimate.** Now **~5.3 weeks out, twelfth iteration running
   inside the brief's "<8 weeks away" threshold**; the T-4-week WRITING-mode
   governor (~Oct 2–3, 2026) is now **~9 days away** — flagged again, still
   not a loop-authored gate. **Not a tier-1 event** — no scoop, no confirmed
   deadline change, no decision only Maria can make; no `NEEDS_INPUT.md`
   change. **Non-empty iteration** (three new citations, one a genuine
   headline find directly on point for the correctness-grading-gate design,
   two data-hygiene fixes, one false lead caught before duplication) —
   empty-iteration counter stays at 0 of 3. Full detail:
   `docs/research/digests/2026-09-23.md`.
1. 2026-09-22 (iter 44): no scope-narrowing decision — the five sub-questions
   reserved for Maria are still unanswered; `DECISIONS.md`, `NEEDS_INPUT.md`,
   and `docs/research/from-science/` unchanged since 2026-09-07/2026-09-17.
   Local `main` was detached HEAD pinned exactly on `origin/main`'s tip
   (`d05b877`, iter 43's own commit) at the start of this iteration — same
   harmless detached-HEAD pattern noted in prior iterations, not drift;
   checked out `main` and fast-forwarded, nothing lost, nothing new folded
   in. Did iteration 43's queued next action: swept the
   **edge-VLM/medical-structured-generation angle** (last touched iter 40,
   was the most stale of the four rotating angles). Six new-to-`seen.json`
   items. **Headline find: SCOPE** (arXiv 2606.02951, ACM/IEEE HRI 2026) —
   differentiate-from/cite, a modular edge-deployed natural-language camera
   agent (PTZ control + scene understanding) pairing Qwen3 SLMs with
   Moondream/Qwen VLMs, real physical edge hardware, robotics venue — the
   strongest edge-robotics-VLM precedent found since Lite VLA (iter 40),
   this time for perception/camera-control rather than action; not a
   mechanism competitor (no grammar/mask angle, no Joules telemetry). Five
   more items, all **ignore** and filed to avoid re-discovery, continuing
   the standing pattern (iters 10/17/21/24/26/28/32/36/40) that
   accuracy/robustness-focused medical- and surgical-VLM literature has
   exactly one differentiate-from paper on file and zero systems-cost
   competitors: Glaucoma OCT structured-report MM-LLM (arXiv 2510.02403),
   EndoVLA (arXiv 2505.15206, action model not structured-text generation),
   a MICCAI 2026 temporal-VLM robustness study (arXiv 2608.14262), a
   surgical-feedback NLG pipeline (arXiv 2511.15159), and SurgXBench (arXiv
   2505.10764). **Two data-hygiene fixes** (same class as iters 22/29/31/
   32/33/36/38/41/42/43): **arXiv 2505.21594** ("Fast and Cost-effective
   Speculative Edge-Cloud Decoding with Early Exits," TMLR, bare since
   iter 1) — now annotated, an edge-draft/cloud-verify speculative-decoding
   system demoed on a Unitree Go2 quadruped's on-device VLM control loop —
   differentiate-from/ignore-for-Route-A, same weakens-not-serves-
   on-device-privacy shape as DynaSplit/CoVSpec/INAR-VL/LAST/SLED, no
   cluster-count change. **DOI 10.1145/3796263.3796284** ("Edge-Deployed
   Lightweight LLMs for Medical Imaging," 2026 conf., bare since ~iter 4) —
   now annotated, a real Jetson-Orin-Nano-Super/Raspberry-Pi-4/5 dual
   medical-report pipeline with power modeling claiming 35–50% energy
   reduction under INT4 — differentiate-from/cite, the most concrete
   "edge-deployed + medical structured report + energy modeling" precedent
   on file for this angle (full text `EGRESS_BLOCKED` from dl.acm.org, so
   its "no grammar/mask angle" read is provisional, based on secondary
   summaries only). **MLSys 2027:**
   `mlsys.org/Conferences/2027/CallForResearchPapers` and
   `openreview.net/group?id=MLSys.org/2027/Conference` both tried directly
   this iteration — both still `EGRESS_BLOCKED`. WebSearch reconfirmed the
   **Oct 30, 2026** submission-deadline figure, this time as "12:00 PM PDT"
   (≈19:00 UTC) — the same ~1-hour secondary-source quibble against the
   standing "20:00 UTC" figure first flagged iter 40, not adopted, not a new
   contradiction. **No change to the Oct 30, 2026 estimate.** Now **~5.4
   weeks out, eleventh iteration running inside the brief's "<8 weeks away"
   threshold**; the T-4-week WRITING-mode governor (~Oct 2–3, 2026) is now
   **~10 days away** — flagged again, still not a loop-authored gate. **Not
   a tier-1 event** — no scoop, no confirmed deadline change, no decision
   only Maria can make; no `NEEDS_INPUT.md` change. **Non-empty iteration**
   (one genuinely useful edge-robotics-VLM citation, five items filed to
   avoid re-discovery, two data-hygiene fixes) — empty-iteration counter
   stays at 0 of 3. Full detail: `docs/research/digests/2026-09-22.md`.
1. 2026-09-21 (iter 43): no scope-narrowing decision — the five sub-questions
   reserved for Maria are still unanswered; `DECISIONS.md`, `NEEDS_INPUT.md`,
   and `docs/research/from-science/` unchanged since 2026-09-07/2026-09-17.
   Local `main` matched `origin/main` exactly at the start of this iteration
   (`19ac52e`, iter 42's own commit) — no drift, no stale checkout. Did
   iteration 42's queued next action: swept the **harness-OS-thesis angle**
   (last touched iter 39, was the most stale of the four rotating angles).
   No direct competitor to G1's mechanism or bigos's Harness-OS frame
   surfaced. Two genuinely useful telemetry-methodology finds: **LEAF**
   (doi:10.3390/make8020048, MDPI MAKE journal, Feb 2026) — differentiate-
   from/cite, real Raspberry Pi 4/5 + Jetson Nano + desktop-GPU hardware,
   Ollama-served 4-bit models, sustainability metrics (Circular Economy
   Score, Energy-per-Token) — a fourth independent Jetson-family + new
   Raspberry-Pi data point, not on arXiv so missed by prior arXiv-only
   sweeps. **Code-SLM serving-configuration energy study** (arXiv 2412.15441)
   — differentiate-from/cite, 12 code-gen SLMs × runtime-engine/execution-
   provider combinations, serving config alone cuts energy up to 89% —
   directly on point for the harness-OS thesis's core "runtime choices
   change SLM inference cost" claim (ONNX Runtime, not llama.cpp; no
   grammar/mask angle). Three more citable as related-work anchors: Network
   Edge Inference survey (2604.22906), Collaborating Small/Large LMs survey
   (2510.13890, single anchor for the whole DynaSplit/CoVSpec/INAR-VL/LAST/
   SLED/Token-Level-Routing edge-cloud cluster), Large Models for Small
   Devices survey (2608.15693). Three edge-cloud-dependent papers filed
   **ignore** (Token Level Routing 2504.07878, Distributed Speculative
   Decoding 2512.16273, Speculation at a Distance 2606.25091) — same
   DynaSplit-shaped weakens-not-serves-on-device-privacy pattern as the
   standing cluster. One pre-2026 survey (2409.00088) filed with no
   discussion, same class as iter 32's Mobile-Foundation-Model-as-Firmware
   filing. One non-paper terminology finding logged for data hygiene only:
   a thebytedive.com blog post reporting "harness engineering" was formally
   coined by Mitchell Hashimoto (HashiCorp co-founder) in Feb 2026 —
   industry-wide usage distinct from bigos's own "Harness OS" framing, not
   a competitor. **EffGen follow-up check** (named candidate, iter 8):
   confirmed ICML 2026 acceptance, PyPI tooling additions only — clean
   negative result, no new mechanism or edge-hardware measurement, earlier
   read still holds. **Data-hygiene fix** (same class as iters 22/29/31/32/
   33/36/38/41): **ELIB/MBU** (arXiv 2508.11269) had sat bare-date-only since
   iter 1 — now annotated (benchmarking tool + memory-bandwidth-utilization
   metric, 3 edge platforms × 5 quantized models, differentiate-from/cite,
   no cluster-count change). **MLSys 2027:**
   `mlsys.org/Conferences/2027/CallForResearchPapers` still `EGRESS_BLOCKED`
   (direct WebFetch tried again); WebSearch aggregators reconfirmed the
   **Oct 30, 2026, 20:00 UTC** deadline, submissions open Oct 10, notification
   Feb 28, 2027 (matches iter 34/35, still a three-way spread against the
   standing Nov 30/Jan 18 figures, not newly contradicted) — one new detail,
   program/conference dates "not set yet," consistent with the estimate on
   file. **No change to the Oct 30, 2026 estimate.** Now **~5.6 weeks out,
   tenth iteration running inside the brief's "<8 weeks away" threshold**;
   the T-4-week WRITING-mode governor (~Oct 3, 2026) is now **~12 days
   away** — flagged again, still not a loop-authored gate. **Not a tier-1
   event** — no scoop, no confirmed deadline change, no decision only Maria
   can make; no `NEEDS_INPUT.md` change. **Non-empty iteration** (two
   citations directly on point for the harness-OS thesis's core claim, three
   related-work survey citations, three items filed to avoid re-discovery,
   one data-hygiene fix, one clean-negative-result follow-up check, one
   terminology finding logged) — empty-iteration counter stays at 0 of 3.
   Full detail: `docs/research/digests/2026-09-21.md`.
1. 2026-09-20 (iter 42): no scope-narrowing decision — the five sub-questions
   reserved for Maria are still unanswered; `DECISIONS.md`, `NEEDS_INPUT.md`,
   and `docs/research/from-science/` unchanged since 2026-09-07/2026-09-17.
   Local `main` matched `origin/main` exactly at the start of this iteration
   (`6be8c30`, iter 41's own commit) — no drift, no stale checkout. Did
   iteration 41's queued next action: swept the **algorithmic mask-speedup
   cluster** (last touched iter 38, was the most stale of the four rotating
   angles). **No new numbered cluster member** — third consecutive sweep
   (iters 33, 38, 42) with no new entrant, cluster stays at 11 members;
   flagged as a possible saturation signal, not a conclusion. One false lead
   caught before it became a duplicate: search results repeatedly surfaced
   "GreatGramma" (17.71x offline-preprocessing speedup) — checked against
   `seen.json` and confirmed this is the *tool name* inside the
   already-tracked founding-lineage paper 2502.05111, not a new paper; no
   entry added. **GLRMask** (github.com/IsaacBreen/glrmask) — an actively
   maintained Rust/Python GCD library, AWS-CPU-server-benchmarked, no
   accompanying paper — ignore, cloud-serving-shaped tooling, logged for
   data hygiene only. A zolotukhin.ai blog post on llama.cpp/GBNF mask
   overhead on a **local, non-datacenter GPU** looked potentially useful
   (closer to the lab's own regime than the cluster's cloud-serving papers)
   but `zolotukhin.ai` is `EGRESS_BLOCKED` from this session — filed
   unscored. A related WebSearch snippet independently reports llama.cpp's
   naive grammar sampler doing a full CPU vocabulary walk per token, with a
   concrete number (~25ms/token decode on a consumer GPU) putting mask
   overhead in the same order of magnitude as decode itself on non-
   datacenter hardware — resonant with G1's own founding observation, not a
   scoop (G1's contribution is the shadow-mask pipelining fix + edge Joules
   telemetry, not the bare overhead fact). **Vectorizing the Trie / STATIC**
   (arXiv 2602.22647) — trie-to-sparse-matrix vectorization for TPU/GPU
   constrained decoding, up to 948x speedup — differentiate-from, same
   trie-masking family as the cluster's own Trie Automata member but wrong
   task domain (closed-corpus generative retrieval, not general JSON/tool-
   call schema) and wrong hardware regime (TPU/GPU accelerator); not a 12th
   member. **Two adjacent finds outside the cluster, both citable:** "Lost
   in Space" / "Finding the Right Tokens for Structured Output" (arXiv
   2502.14969, Cornell) — differentiate-from/cite for the
   **correctness-tax cluster**, a fourth "phrasing changes correctness"
   mechanism (joins schema-key-wording 2604.14862, "Your Prompt Is Not the
   Only Prompt" 2608.08254), and the only one of the four to explicitly
   flag the effect as **largest for smaller/local-laptop-scale models** —
   directly on point for Maria's own sub-1B regime. "Keyword Matters:
   Unveiling the Energy Sensitivity of On-Device LLM Prompting" (arXiv
   2607.22568) — differentiate-from/cite, real smartphone power
   measurements showing prompt wording changes decoding length and energy —
   a new mechanism family (prompt phrasing as an energy channel), same
   on-device-real-power telemetry-methodology shape as lm-Meter/Sustainable-
   LLM-Inference-for-Edge-AI. Two more filed as **ignore**: a wireless-
   transmission-energy speculative-decoding paper (2508.12590, wrong energy
   framing) and an edge-cloud split-computing paper (2511.04002, same
   DynaSplit-shaped privacy-weakening pattern as CoVSpec/INAR-VL/LAST).
   **MLSys 2027:** `mlsys.org/Conferences/2027/CallForResearchPapers` (both
   direct WebFetch and WebSearch) still `EGRESS_BLOCKED`; this iteration's
   new target (a "call for reviewers"/social-announcement search) surfaced
   nothing dated. WebSearch reconfirmed the **Oct 30, 2026, 20:00 UTC**
   submission-deadline figure — no change — and re-surfaced the "Oct 10
   opens / Feb 28 2027 notification" pairing (matching iter 34/35's figures,
   not a new fifth data point). **No change to the Oct 30, 2026 estimate on
   file.** Now **~5.7 weeks out, ninth iteration running inside the brief's
   "<8 weeks away" threshold**; the T-4-week WRITING-mode governor (~Oct 3,
   2026) is now **~13 days away** — flagged again, still not a loop-authored
   gate. **Not a tier-1 event** — no scoop, no confirmed deadline change, no
   decision only Maria can make; no `NEEDS_INPUT.md` change. **Non-empty
   iteration** (one citation directly on point for the sub-1B/correctness-
   tax framing, one adjacent telemetry-methodology citation, one false lead
   caught before duplication, two filed to avoid re-discovery) —
   empty-iteration counter stays at 0 of 3. Full detail:
   `docs/research/digests/2026-09-20.md`.
1. 2026-09-19 (iter 41): no scope-narrowing decision — the five sub-questions
   reserved for Maria are still unanswered; `DECISIONS.md`, `NEEDS_INPUT.md`,
   and `docs/research/from-science/` unchanged since 2026-09-07. Local `main`
   was 2 commits behind `origin/main` at the start of this iteration (iters
   39/40's own commits) — this session's own stale starting checkout, not
   new remote work; fast-forwarded via `git checkout main && git pull`,
   nothing lost. Did iteration 40's queued next action: swept the
   **correctness-tax cluster** (last touched iter 37, was the most stale of
   the four rotating angles). Six new-to-`seen.json` items, none competitive
   with G1/the bigos mechanism layer, all reinforcing the cluster's own
   problem/fix-side taxonomy: **Approximately Aligned Decoding / AprAD**
   (arXiv 2410.01103, Amazon, Oct 2024) — differentiate-from/cite, a
   speculative-decoding-inspired resampling fix predating the 2026 cluster
   by over a year, same "foundational-precursor-found-late" shape as
   CRANE/GAD/SynCode/PSC, a fourth independent fix-side mechanism.
   **Quantifying the Impact of Structured Output Format via Causal
   Inference** (arXiv 2509.21791, EACL 2026 Findings) — differentiate-from/
   cite, a methodologically stronger (causal, not correlational)
   confirmation of the standing problem. **Empirical Study for Structured
   Output Control... for Software Engineering** (arXiv 2606.09395) —
   differentiate-from/cite, independent SE-domain replication of the
   validity-vs-correctness split. **Where vs What** (arXiv 2608.25358) —
   differentiate-from/cite, **methodology directly reusable** for the
   harness-OS paper's own planned correctness-grading gate (its
   Structure-Content Decomposition separates placement errors from value
   errors). **Your Prompt Is Not the Only Prompt** (arXiv 2608.08254) —
   differentiate-from/cite, a third mechanism family (schema descriptions as
   a second, sometimes-dominant instruction channel), joins schema-key-
   wording (2604.14862). **The Architecture of Errors** (arXiv 2605.30628)
   — ignore, general LLM-reliability theory, not structured-output-specific,
   filed to avoid re-discovery. Also fixed two bare-date-only `seen.json`
   entries (same data-hygiene class as iters 22/29/31/32/33/36/38):
   **JSONSchemaBench** (arXiv 2501.10868) — differentiate-from/cite,
   methodology-relevant (benchmarks llama.cpp's own grammar engine among
   six frameworks, no edge-energy axis); **Attention Meets Reachability**
   (arXiv 2603.05540) — differentiate-from, a theoretical lineage paper for
   the *algorithmic mask-speedup cluster* (not correctness-tax), no
   cluster-count change. **MLSys 2027:**
   `mlsys.org/Conferences/2027/CallForResearchPapers`, `mlsys-deadlines.
   github.io`, and `mldeadlines.com/conference/mlsys-2027/` all still
   `EGRESS_BLOCKED`; tried a "call for reviewers/program committee" search
   as a new target per iter 40's queued action — nothing dated surfaced.
   WebSearch reconfirmed the **Oct 30, 2026** submission-deadline figure —
   **no change** — and repeated iter 40's "submissions open Oct 10" and
   "author reviews Jan 18, 2027" secondary claims, no new contradiction, no
   new convergence. Now **~5.9 weeks out, eighth iteration running inside
   the brief's "<8 weeks away" threshold**; the T-4-week WRITING-mode
   governor (~Oct 3, 2026) is now **~14 days away** — flagged again per
   standing practice, still not a loop-authored gate. **Not a tier-1
   event** — no scoop, no confirmed deadline change, no decision only Maria
   can make; no `NEEDS_INPUT.md` change. **Non-empty iteration** (six new
   citations, one a genuine fourth fix-side mechanism, one flagged as
   directly reusable for the correctness-grading-gate design, two
   data-hygiene fixes) — empty-iteration counter stays at 0 of 3. Full
   detail: `docs/research/digests/2026-09-19.md`.
1. 2026-09-18 (iter 40): no scope-narrowing decision — the five sub-questions
   reserved for Maria are still unanswered; `DECISIONS.md`, `NEEDS_INPUT.md`,
   and `docs/research/from-science/` unchanged since 2026-09-07. Local `main`
   matched `origin/main` exactly at the start of this iteration (`7a43e15`,
   iter 39's own commit) — no drift, no stale checkout. Did iteration 39's
   queued next action: swept the **edge-VLM/medical-structured-generation
   angle** (last touched iter 36, was the most stale of the four rotating
   angles). Ten new-to-`seen.json` items. Six accuracy/hallucination-focused
   medical-VLM papers (ARCD 2512.17189, VALOR 2512.16201, Anatomy-VLM
   2511.08402, a 34-page adaptation-strategies survey 2506.18378, two
   cloud-API clinical-report tools 2509.06033/2509.13590) all scored
   **ignore**, continuing the standing pattern (iters 10/17/21/24/26/28/32/
   36) that this literature area has exactly one differentiate-from paper on
   file and zero systems-cost competitors. One training-time diffusion
   image-*synthesis* paper (2603.10519) filed to avoid re-discovery — wrong
   task entirely (generation, not report/interpretation). **Two genuinely
   useful finds:** **Lite VLA** (arXiv 2511.05642, "Efficient
   Vision-Language-Action Control on CPU-Bound Edge Robots") —
   differentiate-from/cite, the strongest "robot that sees, on-device, real
   hardware" precedent found yet for the paper's edge×robotics motivation
   (GPS-denied/no-cloud-connectivity framing, simultaneous movement +
   reasoning) — not a mechanism competitor (VLA action model, not
   structured text/JSON generation, no grammar/mask angle). **"Sometimes
   Painful but Certainly Promising"** (arXiv 2503.09114, ACM TECS) — cite, a
   general (text) on-device SLM feasibility/trade-off characterization
   across quantization schemes, same methodological shape as 2511.11624 and
   iter 27's Sustainable-LLM-Inference-for-Edge-AI citation, another
   telemetry-methodology precedent. Also filed one 2023 paper ("Mobile
   Foundation Model as Firmware," arXiv 2308.14363) with no digest
   discussion — predates the current SLM-harness literature wave by two
   years, not worth analysis time. **MLSys 2027:**
   `mlsys.org/Conferences/2027/CallForResearchPapers` and (a new target this
   cluster of iterations) `mldeadlines.com/conference/mlsys-2027/` both
   still `EGRESS_BLOCKED`; WebSearch reconfirmed the **Oct 30, 2026**
   submission-deadline figure (no change), this time phrased as "12:00 PM
   PDT" (≈19:00 UTC) vs. iter 37's "20:00 UTC" — a ~1-hour secondary-source
   quibble, not adopted. Also surfaced a **new notification-adjacent claim,
   "author reviews available Jan 18, 2027"** — a third distinct figure in
   the notification-date family (alongside the standing "Nov 30, 2026" and
   the once-seen "Feb 28, 2027") — not adopted, filed as further widening of
   that specific secondary-source spread; **the submission-deadline figure
   itself has stayed stable across ten iterations running.** **No change to
   the Oct 30, 2026 estimate on file.** Now **~6.0 weeks out, seventh
   iteration running inside the brief's "<8 weeks away" threshold**; the
   T-4-week WRITING-mode governor (~Oct 3, 2026) is now **~15 days away** —
   flagged plainly this iteration that at the current pace this governor
   will likely be crossed before Maria answers sub-question #2
   (venue/timing), the sub-question that determines whether "WRITING mode"
   means anything yet. **Not a tier-1 event** — no scoop, no confirmed
   deadline change, no decision only Maria can make; no `NEEDS_INPUT.md`
   change (per Maria's standing instruction, no re-ping on the five reserved
   sub-questions). **Non-empty iteration** (ten new citations, two genuinely
   useful — one edge-robotics-VLM precedent, one general-SLM-edge
   characterization precedent) — empty-iteration counter stays at 0 of 3.
   Full detail: `docs/research/digests/2026-09-18.md`.
1. 2026-09-17 (iter 39): no scope-narrowing decision — the five sub-questions
   reserved for Maria are still unanswered; `DECISIONS.md`, `NEEDS_INPUT.md`,
   and `docs/research/from-science/` unchanged since 2026-09-07. Local `main`
   matched `origin/main` exactly at the start of this iteration (`0312b13`,
   iter 38's own commit) — no drift, no stale checkout. Did iteration 38's
   queued next action: swept the **harness-OS-thesis angle** (last touched
   iter 35, was the most stale of the four rotating angles). Eleven new-to-
   `seen.json` items. **Headline find: DVFSLM** (arXiv 2609.13153, "DVFS for
   Small Language Model Inference on Mobile Edge Devices," City University of
   Hong Kong, companion MobiSys'26 workshop paper) — a runtime DVFS governor
   purpose-built for SLM decode workloads, dynamically adjusting CPU/GPU/EMC
   frequencies to minimize energy-per-token while meeting a token-generation
   deadline. **Differentiate-from/cite** — the strongest harness-OS-thesis
   find in ~10 iterations of sweeping this angle: the first paper found that
   sits at the actual systems/runtime layer (frequency-scaling governance)
   rather than the agent-orchestration/application layer every prior
   harness-taxonomy find (iters 30-35) sat at. **Not a scoop** of G1 (power/
   frequency governance is a different mechanism from constrained-decoding
   mask-cost tax + shadow-mask pipelining) or of the bigos umbrella (one
   candidate module, not the whole frame) — but it sharpens the paper's need
   to state its systems-layer position precisely, same class of finding as
   AgentStop (iter 32) and lm-Meter (iter 27). Two adjacent DVFS-thread
   citations (arXiv 2507.02135, a characterization study motivating DVFSLM's
   gap; arXiv 2604.15357, a frequency-aware latency-*estimation* paper,
   distinct role from DVFSLM's governance). Eight more items, all
   **ignore**, continuing the standing pattern that "agent harness"
   literature stays at the application/orchestration level: another
   harness-taxonomy survey (TMLR 2026 submission, ETCLOVG taxonomy, likely
   the same underlying project as the already-tracked 2605.29682) plus seven
   more application-level harness papers (agent-system survey, failure-
   localization taxonomy, category-theory framing, code-as-harness pattern,
   determinism-enforcing framework, a harness-construction foundry, a
   harness-vs-model ablation study) — none touch edge hardware or systems
   cost. **MLSys 2027:** `mlsys.org/Conferences/2027/CallForResearchPapers`,
   `mlsys.org/Conferences/2027/Dates`, and `mlsys-deadlines.github.io` all
   still `EGRESS_BLOCKED`; WebSearch reconfirmed the Oct 30, 2026, 20:00 UTC
   submission-deadline figure (no change) and repeated the unresolved "Oct 10
   opens / Feb 28 2027 notification" secondary-source pairing first seen iter
   34/35 (still doesn't match the Nov 30, 2026 notification figure iter
   28/36 reaffirmed) — same spread as before, not a new contradiction, **no
   change to the Oct 30–31, 2026 estimate on file.** Now **~6.1 weeks out,
   sixth iteration running inside the brief's "<8 weeks away" threshold**;
   the T-4-week WRITING-mode governor (~Oct 3, 2026) is now **~16 days
   away**. **Not a tier-1 event** — no scoop, no confirmed deadline change,
   no decision only Maria can make; no `NEEDS_INPUT.md` change. **Non-empty
   iteration** (one genuine systems-layer citation directly relevant to the
   harness-OS thesis, two adjacent DVFS-thread citations, eight
   application-layer surveys filed to avoid re-discovery) — empty-iteration
   counter stays at 0 of 3. Full detail: `docs/research/digests/
   2026-09-17.md`.
1. 2026-09-16 (iter 38): no scope-narrowing decision — the five sub-questions
   reserved for Maria are still unanswered; `DECISIONS.md`, `NEEDS_INPUT.md`,
   and `docs/research/from-science/` unchanged since 2026-09-07. Local `main`
   was 7 commits behind `origin/main` at the start of this iteration (a
   detached-HEAD checkout landing exactly on iter 37's own commit `2172df0`
   — this session's own stale starting state, not new remote work);
   fast-forwarded via `git checkout main && git pull`, nothing lost, nothing
   new folded in. Did iteration 37's queued next action: swept the
   **algorithmic mask-speedup cluster** (last touched iter 33, was the most
   stale of the four rotating angles). **No new cluster member** — the
   cluster stays at 11 numbered members. Four new-to-`seen.json` items, none
   competitive: **SLED** (arXiv 2506.09397, "A Speculative LLM Decoding
   Framework for Efficient Edge Serving") — local draft + edge-server
   verify speculative decoding, tested on **Jetson Orin Nano and Raspberry
   Pi 4B/5** (2.2x throughput, 2.8x capacity vs. baselines); no grammar/mask
   angle, verification leaves the device — same weakens-not-serves
   on-device-privacy shape as DynaSplit/CoVSpec/INAR-VL/LAST — scored
   **ignore** for Route A framing, but a genuinely useful **third
   independent Jetson Orin Nano** data point (after 2511.11624 iter 31,
   EdgeCIM iter 32) and first Raspberry Pi 4B source for the still-postponed
   Jetson-purchase sub-question. **The Extrapolation Cliff...** (arXiv
   2605.08737, NTU Singapore) — a training-time (on-policy-distillation)
   format-collapse threshold for structured/JSON outputs, a third layer
   distinct from both standing clusters (decode-time mask cost;
   inference-time accuracy loss) — scored **ignore**, filed only to avoid
   re-discovery. **ARCS** (arXiv 2603.29068) — analog-circuit-synthesis
   generator using grammar-constrained decoding as one component; wrong
   domain entirely (EDA, not edge/robotics/medical) — scored **ignore**.
   **Data-hygiene fix** (same class as iters 22/29/31/32/33/36): **"Future
   Validity is the Missing Statistic..." / Φ-estimation** (arXiv 2605.07698)
   had sat in `seen.json` since iteration 3 as a bare date despite already
   being described in that iteration's own digest — now annotated
   (speculative-decoding + grammar sampling-correctness paper, different
   mechanism from G1, never a numbered cluster member) — **differentiate-
   from**, no cluster-count change. **MLSys 2027:**
   `mlsys.org/Conferences/2027/CallForResearchPapers` and the OpenReview
   2027 group page both still `EGRESS_BLOCKED`; WebSearch reconfirmed the
   "Oct 30, 2026, 20:00 UTC" figure iter 37 first surfaced (no new
   contradiction) — **no change to the Oct 30–31, 2026 estimate on file.**
   Now **~6.3 weeks out, fifth iteration running inside the brief's "<8
   weeks away" threshold**; the T-4-week WRITING-mode governor (~Oct 3,
   2026) is now **~17 days away**. **Not a tier-1 event** — no scoop, no
   confirmed deadline change, no decision only Maria can make; no
   `NEEDS_INPUT.md` change. **Non-empty iteration** (one data-hygiene fix,
   three new citations filed to avoid re-discovery, one genuinely useful
   third-Jetson-Orin-Nano hardware data point) — empty-iteration counter
   stays at 0 of 3. Full detail: `docs/research/digests/2026-09-16.md`.
1. 2026-09-15 (iter 37): no scope-narrowing decision — the five sub-questions
   reserved for Maria are still unanswered; `DECISIONS.md`, `NEEDS_INPUT.md`,
   and `docs/research/from-science/` unchanged since 2026-09-07/2026-09-09.
   Local `main` matched `origin/main` exactly at the start of this iteration
   (`cc01d10`, iter 36's own commit) — no drift, no stale checkout. Did
   iteration 36's queued next action: swept the **correctness-tax cluster**
   (last touched iter 34, the more stale of the two options queued, vs.
   harness-OS-thesis at iter 35). Two new-to-`seen.json` items, both
   pre-2026 foundational precursors rather than new competitors: **Grammar-
   Aligned Decoding** (arXiv 2405.21047, NeurIPS 2024) — scored
   **differentiate-from**, joins the cluster's fix-side (with CRANE,
   ATLAS-RTC, DCCD) as a *second* independent mechanistic explanation for
   the correctness-tax gap — grammar-constrained decoding distorts the
   LLM's own output distribution (a sampling-bias account), distinct from
   CRANE's reasoning-token-starvation account — third "foundational-
   precursor-found-late" case after PSC (mask-speedup cluster) and CRANE
   (correctness-tax cluster) itself. **SynCode** (arXiv 2403.01632, UIUC)
   — scored **differentiate-from**, a lineage/precursor citation for the
   **algorithmic mask-speedup cluster** (same non-numbered-member role as
   2502.05111): DFA-based offline mask-store for CFG-constrained decoding,
   eliminates JSON syntax errors entirely, no edge/Joules angle. Neither
   item is a scoop or a new active-development competitor — both are
   background/lineage citations the 2026 clusters keep citing. **MLSys
   2027:** `mlsys.org/Conferences/2027/CallForResearchPapers` and the
   OpenReview 2027 group page both still `EGRESS_BLOCKED`; WebSearch
   aggregators produced a slightly more precise time-of-day figure ("Oct 30,
   2026, 20:00 UTC" vs. the "11:59 AM UTC/AoE" figure on file since iter
   5/28) — a minor secondary-source discrepancy, not adopted, filed as
   noise. **No change to the Oct 30–31, 2026 estimate on file.** Now **~6.4
   weeks out, fourth iteration running inside the brief's "<8 weeks away"
   threshold**; the T-4-week WRITING-mode governor (~Oct 3, 2026) is now
   **~18 days away**. **Not a tier-1 event** — no scoop, no confirmed
   deadline change, no decision only Maria can make; no `NEEDS_INPUT.md`
   change. **Non-empty iteration** (two new citations, one a genuine
   second-mechanism strengthening of the correctness-tax gap's causal
   story) — empty-iteration counter stays at 0 of 3. Full detail:
   `docs/research/digests/2026-09-15.md`.
1. 2026-09-14 (iter 36): no scope-narrowing decision — the five sub-questions
   reserved for Maria are still unanswered; `DECISIONS.md`, `NEEDS_INPUT.md`,
   and `docs/research/from-science/` unchanged since 2026-09-07. Local `main`
   was 5 commits behind `origin/main` at the start of this iteration (landed
   exactly on iter 35's own commit `49cefca` — this session's own checkout
   starting stale, not new remote work); fast-forwarded via `git checkout
   main && git pull`, nothing lost. Did iteration 35's queued next action:
   swept the **edge-VLM/medical-structured-generation angle** (last touched
   iter 32, the more stale of the two general-sweep options). Four new-to-
   `seen.json` items: three edge-cloud split-computing papers (**CoVSpec**
   2605.02218, **INAR-VL** 2605.18853, **LAST** 2607.27952) all scored
   **ignore** — same DynaSplit-shaped pattern, the target/large model never
   runs on-device in any of them, which weakens rather than serves an
   on-device-privacy framing. **Phase Matters** (arXiv 2606.27906, MobiSys
   Workshop '26) scored **differentiate-from/cite** — the strongest edge-VLM
   characterization-methodology precedent found yet: real phone-SoC
   (Snapdragon 8 Elite) hardware-in-the-loop thermal+energy+phase-level
   telemetry, and an independent confirmation (after Ghidorah, iter 19) that
   hetero-core speedup is phase-dependent (NPU: 1.64x prefill, only 1.18x
   decode) rather than uniform. Also fixed four bare-date-only `seen.json`
   entries left unannotated since iter 1-2 (same data-hygiene class as iters
   22/29/31/32/33): a VLM-edge survey (2502.07855, cite); a robotic-VLM
   edge-server paper (2601.14921, ignore — **and a name collision worth
   remembering: it tests on a Unitree "G1" humanoid robot, unrelated to
   bigos's own G1 module**, second such collision after HARBOR/iter 35);
   LiteVLM (2506.07416, differentiate-from — genuine on-device edge+robotics
   VLM latency optimization on NVIDIA DRIVE Thor, different mechanism axis
   than G1); and, most operationally useful, **arXiv 2603.23640** ("LLM
   Inference at the Edge... Under Sustained Load") which benchmarks **Maria's
   own hardware, iPhone 16 Pro**, showing it loses ~40% peak throughput within
   3 iterations to thermal throttling — directly relevant to BRIEF.md's
   paper-grade thermal-precondition requirement for any future on-device
   measurement session on the phone rather than the M1. **MLSys 2027:** all
   of `mlsys.org/Conferences/2027/CallForResearchPapers`, the newly-discovered
   `openreview.net/group?id=MLSys.org%2F2027%2FConference` page, and four
   secondary aggregators (`mlsys-deadlines.github.io`, `mldeadlines.com`,
   `mlciv.com`, `trybibby.com`) are all still `EGRESS_BLOCKED` — still no
   verified primary source. This iteration's WebSearch surfaced one
   internally-inconsistent noise artifact (a claimed "MLSys 2026 submission
   deadline of October 30, 2026" — not adopted, looks like a year-label
   mix-up in the search summarizer) and one new unconfirmed claim
   ("supplementary material due July 31, 2026," already past, filed as
   noise) — but also **reaffirmed** notification date Nov 30, 2026 and
   conference dates Feb 16–23, 2027, Montréal, matching iter 28's original
   figures rather than iter 34's contradictory "Feb 28, 2027" claim, which
   **narrows** the secondary-source spread iter 34/35 flagged as growing.
   **No change to the Oct 31, 2026 estimate on file.** Now **~6.7 weeks
   out**, third iteration running inside the brief's "<8 weeks away"
   threshold, T-4-week WRITING-mode governor (~Oct 3, 2026) now ~19 days
   away. **Not a tier-1 event** — no scoop, no confirmed deadline change, no
   decision only Maria can make; no `NEEDS_INPUT.md` change. **Non-empty
   iteration** (four new citations, four data-hygiene fixes, one directly
   useful hardware-thermal finding, one MLSys secondary-source contradiction
   resolved rather than deepened) — empty-iteration counter stays at 0 of 3.
   Full detail: `docs/research/digests/2026-09-14.md`.
1. 2026-09-13 (iter 35): no scope-narrowing decision — the five sub-questions
   reserved for Maria are still unanswered; `DECISIONS.md`, `NEEDS_INPUT.md`,
   and `docs/research/from-science/` unchanged since 2026-09-07. Local `main`
   matched `origin/main` exactly at the start of this iteration (`bb6a6d3`,
   iter 34's own commit) — no drift this time, the intermittent leftover
   detached-HEAD/stale-checkout pattern did not recur. Did iteration 34's
   queued next action: swept the **harness-OS-thesis angle** (last properly
   touched iter 31, briefly iter 32 — the more stale of the two live general
   angles) and did the queued CRANE citing-work check. Eight new-to-
   `seen.json` items, none competitive with bigos's own mechanism layer.
   Most operationally useful find: **"HARBOR" is now a three-way (really
   four-way, counting Harbor-Index) name collision** — arXiv 2606.08610
   ("HARBOR: A Harness Framework for Agentic Robot Reinforcement Learning,"
   a robot-RL sim-to-real pipeline-automation system) and arXiv 2609.04298
   ("Harbor Adapters and Harbor-Index," an 80+-benchmark agentic-eval
   infrastructure) are both unrelated to the already-tracked HARBOR
   (2604.20938, an agent-config Bayesian-optimization solver) — one of
   Maria's three named candidates from iter 30 needs disambiguation if it
   resurfaces in conversation; both new HARBORs scored **ignore** (wrong
   layer: RL-training-pipeline automation and eval-infrastructure
   respectively, neither an SLM inference harness or edge mechanism). Two
   terminology-precedent finds joining the harness-taxonomy-survey line:
   **"What makes a harness a harness"** (arXiv 2606.10106) — a conceptual-
   analysis paper proposing a reference definition of "agent harness" via a
   horse-tack analogy, scoped to coding agents — **cite**, a third citable
   source (with 2604.08224, 2604.18071) for how the harness-OS paper should
   define "harness" precisely; and **"Harness Engineering..."** (arXiv
   2609.00006, 11-coding-agent-system source-code study) — **ignore**, same
   coding-agent-domain pattern. **Pico** (arXiv 2509.16413) is a *training*-
   time SLM-research sandbox, not an inference/deployment harness despite
   the name-adjacent phrasing — **ignore**. Three more cascade/escalation-
   routing papers (**UCCI** 2605.18796, **Signed Rescue Routing** 2609.07786,
   **Routing, Cascades, and User Choice** 2602.09902) all scored
   **differentiate-from**, extending the standing escalation/cascade-routing
   cluster to 5 members, all cloud/GPU-serving-shaped with no edge angle.
   **CRANE citing-work check** (arXiv 2502.09061, queued by iter 34): clean
   negative result — no follow-up work found applying CRANE's reasoning-
   preserving-grammar fix to an edge/latency-sensitive setting. **MLSys
   2027:** `mlsys.org/Conferences/2027/CallForResearchPapers` still
   `EGRESS_BLOCKED`; WebSearch surfaced a new internal inconsistency among
   secondary sources — the unconfirmed "submission opens Oct 10, 2026" claim
   (first seen iter 34) now paired with a "notification Feb 28, 2027" figure
   that does **not** match the "notifications Nov 30, 2026" figure iter 28
   recorded as part of the converged estimate — a growing spread across
   secondary sources, not a confirmed change. **No change to the Oct 31,
   2026 estimate on file.** **Timing note, not a tier-1 event:** today's
   date puts that estimate at **~6.5 weeks out**, inside the brief's "<8
   weeks away" threshold for the second iteration running and closing on the
   T-4-week WRITING-mode governor (~Oct 3, 2026) — flagged again for Maria's
   own awareness given the secondary-source uncertainty is growing rather
   than resolving, but still not a loop decision or confirmed deadline
   change, so no `NEEDS_INPUT.md` action. **Not a tier-1 event** — no scoop,
   no deadline change, no decision only Maria can make; no `NEEDS_INPUT.md`
   change. **Non-empty iteration** (eight new citations, one operationally
   useful naming-collision finding, one clean negative result on the CRANE
   check) — empty-iteration counter stays at 0 of 3. Full detail:
   `docs/research/digests/2026-09-13.md`.
2. 2026-09-12 (iter 34): no scope-narrowing decision — the five sub-questions
   reserved for Maria are still unanswered; `DECISIONS.md` and
   `docs/research/from-science/` unchanged since 2026-09-07. Local `main` was
   3 commits behind `origin/main` at the start of this iteration (the same
   recurring leftover detached-HEAD/stale-checkout pattern — this session's
   own starting state, not new remote work) — fast-forwarded via `git
   checkout main && git pull`, nothing lost, nothing new folded in (the 3
   "new" commits were iter 33's own already-known content). Did iteration
   33's queued next action: swept the **correctness-tax cluster** (last
   touched iter 27, was the most stale of the three original clusters, 9
   members). Four new-to-`seen.json` items, one a genuine mechanism-level
   find: **CRANE** (arXiv 2502.09061, ICML 2025 poster) — a formal proof
   (Turing-machine-simulation argument) for *why* constrained decoding loses
   functional accuracy (a restrictive grammar denies room for the
   intermediate reasoning tokens autoregressive LMs need), plus a fix
   (grammar augmented with reasoning-preserving rules, up to 10pp accuracy
   gain) — differentiate-from, joins the cluster's fix-side (with ATLAS-RTC,
   DCCD) as the clearest mechanistic explanation on file yet for the
   correctness-tax gap this loop has cited since iter 7; predates the 2026
   cluster (Feb 2025) but was never entered until now, same
   "foundational-precursor-found-late" shape as PSC in the mask-speedup
   cluster. Three more items scored **ignore**, all name-collisions or
   paradigm-mismatches rather than competitors: **Constraint Decay: The
   Fragility of LLM Agents in Backend Code Generation** (arXiv 2605.06445,
   software-architecture constraints in agentic code-gen, not grammar/schema
   decoding — different meaning of "constraint" entirely); **DINGO:
   Constrained Inference for Diffusion LLMs** (arXiv 2505.23061, NeurIPS
   2025, same diffusion/non-autoregressive-paradigm-mismatch shape as LAVE,
   iter 25); **The LLM Data Auditor** (arXiv 2601.17717, a synthetic
   training-data quality/trustworthiness survey, unrelated to
   constrained-decoding correctness or edge systems). **MLSys 2027:**
   `mlsys.org/Conferences/2027/CallForResearchPapers` still
   `EGRESS_BLOCKED`; WebSearch again surfaced the same mixed secondary
   signals as iter 30-33 (dates "TBD" alongside an unconfirmed "submission
   opens Oct 10, 2026" claim) — **no change to the Oct 31, 2026 estimate on
   file.** **Timing note, not a tier-1 event:** today's date puts that
   estimate at ~7 weeks out — the first iteration where it falls inside the
   brief's "<8 weeks away" verify-every-iteration threshold; flagged for
   Maria's own awareness (sharpens the practical cost of sub-question #2,
   venue/timing, staying open) but not a loop decision or a confirmed
   deadline change, so no `NEEDS_INPUT.md` action. **Not a tier-1 event** —
   no scoop, no deadline change, no decision only Maria can make; no
   `NEEDS_INPUT.md` change. **Non-empty iteration** (four new citations, one
   a genuine mechanism-level strengthening of the correctness-tax gap
   argument) — empty-iteration counter stays at 0 of 3. Full detail:
   `docs/research/digests/2026-09-12.md`.
2. 2026-09-11 (iter 33): no scope-narrowing decision — the five sub-questions
   reserved for Maria are still unanswered; `DECISIONS.md` and
   `docs/research/from-science/` unchanged since 2026-09-07. Local `main` was
   2 commits behind `origin/main` at the start of this iteration (the
   recurring leftover detached-HEAD/stale-checkout pattern — this session's
   own starting state, not new remote work) — fast-forwarded via `git
   checkout main && git pull`, nothing lost. Did iteration 32's queued next
   action: swept the **algorithmic mask-speedup cluster** (last touched iter
   29, was the most stale of the three original clusters, 10 members) and
   did the two follow-ups it queued. **One new cluster member:
   Earley-Driven Dynamic Pruning / Formatron** (arXiv 2506.01151, ICML 2025
   poster) — dynamic pruning of Earley parser sets/states for CFG mask
   computation, up to 2× faster constrained decoding, no accuracy loss —
   differentiate-from, 11th member, same defining trait as the other ten
   (cloud/GPU-serving-baseline speedup, no edge/Joules/single-stream angle).
   **AgentStop follow-up:** no citing/follow-up work found yet (expected,
   paper is only ~4 months old); fetched its GitHub repo
   (`brave-experiments/AgentStop`) directly and confirmed no decode-time/
   kernel-level mechanism exists beyond the log-prob-based agent-loop
   early-termination policy already on file — the "one layer down" framing
   in "Known competitive pressure" now checked against the source, not just
   the abstract. **MLSys 2027:** tried three targets this iteration
   (`mlsys.org/Conferences/2027/CallForResearchPapers`, an OpenReview
   submission-site URL surfaced by WebSearch, and `mlsys-deadlines.
   github.io`) — all three `EGRESS_BLOCKED`; WebSearch's own summary now
   states dates are genuinely TBD rather than converging on a figure, but
   this is a softer restatement of iter 31's phrasing, not a new
   contradiction — **no change to the Oct 31, 2026 estimate on file.** Also
   fixed two more bare-date-only `seen.json` entries (same data-hygiene
   class as iters 29/31/32's fixes): **2502.05111** ("Flexible and Efficient
   Grammar-Constrained Decoding," bare since iter 1) — the cluster's
   foundational lineage precursor, now annotated, no cluster-count change
   (never counted as a numbered member). **2605.29986** ("Accelerating
   Constrained Decoding with Token Space Compression"/CFGzip, bare since
   iter 5) — the cluster's already-named "Token-Space-Compression" member,
   now annotated, no cluster-count change (already counted under that name).
   **Not a tier-1 event** — no scoop, no deadline change, no decision only
   Maria can make; no `NEEDS_INPUT.md` change. **Non-empty iteration** (one
   new cluster-member citation, one mechanism-depth check on AgentStop
   confirming the standing "one layer down" claim, two data-hygiene fixes)
   — empty-iteration counter stays at 0 of 3. Full detail:
   `docs/research/digests/2026-09-11.md`.
2. 2026-09-10 (iter 32): no scope-narrowing decision — the five sub-questions
   reserved for Maria are still unanswered; `DECISIONS.md` and
   `docs/research/from-science/` unchanged since 2026-09-07. Local `main` was
   1 commit behind `origin/main` at the start of this iteration (iter 31's
   own commit, `cf6df45`) — fast-forwarded via `git checkout main && git
   pull`, nothing lost. Did iteration 31's queued next action: swept the
   **edge-VLM/medical-structured-generation** angle (the more stale of the
   two options, last touched iter 28) and made one more explicit attempt at
   the "SLM harness + edge-hardware telemetry in the same paper" search.
   **Headline find: AgentStop** (arXiv 2605.15206, ACM CAIS'26) — the first
   paper found that actually pairs an agent-harness-level mechanism
   (preemptive early-termination of unlikely-to-succeed trajectories, using
   token-level log-prob signals) with real on-device energy telemetry (GPU
   power/temp/battery on consumer hardware), reporting 15–20% energy savings
   at <5% utility cost — directly answers the question iter 31 flagged as
   under-searched. **Not a scoop of G1** — its mechanism is an agent-loop
   orchestration policy (when to stop a trajectory), not a decode-time/kernel
   mechanism like G1's mask-cost tax + shadow-mask pipelining — but it is the
   strongest concrete precedent yet that "harness mechanism + edge telemetry"
   papers exist and get accepted at real venues, which sharpens (does not
   threaten) the requirement that bigos's contribution be explicit about
   sitting one layer down (kernel/runtime/decode-time) from AgentStop's
   agent-loop-control layer. Eight more new-to-`seen.json` items: **CLONE**
   (arXiv 2506.02847, USENIX ATC'25) — joint offline model-tailoring +
   online latency-aware system optimization for edge LLM inference (up to
   11.92x speedup / 7.36x energy savings on off-the-shelf edge platforms),
   differentiate-from, joins the general-purpose inference-engine speedup
   cluster as its most complete framework yet (different mechanism class,
   still no grammar/mask angle); a third harness-taxonomy survey (**Agent
   Harness for LLM Agents: A Survey**, arXiv 2605.29682, 110+ papers/23
   systems, "ETCSLV" taxonomy) and one more architecture-pattern paper
   (**Harnessing Agent Skills**, arXiv 2606.20631) continuing the standing
   pattern that harness-taxonomy literature stays at the application/
   agent-orchestration level with zero edge/telemetry angle; an edge-cloud
   ECG diagnostic system (Sensors 2026, doi:10.3390/s26123753) scored
   ignore (the LLM part runs in the cloud, not on-device — same
   split-computing shape as DynaSplit); three background surveys (SLM survey
   2410.20011, edge-LLM review 2410.11845, energy-efficient co-design survey
   2603.23668) filed as cite for related-work framing; one VLA-redundancy
   paper (2606.27755) filed as ignore, tangential to the medical/text scope.
   Also **fixed a data-hygiene gap**: **EdgeCIM** (arXiv 2604.11512, a custom
   compute-in-memory ASIC for SLM decode benchmarked against a Jetson Orin
   Nano, up to 49.6x energy efficiency over it) had sat in `seen.json` since
   iteration 1 (2026-08-10) as a bare date with no annotation — now
   annotated as differentiate-from (silicon/hardware layer, not
   software/harness, not a real competitor) and flagged as a second
   independent Jetson Orin Nano energy data point (alongside iter 31's
   2511.11624) for whenever the postponed Jetson purchase resurfaces. MLSys
   2027: `mlsys.org/Conferences/2027/CallForResearchPapers` still
   `EGRESS_BLOCKED`; no change to the Oct 31, 2026 estimate on file. **Not a
   tier-1 event** — no scoop, no deadline change, no decision only Maria can
   make; no `NEEDS_INPUT.md` change. **Non-empty iteration** (one
   high-priority new citation directly resolving an open search question,
   eight more new citations, one data-hygiene fix) — empty-iteration counter
   stays at 0 of 3. Full detail: `docs/research/digests/2026-09-10.md`.
2. 2026-09-09 (iter 31): no scope-narrowing decision — the five sub-questions
   reserved for Maria (paper scope, venue/timing, starting use case,
   co-author, Jetson) are still unanswered; `DECISIONS.md` and
   `docs/research/from-science/` unchanged since 2026-09-07, no new `docs/`
   artifact. Local `main` matched `origin/main` exactly at `ef30c22` (iter
   30's own commit) at the start of this iteration — no drift, no stale
   checkout this time. Per Maria's own instruction, this loop does not gate
   or re-ping on the five sub-questions. Did iteration 30's queued next
   action: continued the broadened harness-OS-thesis literature sweep (a-c)
   plus one touch on the older correctness-tax cluster (d, most stale of the
   three original clusters, last swept iter 27). Five new-to-`seen.json`
   items, all reinforcing rather than threatening the paper's gap: two
   harness-taxonomy surveys/studies at the agent-orchestration level
   (**"Externalization in LLM Agents"** 2604.08224 — cite, names "harness
   engineering" as a unifying term, useful terminology precedent;
   **"Architectural Design Decisions in AI Agent Harnesses"** 2604.18071 —
   differentiate-from, 70-project taxonomy of harness design dimensions) and
   three application-level "harness" papers scored ignore (AutoHarness
   2603.03329 — code harness for game-move legality; Evo-Harness 2608.15071 —
   skill-compilation harness for self-evolving agents, EMNLP 2026; SCG-MEM
   2604.20117 — schema-constrained decoding for agent-memory key lookup).
   None touch edge hardware, SLM-specific systems cost, or a low-level
   mechanism — the gap stays open and is easier to state precisely now:
   "SLM harness" literature is application-level, "SLM on edge" literature is
   characterization-level (see next item), nothing sits at their
   intersection with a systems mechanism. **"Last Harness"** (iter 30's
   carried-over unresolved thread) still not found as an actual paper after
   several more phrasings — recommend dropping this thread unless Maria
   clarifies where the name came from; the "harness" literature space is now
   well-mapped and nothing under that title or an obvious alias exists.
   Citing-work check on 2605.12129 came up empty (expected, paper is only 4
   months old) — re-check in a few weeks, not every iteration. Also fixed two
   under-annotated `seen.json` entries left bare-date-only since iter 1-2
   (same data-hygiene class as iters 22/29's PSC fix), no cluster-count
   change: **2511.11624** ("Characterizing... Energy Footprint... of SLM on
   Edges" — benchmarks Llama 3.2/Phi-3 Mini/TinyLlama/Gemma 2 on Raspberry Pi
   5, Jetson Nano, and **Jetson Orin Nano** — directly useful diligence for
   DECISIONS.md's still-postponed Jetson purchase, Orin Nano+GPU has the best
   energy-to-performance ratio of the three boards; characterization-only, no
   mechanism, not a threat) and **2606.25605** ("Constraint Tax in Open-Weight
   LLMs" tool-calling-suppression study — a distinct correctness-tax
   mechanism, joins that cluster's problem side). MLSys 2027:
   `mlsys.org/Conferences/2027/CallForResearchPapers` still `EGRESS_BLOCKED`;
   this iteration's WebSearch was less specific than iter 28-30's converged
   figure ("planning underway, dates TBD" rather than repeating Oct 31 2026)
   but nothing contradicted the standing estimate — **no change to the
   figure on file**. **Non-empty iteration** (five new citations, two
   data-hygiene fixes, one dead-end thread narrowed further) —
   empty-iteration counter stays at 0 of 3. Full detail:
   `docs/research/digests/2026-09-09.md`.
2. 2026-09-08 (iter 30): **GATE CLOSED — reconciliation iteration**, same
   shape as iter 8/13. Route-confirmation gate (opened iter 8) was answered
   by Maria via direct chat on 2026-09-07 at 23:23 UTC, in commit `fd8833d`
   (~17h after iter 29's own commit `97b76a9`, which had correctly reported
   the gate still open as of its own run) — not through this loop's own
   tracking, so STATUS.md never caught up until now. Her decision
   (`docs/research/DECISIONS.md`, "KIERUNEK: bigos = Harness OS for Small
   Models"; full spec in a separate repo, `edge-bigos/docs/
   BIGOS-HARNESS-OS.md`, not accessible from this session): the confirmed
   frame is not a single Route A/B/C/D pick but an **umbrella** — bigos
   becomes a "Harness OS for Small Models," G1 (mask-cost tax + shadow-mask
   pipelining) is one module inside it, the medical use case is the
   application axis (text now, VLM later, matching the paper shape agreed
   2026-08-14: characterization + mechanism + optimization method, with a
   correctness-grading gate). Five sub-questions are explicitly reserved for
   Maria in conversation, **not this loop** — paper scope (whole harness vs.
   G1-first), venue/timing (MLSys 2027 vs. preprint+workshop), starting use
   case (text vs. VLM), co-author, Jetson (still postponed). Per her own
   instruction (`NEEDS_INPUT.md`, "Instrukcja dla pętli"): the narrow
   literature-only restriction is **lifted in favor of a broader one** — no
   more re-ping on the closed A/B/C/D question, and the general sweep angle
   should now track the harness-OS thesis itself (SLM harnesses, mechanism
   cost, escalation/cascade routing, and three named candidates — EffGen,
   HARBOR, "Last Harness") rather than staying anchored to the mask-speedup
   cluster alone. **Mac-hardware experiments and drafting stay on hold**
   until Maria answers the five sub-questions — that restriction is
   unchanged, only the literature scope widened. Did this iteration's first
   harness-thesis sweep per the new instruction: eight new-to-`seen.json`
   items, headlined by **"It's Not the Size: Harness Design Determines
   Operational Stability in Small Language Models"** (arXiv 2605.12129) —
   the closest-yet paper to bigos's own founding claim, cited as high
   priority (validates the premise, doesn't touch the low-level-systems
   mechanism bar). Also found HARBOR (2604.20938), EffGen (2602.00887,
   one of Maria's three named candidates), "Better Harnesses, Smaller
   Models" (2607.08938), two escalation/cascade-routing papers (2606.27457,
   2605.06350), and two open-source PyPI packages (`small-model-harness`,
   `tinyedge-runtime`) — none compete on mechanism (no edge-hardware energy/
   latency telemetry, no grammar/mask angle, no runtime/OS contribution
   found anywhere), but together they establish that "harness beats model
   size for SLMs," stated bare, is **not novel** — sharpening rather than
   threatening the paper's required contribution (the specific low-level
   mechanisms that make a harness measurably cheaper/faster on real edge
   hardware). "Last Harness" not found as a paper on arXiv — unresolved, not
   worth a repeat query this iteration. **Not a tier-1 event** — no new
   `NEEDS_INPUT.md` gate opened; the five reserved sub-questions stay with
   Maria via conversation per her own instruction, not via a loop-authored
   gate. MLSys 2027: `mlsys.org/Conferences/2027/CallForResearchPapers`
   still `EGRESS_BLOCKED`; WebSearch surfaced a secondary, inconsistent
   estimate ("submission opens Oct 10, 2026") alongside the long-standing
   Oct 31, 2026 11:59 AM UTC/AoE consensus — flagged, not adopted, no change
   to the figure on file. Repo state: local `main` matched `origin/main`
   exactly at `fd8833d`, no drift; `experiments/results/` unchanged.
   **Non-empty iteration** (gate reconciled, eight new citations, one
   high-priority "must cite" finding) — empty-iteration counter resets to 0
   of 3. Full detail: `docs/research/digests/2026-09-08.md`.
2. 2026-09-07 (iter 29): no scope-narrowing decision — `NEEDS_INPUT.md`
   (route confirmation, opened iter 8) is still open, gate age now ~23 days.
   Local `main` branch pointer was 3 commits stale (`2ad34d0`, iter 25's own
   commit) while `origin/main` and this session's own starting detached HEAD
   were already at `2922da6` (iter 28's own commit) — the same recurring
   leftover detached-HEAD/stale-branch-pointer pattern as iterations
   8/14/20/25/27/28. Fast-forwarded via `git checkout main && git pull`,
   nothing lost, nothing new folded in (the 3 "new" commits were iters
   26–28's own already-known content). Checked `DECISIONS.md` (still ends
   the 2026-08-14 "na razie M1" entry, unchanged), `docs/research/
   from-science/` (same seven files, unchanged since 2026-08-19), and
   `experiments/results/` (no new run directories or NIGHT-REPORTs) —
   nothing new since iter 28. **No re-ping** — same reasoning as iter
   10–28. Did iter 28's queued next action, option (a): swept the
   **algorithmic mask-speedup cluster** (last touched iter 25, 4 iterations
   stale, was 9 members) for new entrants. One new-to-`seen.json` cluster
   member: **WGRAMMAR** (arXiv 2507.16768) — decomposes grammar constraints
   into precompiled-offline static + runtime-instantiated dynamic
   components, up to 250× speedup (the cluster's largest raw number yet) —
   differentiate-from, 10th member, no edge/Joules/single-stream angle,
   same defining trait as the other nine. Also fixed a data-hygiene gap
   (same class as iter 22's fix): the cluster's founding member, **PSC**
   (arXiv 2608.03065, "Efficient Grammar-Constrained Decoding via Parser
   Stack Classification"), had been referenced by that acronym since
   iteration 2/12 but its actual arXiv ID was never entered into
   `seen.json` — now recorded with full annotation; no cluster-size change
   from this fix, PSC was already counted among the 9 (now 10) members.
   MLSys 2027: `mlsys.org/Conferences/2027/CallForResearchPapers` still
   `EGRESS_BLOCKED`; WebSearch aggregators still converge on the same Oct
   31, 2026, 11:59 AM UTC/AoE estimate on file since iter 5/28, still
   third-party consensus not a verified primary source. **Non-empty
   iteration** (one new cluster-member citation, one data-hygiene fix) —
   empty-iteration counter stays at 0 of 3. Full detail:
   `docs/research/digests/2026-09-07.md`.
2. 2026-09-06 (iter 28): no scope-narrowing decision — `NEEDS_INPUT.md`
   (route confirmation, opened iter 8) is still open, gate age now ~22 days.
   Local `main` was 2 commits behind `origin/main` at the start of this
   iteration (the same intermittent leftover-detached-HEAD-checkout pattern
   as iterations 8/14/20/25/26/27 — this session's own checkout starting
   stale, not new remote work); fast-forwarded via `git checkout main &&
   git pull` to `80e5684` (iter 27's own commit), nothing lost. Checked
   `DECISIONS.md` (still ends the 2026-08-14 "na razie M1" entry, unchanged),
   `docs/research/from-science/` (same seven files, unchanged), and
   `experiments/results/` (no new run directories or NIGHT-REPORTs) —
   nothing new since iter 27. **No re-ping** — same reasoning as iter 10–27.
   Did iter 27's queued next action, option (a): swept the **edge-VLM/
   medical-structured-generation** angle (last touched iter 26, 2 iterations
   stale). Three new-to-`seen.json` items, all **ignore** — none touch edge
   deployment, latency, energy, or constrained/structured-decoding
   mechanics: **C-SRRG** (arXiv 2510.00428) — chest-X-ray structured-report
   dataset curation (multi-view images + clinical indication + imaging
   technique + prior studies) fixing temporal hallucinations, benchmarked
   vs. SOTA multimodal LLMs, accuracy/data-curation focus. **Simple
   Token-Efficient VLM for Pathology Synoptic Reports** (arXiv 2605.30716) —
   minimal 3-component architecture for gigapixel whole-slide-image case
   reports; its "constrained GPU memory" is *training*-time memory for long
   visual-token sequences, not edge deployment. **Vision-language Models for
   Structured Report Generation in Radiology** (Springer, BVM 2026) —
   introduces SRRG-benchmark for evaluating LLMs converting free-text CXR
   reports to structured form, a benchmark/evaluation contribution. All
   three continue the standing pattern (iters 10/17/21/24/26): this
   literature area has exactly one differentiate-from paper on file (the ACM
   edge-deployed dual-report paper, iter 10) and zero systems-cost
   competitors — Route A's gap stays open. MLSys 2027: `mlsys.org/
   Conferences/2027/CallForResearchPapers` still `EGRESS_BLOCKED`; WebSearch
   aggregators now converge on a slightly more specific figure (submission
   deadline Oct 31, 2026, 11:59 AM UTC/AoE, notifications Nov 30 2026,
   conference Feb 16–23 2027 Montréal) — a one-day refinement of the ~Oct
   30–31 estimate on file since iter 5, still third-party consensus not a
   verified primary source. **Non-empty iteration** (three new citations,
   all reinforcing the standing gap) — empty-iteration counter stays at 0 of
   3. Full detail: `docs/research/digests/2026-09-06.md`.
2. 2026-09-05 (iter 27): no scope-narrowing decision — `NEEDS_INPUT.md`
   (route confirmation, opened iter 8) is still open, gate age now ~21 days.
   Local `main` was 1 commit behind `origin/main` at the start of this
   iteration (the same intermittent leftover-detached-HEAD-checkout pattern
   as iterations 8/14/20/25/26 — this session's own checkout starting stale,
   not new remote work); fast-forwarded via `git checkout main && git pull`
   to `8c0ab2f` (iter 26's own commit), nothing lost. Checked
   `DECISIONS.md` (still ends the 2026-08-14 "na razie M1" entry, 7 sections
   total, unchanged), `docs/research/from-science/` (same seven files,
   unchanged), and `experiments/results/` (no new run directories or
   NIGHT-REPORTs) — nothing new since iter 26. **No re-ping** — same
   reasoning as iter 10–26. Did iter 26's queued next action, option (a):
   swept the **correctness-tax cluster** (9 members as of iter 24, 3
   iterations stale) for new entrants. Four new-to-`seen.json` items, none
   competitive: **Correctness-Guaranteed Code Generation via Constrained
   Decoding** (arXiv 2508.15866, COLM 2025) and **AdapTrack** (arXiv
   2510.17376, ICSE 2026) — both code-generation-domain constrained/
   backtracking-decoding papers, no edge/latency angle — ignore, continuing
   the standing pattern that code-gen-domain constrained-decoding work
   (Weave of Formal Thought, Type-Constrained Code Generation) never
   competes with G1/Route A. **lm-Meter** (arXiv 2510.06126, ACM/IEEE SEC
   2025) — an on-device kernel-level LLM latency profiler, no grammar/mask
   angle — differentiate-from/cite, the closest-yet paper to this harness's
   own telemetry-methodology side (worth a related-work sentence on
   instrumentation, not a threat to the scientific claim). **Sustainable
   LLM Inference for Edge AI** (arXiv 2504.03360) — 28 quantized Ollama
   models benchmarked on a Raspberry Pi 4 for energy/accuracy/latency using
   real hardware power meters — differentiate-from/cite, quantization
   mechanism (not constrained decoding), same shape as the already-filed
   CO2-Meter methodology citation (iter 9). MLSys 2027: tried
   `mlsys.org/Conferences/2027/CallForResearchPapers` again — still
   `EGRESS_BLOCKED`; WebSearch confirms no official CFP text yet, no change
   to the ~Oct 30–31, 2026 estimate. **Non-empty iteration** (four new
   citations, two of them methodology-adjacent to this harness) —
   empty-iteration counter stays at 0 of 3. Full detail:
   `docs/research/digests/2026-09-05.md`.
2. 2026-09-04 (iter 26): no scope-narrowing decision — `NEEDS_INPUT.md`
   (route confirmation, opened iter 8) is still open, gate age now ~20 days.
   Local `main` matched `origin/main` exactly (`2ad34d0`, iter 25's own
   commit) — no drift, nothing new to fold in. Checked `DECISIONS.md` (still
   ends the 2026-08-14 "na razie M1" entry), `docs/research/from-science/`
   (same seven files, unchanged — `find -newer` flagged them only from this
   session's checkout refreshing mtimes, not real content changes), and
   `experiments/results/` (no new run directories or NIGHT-REPORTs) —
   nothing new since iter 25. **No re-ping** — same reasoning as iter 10–25.
   Did iter 25's queued next action, option (c): swept the **edge-VLM/
   medical-structured-generation** angle (last touched iter 21, 5 iterations
   stale). Three new-to-`seen.json` items: **EdgeFM** (arXiv 2604.27476) — a
   lightweight agent-driven VLM/LLM inference runtime for industrial edge
   deployment, single-request low-latency focus, kernel-dispatch
   optimization — differentiate-from, first VLM-specific member of the
   **general-purpose inference-engine speedup cluster** (BaseRT/MetalRT,
   Ollama's MLX backend), no grammar/mask angle. **Progressive Semantic
   Communication for Efficient Edge-Cloud VLMs** (arXiv 2604.26508) — an
   edge-cloud split-computing paper (compressed visual tokens still leave
   the device) — ignore, same shape as DynaSplit, weakens rather than serves
   Route A's on-device-privacy framing. **Reason Like a Radiologist /
   BoxMed-RL** (arXiv 2504.18453) — CoT+RL training for spatially-verifiable
   radiology reports — ignore, accuracy-focused, cloud-scale, continues the
   standing pattern (iters 10/17/21/24) that this literature area has
   exactly one differentiate-from paper and no systems-cost competitor.
   MLSys 2027: tried the more specific
   `mlsys.org/Conferences/2027/CallForResearchPapers` URL (surfaced by
   WebSearch as a real page title) per iter 25's queued next-action — still
   `EGRESS_BLOCKED`; no change to the ~Oct 30–31, 2026 estimate.
   **Non-empty iteration** (one new cluster-member citation, two ignores
   reinforcing the standing pattern) — empty-iteration counter stays at 0 of
   3. Full detail: `docs/research/digests/2026-09-04.md`.
2. 2026-09-01 (iter 25): no scope-narrowing decision — `NEEDS_INPUT.md`
   (route confirmation, opened iter 8) is still open, gate age now ~17 days.
   Local checkout started detached at `0845ba7` (iter 24's own commit,
   matching `origin/main` after `git fetch`) — `git checkout main` moved off
   detached HEAD, no drift, nothing lost. Checked `DECISIONS.md` (still ends
   `55050ff`, 2026-08-14), `docs/research/from-science/` (same seven files,
   unchanged), and `docs/` broadly (`find docs -newer` the iter-24 digest,
   and `experiments/results/` too, both returned nothing) — nothing new
   since iter 24. **No re-ping** — same reasoning as iter 10–24. Did iter
   24's queued next action, option (a): swept the **algorithmic
   mask-speedup cluster** (8 members as of iter 24 after Stay Within Your
   Bounds, stale 4 iterations) for new entrants. Two new-to-`seen.json`
   items: **Trie Automata for Constrained Decoding over Large Finite Sets**
   (arXiv 2608.12574, Amazon) — Aho-Corasick trie-based per-node mask
   precomputation for large finite-set constraints (K≥300), reports 7×
   faster per-step mask computation than XGrammar, benchmarked vs.
   vLLM/SGLang — differentiate-from, joins the algorithmic mask-speedup
   cluster as its **9th member** (largest raw-speedup number in the cluster
   so far, still no edge/Joules/single-stream angle). **Lookahead-then-
   Verify / LAVE** (arXiv 2602.00612) — constrained decoding for diffusion
   LLMs, a different generation paradigm entirely (parallel dLLM sampling,
   not autoregressive) — ignore, filed only to avoid re-discovery. MLSys
   2027: tried `mlsys.org/Conferences/2027/Dates` specifically per iter 24's
   queued next-action — still `EGRESS_BLOCKED`; WebSearch confirms the page
   exists but is unreadable from this session, no change to the ~Oct 30–31,
   2026 estimate. **Non-empty iteration** (one new cluster-member citation,
   one new-paradigm item filed) — empty-iteration counter stays at 0 of 3.
   Full detail: `docs/research/digests/2026-09-01.md`.
2. 2026-08-31 (iter 24): no scope-narrowing decision — `NEEDS_INPUT.md`
   (route confirmation, opened iter 8) is still open, gate age now ~16 days.
   Local `main` matched `origin/main` at the start of this iteration (both
   `fc9537b` after `git fetch`, no drift). Checked `DECISIONS.md` (still ends
   `55050ff`, 2026-08-14), `docs/research/from-science/` (same seven files,
   unchanged since iter 22), and `docs/` broadly (`find docs -newer` the
   iter-23 digest returned nothing) — nothing new since iter 23. **No
   re-ping** — same reasoning as iter 10–23. Did iter 23's queued next
   action, option (a): swept the **correctness-tax cluster** (8 members as
   of iter 22, stale since then) for new entrants. Two new-to-`seen.json`
   items: **The Structured Output Benchmark (SOB)** (arXiv 2604.25359) — a
   7-metric multi-source (text/image/audio) benchmark showing Value Accuracy
   drops 84.5%→67.2%→23.7% across those input modalities; cloud-scale models
   (GPT-5, Phi-4), no edge/latency angle — cite, joins the correctness-tax
   cluster's problem side as a new benchmark contribution, adds an
   input-modality-degradation axis the cluster hadn't captured yet, and is a
   citation to watch since its image-modality angle sits adjacent to Route
   A's own vision choice. **Stay Within Your Bounds: Distance-Guided
   Decoding for Guaranteed Context-Free Grammar Compliance** (arXiv
   2608.28229, posted ~3 days before this sweep — the newest item swept to
   date) — a pushdown-automata-reachability-distance decoder guaranteeing
   CFG compliance with improved completion quality on JSON/SQL/LTL; no
   edge/Joules/mask-cost angle — differentiate-from, joins the **algorithmic
   mask-speedup cluster** as its 7th member. MLSys 2027: `mlsys.org` and
   `mlsys-deadlines.github.io` both still `EGRESS_BLOCKED`; WebSearch
   surfaced a more specific target for next time —
   `mlsys.org/Conferences/2027/Dates` — not yet readable this iteration, no
   change to the ~Oct 30–31, 2026 estimate. **Non-empty iteration** (two new
   cluster-member citations) — empty-iteration counter stays at 0 of 3. Full
   detail: `docs/research/digests/2026-08-31.md`.
2. 2026-08-30 (iter 23): no scope-narrowing decision — `NEEDS_INPUT.md`
   (route confirmation, opened iter 8) is still open, gate age now ~15 days.
   Local `main` matched `origin/main` at the start of this iteration (both
   `2c7be4c` after `git fetch`, no drift). Checked `DECISIONS.md` (still ends
   `55050ff`, 2026-08-14), `docs/research/from-science/` (same seven files,
   unchanged since iter 22), and `docs/` broadly — nothing new since iter 22.
   **No re-ping** — same reasoning as iter 10–22. Did iter 22's queued next
   action: re-checked the **general-purpose inference-engine speedup
   cluster** (BaseRT/MetalRT, Ollama's own MLX backend), stale since iter 16.
   Found a real update, but a reassuring one: **Ollama v0.30.0 (May 2026)**
   promoted its Apple-Silicon MLX backend from preview to the **default fast
   path on capable Macs** (llama.cpp's Metal backend now layered in
   alongside it for wider-hardware GGUF support), and widened MLX
   architecture coverage from one model family at preview to **six** (Gemma
   3, GLM-4 MoE Lite, full Llama series, Qwen 3, Qwen 3.5, Qwen 3.5 MoE) —
   three of G1's four measured families are now architecture-eligible.
   **But the 32GB+ unified-memory gate is confirmed unchanged** across three
   independent 2026 sources, and G1's actual measurement machine
   (`Marias-MacBook-Air.local`, 16GB RAM) sits well under it regardless of
   Ollama version — so despite the architecture-coverage widening, **no
   baseline-drift risk exists for G1's existing numbers or for a near-term
   re-run on the same hardware.** Corrected the stale "only one architecture"
   wording in "Known competitive pressure" below to reflect the current
   picture. BaseRT/MetalRT: no update since iter 11's original citation.
   MLSys 2027: `mlsys.org` still `EGRESS_BLOCKED`, no change to the ~Oct
   30–31, 2026 estimate. **Non-empty iteration** (one baseline-drift watch
   item re-verified and corrected) — empty-iteration counter stays at 0 of 3.
   Full detail: `docs/research/digests/2026-08-30.md`.
2. 2026-08-29 (iter 22): no scope-narrowing decision — `NEEDS_INPUT.md`
   (route confirmation, opened iter 8) is still open, gate age now ~14 days.
   Local `main` matched `origin/main` at the start of this iteration (both
   `5b5d8fa`, no drift). Checked `DECISIONS.md` (still ends `55050ff`,
   2026-08-14), `docs/research/from-science/`, and `docs/` broadly —
   nothing new since iter 21. **No re-ping** — same reasoning as iter
   10–21. Did iter 21's queued next action: swept the **correctness-tax
   cluster** (stale since iter 14) for new entrants. Three new-to-
   `seen.json` items, none competitive with G1/Route A (no edge/Joules/
   per-token-mask-cost angle in any): **Structured Output Collapses Answer
   Diversity Across 44 Language Models** (arXiv 2607.18476) — large-scale
   (44-model) study showing a bare JSON-format instruction collapses answer
   diversity, a new mechanism (diversity collapse, not just accuracy loss)
   — cite. **Alignment Tax of Constrained Decoding in LLM Reflection**
   (arXiv 2604.06066) — Outlines-constrained self-correction on Qwen3-8B
   triggers a new failure mode ("structure snowballing") instead of fixing
   the original one — cite. **When Correct Isn't Usable** (arXiv 2605.02363)
   — reports the cluster's largest latency number yet (3.6×–8.2×
   overhead), checked closely as a possible scoop risk given the size of
   that number; **not a scoop** — it's a two-pass reference-then-constrain
   baseline on cloud-class 7–9B models, not per-token mask cost, no edge
   angle; joins the cluster's fix-side sub-group with ATLAS-RTC.
   Also caught a small data-hygiene gap: iter 21's own digest had already
   referenced 2605.02363 as "already tracked," but it was never actually in
   `seen.json` — now recorded for the first time with full annotation. The
   correctness-tax cluster is now 8 members total (4 original problem-side +
   2 new problem-side + 2 fix-side) — see "Known competitive pressure"
   below. **Non-empty iteration** (three new cluster-member citations, one
   data-hygiene fix) — empty-iteration counter stays at 0 of 3. Full detail:
   `docs/research/digests/2026-08-29.md`.
2. 2026-08-28 (iter 21): no scope-narrowing decision — `NEEDS_INPUT.md`
   (route confirmation, opened iter 8) is still open, gate age now ~13 days.
   Local `main` matched `origin/main` at the start of this iteration (both
   `25dbc61`, no drift — the stale-checkout pattern from iterations 8, 14, 20
   did **not** recur). Checked `DECISIONS.md` (still ends `55050ff`,
   2026-08-14), `docs/research/from-science/`, and `docs/` broadly —
   nothing new since iter 20. **No re-ping** — same reasoning as iter
   10–20. Did iter 20's queued next action: switched the general sweep
   angle back to **edge-VLM/medical-structured-generation** (last touched
   iter 17, now stale 4 iterations, per the no-repeat-two-iterations-running
   rule after iters 18–20 covered the hetero-core mask-pipelining question).
   Four new-to-`seen.json` items: **LEAD** (arXiv 2602.04617) and
   **Resource-Efficient Medical Report Generation** (arXiv 2410.15642) are
   both accuracy/model-size-focused medical-report papers with no edge-
   systems or decode-time-latency angle — ignore, continuing the standing
   pattern that this literature area has exactly one differentiate-from
   paper on file (the ACM edge-deployed dual-report paper, iter 10).
   **Adaptive Constraint Propagation / MetaJuLS** (arXiv 2601.00095, ACL
   2026) is a genuine constrained-generation speedup method (meta-RL
   constraint-propagation policy, 1.5–2.0× vs. GPU-optimized baselines) but
   benchmarked against cloud/GPU baselines with no edge/single-stream/
   Joules angle — differentiate-from, joins the algorithmic mask-speedup
   cluster (now six members, all cloud/GPU-baseline-shaped). **Towards
   Efficient Large Vision-Language Models** (arXiv 2603.27960) is a survey
   taxonomizing LVLM inference acceleration into four dimensions including
   decoding strategies — filed as a citable background reference for
   Route A's eventual related-work section, not a competing result. MLSys
   2027: `mlsys.org` still `EGRESS_BLOCKED`, WebSearch confirms no official
   CFP yet, no change to the ~Oct 30–31, 2026 estimate. **Non-empty
   iteration** (one new cluster-member citation, one new background survey
   filed) — empty-iteration counter stays at 0 of 3. Full detail:
   `docs/research/digests/2026-08-28.md`.
2. 2026-08-27 (iter 20): no scope-narrowing decision — `NEEDS_INPUT.md`
   (route confirmation, opened iter 8) is still open, gate age now ~12 days.
   Local `main` was 2 commits behind `origin/main` at the start of this
   iteration (a leftover detached-HEAD checkout, same recurring pattern as
   iterations 8 and 14 — this session's own checkout starting stale, not new
   remote work) — fast-forwarded via `git fetch` + `git merge --ff-only` to
   `f1af906` (iter 19's own commit), nothing lost. Checked `DECISIONS.md`
   (still ends `55050ff`, 2026-08-14), `docs/lekcje/`,
   `docs/WSPOLPRACA-BADAWCZA.md`, `docs/MAPA-WIEDZY.md` (still just
   `6c8109a`, 2026-08-19), and `docs/` broadly — nothing new since iter 19.
   **No re-ping** — same reasoning as iter 11–19. Did iter 19's queued next
   action: checked whether Ghidorah's HCMP/ARCA hetero-core-scheduling
   technique (or an equivalent) has been applied by anyone to
   **grammar/schema mask computation** specifically, since Ghidorah's own
   publication. **Result: still no** — a second independent negative result
   agreeing with iter 19's own CPU-only-mask-pipelining check. Found a
   well-populated hetero-core **speculative decoding** cluster (Dovetail,
   AHASD, OpenPangu-NPU — all new to `seen.json`, all differentiate-from,
   none touching grammar/mask) but zero hits combining hetero-core
   scheduling with mask/grammar computation. Also closed out an old,
   under-annotated `seen.json` entry: **APEX** (arXiv 2506.03296, tracked
   since iter 2 with no notes) is a KV-cache-offload scheduler for
   data-center GPUs (T4/A10), same shape as NEO (iter 19) — confirmed not
   competitive, no grammar/edge angle. MLSys 2027: `mlsys.org` still
   `EGRESS_BLOCKED`, WebSearch confirms no official CFP yet, no change to
   the ~Oct 30–31, 2026 estimate. **Non-empty iteration** (two negative
   results that together further strengthen confidence Route A's
   CPU-decode mask-pipelining gap is genuinely open, four new cluster-member
   citations found, one old entry's status clarified) — empty-iteration
   counter stays at 0 of 3. Full detail:
   `docs/research/digests/2026-08-27.md`.
2. 2026-08-26 (iter 19): no scope-narrowing decision — `NEEDS_INPUT.md` (route
   confirmation, opened iter 8) is still open, gate age now ~11 days. Local
   `main` was 1 commit behind `origin/main` at the start of this iteration (a
   leftover detached-HEAD checkout carried over from how this session
   started, not new remote work) — fast-forwarded via `git checkout main &&
   git pull`, nothing lost, landed at iter 18's own commit `b313e1d`. Checked
   `DECISIONS.md`, `docs/research/from-science/`, `docs/lekcje/`, and `docs/`
   broadly — nothing new since iter 18. **No re-ping** — same reasoning as
   iter 11–18. Did iter 18's queued next action: checked whether anyone
   publishes **CPU-only** mask/forward-pass pipelining (as opposed to the
   already-known CPU-mask/GPU-forward-pass trick), for the case where the LM
   forward pass itself runs on CPU (per iter 18's "Challenging GPU Dominance"
   find). **Result: no direct hit found — nobody has published this specific
   mitigation.** A genuine negative result, not a dead end: it means the gap
   Route A's design question opened (iter 18) is still unclaimed. Two new
   items entered `seen.json`: NEO (arXiv 2411.01142, MLSys 2025) is a
   cloud-serving batch-throughput CPU-offload paper on data-center GPUs
   (T4/A10G/H100) — ignore, same shape as the already-tracked batch-serving
   cluster. **Ghidorah** (arXiv 2505.23219) is more useful: on-device,
   unified-memory hardware, tested specifically on **Jetson NX** (Route A's
   own planned-but-postponed replication hardware), combines speculative
   decoding with hetero-core (CPU/GPU/NPU) parallelism for 7.6× decode
   speedup — not a grammar-masking paper, but (a) reviewer-defense ammunition
   alongside BaseRT that Jetson-class hardware is an active optimization
   target, and (b) its HCMP/ARCA technique (profile-then-assign pipeline
   stages to different cores) is a plausible **design template** for the
   still-open CPU-decode mask-pipelining question — worth revisiting if/when
   Route A's mechanism gets designed. Also noted a minor, non-actionable
   inconsistency between two third-party MLSys-2027-deadline aggregators on
   the *conference* dates (Feb Montréal vs. May) — both still agree on the
   Oct 31, 2026 *submission*-deadline estimate, no change to the figure on
   file. **Non-empty iteration** (one negative result that sharpens rather
   than closes Route A's open design gap, one new hardware-class/technique
   citation found) — empty-iteration counter stays at 0 of 3. Full detail:
   `docs/research/digests/2026-08-26.md`.
2. 2026-08-25 (iter 18): no scope-narrowing decision — `NEEDS_INPUT.md` (route
   confirmation, opened iter 8) is still open, gate age now ~10 days. Verified
   local `main` matched `origin/main` at start of iteration (both `fa2db03`,
   no drift). Checked `DECISIONS.md`, `docs/research/from-science/`, and
   `docs/` broadly — nothing new since iter 17. **No re-ping** — same
   reasoning as iter 11–17. Did iter 17's queued next action: a direct
   novelty check on Route A's own "hero optimization" (mask-generation
   pipelined with the forward pass across the CPU/GPU sync boundary).
   **Result: not a new find, and not a scoop — the mechanism has been on
   file since iteration 2.** Rereading `docs/research/digests/2026-08-10-2.md`
   confirmed **XGrammar** (arXiv 2411.15100) already documents the exact
   mechanism ("the mask is computed on the CPU in parallel with the GPU's
   forward pass"), and it was swept and filed on day one of this loop —
   before Route A's pipelining framing existed (that came iter 6,
   2026-08-14). This iteration's actual contribution is narrower: (a) fixed
   an ambiguity in this file's "Known competitive pressure" section, which
   named only XGrammar-2 as a tracked cluster member and could read as
   implying no cluster member does CPU/GPU pipelining, when the original
   XGrammar (not XGrammar-2) is exactly that; (b) confirmed llama.cpp/Ollama
   — G1's actual measured engine — still doesn't have this (XGrammar is a
   separate, non-integrated engine; Ollama declined it, iter 14) so the
   open contribution is porting + characterizing where it breaks, not
   inventing it; (c) found one genuinely new, relevant paper — "Challenging
   GPU Dominance: When CPUs Outperform for On-Device LLM Inference" (arXiv
   2505.06461) — showing llama.cpp on an iPhone 15 Pro with a 1B model runs
   *faster on CPU-only than with GPU acceleration* (17 vs 12.8 tok/s). This
   matters directly for Route A's design: the CPU/GPU pipelining trick
   assumes mask-on-CPU and forward-pass-on-GPU are genuinely separate
   resources to hide one under the other; if small on-device models are
   already running their forward pass on CPU (because CPU wins there), that
   assumption doesn't hold and the pipelining fix needs a CPU-decode answer,
   not just a GPU-decode one — sharpens rather than threatens the open gap.
   Four more new-to-`seen.json` items, all low-relevance (code-gen domain or
   correctness-tax-cluster fix-side, no hardware-cost angle) — see digest.
   **Non-empty iteration** (one design-relevant new paper found, one
   standing ambiguity in this file fixed) — empty-iteration counter stays
   at 0 of 3. Full detail: `docs/research/digests/2026-08-25.md`.
2. 2026-08-24 (iter 17): no scope-narrowing decision — `NEEDS_INPUT.md` (route
   confirmation, opened iter 8) is still open, gate age now ~9 days. Verified
   local `main` matched `origin/main` at start of iteration (both `500a0f9`,
   no drift). Checked `DECISIONS.md`, `docs/research/from-science/`,
   `docs/lekcje/`, and `docs/` broadly via `find -newer` against the iter-13
   collab-protocol commit — nothing new since iter 13, no second lesson file,
   no Monday-block package yet. **No re-ping** — same reasoning as iter
   11–16. Did the three items STATUS.md queued: (1) MLSys 2027 CFP re-check —
   `mlsys.org` still `EGRESS_BLOCKED`, no change to the ~Oct 30–31, 2026
   estimate. (2) **Resolved iter 16's carried-over "Seeing is Free, Speaking
   is Not" (2607.09520) citing-work check** — the earlier plain-title search
   collided with unrelated free-speech-law content; a quoted-full-title retry
   resolved correctly to the actual paper and found no citing/follow-up work
   yet, a clean negative result (Route A's claim to be first-to-extend stands
   unchallenged) rather than an inconclusive one. (3) General sweep on the
   edge-VLM/medical-structured-generation angle (last touched iter 11, per
   the no-repeat-two-iterations-running rule): four new-to-`seen.json` items,
   all 2024-era accuracy/fine-tuning-focused medical-VLM papers with no
   edge-systems angle — all scored ignore. Confirms the standing pattern:
   this literature area has exactly one differentiate-from paper (the ACM
   edge-deployed dual-report paper, found iter 10) and no systems-cost/
   constrained-decoding work — the gap Route A would fill stays open.
   **Non-empty iteration** (the 2607.09520 citing-work item, open since iter
   16, was definitively closed) — empty-iteration counter stays at 0 of 3.
   Full detail: `docs/research/digests/2026-08-24.md`.
2. 2026-08-23 (iter 16): no scope-narrowing decision — `NEEDS_INPUT.md` (route
   confirmation, opened iter 8) is still open, gate age now ~8 days. Verified
   local `main` matched `origin/main` at start of iteration (both `dcbccc5`,
   no drift). No `DECISIONS.md` entry or chat/docs decision has answered the
   route question. **No re-ping** — same reasoning as iter 11–15. Closed
   iter 15's flagged Ollama-0.19-MLX baseline-drift-risk item with hard
   evidence instead of leaving it as "gated, unconfirmed": grepped
   `ollama_version` out of every G1 result manifest (002–006) — all ran on
   **Ollama 0.20.6**; the `g1-mask-cost-pilot` manifest records
   `ram_total_gb: 16.0` on `Marias-MacBook-Air.local`, well under the 32GB+
   unified memory Ollama's MLX backend requires; and v0.20.6's actual GitHub
   release notes (Gemma 4 tool calling, streaming tool calls, a Hermes agent
   guide, an image bugfix) mention no MLX/backend changes. **Verdict:
   resolved, not carried forward** — 002–006 definitively ran on the classic
   llama.cpp/Metal backend, no re-run needed to trust their numbers. Did the
   literature sweep on a fresh angle (grammar-constrained decoding cost
   specifically on VLMs, per iter 15's instruction not to repeat a narrow
   query twice running): found nothing new/competitive — every relevant hit
   was already in `seen.json`, one new-to-`seen.json` item (2407.08103, a
   2024 foundational automata-constrained-decoding paper, predates the whole
   2026 cluster) scored differentiate-from. MLSys 2027: `mlsys.org` still
   blocked; WebSearch surfaced a more specific but still-unofficial estimate
   (submission deadline Oct 31 2026 11:59 UTC, conference Feb 16–23 2027
   Montréal) from the same third-party deadline aggregators already on
   file — a 1-day refinement of the ~Oct 30 2026 estimate, not new
   information. A citation search for "Seeing is Free, Speaking is Not"
   (2607.09520) follow-up work came up empty (title collides with unrelated
   free-speech-law content) — not resolved, not worth a second attempt this
   iteration. **Non-empty iteration** (one baseline-drift-risk item fully
   resolved with hard evidence) — empty-iteration counter stays at 0 of 3.
   Full detail: `docs/research/digests/2026-08-23.md`.
2. 2026-08-22 (iter 15): no scope-narrowing decision — `NEEDS_INPUT.md` (route
   confirmation, opened iter 8) is still open, gate age now ~7 days. Verified
   local `main` matched `origin/main` at start of iteration (both `b9ef59a`,
   no drift this time). No `DECISIONS.md` entry or chat/docs decision has
   answered the route question. **No re-ping** — same reasoning as iter
   11–14. Did the two carried-over literature tasks: (1) **resolved iter
   14's open Format Tax citing-work check** — `arxiv.org` and
   `semanticscholar.org` both still blocked, so used a WebSearch quote-check
   instead; "One Token Away from Collapse" (2604.13006) turns out to be
   about *lexical*-constraint fragility (banning a punctuation char/word),
   not a decoder-level extension of Format Tax's prompt-vs-decoder
   argument — closing as differentiate-from, not carrying forward again.
   (2) **Widened sweep to the Apple-Silicon-engine baseline-drift watch**
   (last touched iter 11/12) and found something more concrete than the
   BaseRT/MetalRT third-party watch already on file: **Ollama itself**
   (the actual tool the harness drives, not a third-party alternative)
   shipped **Ollama 0.19 (March 2026)**, switching its own Apple Silicon
   backend from direct Metal to Apple's MLX framework — reported ~2× decode
   speedup, ~60% faster prefill. Gated by two conditions that keep it from
   being an action item today: needs 32GB+ unified memory (unconfirmed
   whether Maria's M1 qualifies, but G1's sub-1B models don't need it
   either way), and at preview only one model architecture (Qwen3.5-35B-A3B,
   none of G1's four families) is MLX-accelerated. Recorded as the concrete
   trigger condition for iter 11's standing "rerun sanity-check if the
   harness's underlying engine is upgraded" note — **not** an action item
   now. MLSys 2027 CFP re-check: still unconfirmed, `mlsys.org` blocked
   again, WebSearch confirms the organizing committee itself isn't
   finalized yet — no change to the ~Oct 30, 2026 extrapolated estimate.
   **Non-empty iteration** (one new baseline-drift-risk event recorded,
   one open item closed) — empty-iteration counter stays at 0 of 3. Full
   detail: `docs/research/digests/2026-08-22.md`.
2. 2026-08-21 (iter 14): no scope-narrowing decision — `NEEDS_INPUT.md` (route
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

## Current stage: **Route-confirmation gate CLOSED (2026-09-07) — now inside Stage 3, five sub-questions reserved for Maria via conversation, not a loop gate.**

Route confirmation resolved, but not as an A/B/C/D pick: Maria's 2026-09-07
decision reframes the whole question one level up. **bigos = "Harness OS for
Small Models"** is the umbrella research/product frame; **G1 (mask-cost tax
+ shadow-mask pipelining) is one module inside it**, not a competing route.
The medical use case is the application axis, text-first with a VLM
extension later (Route A's VLM work is not dropped, just resequenced as
"later" rather than "instead of"). The paper shape agreed 2026-08-14 stands
unchanged: characterization + mechanism + an optimization method, gated by
task-correctness (not just schema-validity) grading.

Experiment 005 already satisfied Stage 4's exit criterion ("a 1-week pilot
supports or kills the hypothesis") for **G1 as a module**: all four tested
sub-1B models show a positive, CI-clean latency tax below ~25ms/token
(qwen2.5:0.5b +1.84%, qwen3:0.6b +2.42%, llama3.2:1b +6.15%, gemma3:1b
+30.36%). That result stands and isn't in question, and now has a clear home
(a mechanism module inside the harness-OS paper) instead of an unresolved
"which paper" question.

**Five sub-questions are explicitly Maria's to answer via conversation, not
this loop's to guess or gate on** (`NEEDS_INPUT.md`, "Instrukcja dla pętli"):
1. Paper scope — the whole harness vs. G1-first within the harness frame.
2. Venue/timing — MLSys 2027 (~Oct 31, 2026) vs. preprint + 2027 workshop +
   a fuller paper later.
3. Starting use case — text (thesis-era medical assistant) vs. VLM
   (Route A's original proposal).
4. Co-author.
5. Jetson purchase — still postponed, unchanged since 2026-08-14.

**What changed for this loop's own behavior:** the narrow literature-only
restriction (mask-speedup cluster + edge-VLM/medical sweeps only) is
replaced by a broader one — sweep for the harness-OS thesis itself (SLM
harnesses, mechanism cost, escalation/cascade routing, named candidates
EffGen/HARBOR/"Last Harness") instead of staying anchored to G1's original
narrow clusters. **Mac-hardware experiments and drafting remain on hold**
until Maria answers the five sub-questions above — only the literature
scope widened, not the experiment/drafting freeze. Do **not** re-open or
re-ping on the closed A/B/C/D question.

<details><summary>Archived: the closed A/B/C/D gate (superseded 2026-09-07, kept for history)</summary>

The from-science strategic review (2026-08-14) scored G1's original
text-only framing as the **weakest** of four routes (relegated to "Trasa D —
fallback… a generalization cell within A") and recommended Route A, the
VLM/medical generalization, as a standalone paper direction. Maria's actual
answer superseded the frame of the question itself rather than picking one
of the four options as originally posed: instead of A *or* D, both become
modules/phases inside one umbrella (bigos), which is why iterations 8–29's
tracking of "is it A, B, C, or D" never resolved — it was asking the wrong
shape of question. See `NEEDS_INPUT.md`'s archived section for the original
four-option framing in full.

</details>

<details><summary>Stage-2 exit justification (honest accounting, unchanged since iter 2)</summary>

~40 sources swept pre-Stage-4 on the original text-LLM framing; the surviving
axis (energy/decoding machinery) cleared the ≥15-source bar comfortably. Axes
1 and 2 were closed at 8–9 sources each by the kill criterion, not exhaustive
sweep. The from-science strategic review (2026-08-14) constitutes a second,
much larger sweep specifically for the route-selection question — see
`docs/research/from-science/2026-08-14-landscape-systems.md` and
`2026-08-14-landscape-robotics-medical.md` for their own source counts.

</details>

## Nearest deadlines (re-checked 2026-10-01, iter 53)
| Venue | Deadline | Note |
|---|---|---|
| ~~ICRA 2027~~ | ~Sep 15, 2026 | now past — closed as a primary-target option regardless of route; from-science review already scored Route B (robot-grammar) as losing to Route A partly for this reason. Stays open as a paper #2 possibility in the spring robotics season. |
| **MLSys 2027** | **Oct 30, 2026, 20:00 UTC — still UNCONFIRMED against a primary source, now ~4.1 weeks out (29 days)** | `mlsys.org/Conferences/2027/Dates` direct `WebFetch` still `EGRESS_BLOCKED` (domain-level block, confirmed iter 50, unchanged). WebSearch surfaced a fuller CFP schedule this iteration (submissions open Oct 10 2026 20:00 UTC, deadline Oct 30 2026 20:00 UTC — resolves iter 52's "12:00 PM PDT" quibble back to the majority figure, reviews Jan 18 2027, author responses Jan 21 2027, notifications Feb 28 2027, **conference June 21–25, 2027, Bellevue, WA** — adopted, supersedes the Feb-2027-Montréal figure). **Twenty-fourth iteration this estimate falls inside the brief's "<8 weeks away" threshold**, and the T-4-week WRITING-mode governor (~Oct 2, 2026) is **~1 day away — still not crossed** (today is 2026-10-01) but will very likely be crossed before iteration 54, before Maria answers sub-question #2 (venue/timing). |
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
**Iter 34 adds the *why*:** CRANE (arXiv 2502.09061, ICML 2025) formally
proves that restrictive grammars strip room for the intermediate-reasoning
tokens autoregressive LMs need, giving the correctness-tax gap a mechanistic
grounding rather than only empirical measurements — useful citation for the
harness-OS paper's own correctness-grading-gate argument. Still needs a
harness change (task-correctness grading) outside this loop's
`docs/research/`-only scope — flagged, not fixed, applies to whichever route
is confirmed. **Iter 45 adds a methodology template:** "Constrained Decoding
Eliminates Structural Failures in Small LLMs but Reveals a Scale-Dependent
Semantic Gap" (arXiv 2609.23742) runs the two-axis (structural-vs-semantic)
evaluation this gap has called for since iter 7, on a 0.6B–4B model range
overlapping Maria's own — its eval design (not its code, unpublished) is a
direct template for the harness change this section keeps flagging as
outstanding.

## Scoop-check status (iter 53: zero new competitive papers — correctness-tax cluster sweep, four new-to-seen.json items, none competitive: headline sub-4B-regime find (VAREX, arXiv 2603.15118, IBM Research — below 4B parameters structured-output compliance not extraction capability is the dominant bottleneck, "schema echo" depresses scores 45-65pp, differentiate-from/cite) plus two independent schema-valid-≠-correct domain confirmations, both titled "ExtractBench" but unrelated papers (arXiv 2602.12247: 90% valid/12.5% pass on one domain, 0% valid on a 369-field schema; arXiv 2607.29677, LlamaIndex: enterprise-document value-F1/grounding/cost benchmark — name collision flagged between the two) plus one ignore (Prompt Complexity Dilutes Structured Reasoning, arXiv 2603.13351, keyword-overlap false lead, wrong sense of "structured"); two data-hygiene fixes on bare-date entries (The Format Tax, arXiv 2604.03616, now annotated with its prompt-vs-decoder cost decomposition, also resolved a second small URL-format-duplicate instance; Diffusion-LM constrained decoding, arXiv 2607.07026, now annotated, wrong generation paradigm); MLSys 2027 estimate unchanged on the deadline (~4.1 weeks out, twenty-fourth iteration inside the <8-week threshold, T-4-week WRITING-mode governor ~1 day away, still not crossed, correcting iter 52's prediction), conference date/location newly converged on Bellevue/June 21-25 2027 via a fuller WebSearch-surfaced CFP schedule (still not a primary-source fetch, domain still EGRESS_BLOCKED); iter 52: zero new competitive papers — edge-VLM/medical-structured-generation sweep, six new-to-seen.json items, none competitive: two real-device VLM-deployment telemetry precedents (Efficient Deployment of VLMs on Mobile Devices / OnePlus 13R study, arXiv 2507.08505, headline — real-phone llama.cpp/MLC-Imp/mllm deployment with CPU/GPU/NPU profiling, closest "our own engine + VLM + real edge device + telemetry" precedent yet, useful Route-A groundwork, no grammar/mask angle; StepX-Edge, arXiv 2607.22708, ICMI 2026, real Snapdragon-8-Gen5 quantized on-device UI VLM deployment) plus one structured-medical-VLM-JSON precedent (Context-Fusion Framework for Endoscopic Polyp Reporting, arXiv 2608.15580, frozen VLM emits structured JSON clinical report, converges with Route A's use-case shape but no edge deployment or constrained decoding); one filed ignore (EndoVLM pretraining model, arXiv 2608.04472, continues zero-systems-cost-competitor pattern); two filed background/no-discussion (surveys 2504.09724, 2412.03772); clean-negative checks (no citing work yet for "Seeing is Free, Speaking is Not" 2607.09520; targeted scoop-shape search surfaced only the already-tracked VLA/Jetson-robot cluster, no new member); MLSys 2027 estimate unchanged (~4.3 weeks out, twenty-second iteration inside the <8-week threshold, T-4-week WRITING-mode governor ~2 days away, still not crossed); iter 51: zero new competitive papers — harness-OS-thesis sweep, eight new-to-seen.json items, none competitive: headline runtime-layer find ("Transition-Aware Backend Dispatch for Edge LLM Inference," arXiv 2607.17415, real Jetson hardware, backend-dispatch policy reduces latency/energy/EDP by 17.4%/14.4%/28.5% over best static policy — differentiate-from/cite, closest runtime-adaptive-decision-backed-by-real-edge-telemetry precedent since DVFSLM, backend-selection layer not decoding-mask layer) plus "The Agent Operating System (AOS)" (arXiv 2608.03214, Open Compute Project, literally named "OS" for agentic systems — 4th independent OS/harness-for-agents naming precedent, application-level only) and "Toward a Modular Architecture for Embedded AI Agent Systems at the Edge" (arXiv 2606.02862, tiered on-device/cloud-augmented-SLM architecture, closer to BRIEF.md's fleet-of-home-robots framing than most harness-taxonomy finds) and "AgenTEE" (arXiv 2604.18231, EuroMLSys 2026, real edge TEE-based confidential agent execution, distinct privacy mechanism) and SWARM-LLM (arXiv 2606.14711, edge-SLM-swarm + optional-cloud routing layer, federated/fleet-axis citation) and SparseDVFS (arXiv 2603.21908, DVFS sub-thread 5th member, converges with iter-47's memory-clock finding); two filed for hygiene only (GPU-DVFS-for-fine-tuning 2607.05933, wrong task; Hybrid-AI-Routers 2504.10519, escalation-cluster shape); MLSys 2027 estimate unchanged (~4.4 weeks out, twentieth iteration inside the <8-week threshold, T-4-week WRITING-mode governor still not crossed, correcting iter 50's prediction that it would be); iter 50: zero new competitive papers — algorithmic mask-speedup cluster sweep, fifth consecutive nothing-new sweep (iters 33/38/42/46/50), two new-to-seen.json items both ignore (Guided Decoding for RAG 2509.06631, ~1yr-old application-level comparison study; Control-DAG 2404.06854, non-autoregressive paradigm mismatch), one OSS tool logged for data hygiene (anishesg/dfa-sample, GPU-fused DFA-mask+sampling kernel, cloud-GPU-shaped, no paper); MLSys 2027 estimate unchanged (~4.3 weeks out, nineteenth iteration inside the <8-week threshold), confirmed mlsys.org is blocked at the domain level (not just /Conferences/2027/* subpaths), closing that open question; iter 47: zero new competitive papers — four new-to-seen.json items from the harness-OS-thesis sweep, none competitive: one headline runtime/kernel-layer find ("Edge-Inference Governors Need Memory-Clock State," arXiv 2606.16106, real Jetson Orin, extends the DVFS sub-thread with a memory-clock factor DVFSLM's own governor doesn't model — differentiate-from/cite, different mechanism layer than G1) plus three telemetry-methodology precedents (Transferable Latency Prediction 2607.21602 — real Pixel/Jetson Nano/Orange Pi 5 Pro/RTX 3090 hardware; Triple-Bottom-Line Sustainability 2609.00665 — SLM-vs-quantized-LLM Holistic Sustainability Score, GPU-only; Adaptive Compression for Edge-based RAG 2608.19535 — real Jetson AGX Thor, a runtime-control-mechanism-adaptive-to-device-telemetry precedent for a different task than G1 but the same contribution shape); three items filed ignore (cloud-native distributed-systems research-agenda survey 2604.17227, geo-distributed multi-edge-node context management DisCEdge 2511.22599, multi-LLM edge-orchestration survey 2507.00672); MLSys 2027 estimate unchanged (~5.0 weeks out, fourteenth iteration inside the <8-week threshold), OpenReview 2027 group page still EGRESS_BLOCKED, but a named-organizer search surfaced real primary-source tweets (Zhihao Jia, Jeff Dean) about MLSys 2026 specifically — corroborates the Oct-30 deadline pattern for the first time via a quasi-primary source, and newly flags the standing "Feb 2027 Montréal" conference-location figure as likely wrong (MLSys evidently runs in May) though unconfirmed for 2027 itself; iter 46: zero new competitive papers — one new-to-seen.json item from the algorithmic-mask-speedup-cluster sweep, DOMINO (arXiv 2403.06988, ETH Zurich, Feb 2024), a lineage/precursor citation (same non-numbered role as 2502.05111/SynCode), not a 12th cluster member — cluster stays at 11, fourth consecutive nothing-new sweep (iters 33/38/42/46), flagged as a strengthening possible-saturation signal not a conclusion; one data-hygiene fix (arXiv 2608.10137, "The Parser Already Knows," bare since iter 1-2, now annotated as a fifth correctness-tax-cluster fix-side mechanism); MLSys 2027 estimate unchanged (~5.1 weeks out, thirteenth iteration inside the <8-week threshold), both the standing CFP-page target and the queued x.com/mlsysconf organizer-account target still EGRESS_BLOCKED; iter 41: zero new competitive papers — six new-to-seen.json items from the correctness-tax cluster sweep, none competitive: four differentiate-from/cite additions strengthening the cluster's own problem/fix-side taxonomy (AprAD 2410.01103, a fourth independent fix-side mechanism predating the 2026 cluster by over a year; a causal-inference confirmation of the problem, 2509.21791; an SE-domain replication, 2606.09395; a third mechanism family, schema-description-as-instruction-channel, 2608.08254) plus one methodology find flagged as directly reusable for the harness-OS paper's own correctness-grading gate (Where vs What / SCD, 2608.25358) and one ignore (The Architecture of Errors, 2605.30628, general reliability theory not structured-output-specific); two data-hygiene fixes on bare-date entries since iter 1-2 (JSONSchemaBench 2501.10868, benchmarks llama.cpp's own grammar engine, no edge-energy axis; Attention Meets Reachability 2603.05540, a mask-speedup-cluster lineage paper not correctness-tax); MLSys 2027 estimate unchanged (~5.9 weeks out, eighth iteration inside the <8-week threshold), all three tried URLs plus a new call-for-reviewers search still yield nothing past secondary-source aggregation; iter 36: zero new competitive papers — four new-to-seen.json items from the edge-VLM/medical-structured-generation sweep: three edge-cloud split-computing papers (CoVSpec 2605.02218, INAR-VL 2605.18853, LAST 2607.27952) all scored ignore, the target/large model never runs on-device in any of them, weakening rather than serving the on-device-privacy framing; one methodology citation (Phase Matters, arXiv 2606.27906, real phone-SoC — Snapdragon 8 Elite — hardware-in-the-loop thermal+energy+phase-level telemetry, independent confirmation that hetero-core speedup is phase-dependent) scored differentiate-from/cite; plus four data-hygiene fixes on bare-date-only entries since iter 1-2 (VLM-edge survey 2502.07855 cite; robotic-VLM edge-server paper 2601.14921 ignore, also a "G1" name collision with an unrelated Unitree humanoid robot; LiteVLM 2506.07416 differentiate-from, genuine on-device edge+robotics VLM optimization but different mechanism axis; and arXiv 2603.23640 cite/high-value, directly benchmarks Maria's own iPhone 16 Pro showing ~40% thermal-throttling throughput loss under sustained load); MLSys 2027 estimate unchanged (~6.7 weeks out, third iteration inside the <8-week threshold), the secondary-source spread iter 34/35 flagged as growing actually narrowed this iteration (Nov 30 2026 notification date and Feb 16-23 2027 Montréal dates reaffirmed, matching iter 28 rather than iter 34's contradictory figure); iter 35: zero new competitive papers — eight new-to-seen.json items from the harness-OS-thesis sweep, none competitive: two unrelated "HARBOR"-named papers (robot-RL pipeline automation 2606.08610, agentic-eval infra 2609.04298) scored ignore but flagged as a name collision worth remembering; two coding-agent-domain harness-terminology papers (2606.10106 cited as a definitional precedent, 2609.00006 ignored); one training-time SLM-research-sandbox paper (Pico, 2509.16413) ignored as a different meaning of "framework"; three more escalation/cascade-routing papers (UCCI, Signed Rescue Routing, Routing/Cascades/User Choice) extending that cluster to 5 members, all cloud-serving-shaped; plus a clean negative result on the queued CRANE citing-work check (no edge-focused follow-up found); MLSys 2027 estimate now ~6.5 weeks out, second iteration inside the brief's <8-week verify threshold, secondary-source spread growing (new notification-date mismatch) but no confirmed change; iter 34: zero new competitive papers — four new-to-seen.json items from the correctness-tax cluster sweep, one a genuine mechanism-level find (CRANE, arXiv 2502.09061, formal proof of why constrained decoding loses functional accuracy + a fix, joins the cluster's fix-side alongside ATLAS-RTC/DCCD) and three ignores (Constraint Decay — a name-collision with unrelated software-architecture constraints; DINGO — diffusion-LLM paradigm mismatch, same shape as LAVE; LLM Data Auditor — unrelated synthetic-training-data survey); MLSys 2027 estimate now ~7 weeks out, first iteration inside the brief's <8-week verify threshold, flagged as a timing note not a deadline change; iter 33: zero new competitive papers — one new algorithmic-mask-speedup-cluster member (Earley-Driven Dynamic Pruning/Formatron, 11th) and two data-hygiene fixes (2502.05111 cluster-lineage precursor, 2605.29986 Token-Space-Compression member, both previously bare-date-only), plus a direct AgentStop-repo mechanism-depth check confirming no decode-time/kernel angle exists there beyond the known agent-loop early-termination policy; iter 32: zero direct scoops of G1's mechanism — nine new-to-seen.json items, headlined by AgentStop (arXiv 2605.15206, harness-level early-termination mechanism + real edge energy telemetry, differentiate-from/high-priority, first-of-its-kind combination found but different mechanism layer than G1) plus CLONE (edge-LLM model-tailoring+system-optimization framework, joins general-purpose speedup cluster), a third harness-taxonomy survey, one architecture-pattern paper, one edge-cloud ECG system (ignore), three background surveys (cite), one VLA-redundancy paper (ignore), and one data-hygiene fix (EdgeCIM's 32-iteration-old bare-date entry now annotated); iter 31: zero new competitive papers — five new-to-seen.json harness-taxonomy/agent-orchestration items (two cite/differentiate-from as terminology precedent, three ignore), two data-hygiene fixes (2511.11624 Jetson/Pi5 energy-characterization paper and 2606.25605 tool-suppression correctness-tax paper, both previously bare-date-only), "Last Harness" thread further narrowed to a dead end; iter 30: zero direct scoops of G1's mechanism or a low-level-systems edge-telemetry contribution, but the bigos harness-OS *umbrella claim* ("harness beats model size for SLMs") is not novel as a bare statement — at least three 2026 papers (2605.12129, 2602.00887, 2607.08938) and two OSS packages already say versions of it; none combine it with edge-hardware energy/latency measurement or a pluggable mechanism module, so not a kill-criterion scoop, but sharpens the contribution requirement; eight new-to-seen.json items from the first harness-OS-thesis sweep, gate reconciliation also folded in Maria's 2026-09-07 decision; iter 29: zero new competitive papers, one new algorithmic-mask-speedup-cluster member (WGRAMMAR, 10th) and one data-hygiene fix (PSC's arXiv ID recorded, no cluster-size change); iter 28: zero new competitive papers, three edge-VLM/medical-structured-generation items, all ignore (no edge/latency/constrained-decoding angle in any), continuing the standing zero-systems-cost-competitor pattern; iter 27: zero new competitive papers, four cluster-relevant citations (two code-gen-domain constrained-decoding papers scored ignore, two edge-hardware-measurement papers — lm-Meter, Sustainable LLM Inference for Edge AI — scored differentiate-from/cite as methodology precedent); iter 26: zero new competitive papers, one cluster-member citation (EdgeFM joins general-purpose inference-engine speedup cluster as its first VLM-specific member), two ignores reinforcing the standing no-systems-cost-competitor pattern in edge-VLM/medical literature; iter 25: zero new competitive papers, one cluster-member citation (Trie Automata joins mask-speedup cluster as 9th member), one new-generation-paradigm item filed as ignore (LAVE/diffusion-LLM decoding); iter 24: zero new competitive papers, two cluster-member citations (SOB joins correctness-tax cluster as 9th member, Stay Within Your Bounds joins mask-speedup cluster as 7th member); iter 23: zero new competitive papers, one baseline-drift watch item re-verified and corrected (Ollama v0.30.0 MLX-default transition, no drift risk confirmed); iter 22: zero new competitive papers, three correctness-tax-cluster citations, one data-hygiene fix; iter 21: zero new competitive papers, one cluster-member citation, one background survey filed; iter 20: zero new competitive papers, a second independent negative result confirming Route A's mask-pipelining gap, three hetero-core speculative-decoding cluster citations, one old entry clarified; iter 19: zero new competitive papers, one negative result strengthening Route A's open gap, one hardware-class/technique citation; iter 18: zero new competitive papers, one design-relevant find, one standing ambiguity fixed; iter 17: zero new competitive papers, one carried-over citing-work check resolved; iter 16: zero new competitive papers, one baseline-drift item fully resolved; iter 15: zero new papers, one new engine-release baseline-drift event; iter 14: two new items, none a scoop; iter 13: one new item, not a scoop; iter 12: three new items, none a scoop; iter 11: six new items, none a scoop; iter 10: zero new items; iter 9: seven new items, none a scoop)
Iter 30 is the first sweep under the new harness-OS thesis (route-
confirmation gate closed 2026-09-07, see "Current stage"). Eight new-to-
`seen.json` items, headlined by **"It's Not the Size: Harness Design
Determines Operational Stability in Small Language Models"** (arXiv
2605.12129) — cite, high priority: the closest-yet paper to bigos's own
founding claim (harness > model size for SLMs), but an evaluation/prompting-
level study (3 harness shapes x 3 SLMs x 24 tasks, task-success-rate only),
no edge hardware, no telemetry, no mechanism-level contribution — validates
the premise without touching the low-level-systems bar Maria requires.
**HARBOR** (arXiv 2604.20938) — differentiate-from, agent-harness
Bayesian-optimization config search, not SLM/edge-specific. **EffGen**
(arXiv 2602.00887, one of Maria's three named candidates) — differentiate-
from, open-source SLM agentic framework, application layer not systems
layer. **Better Harnesses, Smaller Models** (arXiv 2607.08938) —
differentiate-from, third independent "harness beats size" paper. Two
escalation/cascade-routing papers, **Cluster, Route, Escalate** (arXiv
2606.27457) and **Is Escalation Worth It?** (arXiv 2605.06350) —
differentiate-from/cite, cloud multi-model cascade routing, relevant only
if bigos's harness design adds an escalate-to-larger-model feature. Two
open-source PyPI packages, `small-model-harness` and `tinyedge-runtime` —
differentiate-from, confirm "harness for small models" is an active
practitioner/OSS space too, not just academic; `tinyedge-runtime`
("execution kernel and strategy SDK ... edge devices and robots") is the
closest name-and-scope match found, but explicitly disclaims the
telemetry/benchmarking work that is bigos's actual value-add. "Last
Harness" — not found as an arXiv paper, unresolved. Full detail:
`docs/research/digests/2026-09-08.md`.

Iter 27 found no new competitive papers. Swept the correctness-tax cluster
(9 members as of iter 24, 3 iterations stale) and found four new-to-
`seen.json` items, none competitive: **Correctness-Guaranteed Code
Generation via Constrained Decoding** (arXiv 2508.15866, COLM 2025) and
**AdapTrack** (arXiv 2510.17376, ICSE 2026) — both code-generation-domain
constrained/backtracking-decoding papers, no edge or latency angle —
ignore, matching the standing pattern that code-gen-domain constrained-
decoding work never competes with G1/Route A. **lm-Meter** (arXiv
2510.06126, ACM/IEEE SEC 2025) — an on-device kernel-level LLM latency
profiler, no grammar/mask angle — differentiate-from/cite, closest-yet
paper to this harness's own telemetry-methodology side. **Sustainable LLM
Inference for Edge AI** (arXiv 2504.03360) — 28 quantized Ollama models on
a Raspberry Pi 4, energy/accuracy/latency via real hardware power meters —
differentiate-from/cite, quantization mechanism (not constrained decoding),
same shape as the already-filed CO2-Meter methodology citation (iter 9).
Full detail: `docs/research/digests/2026-09-05.md`.

Iter 29 found no new competitive papers. Swept the algorithmic mask-speedup
cluster (4 iterations stale, was 9 members) and found one new-to-`seen.json`
member, **WGRAMMAR** (arXiv 2507.16768) — static/dynamic constraint
decomposition + mask caching, up to 250× speedup, the cluster's largest raw
number yet, benchmarked on cloud/GPU serving with no edge/Joules angle —
differentiate-from, 10th member. Also fixed a data-hygiene gap: the
cluster's founding member **PSC** (arXiv 2608.03065), named by that acronym
since iteration 2/12, had never had its actual arXiv ID entered into
`seen.json` — now recorded, no cluster-size change. Full detail:
`docs/research/digests/2026-09-07.md`.

Iter 28 found no new competitive papers. Swept the edge-VLM/medical-
structured-generation angle (2 iterations stale) and found three new-to-
`seen.json` items, all ignore: **C-SRRG** (arXiv 2510.00428) — chest-X-ray
structured-report dataset curation fixing temporal hallucinations, accuracy/
data-curation focus. **Simple Token-Efficient VLM for Pathology Synoptic
Reports** (arXiv 2605.30716) — minimal architecture for gigapixel WSI case
reports; its "constrained GPU memory" is training-time memory, not edge
deployment. **Vision-language Models for Structured Report Generation in
Radiology** (Springer, BVM 2026) — introduces a benchmark for structured-CXR
conversion, evaluation contribution. All three continue the standing pattern
(iters 10/17/21/24/26): exactly one differentiate-from paper on file in this
area, zero systems-cost competitors. Full detail:
`docs/research/digests/2026-09-06.md`.

Iter 26 found no new competitive papers. Swept the edge-VLM/medical-
structured-generation angle (5 iterations stale) and found three
new-to-`seen.json` items: **EdgeFM** (arXiv 2604.27476) — a lightweight
agent-driven VLM/LLM inference runtime for industrial edge deployment,
single-request low-latency focus — differentiate-from, first VLM-specific
member of the general-purpose inference-engine speedup cluster, no
grammar/mask angle. **Progressive Semantic Communication for Efficient
Edge-Cloud VLMs** (arXiv 2604.26508) — edge-cloud split computing, same
shape as DynaSplit — ignore. **Reason Like a Radiologist / BoxMed-RL**
(arXiv 2504.18453) — CoT+RL training for verifiable radiology reports,
accuracy-focused — ignore. Full detail: `docs/research/digests/2026-09-04.md`.

Iter 25 found no new competitive papers. Swept the algorithmic mask-speedup
cluster (4 iterations stale) and found two new-to-`seen.json` items: **Trie
Automata for Constrained Decoding over Large Finite Sets** (arXiv 2608.12574,
Amazon) — 9th mask-speedup-cluster member, Aho-Corasick trie-based masking
for large finite-set constraints, 7× faster mask computation than XGrammar
(the cluster's largest raw-speedup number to date), benchmarked vs.
vLLM/SGLang, no edge/Joules angle. **Lookahead-then-Verify / LAVE** (arXiv
2602.00612) — constrained decoding for diffusion LLMs, a different
generation paradigm (parallel dLLM sampling, not autoregressive) — ignore,
no relevance to G1/Route A's autoregressive decode-time claim. Full detail:
`docs/research/digests/2026-09-01.md`.

Iter 24 found no new competitive papers. Swept the correctness-tax cluster
(2 iterations stale) and found two new-to-`seen.json` items: **The
Structured Output Benchmark / SOB** (arXiv 2604.25359) — 9th correctness-tax
cluster member, a multi-source (text/image/audio) accuracy benchmark, no
edge angle. **Stay Within Your Bounds** (arXiv 2608.28229) — 7th
algorithmic-mask-speedup-cluster member, a CFG-compliance-guaranteeing
decoder, no edge/Joules angle. Full detail:
`docs/research/digests/2026-08-31.md`.

Iter 23 found no new competitive papers. Re-checked the general-purpose
inference-engine speedup cluster (BaseRT/MetalRT, Ollama's own MLX backend),
stale since iter 16. **Ollama v0.30.0** (May 2026) promoted MLX from preview
to the default fast path on capable Macs and widened architecture coverage
to six families (three of G1's four) — but the 32GB+ unified-memory gate is
confirmed unchanged, and G1's measurement Mac (16GB RAM) sits well under it
regardless of Ollama version, so **no baseline-drift risk** for existing or
near-term G1 numbers. BaseRT/MetalRT: no update since iter 11. Full detail:
`docs/research/digests/2026-08-30.md`.

Iter 22 found no new competitive papers. Swept the correctness-tax cluster
(stale since iter 14) and found three new entrants, all reviewed via
WebSearch (arxiv.org stayed `EGRESS_BLOCKED` all iteration): **Structured
Output Collapses Answer Diversity Across 44 Language Models** (arXiv
2607.18476) — a large-scale (44-model) study showing a bare JSON-format
instruction collapses answer diversity, a new mechanism (diversity
collapse) joining the cluster. **Alignment Tax of Constrained Decoding in
LLM Reflection** (arXiv 2604.06066) — constrained decoding triggers a new
failure mode ("structure snowballing") in LLM self-correction rather than
fixing the hallucination-snowballing problem it targeted. **When Correct
Isn't Usable** (arXiv 2605.02363) — reports the cluster's largest latency
number yet (3.6×–8.2×); checked closely as a possible scoop given the size
of that number, but it's a two-pass reference-then-constrain baseline on
cloud-class 7–9B models, not per-token mask cost, no edge angle — not a
scoop, joins the cluster's fix-side sub-group with ATLAS-RTC. Also fixed a
small data-hygiene gap: iter 21's digest had referenced 2605.02363 as
"already tracked" but it was never actually recorded in `seen.json` — now
entered for the first time with full annotation. The correctness-tax
cluster is now 8 members total. Full detail:
`docs/research/digests/2026-08-29.md`.

Iter 21 found no new competitive papers. The edge-VLM/medical-structured-
generation sweep (last touched iter 17) returned two accuracy/model-size-
focused medical-report papers (LEAD 2602.04617, Resource-Efficient Medical
Report Generation 2410.15642), both ignore — continuing the standing
pattern that this literature area has exactly one differentiate-from paper
on file. One genuine constrained-generation speedup method, **Adaptive
Constraint Propagation / MetaJuLS** (arXiv 2601.00095, ACL 2026), joins the
algorithmic mask-speedup cluster as its sixth member — 1.5–2.0× speedup via
a meta-RL constraint-propagation policy, but benchmarked against cloud/GPU
baselines with no edge/single-stream/Joules angle, so differentiate-from,
not competitive. One survey, "Towards Efficient Large Vision-Language
Models" (arXiv 2603.27960), filed as a citable background reference for
Route A's eventual related-work section. Full detail:
`docs/research/digests/2026-08-28.md`.

Iter 20 found no new competitive papers. Ran a targeted search for whether
Ghidorah-class hetero-core (CPU/GPU/NPU) scheduling has been applied to
grammar/schema **mask computation** specifically (as opposed to speculative
decoding generally) since Ghidorah's own publication — **still no hit**, a
second independent negative result agreeing with iter 19's own CPU-only
mask-pipelining check. Found three new hetero-core speculative-decoding
cluster members (Dovetail 2412.18934, AHASD 2604.25326, OpenPangu-NPU
2603.03383), all differentiate-from — none touch grammar/mask, all confirm
the cluster is populated with speculative-decoding work but not mask-
computation work. One low-relevance item (Diffusion-LLM-on-NPU 2606.13740)
scored ignore — different generation paradigm. Also closed an old
under-annotated `seen.json` entry: APEX (2506.03296, tracked since iter 2)
read for the first time this iteration — confirmed a KV-cache-offload
scheduler for data-center GPUs, same shape as NEO (iter 19), not
competitive. Full detail: `docs/research/digests/2026-08-27.md`.

Iter 19 found no new competitive papers, and — more usefully — a **negative**
result: no published work does CPU-only mask/forward-pass pipelining (the
open half of Route A's design question iter 18 surfaced). NEO (arXiv
2411.01142) is a cloud-serving CPU-offload paper (T4/A10G/H100, no grammar
angle) — ignore. **Ghidorah** (arXiv 2505.23219) is more useful:
on-device/unified-memory, tested on Jetson NX (Route A's own planned
hardware), 7.6× decode speedup via speculative decoding + hetero-core
(CPU/GPU/NPU) scheduling — not grammar-masking, but citable hardware-class
ammunition and a plausible design template (profile-then-assign pipeline
stages to cores) for the still-open CPU-decode mask-pipelining question.
Full detail: `docs/research/digests/2026-08-26.md`.

Iter 18 found no new competitive papers — the mechanism-level check on
XGrammar (2411.15100) confirmed it's been on file, correctly characterized,
since iteration 2; not a scoop, see "Known competitive pressure" above for
the full account. One genuinely new, design-relevant paper: "Challenging GPU
Dominance" (arXiv 2505.06461) shows llama.cpp on iPhone 15 Pro running a 1B
model faster CPU-only than with GPU acceleration — sharpens (doesn't kill)
Route A's mask-pipelining design question. Four more new-to-`seen.json`
items, all low-relevance (code-gen domain or correctness-tax-cluster
fix-side). Full detail: `docs/research/digests/2026-08-25.md`.

Iter 17 found no new competitive papers. Resolved iter 16's carried-over
"Seeing is Free, Speaking is Not" (2607.09520) citing-work check — the
earlier plain-title search collided with unrelated free-speech-law content;
a quoted-full-title retry resolved correctly to the actual paper and found
no citing/follow-up work, a clean negative (Route A's claim to be
first-to-extend stands unchallenged). The general sweep on the edge-VLM/
medical-structured-generation angle (last touched iter 11) returned four
new-to-`seen.json` items, all 2024-era accuracy/fine-tuning-focused medical-
VLM papers with no edge-systems angle — all scored ignore, confirming the
standing pattern that this literature area has exactly one differentiate-
from paper (the ACM edge-deployed dual-report paper, iter 10) and nothing
touching constrained-decoding/systems cost. Full detail:
`docs/research/digests/2026-08-24.md`.

Iter 16 found no new competitive papers, but fully resolved iter 15's
Ollama-0.19-MLX baseline-drift flag: G1 experiments 002–006 all ran on
**Ollama 0.20.6** (confirmed from every `result.json` manifest), and the
`g1-mask-cost-pilot` manifest records `ram_total_gb: 16.0` on
`Marias-MacBook-Air.local` — well under the 32GB+ unified memory Ollama's
MLX backend requires, so this machine cannot have used the MLX path
regardless of version. v0.20.6's actual GitHub release notes confirm no
MLX/backend changes shipped in that release either. **002–006's numbers
need no re-run or caveat** — they're on the classic llama.cpp/Metal backend,
full stop. The literature sweep (grammar-constrained decoding cost on VLMs,
a fresh angle) found nothing new/competitive: one new-to-`seen.json` item
(2407.08103, a 2024 foundational automata-constrained-decoding paper) scored
differentiate-from, everything else already tracked. Full detail:
`docs/research/digests/2026-08-23.md`.

Iter 15 found no new competitive papers, but one new non-paper event worth
recording: **Ollama 0.19** (March 2026) switched Ollama's own Apple Silicon
backend from Metal to MLX (~2× decode speedup), gated by 32GB+ RAM and (at
preview) a single MLX-accelerated model architecture, neither of which
applies to G1's current setup — not an action item, but the concrete
trigger condition for the standing "rerun sanity-check if the harness's
engine is upgraded" note. Also closed iter 14's carried-over Format Tax
citing-work check: "One Token Away from Collapse" (2604.13006) is a
lexical-constraint fragility paper, not a decoder-level extension of Format
Tax's argument — differentiate-from, not carried forward again. Full
detail: `docs/research/digests/2026-08-22.md`.

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

## Known competitive pressure (updated iter 39)
Iter 39 finds the harness-OS-thesis angle's first genuine **systems-layer**
paper after ten iterations of application-layer surveys: **DVFSLM** (arXiv
2609.13153) — a runtime DVFS governor for SLM decode workloads on mobile
edge devices, minimizing energy-per-token under a token-generation-deadline
constraint. Not a scoop of G1 or the bigos umbrella (different mechanism:
frequency-scaling governance, not constrained-decoding mask cost), but the
closest-yet precedent for "a runtime/kernel-level governance module for SLM
edge inference is a publishable systems contribution" — worth a related-work
citation, and worth remembering as a candidate second module if the harness-
OS paper's scope ever widens beyond G1. Eight more items (another harness-
taxonomy survey plus seven application-level harness papers) filed as
ignore, continuing the pattern that this literature stays at the agent-
orchestration level. See the iter-39 scope-decision entry above for full
detail.

Iter 38 finds **no new member** for the **algorithmic mask-speedup
cluster** (stays at 11) — the sweep's most useful result is a hardware data
point, not a competitor: **SLED** (arXiv 2506.09397) speculative-decodes
edge-device-drafted tokens verified on an edge server, tested on **Jetson
Orin Nano and Raspberry Pi 4B/5**, giving the Jetson-purchase question
(sub-question 5) a third independent Orin Nano reference point and a first
Pi 4B one; no grammar/mask angle and verification leaves the device, so it
does not compete with G1/Route A. Two more items (**The Extrapolation
Cliff**, arXiv 2605.08737, a training-time format-collapse threshold for
structured outputs; **ARCS**, arXiv 2603.29068, grammar-constrained
decoding applied to analog-circuit synthesis) are filed as ignore/
out-of-scope, not threats. See the iter-38 scope-decision entry above for
full detail.

Iter 37 adds a *second* independent mechanistic explanation to the
**correctness-tax cluster**'s fix-side: **Grammar-Aligned Decoding** (arXiv
2405.21047, NeurIPS 2024) — proves grammar-constrained decoding distorts the
LLM's own output distribution (grammatical outputs end up with likelihoods
that don't match what the LLM itself would assign), a sampling-bias account
distinct from CRANE's (iter 34) reasoning-token-starvation account; proposes
ASAp, an adaptive algorithm trading efficiency for distributional fidelity.
No edge/hardware angle. Predates the 2026 cluster by ~19 months, third
foundational-precursor-found-late case after PSC and CRANE. Iter 37 also
adds **SynCode** (arXiv 2403.01632, UIUC) to the **algorithmic mask-speedup
cluster** as a lineage/precursor citation (same non-numbered-member role as
2502.05111): DFA-based offline mask-store for CFG-constrained decoding,
eliminates JSON syntax errors entirely, no edge/Joules angle. Neither item
is a scoop or an active competitor — both are background citations the 2026
clusters keep citing in their own related work.

Iter 34 adds one new member to the **correctness-tax cluster**'s fix-side:
**CRANE** (arXiv 2502.09061, ICML 2025 poster) — a formal proof, via a
Turing-machine-simulation argument, that restrictive output grammars deny an
autoregressive LM the intermediate-reasoning tokens it needs for multi-step
computation, which is *why* constrained decoding loses functional accuracy;
fixes this by augmenting the grammar with reasoning-preserving rules (up to
10pp accuracy gain over both unconstrained and prior constrained baselines).
No edge/hardware angle. Joins ATLAS-RTC and DCCD on the cluster's fix-side,
and is the clearest mechanistic (not just empirical) explanation on file for
the correctness-tax gap this loop has cited since iter 7 — predates the 2026
cluster by over a year but was never entered until this sweep, same
foundational-precursor-found-late shape as PSC in the mask-speedup cluster.
Three more items scored ignore, all name-collisions/paradigm-mismatches:
Constraint Decay (2605.06445, software-architecture constraints in agentic
backend code-gen — unrelated despite the name), DINGO (2505.23061,
diffusion-LLM constrained decoding — paradigm mismatch, same shape as LAVE),
and The LLM Data Auditor (2601.17717, synthetic training-data quality
survey — unrelated). See the iter-34 scope-decision entry above for full
detail.

Iter 33 adds an 11th member to the **algorithmic mask-speedup cluster**:
**Earley-Driven Dynamic Pruning / Formatron** (arXiv 2506.01151, ICML 2025
poster) — dynamic pruning of Earley parser sets/states for CFG mask
computation, up to 2× faster constrained decoding with no accuracy loss.
Same defining trait as the other ten members: cloud/GPU-serving-baseline
speedup, no edge hardware, no Joules/single-stream framing. Iter 33 also
directly checked AgentStop's own GitHub repo (`brave-experiments/
AgentStop`) rather than relying on the paper abstract alone — confirmed no
decode-time/kernel-level mechanism exists there beyond the log-prob-based
agent-loop early-termination policy already on file, so the "one layer
down" related-work framing from iter 32 stands verified against the source.
Also fixed two more bare-date-only `seen.json` entries (same class as
iters 29/31/32's fixes): **2502.05111** (the mask-speedup cluster's
foundational lineage precursor, bare since iter 1) and **2605.29986**
(CFGzip, the cluster's already-named "Token-Space-Compression" member,
bare since iter 5) — neither changes the cluster's member count, both were
either already counted (2605.29986) or never counted as a numbered member
(2502.05111, the lineage root).

Iter 32's headline find, **AgentStop** (arXiv 2605.15206, ACM CAIS'26), is
the first paper found that pairs an agent-harness-level mechanism with real
on-device energy telemetry in the same paper — see the iter-32 scope-decision
entry above for full detail. It is not a scoop (different mechanism layer:
agent-loop early-termination vs. G1's decode-time mask-cost/pipelining), but
it is now the standing reference for "closest-yet paper to the harness+
telemetry combination" — any related-work section needs to explain precisely
why bigos's contribution sits one layer lower (kernel/runtime/decode-time,
not agent-loop-control). Iter 32 also adds **CLONE** (arXiv 2506.02847,
USENIX ATC'25) as a new member of the **general-purpose inference-engine
speedup cluster** (BaseRT/MetalRT, Ollama's MLX backend, EdgeFM) — a joint
model-tailoring + latency-aware system-optimization framework for edge LLM
inference, the most complete framework in that cluster to date, still no
grammar/mask angle. A third harness-taxonomy survey (arXiv 2605.29682) and
one more architecture-pattern paper (arXiv 2606.20631) continue the standing
pattern that harness-taxonomy literature stays application/orchestration-
level with zero edge/telemetry angle. Iter 32 also closed a data-hygiene gap
open since iteration 1: **EdgeCIM** (arXiv 2604.11512) was tracked as a bare
date with no annotation for 32 iterations — now annotated as a custom
compute-in-memory ASIC benchmarked against a Jetson Orin Nano (differentiate-
from, silicon layer not software/harness, useful future Jetson-diligence
data point).

Iter 29 adds a 10th member, **WGRAMMAR** (arXiv 2507.16768, "Leverage Prior
Knowledge to Accelerate Structured Decoding"), to the **algorithmic
mask-speedup cluster** (PSC, GRID, Attention-Meets-Reachability,
Token-Space-Compression, Pre³, XGrammar-2, MetaJuLS, Stay Within Your
Bounds, Trie Automata): decomposes grammar constraints into precompiled-
offline static + runtime-instantiated dynamic components, compositional
regular-format operators instead of pushdown automata, plus mask caching —
up to 250× speedup over existing systems, the cluster's largest raw number
to date. Same defining trait as every other member: cloud/GPU-serving
benchmarks, no edge hardware, no Joules, no single-stream framing. Iter 29
also fixed a standing data-hygiene gap (same class as iter 22's fix): the
cluster's *founding* member, **PSC** (arXiv 2608.03065), has been named by
that acronym in this file and in `docs/research/from-science/
2026-08-14-landscape-systems.md` since iteration 2/12, but its actual
arXiv ID was never entered into `seen.json` — now recorded with full
annotation. This is a bookkeeping correction, not a new discovery: PSC was
already counted among the cluster's members all along (9 before this
iteration, 10 now that WGRAMMAR joins).

Iter 28 adds no cluster members — all three new-to-`seen.json` items from
the edge-VLM/medical-structured-generation sweep (C-SRRG 2510.00428, Simple
Token-Efficient VLM for Pathology 2605.30716, the BVM-2026 SRRG-benchmark
chapter) scored ignore, none touching edge deployment, latency, or
constrained/structured-decoding mechanics. This is now the sixth consecutive
sweep of this literature area (iters 10/17/21/24/26/28) to turn up exactly
zero systems-cost competitors — the area is well-populated with accuracy/
data-curation/benchmark work but nobody has yet paired it with edge-hardware
cost or grammar-constrained decoding, which is precisely Route A's proposed
gap.

Iter 27 adds two methodology-precedent citations rather than cluster
members: **lm-Meter** (arXiv 2510.06126) is an on-device kernel-level LLM
latency profiler — no grammar/mask angle, so not competitive, but it's the
closest-yet paper to this harness's own telemetry-instrumentation approach
(phase-level latency measurement on real mobile hardware), worth a
related-work sentence on the measurement-methodology side if Route A's
paper discusses its own harness design. **Sustainable LLM Inference for
Edge AI** (arXiv 2504.03360) benchmarks 28 quantized Ollama models on a
Raspberry Pi 4 for energy/accuracy/latency using real hardware power
meters — mechanism is quantization bit-width, zero overlap with G1/Route
A's per-token mask-cost claim, but it's a citable methodology precedent for
"hardware energy meters on Ollama-served edge SBCs," the same shape as the
already-filed CO2-Meter citation (iter 9). Two more new items (2508.15866,
2510.17376) are code-generation-domain constrained-decoding papers — no
edge angle at all — scored ignore, not added to any cluster.

Iter 26 adds the first VLM-specific member to the **general-purpose
inference-engine speedup cluster** (BaseRT/MetalRT, Ollama's own MLX
backend): **EdgeFM** (arXiv 2604.27476) — a lightweight, agent-driven
VLM/LLM inference runtime for industrial edge deployment, tuned specifically
for single-request low-latency workloads via model-/hardware-/stage-aware
kernel dispatch. Same defining trait as the cluster's other members: a real
runtime-speedup contribution with no grammar/constrained-decoding angle, so
not competitive with Route A's mask-pipelining claim — but it's the
closest-yet paper to Route A's *systems* framing specifically for VLMs
(edge, single-stream, deterministic latency), worth a related-work sentence
if Route A proceeds. Two other new-to-`seen.json` items in the edge-VLM/
medical sweep scored ignore: an edge-cloud split-computing paper (2604.26508,
same shape as DynaSplit) and an accuracy/explainability-focused radiology-RL
training paper (2504.18453) — both reinforce the standing pattern that this
literature area has exactly one differentiate-from paper (the ACM
edge-deployed dual-report paper, iter 10) and no systems-cost competitor.

Iter 25 adds a 9th member, **Trie Automata for Constrained Decoding over
Large Finite Sets** (arXiv 2608.12574, Amazon), to the **algorithmic
mask-speedup cluster** (PSC, GRID, Attention-Meets-Reachability,
Token-Space-Compression, Pre³, XGrammar-2, MetaJuLS, Stay Within Your
Bounds): Aho-Corasick trie-based per-node mask precomputation for
large-finite-set constraints (K≥300), reporting 7× faster per-step mask
computation than XGrammar — the cluster's largest raw-speedup number to
date. Same defining trait as every other member: benchmarked against
vLLM/SGLang cloud-serving backends, no edge hardware, no Joules, no
single-stream framing. Also filed (not a cluster member — different
generation paradigm) **Lookahead-then-Verify / LAVE** (arXiv 2602.00612):
constrained decoding for diffusion LLMs, parallel-sampling-then-verify
rather than autoregressive per-token masking, so it doesn't compete with or
extend either the mask-speedup or correctness-tax clusters — filed as
ignore purely to avoid re-discovery on a future dLLM-adjacent sweep.

Iter 24 adds a 7th member, **Stay Within Your Bounds / Distance-Guided
Decoding** (arXiv 2608.28229), to the **algorithmic mask-speedup cluster**
(PSC, GRID, Attention-Meets-Reachability, Token-Space-Compression, Pre³,
XGrammar-2, MetaJuLS): a pushdown-automata-reachability-distance decoder
guaranteeing CFG compliance with better completion quality on JSON/SQL/LTL.
Same defining trait as the other six — a genuine grammar-decoding-mechanism
contribution with no edge hardware, no Joules, no single-stream framing.
Iter 24 also grows the **correctness-tax cluster** to 9 members with **The
Structured Output Benchmark / SOB** (arXiv 2604.25359): a 7-metric
multi-source (text/image/audio) benchmark showing Value Accuracy drops
84.5%→67.2%→23.7% across those input modalities on cloud-scale models — adds
an input-modality-degradation axis the cluster hadn't captured, and is worth
watching specifically because its image-modality framing sits adjacent to
Route A's own vision choice (a reason to keep Route A's eventual contribution
clearly on the systems/hardware-cost side, not the accuracy-benchmark side,
to stay differentiated from SOB).

Iter 23 corrects a stale detail in the **general-purpose inference-engine
speedup cluster** watch item (BaseRT/MetalRT, Ollama's own MLX backend, last
touched iter 15/16): Ollama's MLX backend is no longer preview-only or
single-architecture. **Ollama v0.30.0** (May 2026) made it the default fast
path on capable (32GB+) Macs, with llama.cpp's Metal backend layered in for
wider GGUF support, and MLX now covers six model architectures (Gemma 3,
GLM-4 MoE Lite, full Llama series, Qwen 3, Qwen 3.5, Qwen 3.5 MoE) — three of
G1's four families. The fact that actually matters is unchanged, though: the
32GB+ RAM gate itself hasn't moved, and G1's measurement Mac (16GB) stays
under it on any Ollama version — so this remains **watch-not-act**, not a
live baseline-drift risk, just a more accurate description of what's being
watched for a future 32GB+ re-run.

Iter 22 grows the **correctness-tax cluster** to 8 members. Original 4
(problem-side): Constraint Tax (2605.26128), DCCD (2603.03305), schema-key-
wording (2604.14862), Format Tax (2604.03616). Plus ATLAS-RTC (2603.27905,
fix-side, iter 14). New this iteration — 2 more problem-side: **Structured
Output Collapses Answer Diversity** (2607.18476, 44-model diversity-collapse
study) and **Alignment Tax of Constrained Decoding in LLM Reflection**
(2604.06066, "structure snowballing" failure mode in self-correction); and
1 more fix-side: **When Correct Isn't Usable** (2605.02363, reports 3.6×–8.2×
overhead for a two-pass baseline, their own fix reaches 0.63×–1.06×) — its
overhead number is large enough that it was checked closely as a possible
scoop of G1/Route A's hardware-cost claim, but the mechanism (two-pass
generation cost, cloud-class models, no edge/Joules angle) is unrelated to
G1's per-token mask-computation-cost claim. All 8 members still share the
cluster's defining trait: no edge hardware, no Joules, no single-stream
framing, no per-token mask-cost mechanism — none compete with G1/Route A,
but the cluster is now large and multi-approach enough (accuracy loss,
diversity collapse, reflection failure, two independent latency-focused
fixes) that any final paper needs to position against all 8, not just the
original 3.

Iter 21 adds a sixth member, **Adaptive Constraint Propagation / MetaJuLS**
(arXiv 2601.00095, ACL 2026), to the **algorithmic mask-speedup cluster**
(PSC, GRID, Attention-Meets-Reachability, Token-Space-Compression, Pre³,
XGrammar-2) — a meta-RL constraint-propagation policy claiming 1.5–2.0×
speedup over GPU-optimized baselines on parsing and LLM-constrained
generation. Same defining trait as every other cluster member: benchmarked
against cloud/GPU baselines, no edge hardware, no Joules, no single-stream
framing — six-for-six now. Also filed one background survey, "Towards
Efficient Large Vision-Language Models" (arXiv 2603.27960), useful for
Route A's eventual related-work positioning but not itself competitive
pressure.

Iter 20 adds three more members (Dovetail 2412.18934, AHASD 2604.25326,
OpenPangu-NPU 2603.03383) to the **hetero-core speculative-decoding**
watch-list opened by Ghidorah (iter 19) and BaseRT (iter 11): the cluster
is well-populated and active, but a targeted second-pass search this
iteration found, again, **zero members that apply hetero-core scheduling to
grammar/mask computation** — every one optimizes speculative decoding's
draft/verify split across CPU/GPU/NPU, not grammar-state advancement or mask
generation. Two independent iterations (19, 20) now agree: the CPU-decode
mask-pipelining gap iter 18 opened stays unclaimed by anyone found so far.

Iter 19 added Ghidorah
(arXiv 2505.23219): 7.6× decode speedup on Jetson NX via
speculative decoding + CPU/GPU/NPU hetero-core scheduling — not a
grammar-masking paper, so not competitive with G1/Route A's hardware-cost
claim, but confirms iter 19's targeted search found **no one has yet applied
this class of hetero-core scheduling to grammar/mask computation
specifically** — the CPU-decode mask-pipelining gap iter 18 opened stays
unclaimed by anyone found so far.

Iter 18 fixed a standing ambiguity in this section: the paragraph below used
to name only **XGrammar-2** (2601.04426) as a tracked cluster member, which
could read as implying no cluster member does CPU/GPU mask/forward-pass
pipelining. That's wrong — the *original* **XGrammar** (2411.15100, on file
since iteration 2, 2026-08-10) is exactly that: it overlaps CPU-side mask
computation with the GPU's forward pass, syncing only before sampling. This
was never a hidden fact (it's spelled out in iter 2's own digest), just
under-flagged once Route A's pipelining framing came into existence four
days after XGrammar was first swept. It is **not a scoop**: XGrammar is a
separate engine, not integrated into llama.cpp/Ollama (G1's actual measured
engine), and Ollama's team explicitly declined to adopt it (confirmed
against `ollama/ollama#7900`, iter 14) — so the open contribution for Route
A is porting the known technique to llama.cpp and characterizing where its
near-zero-overhead assumption breaks on edge hardware, not inventing the
technique itself. Iter 18 also found independent evidence that the
assumption *does* break somewhere relevant: "Challenging GPU Dominance"
(arXiv 2505.06461) shows llama.cpp on an iPhone 15 Pro running a 1B model
*faster on CPU-only than with GPU acceleration* — if the forward pass is
already on CPU for small on-device models, CPU-mask/GPU-forward-pass
pipelining doesn't apply cleanly and needs a CPU-decode design answer. This
sharpens Route A's gap statement; it does not close it.

Iter 16 **closes** iter 15's Ollama-0.19-MLX watch item for the *existing*
data: 002–006 ran on Ollama 0.20.6 on a 16GB machine, definitively under the
32GB+ MLX gate, and 0.20.6's own release notes confirm no backend change —
so those numbers are settled, not a live risk. The **general-purpose
inference-engine speedup cluster** (BaseRT/MetalRT, iter 11; Ollama's own
MLX backend, iter 15) stays on file as a **future** watch-not-act item only:
if Maria's hardware or Ollama version changes before any re-run, re-check
the RAM/version gate again then — it's a baseline-drift risk for *future*
runs, not a caveat on 002–006.

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
  the harness-OS paper's planned controlled matrix once Maria answers
  sub-question 3 (starting use case: text vs. VLM).
- The three draft hypothesis shapes from iteration 3 (pipeline the mask;
  energy-per-valid-output; tail-latency framing) are subsumed by G1's role
  as a module inside the bigos harness-OS frame — the pipelining method is
  still G1's "hero optimization," now inside the umbrella paper rather than
  a standalone route.
- New this iteration (30): the harness-OS umbrella claim itself ("harness
  design beats model size for SLMs") is not novel as a bare statement — see
  `docs/research/digests/2026-09-08.md`. The paper's actual contribution
  needs to be stated explicitly as the low-level mechanism/measurement layer
  (G1's mask-cost/pipelining mechanism + rigorous edge-hardware Joules/ms
  telemetry) that the existing "harness > size" literature doesn't provide —
  worth confirming this framing lands correctly whenever a draft motivation
  section gets written.
- New this iteration (32): whenever a related-work section gets drafted, it
  needs an explicit paragraph distinguishing bigos's mechanism layer
  (kernel/runtime/decode-time — e.g. G1's mask-cost tax + shadow-mask
  pipelining) from AgentStop's (arXiv 2605.15206) agent-loop-control layer
  (early-terminating whole trajectories). Both are "harness mechanisms
  validated with real edge/device telemetry," so the difference is not
  obvious without saying it out loud — this is the single most important
  related-work distinction found so far for defending the paper's novelty.
- New this iteration (31): if/when the Jetson purchase question resurfaces
  (sub-question 5), arXiv 2511.11624 has directly usable diligence — Jetson
  Orin Nano w/ GPU has the best energy-to-performance ratio of Pi 5/Jetson
  Nano/Jetson Orin Nano for this model class. Not an action now, just a
  citation to reach for when that conversation happens.
- New this iteration (38): same Jetson sub-question — SLED (arXiv
  2506.09397) adds a third independent Jetson Orin Nano reference point and
  the first Raspberry Pi 4B one (alongside 2511.11624 iter 31, EdgeCIM iter
  32). Not an action now, just another citation to reach for when that
  conversation happens.
- New this iteration (36): whenever an on-device measurement session runs on
  Maria's iPhone 16 Pro (rather than the M1) — including any future Route A
  `mtmd` smoke test — arXiv 2603.23640 shows this exact device loses ~40% of
  peak throughput within 3 sustained iterations to thermal throttling. This
  is directly actionable for BRIEF.md's paper-grade requirement to record a
  thermal precondition per run: a warm-up/cooldown protocol or an explicit
  thermal-state field in the run manifest is needed before iPhone numbers
  could be trusted, not just an M1 note carried over unchanged.
- New this iteration (39): if the harness-OS paper's scope ever widens to
  include a power/frequency-management module alongside G1's grammar-mask
  module, DVFSLM (arXiv 2609.13153) is the directly relevant related-work
  citation and precedent (runtime DVFS governor for SLM decode workloads,
  energy-per-token under a deadline constraint). Not an action now — this is
  a citation to reach for if/when sub-question 1 (paper scope: whole harness
  vs. G1-first) resolves toward "whole harness."
- New this iteration (40): whenever the paper's introduction/motivation
  section gets drafted with an edge×robotics framing, Lite VLA (arXiv
  2511.05642) is the strongest "robot that sees, on-device, real hardware"
  precedent found so far — worth a sentence distinguishing its VLA/action-
  model shape from bigos's structured text/JSON generation under grammar
  constraints. Not an action now, just a citation to reach for when that
  section gets written.
- New this iteration (41): when the correctness-tax cluster's open item
  (task-correctness grading, DECISIONS.md 2026-08-14 #3) actually gets
  implemented as harness code, **Where vs What / SCD** (arXiv 2608.25358) is
  directly reusable methodology — its Structure-Content Decomposition
  separates placement errors (right value, wrong schema position) from
  value errors (wrong value), which is a sharper grading primitive than a
  flat correctness/incorrectness score. Not an action now (harness code is
  outside this loop's `docs/research/`-only scope), but the strongest
  methodology candidate found so far for that gate's actual implementation.
- New this iteration (42): whenever the correctness-tax related-work section
  gets drafted, **"Lost in Space" / "Finding the Right Tokens for Structured
  Output"** (arXiv 2502.14969) is worth leading with among the four
  "phrasing changes correctness" citations — it's the only one that
  explicitly measures the effect as **largest for smaller/local-laptop-scale
  models**, i.e. Maria's own model class. Not an action now, just a citation
  to reach for.
- New this iteration (42): if a future draft ever discusses prompt-
  engineering as an energy lever alongside G1's decode-time mechanism,
  **"Keyword Matters"** (arXiv 2607.22568) is the directly relevant
  precedent (real smartphone power measurements, prompt wording → energy).
  Not an action now.
- New this iteration (44): whenever the paper's introduction/motivation
  section gets drafted with an edge×robotics framing, **SCOPE** (arXiv
  2606.02951, HRI 2026) is a second strong "robot that sees, on-device"
  precedent alongside Lite VLA (iter 40) — this one for perception/camera-
  control (SLM planner + VLM perception pairing) rather than an action
  model, and from an actual robotics venue. Worth a sentence distinguishing
  its camera-agent shape from bigos's structured text/JSON generation under
  grammar constraints. Not an action now, just a citation to reach for.
- New this iteration (44): if the medical-VLM related-work section ever
  needs a concrete "edge-deployed + energy-modeled" citation rather than
  only accuracy-focused ones, DOI 10.1145/3796263.3796284 ("Edge-Deployed
  Lightweight LLMs for Medical Imaging") is the most on-point find so far —
  real Jetson/Raspberry-Pi target hardware, INT4 quantization, power
  modeling. Caveat: full text was `EGRESS_BLOCKED` (dl.acm.org) this
  iteration, so verify its grammar/mask-angle absence directly before
  citing it as a clean non-competitor.
- New this iteration (47): the DVFS/runtime-governance sub-thread is now
  four papers deep (DVFSLM 2609.13153, 2507.02135, 2604.15357,
  "Edge-Inference Governors Need Memory-Clock State" 2606.16106) — if
  sub-question 1 (paper scope) resolves toward "whole harness," this
  sub-thread needs its own related-work paragraph distinguishing power/
  frequency/memory-clock governance from G1's constrained-decoding
  mask-cost mechanism, same class of need as the AgentStop distinction
  (open-debt item from iter 32). Not an action now, just a citation
  cluster to reach for.
- New this iteration (47): the standing "Feb 16–23, 2027, Montréal"
  MLSys 2027 conference-location figure should be re-verified against a
  primary source before it's used in any travel-planning or paper-
  logistics context — real tweets from Zhihao Jia/Jeff Dean confirm MLSys
  runs in May in other recent years (2024, 2026), which fits the
  previously-not-adopted secondary "May 17–22, Bellevue, WA" claim (iter
  45) better than Montréal/February. Does not affect the Oct 30, 2026
  submission-deadline estimate, which is unchanged and now pattern-
  corroborated.
- New this iteration (49): if the correctness-tax-gap related-work section
  ever discusses fix-side mechanisms beyond decode-time masking, **PA-Tool**
  (arXiv 2510.07248) is directly relevant — a training-free, pre-decode
  schema-*renaming* fix (align schema wording to a small model's pretrained
  vocabulary familiarity) rather than a decoding-algorithm change, explicitly
  validated on small models. Also worth noting alongside it: **"Repair, Not
  Improvement"** (arXiv 2608.13959) tested tool-call *abstention* specifically
  on 0.6B–4B models — if Maria's own agent schemas include a legitimate
  "no tool applies" branch, this is the one citation on file that speaks
  directly to whether constrained decoding helps or hurts that specific case.
  Not an action now, just two citations to reach for when that section gets
  written.

## Next action (iteration 54)
1. **Check for Maria's answers to the five reserved sub-questions first**
   (paper scope, venue/timing, starting use case, co-author, Jetson — see
   "Current stage" above and `NEEDS_INPUT.md`'s current, non-archived
   section). Check `docs/research/DECISIONS.md`, chat/session context, and
   any Monday-block artifact under `docs/` for a fresh entry dated after
   2026-09-07. **This loop does not gate on these or re-ping about them**
   (per Maria's own instruction — they're hers to raise in conversation),
   but should fold in whatever has landed and update "Current stage"
   accordingly if something has. If nothing new: no action needed, this
   is not a stale-gate situation the way A/B/C/D was.
2. **Literature tracking (~15-20 min):** iteration 53 touched the
   **correctness-tax cluster** (four new-to-seen.json items — headline
   sub-4B-regime find VAREX, two independent schema-valid-≠-correct domain
   confirmations under a name-collision-flagged shared title, one ignore —
   plus two data-hygiene fixes). Per the no-repeat-two-iterations-running
   rule, iteration 54 should switch to the **algorithmic mask-speedup
   cluster** (last touched iter 50 — fifth consecutive nothing-new sweep,
   stuck at 11 members — now the most stale of the four rotating angles:
   mask-speedup iter 50, harness-OS-thesis iter 51, edge-VLM iter 52,
   correctness-tax iter 53). Given the cluster has returned nothing new for
   five straight sweeps (iters 33/38/42/46/50), consider trying a genuinely
   new angle rather than repeating the same search shapes again — e.g. a
   citing-work check on the cluster's largest-speedup members (WGRAMMAR,
   Trie Automata) now that they've had over a month to accumulate citations,
   or a fresh look at whether any 2026-Q4 kernel-engineering blog/repo
   activity (the zolotukhin.ai angle from iter 42 is still `EGRESS_BLOCKED`,
   worth one more direct-URL retry in case the block has lifted) has
   surfaced anything closer to the lab's own llama.cpp/edge regime than the
   cluster's cloud-serving-shaped members.
   Newly-opened threads worth folding into future sweeps, not urgent now:
   (a) the escalation/cascade-routing cluster is at 8 recognized members
   (PRISM/Privacy-Preserving-Routing/Edge-Cloud-Expert-Cascades/Hybrid-AI-
   Routers spot-recognized iter 51) — if ever picked as a dedicated angle,
   do a proper sweep-and-count pass rather than continuing to spot-recognize
   members in passing; (b) the DVFS sub-thread is 5 members deep (DVFSLM,
   2507.02135, 2604.15357, 2606.16106, SparseDVFS 2603.21908) — worth a
   dedicated DVFS-thread-only citing-work check next time harness-OS-thesis
   comes up (iter 55, unchanged from iter 51's note), separate from the
   general sweep; (c) 2507.08505 (OnePlus-13R llama.cpp VLM-deployment
   study) is worth a citing-work check in a future harness-OS-thesis or
   edge-VLM sweep once it's had more time to accumulate citations (submitted
   Jul 2026); (d) **new this iteration:** both VAREX (2603.15118) and
   ExtractBench×2 (2602.12247, 2607.29677) are document/field-extraction-
   framed rather than agent/tool-calling-framed correctness-tax evidence —
   worth checking next correctness-tax sweep (iter 57-ish) whether this is
   becoming a distinct "extraction benchmark" sub-thread worth tracking
   separately from the tool-calling/JSON-schema sub-thread, the way DVFS
   split off from the general harness-OS sweep.
   Citing-work re-checks not urgent for any of: 2605.12129 (~9 months old),
   AgentStop (checked iter 33), CRANE (clean negative, iter 35), DVFSLM
   (checked iter 39), EffGen (checked iter 43), "Seeing is Free, Speaking is
   Not" (checked iter 52, still fresh) — none worth repeating soon.
   Also re-check MLSys 2027 CFP status — the standing CFP/Dates/2027-group/
   OpenReview/Wayback-Machine targets are all still `EGRESS_BLOCKED`;
   iteration 50 confirmed the block is domain-level (`mlsys.org` itself), so
   no new subpath is worth trying. The "OpenReview submission portal
   opening" angle (portal opens Oct 10, 2026 per the now-fuller iter-53
   CFP-schedule figure) becomes independently checkable once that date
   passes — not yet as of iter 53 (2026-10-01), worth trying once Oct 10 has
   passed — **this deadline is now ~4.1 weeks out (29 days), inside the
   brief's <8-week verify-every-iteration threshold for the twenty-fifth
   iteration running, and the T-4-week WRITING-mode governor (~Oct 2, 2026)
   is only ~1 day out as of iter 53 and will very likely be crossed by
   iteration 54** — do not let this check lapse even on a literature-only
   iteration, and if the T-4-week mark has now been crossed before Maria
   answers sub-question #2 (venue/timing), flag that explicitly and plainly
   as a crossed (not just approaching) timing marker (still not a
   loop-authored gate, per her standing instruction not to re-ping on the
   five sub-questions).
3. Continue the check-the-working-tree-before-trusting-STATUS discipline —
   check `docs/research/DECISIONS.md`, `NEEDS_INPUT.md`,
   `docs/research/from-science/`, and `docs/` more broadly (real decisions
   can land as direct commits outside `docs/research/` too, even though
   this loop only ever writes inside `docs/research/`) for anything new
   every iteration, not just `experiments/results/`. Also re-verify local
   `main` is not behind `origin/main` before trusting any local file state
   — the leftover detached-HEAD/stale-branch-pointer pattern (iterations 8,
   14, 20, 25, 27, 28, 29, 33, 34, 36, 38, 39-40, 44, 48, 49, 51, 52) has
   been intermittent — keep fetch+ff-merge (or `git checkout main && git
   pull`, or `git checkout -B main origin/main` when local `main` itself is
   stale rather than just detached) as a standing check every iteration
   regardless.
4. **`seen.json` URL-format hygiene note (open since iter 45, second instance
   found iter 49, a related third instance found iter 53, still unfixed):**
   at least two entries are keyed on a `.../pdf/...` URL rather than the
   `.../abs/...` form used almost everywhere else —
   `https://arxiv.org/pdf/2606.25605` (iter 45) and
   `https://arxiv.org/pdf/2605.02363` (iter 49) — each nearly/actually caused
   a bare-date duplicate entry. **Iter 53 found a related but distinct third
   case**: `https://arxiv.org/abs/2604.03616` and
   `https://arxiv.org/html/2604.03616` were two separate bare-date entries
   for the same paper (both `/abs/` and `/html/` forms, not a `/pdf/`-vs-
   `/abs/` mismatch) — fixed this time by annotating both keys identically
   (not deleting either), since both were already present and the paper
   needed annotating anyway this iteration. The `/pdf/`-vs-`/abs/` cases
   remain open and not fixed unilaterally (would mean editing past
   iterations' entry keys) — three independent instances now on file across
   two slightly different key-duplication shapes — worth a standalone
   normalization pass (grep `seen.json` for all `arxiv.org/(pdf|html)/...`
   keys, cross-check each against a same-ID `.../abs/...` key, merge
   annotations, keep one canonical key per paper) sooner rather than later.
5. Pace-governor note: iteration 53 was **non-empty** (four new-to-
   seen.json items — one headline sub-4B-regime finding, two independent
   schema-valid-≠-correct domain confirmations with a name-collision flag,
   one filed ignore — plus two data-hygiene fixes); empty-iteration counter
   stays at 0 of 3.
