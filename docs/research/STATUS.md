# Paper machine — STATUS (updated 2026-09-07, iteration 29)

**Scope decisions so far (newest first):**
1. 2026-09-07 (iter 29): no scope-narrowing decision — `NEEDS_INPUT.md`
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

## Current stage: **3/4 boundary — blocked on route confirmation.**
**NEW GATE OPEN — see `NEEDS_INPUT.md` at repo root.**

Experiment 005 already satisfied Stage 4's exit criterion ("a 1-week pilot
supports or kills the hypothesis") for **G1 in its original text-only
framing**: all four tested sub-1B models show a positive, CI-clean latency
tax below ~25ms/token (qwen2.5:0.5b +1.84%, qwen3:0.6b +2.42%, llama3.2:1b
+6.15%, gemma3:1b +30.36%). That result stands and isn't in question.

What's NOT resolved is which *paper* that result feeds into. The from-science
strategic review scored G1's original text framing as the **weakest** of four
routes (relegated to "Trasa D — fallback… a generalization cell within A")
and recommended **Route A**, the VLM/medical generalization, instead. Per
Stage 3's own exit criterion in `BRIEF.md` — "top candidate passes novelty
check + feasibility + **MARIA CONFIRMS it** ([NEEDS-INPUT] gate — genuinely
hers)" — that confirmation hasn't happened in so many words yet, even though
her subsequent hardware decisions read as compatible with it. See
`NEEDS_INPUT.md` for the exact question and the four options (A/B/C/D) with
their tradeoffs.

<details><summary>Stage-2 exit justification (honest accounting, unchanged since iter 2)</summary>

~40 sources swept pre-Stage-4 on the original text-LLM framing; the surviving
axis (energy/decoding machinery) cleared the ≥15-source bar comfortably. Axes
1 and 2 were closed at 8–9 sources each by the kill criterion, not exhaustive
sweep. The from-science strategic review (2026-08-14) constitutes a second,
much larger sweep specifically for the route-selection question — see
`docs/research/from-science/2026-08-14-landscape-systems.md` and
`2026-08-14-landscape-robotics-medical.md` for their own source counts.

</details>

## Nearest deadlines (re-checked 2026-09-07, iter 29)
| Venue | Deadline | Note |
|---|---|---|
| ~~ICRA 2027~~ | ~Sep 15, 2026 | now ~10 days out — too tight for a primary target regardless of route; from-science review already scored Route B (robot-grammar) as losing to Route A partly for this reason. Stays open as a paper #2 possibility in the spring robotics season. |
| **MLSys 2027** | **Oct 31, 2026, 11:59 AM UTC/AoE — still UNCONFIRMED against a primary source** | `mlsys.org/Conferences/2027/CallForResearchPapers` still blocked by network egress (WebFetch, `EGRESS_BLOCKED`) at iter 28. WebSearch aggregators (mlciv.com, mlsys-deadlines.github.io, trybibby.com) now converge on a specific figure — submission deadline Oct 31 2026 11:59 AM UTC/AoE, supplementary material/code Jul 31 2026 (passed), notifications Nov 30 2026, conference Feb 16–23 2027 Montréal — a one-day refinement of the ~Oct 30–31 estimate carried since iter 5, still third-party consensus not verified primary-source text. Re-check every iteration. (iter 29: unchanged, still `EGRESS_BLOCKED`.) |
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
Still needs a harness change (task-correctness grading) outside this loop's
`docs/research/`-only scope — flagged, not fixed, applies to whichever route
is confirmed.

## Scoop-check status (iter 29: zero new competitive papers, one new algorithmic-mask-speedup-cluster member (WGRAMMAR, 10th) and one data-hygiene fix (PSC's arXiv ID recorded, no cluster-size change); iter 28: zero new competitive papers, three edge-VLM/medical-structured-generation items, all ignore (no edge/latency/constrained-decoding angle in any), continuing the standing zero-systems-cost-competitor pattern; iter 27: zero new competitive papers, four cluster-relevant citations (two code-gen-domain constrained-decoding papers scored ignore, two edge-hardware-measurement papers — lm-Meter, Sustainable LLM Inference for Edge AI — scored differentiate-from/cite as methodology precedent); iter 26: zero new competitive papers, one cluster-member citation (EdgeFM joins general-purpose inference-engine speedup cluster as its first VLM-specific member), two ignores reinforcing the standing no-systems-cost-competitor pattern in edge-VLM/medical literature; iter 25: zero new competitive papers, one cluster-member citation (Trie Automata joins mask-speedup cluster as 9th member), one new-generation-paradigm item filed as ignore (LAVE/diffusion-LLM decoding); iter 24: zero new competitive papers, two cluster-member citations (SOB joins correctness-tax cluster as 9th member, Stay Within Your Bounds joins mask-speedup cluster as 7th member); iter 23: zero new competitive papers, one baseline-drift watch item re-verified and corrected (Ollama v0.30.0 MLX-default transition, no drift risk confirmed); iter 22: zero new competitive papers, three correctness-tax-cluster citations, one data-hygiene fix; iter 21: zero new competitive papers, one cluster-member citation, one background survey filed; iter 20: zero new competitive papers, a second independent negative result confirming Route A's mask-pipelining gap, three hetero-core speculative-decoding cluster citations, one old entry clarified; iter 19: zero new competitive papers, one negative result strengthening Route A's open gap, one hardware-class/technique citation; iter 18: zero new competitive papers, one design-relevant find, one standing ambiguity fixed; iter 17: zero new competitive papers, one carried-over citing-work check resolved; iter 16: zero new competitive papers, one baseline-drift item fully resolved; iter 15: zero new papers, one new engine-release baseline-drift event; iter 14: two new items, none a scoop; iter 13: one new item, not a scoop; iter 12: three new items, none a scoop; iter 11: six new items, none a scoop; iter 10: zero new items; iter 9: seven new items, none a scoop)
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

## Known competitive pressure (updated iter 29)
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
  Route A's planned controlled matrix if that route is confirmed (see above).
- The three draft hypothesis shapes from iteration 3 (pipeline the mask;
  energy-per-valid-output; tail-latency framing) are subsumed by whichever
  route Maria confirms — Route A explicitly inherits the pipelining method as
  its "hero optimization."

## Next action (iteration 30)
1. **Check `NEEDS_INPUT.md` first** (unchanged since iter 13's "Update"
   section — iter 29 added no new update to the file itself, since nothing
   rose to tier-1). Still open as of iter 29 (gate age ~23 days). No
   `DECISIONS.md` entry or chat decision answering the route question yet,
   but Maria remains actively engaged via the collab protocol
   (`docs/WSPOLPRACA-BADAWCZA.md` et al.) — **do not treat continued gate
   silence as grounds to re-ping**; the one constitution-mandated ping
   already fired at iter 10. If Maria has answered by iter 30 (via chat,
   `DECISIONS.md`, or a Monday-block artifact under `docs/`): close the gate
   per her choice, update "Current stage," and — only if she also said the
   literature-only restriction is lifted — resume normal pacing. If not yet
   answered: stay in reduced-pace literature-only mode. Pace-governor note:
   iteration 29 was **non-empty** (one new cluster-member citation, one
   data-hygiene fix); empty-iteration counter stays at 0 of 3.
2. **Literature tracking (~10 min):** re-check MLSys 2027 CFP status — try
   `mlsys.org/Conferences/2027/CallForResearchPapers` again (still
   `EGRESS_BLOCKED` at iter 29, but blocked domains are inconsistent run to
   run) in case the organizing committee/CFP has been announced, and confirm
   whether the Oct 31 2026 11:59 AM UTC/AoE figure (WebSearch consensus
   since iter 5, unchanged through iter 29) ever gets corroborated by a
   primary source. For the general sweep: **switch angles** — the
   algorithmic mask-speedup cluster (this iteration) is now fresh again;
   per the no-repeat-two-iterations-running rule, iteration 30 should move
   to a different angle. Candidates: (a) the **correctness-tax cluster**
   (last swept iter 27, will be 3 iterations stale by iter 30, currently 9
   members) — the most stale of the active general-sweep angles; (b) the
   **edge-VLM/medical-structured-generation** angle (last swept iter 28, 2
   iterations stale by iter 30, six consecutive sweeps with zero
   systems-cost competitors); (c) the **hetero-core mask-pipelining
   question** (iters 18–20, stale 11 iterations) — only worth a fresh check
   if something concrete has changed, since two independent negative
   results already stand. Default to (a) since it's the most stale with no
   new trigger needed.
3. Continue the check-the-working-tree-before-trusting-STATUS discipline —
   check `docs/research/DECISIONS.md`, `docs/research/from-science/`, and
   `docs/` more broadly (real decisions can land as direct commits outside
   `docs/research/` too, even though this loop only ever writes inside
   `docs/research/`) for anything new every iteration, not just
   `experiments/results/`. Also re-verify local `main` is not behind
   `origin/main` before trusting any local file state — the leftover
   detached-HEAD/stale-branch-pointer pattern (iterations 8, 14, 20, 25, 27,
   28, 29) has been intermittent, not fixed, so keep fetch+ff-merge (or
   `git checkout main && git pull`) as a standing check every iteration
   regardless, even on iterations where it doesn't recur.
