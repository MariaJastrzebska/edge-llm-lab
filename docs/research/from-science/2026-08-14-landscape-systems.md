# Systems-venue landscape: what is saturated vs open in edge/on-device LLM inference (2026-08-14)

*Author: systems-ML research strategist agent. Method: five parallel deep web surveys
(MLSys 2023–2026, OSDI/NSDI/ASPLOS/EuroSys/ATC, MobiCom/MobiSys/SenSys, USENIX
Sec/S&P/CCS, plus arXiv 2023–2026), cross-checked against this repo's own scoop
history (`STATUS.md`, `2026-08-14-mechanism-and-gaps.md`, `2026-08-12-G1-go-no-go-agent.md`).
All gaps are judged against Maria's actual rig: **one MacBook Air M1, single-stream,
per-token latency + memory + Joules (powermetrics; RAPL/tegrastats backends exist),
sub-1B–4B (VLM: up to ~7B) models via llama.cpp/Ollama, ~3-month runway.***

*ID-confidence note: 2026-prefixed arXiv IDs (26xx.xxxxx) came from live searches and
most were abs-page-verified by the survey agents; the load-bearing ones (2607.09520,
2604.27476, 2604.24447, 2604.18231, 2605.27599) were individually confirmed. Re-verify
any ID before it enters a related-work section. Two of five survey passes (the
KV/quant/spec-decode cluster deep-dive and the venue-deadline confirmation pass) had
not returned at write time; those sections below rest on the repo's prior scoop
checks + this session's direct searches and are marked accordingly.*

---

## 1. Saturation map

Legend: **RED** = saturated/owned, do not enter as headline. **YELLOW** = crowded but
with a rig-shaped open slice. **GREEN** = open. **BLUE OCEAN** = important AND
essentially unoccupied.

### 1.1 Core efficiency cluster (KV-cache, quantization, speculative decoding, sparsity/offload, unified memory)

*(Verdicts from repo scoop history + direct searches; the dedicated deep-dive pass was
still running at write time — treat as high-confidence directionally.)*

| Sub-area | Verdict | Key papers / owners | Rig-shaped opening? |
|---|---|---|---|
| KV-cache compression/eviction/quantized-KV | **RED** | H2O, StreamingLLM, SnapKV, KIVI, Quest; FlashInfer (MLSys'25 best paper). Owned by the big serving groups (Berkeley Sky, CMU/MLC, UW). | None as headline. KV-budget is at most a *knob* inside a deadline/energy study. |
| Weight/activation quantization for edge | **RED** | GPTQ, AWQ (MLSys'24 best paper), SmoothQuant, QuIP#; ~3.5 BPW threshold (2505.15030) is prior art; "Sustainable LLM Inference for Edge AI" (2504.03360) already did 28 Ollama-quantized models with J/token. | Only as an axis (quant tier as an energy-utility frontier), never the contribution. |
| Speculative decoding | **RED, and its edge-ENERGY slice is now contested** | Medusa, EAGLE-1/2/3; on consumer/Apple hardware: "Lossless but Not Free" (2607.17283), cross-family spec-decode on Apple Silicon MLX (2604.16368), SLED (2506.09397). | Avoid. The "spec-decode Joules on consumer hardware" square got taken in 2026. |
| Sparsity / MoE offload / flash offload | **RED** | PowerInfer/PowerInfer-2 (SJTU), LLM in a Flash (Apple, 2312.11514), EdgeMoE; "SSD Offloading for MoE Considered Harmful in Energy" (2508.06978, SNU) already owns the storage-tier-energy angle. | None on a 8–16 GB M1 Air. |
| Unified-memory scheduling / heterogeneous SoC | **YELLOW** | Agent.xpu (2506.24045, PKU — reactive/proactive agentic flows on Intel SoC, kernel-level preemption), ActiveFlow (2504.08378, MSRA), KAIROS/BiScale-class schedulers. Generic edge scheduling is crowded (STATUS.md already flags this). | **Open slice: nobody has published the energy/wall-clock cost curve of MODEL SWAPPING on unified memory** (llama-swap / Ollama keep_alive are universally used, never characterized) — see Gap 5. |

### 1.2 Energy / power / thermal (dedicated survey pass)

| Sub-area | Verdict | Key papers / owners | Rig-shaped opening? |
|---|---|---|---|
| Generic J/token, tokens/Wh, EDP on edge | **RED — closed in 2025–26** | EnerInfer (2606.23001), MEADOW (MLSys'25, 2503.11663), Camel (2508.09173, Jetson bandit-tuned EDP), SLM-energy-on-edges (2511.11624), 2504.03360, "Energy-per-Token" position (EuroMLSys'25, 2603.20224). Owners: Haoxin Wang (GSU), Yale+IBM, Purdue (Y.C. Hu), Michigan ML.ENERGY. | "Yet another J/token characterization" is dead. The substrate value survives only via a *mode-specific* axis (constrained decoding, agent loops, deadlines). |
| DVFS / battery-aware scheduling | **RED on phones; infeasible-as-stated on Mac** | CORE (MLSys 2026, Purdue — the headline mobile-LLM DVFS paper), MNN-AECS (2506.19884, Alibaba — engine-level core selection, already covers iOS). | Apple Silicon exposes no public DVFS knob. Only an observational P/E-core-placement energy study remains — narrow, not a headline. |
| Thermal-throttling-aware inference | **GREEN (thin)** | Sustained-load edge study (2603.23640); EnerInfer's thermal objective. No lab owns it. | **Fanless M1 Air is a natural passive-cooling testbed**: thermal-throttle-induced J/token drift over long generations on Apple Silicon is unclaimed. Best used as a *section*, not a paper. |
| Energy measurement methodology | **YELLOW → GREEN for Apple Silicon** | ML.ENERGY/Zeus (2505.06371, Michigan — has `zeus-apple-silicon` + Jetson support), TokenPowerBench (2512.03024), ELANA (2512.09946). Established finding: decode ≥96% of energy for typical gen tasks (2606.17104). | **Open: the M1 single-stream measurement-validity study** — powermetrics sample-alignment to token boundaries, missing DRAM/ANE counters on M1, cross-validation against RAPL + tegrastats on the same llama.cpp build. High reuse value; makes every other result more credible. See Gap 6. |

### 1.3 Structured / constrained decoding overhead (where G1 sits)

**Verdict: the ALGORITHM space is RED and consolidating; the ENERGY axis is empty
even on servers; the edge single-stream square is (still) unoccupied.**

- Owners: **XGrammar/XGrammar-2** (2411.15100, 2601.04426 — CMU/MLC, now inside
  SGLang/vLLM/TensorRT-LLM), **SGLang compressed-FSM/jump-forward** (2312.07104,
  LMSYS), **llguidance** (2502.05111, Microsoft), **Outlines** (.txt), plus the
  mask-cost academics: CFGzIP (2605.29986), GRID (2607.11951), Pre³ (2506.03887),
  PSC (2608.03065-class), SynCode (2403.01632).
- **Every one reports latency/throughput. No paper reports Joules for grammar
  masking — not on server, not on edge.** (Re-confirmed by two independent survey
  passes this session; consistent with the repo's iter-7 scoop check.)
- The correctness-tax cluster (Constraint Tax 2605.26128, "When Correct Isn't
  Usable" 2605.02363, "The Parser Already Knows" 2608.10137) is a different axis —
  citations, not scoops.
- **G1's square — constrained-decoding tax in ms AND Joules, single-stream edge,
  mechanism = un-overlapped O(vocab) CPU mask-walk in llama.cpp — remains unscooped
  as of 2026-08-14.** The risk to G1 is not scooping; it is *narrowness* (one
  engine, one mechanism, arbitrary JSON schema) and the internal mechanism-attribution
  problem (the 28× spread; see `2026-08-14-mechanism-and-gaps.md` attack #9).

### 1.4 Privacy / confidential / federated

| Sub-area | Verdict | Key papers / owners | Rig-shaped opening? |
|---|---|---|---|
| Privacy as a *measured* axis vs energy/latency | **BLUE OCEAN** | Privacy-as-motivation is universal boilerplate; only "Sustainability Is Not Linear" (2603.26603) and "Securing LLMs in the Wild" (2607.13088 — proposes the SOES metric but never instantiates it on hardware) even pose the trade-off. Zero-egress psychiatric AI (2604.18302) and clinical SLM extraction (2605.04221) are architecture/F1 only. **No lab owns "privacy-cost-in-Joules."** | **Yes — the field wrote the metric and never measured it.** See Gap 3. |
| TEE / confidential inference | **RED datacenter, thin edge, INFEASIBLE on M1** | ETH SPCL CPU/GPU-TEE study (2509.18886), TZ-LLM (TrustZone llama.cpp, 2511.13717), AgenTEE (2604.18231, Imperial/Haddadi, EuroMLSys'26), TensorShield (CCS'25, 2505.22735 — the only edge-TEE-with-energy paper, and it's CNNs). | M1 Secure Enclave cannot host a model — **do not build an enclave**. Open adjacent slice: the *Joules tax of confidentiality* (every confidential-LLM paper omits energy); reachable only with a ~$100 TrustZone SBC added to the rig. |
| Federated LLM at the edge | **RED and structurally wrong for the rig** | FwdLLM (ATC'24), FlowerTune benchmark (2506.02961, Flower/Cambridge — includes medical), EdgeFlowerTune (2605.08636), federated sparse-MoE (EuroSys'26). Owners: Mengwei Xu (BUPT), Nic Lane (Cambridge/Flower). | A single M1 cannot exhibit federated dynamics; a one-device "federated" paper gets desk-rejected. **Do not use this framing.** |

### 1.5 Agents / tool-calling / orchestration / real-time on edge

| Sub-area | Verdict | Key papers / owners | Rig-shaped opening? |
|---|---|---|---|
| Tool-calling ACCURACY on edge | **RED** | BFCL v4, τ-Bench, Gorilla, Octopus v1–v4 (2404.01744); MLPerf Edge Agentic (v6.1, Qwen3.6-27B GGUF function-calling) now owns generic benchmarks. | None. Do not compete with leaderboards. |
| Agent/tool-call ENERGY on device | **BLUE OCEAN** | CarbonCall (2504.20348, Jetson, aggregate power only), "Less is More" (DATE'25, 2411.15399), AgentStop (2605.15206, trajectory-level early stop), server-side "Energy per Successful Goal" (2605.22883) & "Watts and Debts of Agentic Frameworks" (2606.10702). **Key validator: "The Energy Blind Spot" (2605.27599) — Jetson cannot do process-level energy attribution; Apple Silicon powermetrics CAN.** | **Yes — Joules per successful tool call / per agent step, including retry/re-plan waste, as an energy-fidelity Pareto frontier. Nobody normalizes edge energy by task success.** Maria's fidelity+judge harness IS the missing correctness axis. See Gap 2. |
| Multi-model orchestration on one device | **YELLOW** | RouteLLM (ICLR'25), FrugalGPT, Hybrid-LLM — all optimize $-cost, not device Joules; ActiveFlow, LLM-in-a-flash are single-model. | **Open: swap-energy curve + small→big handoff break-even in J-per-correct-answer** (Gap 5). |
| Deadline/SLO-driven serving | **RED datacenter; GREEN single-stream** | Sarathi-Serve, DistServe, Llumnix (all OSDI'24), Splitwise (ISCA'24), Andes (QoE), SOLA (MLSys'25) — all multi-request goodput; their levers (batching, chunked prefill, disaggregation) collapse to nothing at N=1. | **Open: the (deadline attainment × Joules × task fidelity) Pareto frontier for a single stream on one device.** Safety framing: deadline miss = safety event (surgical/robotic), which datacenter goodput papers structurally lack. Feeds Gap 2/4. |

### 1.6 VLM axis (added at Maria's request): on-device vision-language inference

**Overall: edge-VLM *models* are crowded; edge-VLM *systems measurement* is nearly
empty; the medical-vision + energy + privacy intersection is the emptiest cell found
in this entire survey.**

| Sub-area | Verdict | Key papers / owners | Notes |
|---|---|---|---|
| Edge VLM energy/latency/thermal | **GREEN — exactly ONE direct paper** | **"Seeing is Free, Speaking is Not" (2607.09520, ACM MM'26)** — first systematic edge-VLM energy profile (5 models 1B–8B, Jetson Orin NX + RTX 3070). Findings: **decode dominates energy — each output token costs 11–39× an input token; visual-token pruning saves ≤10%; output-length capping saves up to 97%.** Also EdgeFM (2604.27476 — VLM inference framework, speedups not Joules), MELTing point (MobiCom'24, 2403.12844 — text-only, the methodology template), MobileAIBench (2406.10290). | 2607.09520 owns the *generic* encoder-vs-decode energy split — but measured on discrete-GPU hardware. **Whether decode-dominance holds on unified-memory Apple Silicon is an open empirical question — no Apple-Silicon+powermetrics+VLM paper exists (zero hits).** |
| On-device medical/surgical VLMs | **Accuracy RED / systems BLUE OCEAN** | Surgical-LLaVA (2410.09750), LLaVA-Surg (2408.07981), GP-VLS (Stanford, 2407.19305), EndoChat (2501.11347, CUHK/TUM/Strasbourg), LLaVA-Med (2306.00890), MedVLM-R1 (2502.19634). Efficiency-*flavored* but zero device numbers: Med-MoE (EMNLP'24), LiteMedCoT-VL (2605.09384 — claims "portable clinical devices," measures nothing on-device). | **Zero papers measure latency/memory/Joules for a medical/surgical VLM on real edge hardware.** The field talks deployment and measures accuracy. |
| Constrained/structured decoding ON VLMs | **BLUE OCEAN (n≈1 applied, 0 measuring cost)** | Multimodal Structured Generation (CVPR'24 MMFM challenge report, 2406.11403) — accuracy only, engine unnamed. Nothing else on-target. | Constrained decoding on VLMs is barely explored; **measuring its systems cost on VLMs is a literal zero.** |
| VLA action-model edge energy | **RED — late-stage crowding; also weakest M1 fit** | "Characterizing VLA across XPUs" (2604.24447 — the "Char-VLA-XPU" shorthand; Cost/Energy/Time on GPUs/NPUs/edge), QuantVLA (2602.20309), OpenVLA-OFT (2502.19645), FAST/π0-FAST (2501.09747), Deltoris (2608.04428). **"Green-VLA" (2602.00919) is a red herring — "Green" is a robot platform codename, not energy.** | TAKEN: Jetson VLA latency/energy characterization, PTQ-for-VLA, "action-gen is the bottleneck." OPEN but ill-fitting: Apple-Silicon VLA energy — reviewers at robotics venues will call a robot-less M1 unrepresentative. **Scope to VLM, not VLA.** |

**VLM-vs-text verdict (the key question):** the VLM framing **strengthens** the
constrained-decoding-tax story rather than diluting it. The worry that the heavy
vision encoder dilutes the relative tax is **refuted by 2607.09520's own data**:
decode — not the encoder — dominates edge VLM *energy*, and the constrained-decoding
CPU mask-walk lands per-output-token exactly in that dominant phase. Meanwhile
medical vision supplies the **non-arbitrary schema** the text version lacks
(synoptic operative reports, endoscopy findings templates are externally defined
clinical standards, not author-invented JSON) and privacy supplies the reason the
frame stays on-device. Three residual cautions: (a) VLM support in llama.cpp
(mtmd) must be smoke-tested on the M1 for grammar+image workloads before
committing; (b) 2607.09520 must be cited as the generic-profile owner — the
contribution is the constrained-decoding tax + unified-memory replication, not a
re-profile; (c) stay VLM/medical, not VLA/robot.

---

## 2. Ranked OPEN gaps measurable on Maria's rig

Ranking = importance × emptiness × fit-to-rig × fit-to-interests (privacy, medical,
energy, bigos) × 3-month feasibility.

**Gap 1 — The structured-decoding energy tax on edge VLMs (G1 upgraded to vision).**
Measure the grammar/JSON-constrained decoding tax in ms/token AND Joules on edge
VLMs (LLaVA/MiniCPM-V/Qwen2-VL/moondream class in llama.cpp) emitting real clinical
synoptic-report/tool-call schemas from medical images; replicate-or-overturn
2607.09520's decode-energy-dominance on unified-memory Apple Silicon; apply the
already-designed shadow-mask pipelining as the optimization leg.
*Why top paper:* it sits at the intersection of four independently verified empty
cells (constrained-decoding energy = unmeasured anywhere; constrained decoding on
VLMs ≈ n=1; medical-VLM systems numbers = zero; Apple-Silicon VLM energy = zero),
keeps everything G1 already built (mechanism, matrix, pipelining), and fixes G1's
two weaknesses — arbitrary schema and narrowness — in one move.

**Gap 2 — Joules per correctly-completed task: the on-device agent energy-fidelity
frontier (G1′/G3 generalized).**
Make "J per successful tool call / per agent step, retry and re-plan loops
included" a first-class metric; sweep model size × quant × constraint mode on the
M1; her tool-calling-fidelity + LLM-judge harness supplies the correctness
denominator every existing energy paper lacks ("The Energy Blind Spot" 2605.27599
proves the field wants per-step attribution and can't get it on Jetson — powermetrics can).
*Why top paper:* nobody normalizes edge energy by task success; it is the exact
scheduling primitive an "OS for agents on edge" needs, and every crowded literature
(energy, serving, routing, agents) supplies one axis and omits the fusion.

**Gap 3 — The energy cost of privacy: on-device vs cloud for medical extraction.**
Instantiate the SOES-style trade-off nobody has measured: J and ms premium of the
zero-egress on-device path vs a sanitize-then-cloud baseline, on the same medical
field-extraction (or image→report) workload, across model/quant tiers as a
privacy-utility-energy frontier.
*Why top paper:* "Joules per token of privacy" is a number the field explicitly
set up (2607.13088, 2603.26603) and never produced; it is the missing quantitative
argument for the entire on-device-AI privacy narrative.

**Gap 4 — Deadline-driven single-stream inference: the (deadline × Joules ×
fidelity) frontier.**
Under a hard per-token/per-response deadline + energy budget, which knob (model
choice, quant, KV budget, constraint mode, early-stop) wins? Datacenter SLO
scheduling collapses at N=1, so the single-device version is unowned; safety
framing (deadline miss = safety event) gives it robotics teeth without needing a robot.
*Why top paper:* first real-time treatment of edge LLM/VLM inference where energy
and correctness are co-constraints, not afterthoughts.

**Gap 5 — The cost of a model swap on unified memory.**
Characterize J + wall-clock of unload-A/load-B on Apple Silicon (size × quant ×
mmap/mlock × cold/warm), then the small→big handoff break-even in J-per-correct-answer —
reframing RouteLLM/FrugalGPT's $-cost logic for the edge.
*Why top paper:* llama-swap and Ollama keep_alive are used by everyone and
characterized by no one; it is the multi-model scheduling primitive bigos needs.

**Gap 6 — The M1 measurement-validity substrate (methodology paper or killer §2).**
Powermetrics sample-to-token alignment at 10–40 tok/s, M1's missing DRAM/ANE
counters, package-vs-GPU accounting, cross-validation vs RAPL and tegrastats on
one llama.cpp build.
*Why top paper (or paper-section):* Zeus provides primitives but no one has
published the Apple-Silicon single-stream validity study; it hardens every other
gap's numbers and answers the "n=1 hardware" reviewer attack.

**Explicitly avoid as headline:** generic J/token characterization, speculative-decoding
energy on consumer hardware (taken in 2026), prompt-wording energy (2607.22568 owns it),
KV/quant/sparsity algorithms, federated framing, TEE mechanism on M1, VLA-on-M1,
tool-calling accuracy benchmarks, datacenter SLO serving.

---

## 3. Venue fit and deadlines (~3-month runway from mid-Aug 2026)

*(Deadline-confirmation pass still pending at write time; dates below are from
STATUS.md's repeated checks + this session's searches. MLSys 2027 CFP remains
unconfirmed on mlsys.org — treat Oct 30 as planning date, re-check weekly.)*

| Venue | Deadline (est.) | Fit for single-M1, single-stream, 3 months |
|---|---|---|
| **MLSys 2027** | **~Oct 30, 2026 (UNCONFIRMED)** | **Primary target.** MLSys accepts rigorous measurement+mechanism+optimization on real hardware (MEADOW, CORE precedents). Tight but feasible if the pilot data already exists (it does for G1; VLM leg needs ~4–6 weeks). |
| **EuroMLSys (@EuroSys 2027)** | ~Jan–Feb 2027 | **Excellent fit** — workshop explicitly welcomes single-device edge measurement (energy-per-token position paper, AgenTEE precedents). Ideal fallback / fast first publication. |
| HotMobile 2027 | ~Nov 2026 | Good fit for a provocative short paper (Gap 3 or 5 as an 8-pager). |
| MobiSys 2027 | ~Nov–Dec 2026 | Fits if framed as mobile-systems measurement; expects more device breadth (add Jetson via tegrastats backend). |
| RA-L | rolling | **Absorbs any slip**; takes the robotics/deadline framing (Gap 4) directly. Keep as the standing backup per the go/no-go doc. |
| OSDI / NSDI / ATC 2027 | spring 2027 | **Poor fit** — expect distributed scale; single-device measurement studies are structurally disadvantaged. |
| ASPLOS / EuroSys 2027 | multiple cycles / ~Oct 2026 | Marginal — possible only with a strong architectural mechanism story (the unified-memory contention analysis could qualify at ASPLOS, but it's a stretch in 3 months). |
| IISWC / ISPASS 2027 | ~spring 2027 | Natural home for the characterization if MLSys slips; measurement-first venues. |

**Timeline reality:** the only combination that hits MLSys 2027 is one that reuses
the existing harness ≥80%. Gap 1 (VLM constrained-decoding tax) and Gap 2
(J-per-correct-task) both do; Gaps 3–5 are EuroMLSys/HotMobile-sized in this runway.

---

## 4. Bottom line vs G1

G1's square is real and still unscooped, but as a text-only, llama.cpp-only,
arbitrary-JSON story it is narrow for a top venue — Maria's instinct is right.
The survey's answer is not to abandon it but to **re-platform it (Gap 1) and
re-denominate it (Gap 2)**: the same mechanism, matrix, and shadow-mask-pipelining
optimization, executed on medical VLM workloads and reported in Joules-per-correct-
clinical-output, with the privacy premium (Gap 3) as a section rather than a paper.
That composite keeps every asset in the repo, occupies four empty cells at once,
and reads as "the energy/latency systems cost of private, structured, on-device
medical vision" — a claim no group currently owns.
