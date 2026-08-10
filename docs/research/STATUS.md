# Paper machine — STATUS (updated 2026-08-10, iteration 1.5)

**Scope decisions so far (newest first):**
1. 2026-08-10: direction = **edge × robotics × privacy and/or medical**; must be
   low-level/systems-strong; should ideally (not necessarily) feed bigos later.
2. 2026-08-10: "multi-turn agentic eval" framing DEMOTED (not low-level enough).

**Current stage: 2** — gap sweeps on the candidate axes below.

## Nearest deadlines
| Venue | Deadline | Note |
|---|---|---|
| ICRA 2027 | **~Sep 17–22, 2026** | 5 weeks — only if a validated idea appears fast |
| **MLSys 2027** | **Oct 30, 2026** | 11 weeks — prime target for low-level systems work |
| RA-L | rolling | always-open fallback |
| IROS 2027 / RO-MAN / ICORR | ~Feb–Mar 2027 | comfortable; medical-HRI friendly |

## Top candidate axes (Maria's leads + prior systems ideas, pre-gap-sweep)
1. **Federated small language models on edge** — privacy-preserving on-device
   personalization/learning; medical fit (patient data never leaves), robotics
   fit (fleet of home/care robots learning collectively), reuses her FL thread +
   harness. Gap-sweep Q: what's actually published on federated SLM fine-tuning
   on real edge hardware (not simulation)? Where's the systems contribution —
   memory/energy cost of on-device training steps?
2. **On-device visual/VLM inference for home & care robots under privacy
   constraints** — the "robot that sees" future; video = most privacy-critical
   modality; systems angle: efficient VLM serving on edge SoCs (latency/energy/
   memory), medical angle: visual monitoring of patients at home. Newest, least
   prior-art-checked.
3. **Energy/battery-aware inference for robot autonomy** — battery = mission
   time; mechanism+policy work (scheduling, quantization, KV policies) with
   Joules-per-task on robot workloads. Builds most directly on her telemetry.
4. (held from iter 1, still alive as *mechanisms* inside 1–3: constrained
   decoding performance, KV-cache management, heterogeneous-core scheduling)

## Known competitive pressure
MLPerf Edge Agentic owns generic tool-calling benchmarks; ~3.5 BPW threshold
(arXiv 2505.15030) is prior art. Benchmark-framing = dead; mechanisms = open.

## Next action (iteration 2)
Stage-2 gap sweep on axes 1–3 (federated SLM on-device training systems; edge
VLM serving for robots; energy-aware robot inference) → rank into hypotheses
with experiment sketches + venue mapping (MLSys Oct 30 is the clock).
