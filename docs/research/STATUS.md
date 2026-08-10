# Paper machine — STATUS (updated 2026-08-10, iteration 1)

**Current stage: 1 → 2** (venue table seeded; gap analysis started; earlier
"multi-turn eval" idea DEMOTED per Maria — not low-level enough).

## Nearest deadlines
| Venue | Deadline | Note |
|---|---|---|
| ICRA 2027 | **~Sep 17–22, 2026** | 5 weeks — only if a validated low-level idea appears fast |
| **MLSys 2027** | **Oct 30, 2026** | 11 weeks — best fit for low-level systems work; serious candidate |
| RA-L | rolling | fallback, always open |
| IROS 2027 / RO-MAN / ICORR | ~Feb–Mar 2027 | comfortable; RO-MAN = medical-HRI angle |

## Top-3 topic candidates (Stage-3 drafts — to be refined by gap analysis)
1. **Grammar-constrained decoding, fast**: low-level llama.cpp work on
   constrained/structured-output decoding for tool calls (kernel/sampler level;
   the recent GBNF fixes in bindings show the machinery is immature). Medical
   fit: guaranteed-valid structured medical records on device. Venue: MLSys.
2. **Energy-aware inference scheduling on heterogeneous edge SoCs** (P/E cores,
   NPU/GPU offload for llama.cpp): mechanism + policy, Joules-per-valid-tool-call
   as the metric — builds directly on the telemetry layer. Venue: MLSys/SenSys.
3. **KV-cache management under memory pressure for long-running on-device
   agents** (unified memory, quantized KV, eviction policies for multi-turn
   sessions): the iPhone app is the living testbed. Venue: MLSys/EuroSys-ws.

## Known competitive pressure (digest 2026-08-10)
MLPerf Edge Agentic standardizes generic tool-calling benchmarks — benchmark-
framing papers are dead; mechanism papers are the open ground. ~3.5 BPW
quantization threshold (arXiv 2505.15030) is prior art to build past, not repeat.

## Next action (iteration 2)
Stage 2 sweep for candidates 1–3: what exists in constrained decoding
performance, llama.cpp scheduling on heterogeneous cores, and KV-cache policies —
then rank survivors into hypotheses with experiment sketches.
