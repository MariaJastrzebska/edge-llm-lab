# Standing auto-research brief — edge-llm-lab

Purpose: keep Maria's paper work informed weekly, without her searching. A worker
(or Claude session) answers these questions, filters against `seen.json`, and
appends a dated digest to `docs/research/digests/`. Only NEW-since-last-time items
enter a digest. Ping Maria (Telegram/alert) ONLY for tier-1 events.

## Research questions (ranked)
1. New papers/benchmarks on **tool-calling / function-calling fidelity of small
   quantized LLMs on edge devices** (keywords: BFCL, function calling, GGUF,
   quantization, on-device, edge agentic).
2. **Energy/latency/memory accounting for on-device LLM inference** — especially
   anything doing integrated energy (Joules), not instantaneous power.
3. **LLMs as robot command interpreters / task planners on embedded hardware**
   (Pi, Jetson, phones; home/assistive/care robots).
4. **Conversational agents in assistive & medical robotics** (HRI venues, RO-MAN,
   ICORR) — for paper #2.
5. Venue signals: CFPs, deadline changes, relevant ICRA/IROS **workshops**.
6. llama.cpp / fllama ecosystem changes that affect the harness (server timings
   API, new quant types, KV-cache options).

## Tier-1 events (interrupt Maria)
- A published paper that does OUR planned study (quantization × runtime flags ×
  tool-calling fidelity × energy on edge devices, multi-turn) — scoop risk.
- MLPerf Edge Agentic results release or major benchmark standardization.
- A deadline moved/announced that affects the ICRA/RA-L plan.

## Output contract
- Digest file: `docs/research/digests/YYYY-MM-DD.md`, sections per question,
  each item: title, link, 2-line relevance note, verdict (cite / build-on /
  differentiate-from / ignore).
- Update `docs/research/seen.json` (url → date) — dedupe against it first.
- Max ~15 items per digest; relevance over completeness.
