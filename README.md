# edge-llm-lab
Framework for systematic evaluation of low-latency inference techniques in small language models. Includes caching, FlashAttention, continuous batching, and speculative decoding strategies for on-device LLMs.
# Edge LLM Optimization Lab

This repository explores **systematic inference optimization** for small and medium-sized language models (SLMs/LLMs) running on **edge or mobile devices**, using **[llama.cpp](https://github.com/ggerganov/llama.cpp)** as the core inference engine.

We perform controlled experiments to measure the impact of different optimization flags and combinations — including caching, FlashAttention, continuous batching, and speculative decoding — on **latency, memory, and quality**.

---

## 🧠 Motivation

Running LLMs locally on constrained hardware (e.g., Metal, CoreML, mobile GPUs) presents unique challenges:
- Limited VRAM and memory bandwidth  
- Power and temperature constraints  
- Reduced throughput due to limited parallelism  

The goal of this research is to identify **which optimizations deliver the best latency-to-quality trade-off** and to understand **how combinations of techniques interact**.

---

## ⚙️ Optimization Techniques (llama.cpp flags)

Each optimization (or combination) is tested in isolation and cumulatively through an automated experiment pipeline.

| Category | Example Flags | Description |
|-----------|----------------|-------------|
| **Baseline** | `{}` | Unoptimized reference setup |
| **Caching** | `--kv-cache`, `--cache-type-k`, `--cache-type-v` | Key/Value caching variants |
| **Parallelization** | `--flash-attn`, `--cont-batching` | FlashAttention and continuous batching |
| **Speculative Decoding** | `--prompt-lookup`, `--prompt-lookup-ngram`, `--speculative-decoding` | Prompt lookup and adaptive decoding strategies |
| **Hybrid Combinations** | multiple flags | Combined optimizations for mobile use cases |

Each configuration is evaluated for **token latency, throughput, and memory footprint**.

---

## 🧩 Example: Individual Optimization Setups

```python
individual_optimizations = [
    {},  # Baseline
    {"--kv-cache": kv_cache},
    {"--flash-attn": None},
    {"--cont-batching": None},
    {"--prompt-lookup-ngram": 3},
    {"--speculative-decoding": "adaptive"},
    {
        "--cache-type-k": kv_cache,
        "--cache-type-v": kv_cache,
        "--flash-attn": None,
        "--speculative-decoding": "adaptive",
        "--memory-budget": "mobile"
    }
]
