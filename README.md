# edge-llm-lab
Framework for systematic evaluation of low-latency inference techniques in small language models. Includes caching, FlashAttention, continuous batching, and speculative decoding strategies for on-device LLMs.

# 🧠 Edge LLM Optimization Lab

A cross-platform research framework for **systematic inference optimization** of small and medium language models (SLMs/LLMs) running **locally** on both **desktop (llama.cpp)** and **mobile (fllama)** environments.

The framework allows you to:
- 🧩 **Test the same set of optimizations** on desktop and mobile backends.  
- ⚙️ **Plug in your own prompt, model, or inference tool** and measure latency, throughput, and memory.  
- 📊 **Compare** how each optimization — or their combination — impacts efficiency and quality.

---

## 🎯 Motivation

Running LLMs efficiently on local devices (laptops, phones, edge boards) requires a deep understanding of how inference optimizations interact under different hardware constraints.

This project aims to build an **open, extensible benchmarking framework** to analyze latency-vs-quality trade-offs in local inference pipelines.

---

## 🧩 Architecture Overview

Two complementary tracks share the same configuration schema and test logic:

| Environment | Backend | Purpose |
|--------------|----------|----------|
| **Desktop** | [llama.cpp](https://github.com/ggerganov/llama.cpp) | Python-based experiments for detailed, repeatable measurements. |
| **Mobile** | [fllama](https://github.com/Telosnex/fllama) | Flutter/Dart integration to validate results directly on phones (iOS/Android). |

Both environments execute **identical rounds of optimizations** — allowing fair comparison between hardware classes.

---

## ⚙️ Optimization Techniques

Each experiment round applies a defined combination of flags.  
The default pipeline includes:

| Category | Example Flags | Description |
|-----------|----------------|-------------|
| **Baseline** | `{}` | Unoptimized reference setup |
| **Caching** | `--kv-cache`, `--cache-type-k`, `--cache-type-v` | Key/Value caching and variants |
| **Parallelization** | `--flash-attn`, `--cont-batching` | FlashAttention and continuous batching |
| **Speculative Decoding** | `--prompt-lookup`, `--prompt-lookup-ngram`, `--speculative-decoding` | Prompt lookup and adaptive decoding strategies |
| **Hybrid Combinations** | multiple flags | Combined setups tuned for edge/mobile constraints |

All configurations can be extended by adding new rounds or custom flags in YAML.


