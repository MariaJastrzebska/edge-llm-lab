# Edge LLM Lab

A specialized framework for evaluating Large Language Models (LLMs) on edge devices. This system provides a standardized environment for performance benchmarking, accuracy evaluation, and comparative analysis across desktop and mobile hardware configurations, specifically targeting resource-constrained environments.

## Technical Capabilities

- **Modular Agent Architecture**: Configuration-driven agent definitions with distinct system prompts and JSON schemas.
- **Cross-Platform Benchmarking**: Support for automated evaluation on desktop (via `llama-server`) and mobile platforms.
- **Multifactor Metrics Layer**: Implementation of semantic evaluation (BERTScore P/R), structural validation (Levenshtein distance), and exact tool argument matching.
- **Hardware Diagnostics**: Real-time monitoring of system health, memory allocation (RAM/SWAP), and CPU frequency/thermal throttling rounds.
- **Inference Optimization Tracking**: Systematized evaluation of KV cache quantization, Flash Attention, and batching strategies.

## Setup & Prerequisites

### 1. Environment Configuration
The project uses Poetry for dependency management.

```bash
git clone https://github.com/MariaJastrzebska/edge-llm-lab.git
cd edge-llm-lab
poetry install
source .venv/bin/activate
```

### 2. Local Inference Server (llama.cpp)
A local `llama-server` binary is required for desktop-based evaluation. It should be built within `examples/desktop/llama.cpp`:

```bash
cd examples/desktop
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DGGML_METAL=ON
cmake --build . --config Release -j
```

### 3. Environment Variables
Configure `.env` in the project root:
```bash
OPENAI_API_KEY=your_key_here
NEPTUNE_API_TOKEN=your_token
NEPTUNE_PROJECT=workspace/project
```

## Evaluation Pipeline

The primary entry point for multi-stage evaluation is `run_evaluation_pipeline.py`. It executes a structured sequence: **Model Selection → Quantization Analysis → Optimization Tracking**.

### Pipeline Execution
```bash
poetry run python examples/desktop/run_evaluation_pipeline.py --agent constant_data_en --mode logs_and_viz
```

### Available Modes
- `logs_only`: Metrics collection and local logging.
- `logs_and_viz`: Evaluation with real-time visualization and Neptune.ai integration.
- `viz_only`: Post-hoc visualization from existing datasets.

### Reference Dataset Generation
To generate "gold" responses for referenced evaluation, run the interactive session:
```bash
python src/edge_llm_lab/core/referenced_clean.py
```

## Technical Overview

### Semantic and Structural Metrics
- **BERTScore**: Evaluates semantic precision (Hallucination detection) and recall (Completeness) against reference data.
- **Tool Argument Matching**: Case-insensitive exact matching of LLM-generated tool arguments to verify data integrity.

### Hardware Resource Analytics
The framework generates temporal analysis plots for hardware resources, specifically monitoring the RAM/SWAP threshold and CPU frequency oscillations to detect performance degradation during extended inference sessions.

### Experiment Tracking
Integration with Neptune.ai allows for persistent storage of experiment metadata, enabling comparative analysis of different model architectures and quantization levels across varied hardware.

## Development

```bash
# Running unit tests
pytest tests/unit

# Code quality
black src/
mypy src/
```

## License

This project is licensed under the MIT License.

## Support

- **Issues**: [GitHub Issues](https://github.com/MariaJastrzebska/edge-llm-lab/issues)
- **Contact**: maria.jastrzebska.ai@gmail.com
