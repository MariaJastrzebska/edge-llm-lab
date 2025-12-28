# Edge LLM Lab

A comprehensive framework for evaluating Large Language Models (LLMs) on edge devices, including desktop and mobile platforms. This framework provides tools for performance benchmarking, accuracy evaluation, and comparative analysis across different hardware configurations.

## 🚀 Features

- **Configurable Agent System**: Define custom agents with their own prompts and schemas
- **Multi-Platform Support**: Evaluate models on desktop and mobile devices
- **Comprehensive Metrics**: BERTScore (P-BERT/R-BERT) for semantics, Levenshtein for structure, exact tool argument value matching, ROUGE scores, and more.
- **Advanced Visualizations**: Detailed resource health checks and round-by-round throttling timelines (RAM/SWAP/CPU).
- **Efficient Reference Caching**: Global session caching and model-independent API cache for faster repeated evaluations.
- **Flexible Evaluation Strategies**: Support for referenced and unreferenced evaluation
- **Easy Integration**: Simple configuration-based setup

## 🏃 Running the Evaluation Pipeline

### Main Pipeline
The primary script for running the multi-stage evaluation pipeline (Model Selection -> Quantization -> Optimization):

```bash
# Run the pipeline for a specific agent (e.g., constant_data_en)
poetry run python examples/desktop/run_evaluation_pipeline.py --agent constant_data_en
```

### Modes
The pipeline supports different modes in `run_evaluation_pipeline.py`:
- `logs_only`: Perform evaluation and metrics collection.
- `logs_and_viz`: (Default) Evaluation + generate comparison plots + upload to Neptune.
- `viz_only`: Re-generate plots from existing local or Neptune logs.

---

### Method 1: Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/MariaJastrzebska/edge-llm-lab.git
cd edge-llm-lab

# Configure Poetry to create virtualenv in project folder
poetry config virtualenvs.in-project true

# Install dependencies (automatically creates .venv)
poetry install

# Activate environment
source .venv/bin/activate
```

### Method 2: Standard pip

```bash
pip install -e .
```

## ⚙️ Prerequisites & Setup

### 1. Build Local llama.cpp (Required for Desktop Evaluation)
To use advanced optimizations, you need a local `llama-server` binary built within the project:

```bash
cd examples/desktop
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DGGML_METAL=ON  # Use -DGGML_CUDA=ON for NVIDIA GPUs
cmake --build . --config Release -j
```

### 1a. Updating llama.cpp
If you need to update to the latest version of `llama.cpp`:
```bash
cd examples/desktop/llama.cpp
git pull
rm -rf build
mkdir build && cd build
cmake .. -DGGML_METAL=ON
cmake --build . --config Release -j
```

### 2. Environment Variables
Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=your_key_here
OLLAMA_HOST=http://localhost:11434
```

## 🏗️ Quick Start

### Command Line Usage

```bash
# List all available optimizations
python examples/desktop/basic_evaluation.py --list-optimizations

# Run with desktop optimizations
python examples/desktop/basic_evaluation.py \
  --model llama3:8b \
  --agent data_collection_agent \
  --mode desktop

# Run with specific optimization IDs
python examples/desktop/basic_evaluation.py \
  --model llama3:8b \
  --agent data_collection_agent \
  --mode test \
  --optimizations 0,2,5

# Run with custom optimization file
python examples/desktop/basic_evaluation.py \
  --model llama3:8b \
  --agent data_collection_agent \
  --optimizations examples/desktop/config/custom_optimizations.yaml

```bash
+# Run referenced evaluation (interactive reference creation if missing)
+python src/edge_llm_lab/core/referenced_clean.py
+```

### 3. Creating Reference Conversations
When running `referenced_clean.py` for the first time for a specific agent, the framework will:
1. Detect that a reference file is missing (or empty).
2. Start an **interactive session** using ChatGPT.
3. Prompt you to enter user messages to guide the "perfect" conversation.
4. Save the result to `source/output/agents/<agent>/referenced/reference/`.

### 1. Create Configuration

Create a configuration file for your evaluation setup:

```yaml
# config/my_evaluation.yaml
source_path: "data"
inference_params_path: "config/inference_params.yaml"
evaluator_model: "gpt-4o-mini"
ollama_host: "http://localhost:11434"

agents:
  my_agent:
    name: "My Custom Agent"
    description: "Collects structured data from user input"
    prompt_path: "prompts/my_agent.txt"
    schema_path: "schemas/my_agent.json"
    tool_name: "send_data"
```

### 2. Basic Usage

```python
from edge_llm_lab.core.base_evaluation import BaseEvaluation
from edge_llm_lab.core.agent_config import ConfigLoader

# Load configuration
config = ConfigLoader.load_from_yaml("config/my_evaluation.yaml")

# Initialize evaluator
evaluator = BaseEvaluation(
    model_name="llama3.1:8b",
    agent_type="my_agent",
    config=config
)

# Evaluate a response
metrics = evaluator.evaluate_response(
    response={"name": "Alice", "age": 28},
    user_input="Hi, I'm Alice, 28 years old",
    ground_truth={"name": "Alice", "age": 28}
)

print(metrics)
```

## 📈 New Metrics & Features

### Experiment Tracking with Neptune
The framework integrates with Neptune.ai for real-time tracking of evaluation metrics, hardware usage, and visualizations.

1. Ensure your `.env` file has:
```bash
NEPTUNE_API_TOKEN=your_token
NEPTUNE_PROJECT=workspace/project
```
2. Run with plotting enabled:
```bash
poetry run python examples/desktop/run_evaluation_pipeline.py --agent constant_data_en
```
3. View results at `app.neptune.ai`.

---

### BERTScore (Hallucination & Completeness)
We use `bert-score` to provide semantic evaluation:
- **P-BERT (Precision)**: Measures how much of the model's response is supported by the reference (Hallucination detection).
- **R-BERT (Recall)**: Measures how much of the reference information was captured by the model (Completeness).

### Tool Argument Value Matching
Beyond structure, we now perform exact value matching for tool arguments (`tool_args_values`). This is case-insensitive and ignores minor formatting differences to ensure the data itself is correct.

### Throttling Visualizations
The system now generates a `throttling_timeline_*.png` for each optimization. It plots:
- **RAM vs SWAP**: See exactly when the system starts swapping.
- **CPU Frequency**: Monitor thermal throttling and performance drops round-by-round.

### Smart Caching
- **API Cache**: Reference generation is now model-independent, meaning once you generate a gold response for an agent, it is cached for all subsequent models tested.
    ```bash
    OPENAI_API_KEY="your_openai_key"
    NEPTUNE_API_TOKEN="your_neptune_token"
    NEPTUNE_PROJECT="maria.jastrzebska.ai/edge-llm-lab"
    ```

## 📋 Configuration Guide

### Agent Configuration

Each agent requires:
- **prompt_path**: Path to the system prompt file
- **schema_path**: Path to the JSON schema for structured output
- **tool_name**: Name of the function/tool for the agent

### Inference Parameters

Control model behavior with parameters:
```yaml
temperature: 0.7
top_p: 0.9
max_tokens: 1000
context_size: 4000
```

## 🔧 Advanced Usage

### Custom Evaluation Metrics

```python
# Extend the base evaluator for custom metrics
class CustomEvaluator(BaseEvaluation):
    def evaluate_response(self, response, user_input, ground_truth=None):
        metrics = super().evaluate_response(response, user_input, ground_truth)
        
        # Add custom metrics
        metrics.additional_metrics["custom_score"] = self.calculate_custom_score(response)
        
        return metrics
```

### llama-server Optimizations

The framework provides pre-configured optimization presets for `llama-server` inference:

```python
from edge_llm_lab.utils.optimization import (
    get_optimisations,
    get_mobile_optimizations,
    get_desktop_optimizations,
    get_optimal_kv_cache_type
)

# Get optimal KV cache type for your model
kv_cache = get_optimal_kv_cache_type("q8_0")  # For 8-bit quantized models

# Get all available optimizations (test mode vs production mode)
individual_opts, selected_opts = get_optimisations(kv_cache)

# Use desktop-optimized configurations
desktop_opts = get_desktop_optimizations(kv_cache)
evaluator.pipeline_eval_model(mode='logs_and_viz', optimisations=desktop_opts)

# Use mobile-optimized configurations
mobile_opts = get_mobile_optimizations(kv_cache)
evaluator.pipeline_eval_model(mode='logs_and_viz', optimisations=mobile_opts)

# Test all optimizations systematically
evaluator.pipeline_eval_model(mode='logs_only', optimisations=individual_opts)
```

**Available optimization parameters:**
- `--n-gpu-layers`: Number of layers to offload to GPU
- `--cache-type-k`, `--cache-type-v`: KV cache quantization
- `--flash-attn`: Enable Flash Attention
- `--cont-batching`: Enable continuous batching
- `--threads`: Number of threads
- `--batch-size`, `--ubatch-size`: Batch sizes
- `--no-mmap`, `--no-kv-offload`: Memory management
- And many more...

See `src/edge_llm_lab/utils/optimization.py` for the full list of optimizations.

## 📊 Evaluation Results

The framework generates comprehensive evaluation logs including:

- **Performance Metrics**: Latency breakdown, throughput, token statistics
- **Quality Metrics**: Accuracy, ROUGE scores, format validation
- **Device Information**: CPU, RAM, OS, architecture, power consumption
- **Resource Monitoring**: Memory usage, swap, energy deltas
- **Session Metadata**: Model info, quantization, timestamps

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Submit Evaluation Results**: Share your model evaluation results
2. **Add New Agents**: Create new agent types for different domains
3. **Improve Metrics**: Enhance evaluation algorithms
4. **Documentation**: Help improve documentation and examples

### How to Contribute Results

1. Fork the repository
2. Add your evaluation results to `contributions/`
3. Include:
   - Model information
   - Device specifications
   - Evaluation configuration
   - Results and metrics
4. Submit a pull request

### Example Contribution Structure

```
contributions/
├── your_username/
│   ├── model_evaluations/
│   │   ├── llama3.1_8b_macbook_pro.json
│   │   └── gpt4_mini_iphone_15.json
│   ├── device_specs/
│   │   └── macbook_pro_m3.json
│   └── README.md
```

## 📚 Examples

Check the `examples/` directory for:
- **Desktop Evaluation**: Python-based evaluation scripts
- **Mobile Integration**: Flutter/Dart integration examples (coming soon)
- **Custom Agents**: Domain-specific agent implementations

## 🛠️ Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/

# Type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for API access
- Ollama for local model serving
- The open-source LLM community

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/MariaJastrzebska/edge-llm-lab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MariaJastrzebska/edge-llm-lab/discussions)
- **Email**: maria.jastrzebska@example.com

---

**Made with ❤️ for the LLM evaluation community**