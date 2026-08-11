# Edge-LLM-Lab: A Research Framework for the Quantitative Evaluation of Large Language Models on Resource-Constrained Hardware

Edge-LLM-Lab is a comprehensive scholarly framework designed for the rigorous evaluation of Large Language Models (LLMs) deployed on edge compute environments, bridging the gap between desktop and mobile platforms. This framework facilitates high-resolution performance benchmarking, semantic accuracy assessment, and comparative analysis of "mobile readiness" across heterogeneous hardware architectures.

## Core Capabilities

- **Modular Agent-Based Architecture**: Enables the definition of specialized autonomous agents through granular prompt engineering and JSON-based schema validation.
- **Cross-Platform Benchmarking Protocol**: Standardized evaluation metrics across desktop (macOS/Linux) and mobile environments.
- **Multi-Factor Analytic Metrics**: Implementation of semantic fidelity measures (P-BERT/R-BERT), structural integrity metrics (Levenshtein distance), and precise tool-argument verification.
- **High-Resolution Resource Monitoring**: Temporal analysis of hardware telemetry, including round-by-round monitoring of RAM occupancy, SWAP pressure, and CPU thermal throttling.
- **Persistent Reference Caching Layer**: Session-aware caching mechanisms and model-agnostic API response persistence to ensure reproducibility and computational efficiency.
- **Hybrid Evaluation Methodology**: Robust support for both referenced (gold-standard) and unreferenced (exploratory) evaluation strategies.
- **Extensible Configuration System**: Declarative YAML-based configuration for streamlined experimental reproducibility.

## Experimental Methodology and Workflow

### Core Evaluation Pipeline
The framework employs a multi-stage pipeline encompassing model selection, quantization validation, and runtime optimization. The primary execution interface is as follows:

```bash
# Execute the full research pipeline for a designated agent configuration
poetry run python examples/desktop/run_evaluation_pipeline.py --agent constant_data_en
```

### Credentials
Tokens (`NEPTUNE_API_TOKEN`, `OPENAI_API_KEY`) resolve environment-first, then from
GCP Secret Manager, so an exported variable or a local `.env` always wins and a
device without GCP configured runs unchanged:

```bash
export GCP_PROJECT_ID=other-project           # OPTIONAL — overrides llm-apps-425420
gcloud auth application-default login         # local credentials
poetry run python -m edge_llm_lab.utils.gcp_secrets   # verify; prints no values
```

The environment-variable-to-secret-name mapping lives in
`src/edge_llm_lab/utils/gcp_secrets.py` (`SECRETS`): `NEPTUNE_API_TOKEN` and
`OPENAI_API_KEY` map to secrets of the same name, and `TELEGRAM_BOT_TOKEN` /
`TELEGRAM_CHAT_ID` map to `Edge_lab_fed_bot` / `EDGE_LAB_CHAT_ID` for reporting
sweep progress. `source_of()` reports where a credential came from (`env` /
`secret-manager` / `missing`) for inclusion in a run manifest.

### Operational Modes
The `run_evaluation_pipeline.py` script supports distinct operational modes for granular data collection:
- `logs_only`: Focuses on empirical data collection and primary metric calculation.
- `logs_and_viz`: (Default) Comprehensive data collection with automated generation of comparative visualizations and synchronization with the Neptune research repository.
- `viz_only`: Facilitates post-hoc visualization and statistical re-analysis from existing telemetry logs.

