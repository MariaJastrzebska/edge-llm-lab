import yaml
import os
from typing import Dict, Any, List

class UnifiedConfig:
    """Unified configuration loader for the evaluation pipeline."""
    
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.config_dir = os.path.join(root_dir, "examples/desktop/config")
        self.agent_dir = os.path.join(root_dir, "examples/desktop/input/agents")

    def load_pipeline_config(self) -> Dict[str, Any]:
        """Loads the multi-stage pipeline configuration."""
        # For now, we will construct it dynamically as requested
        return {
            "stages": [
                {
                    "id": 1,
                    "name": "Model Selection",
                    "config_path": os.path.join(self.agent_dir, "constant_data_en/evaluation_config/config.yaml")
                },
                {
                    "id": 2,
                    "name": "Quantization Analysis",
                    "config_path": os.path.join(self.agent_dir, "constant_data_en/evaluation_config/config_quantized.yaml")
                },
                {
                    "id": 3,
                    "name": "Optuna Optimization",
                    "config_path": os.path.join(self.config_dir, "custom_optimizations.yaml")
                }
            ]
        }

    def load_yaml(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_optimizations(self) -> List[Dict[str, Any]]:
        path = os.path.join(self.config_dir, "custom_optimizations.yaml")
        config = self.load_yaml(path)
        return config.get("optimizations", [])
