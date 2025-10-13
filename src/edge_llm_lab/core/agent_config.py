"""
Configuration system for LLM evaluation agents.
Allows flexible configuration of agents, prompts, and schemas.
"""

from typing import Dict, Any, Optional, Type
from pydantic import BaseModel, Field
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentConfig:
    """Configuration for a specific agent type."""
    name: str
    description: str
    prompt_path: str
    schema_path: str
    validation_prompt_path: Optional[str] = None
    validation_schema_path: Optional[str] = None
    display_prompt_path: Optional[str] = None
    pydantic_model: Optional[Type[BaseModel]] = None
    tool_name: str = "send_data"
    
    def __post_init__(self):
        """Validate paths after initialization."""
        if not Path(self.prompt_path).exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_path}")
        if not Path(self.schema_path).exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")


@dataclass
class EvaluationConfig:
    """Configuration for evaluation framework."""
    source_path: str
    inference_params_path: str
    agents: Dict[str, AgentConfig]
    evaluator_model: str = "gpt-4o-mini"
    ollama_host: str = "http://localhost:11434"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not Path(self.source_path).exists():
            raise FileNotFoundError(f"Source path not found: {self.source_path}")
        if not Path(self.inference_params_path).exists():
            raise FileNotFoundError(f"Inference params not found: {self.inference_params_path}")


class ConfigLoader:
    """Loads configuration from YAML files or dictionaries."""
    
    @staticmethod
    def load_from_yaml(config_path: str) -> EvaluationConfig:
        """Load configuration from YAML file."""
        import yaml
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return ConfigLoader.load_from_dict(config_data)
    
    @staticmethod
    def load_from_dict(config_data: Dict[str, Any]) -> EvaluationConfig:
        """Load configuration from dictionary."""
        agents = {}
        
        for agent_name, agent_data in config_data.get('agents', {}).items():
            agents[agent_name] = AgentConfig(
                name=agent_data['name'],
                description=agent_data['description'],
                prompt_path=agent_data['prompt_path'],
                schema_path=agent_data['schema_path'],
                validation_prompt_path=agent_data.get('validation_prompt_path'),
                validation_schema_path=agent_data.get('validation_schema_path'),
                display_prompt_path=agent_data.get('display_prompt_path'),
                tool_name=agent_data.get('tool_name', 'send_data')
            )
        
        return EvaluationConfig(
            source_path=config_data['source_path'],
            inference_params_path=config_data['inference_params_path'],
            agents=agents,
            evaluator_model=config_data.get('evaluator_model', 'gpt-4o-mini'),
            ollama_host=config_data.get('ollama_host', 'http://localhost:11434')
        )


def create_example_config() -> Dict[str, Any]:
    """Create example configuration for demonstration."""
    return {
        "source_path": "examples/desktop/source",
        "inference_params_path": "examples/desktop/config/inference_params.yaml",
        "evaluator_model": "gpt-4o-mini",
        "ollama_host": "http://localhost:11434",
        "agents": {
            "data_collection": {
                "name": "Data Collection Agent",
                "description": "Collects structured data from user input",
                "prompt_path": "examples/desktop/prompts/data_collection.txt",
                "schema_path": "examples/desktop/schemas/data_collection.json",
                "validation_prompt_path": "examples/desktop/prompts/validation.txt",
                "validation_schema_path": "examples/desktop/schemas/validation.json",
                "display_prompt_path": "examples/desktop/prompts/display.txt",
                "tool_name": "send_collected_data"
            },
            "qa_agent": {
                "name": "Q&A Agent",
                "description": "Answers questions based on context",
                "prompt_path": "examples/desktop/prompts/qa.txt",
                "schema_path": "examples/desktop/schemas/qa.json",
                "tool_name": "send_answer"
            }
        }
    }
