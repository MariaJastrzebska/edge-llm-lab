"""
Generalized base evaluation framework for LLM models.
Supports configurable agents, prompts, and evaluation strategies.
"""

from __future__ import annotations

import sys
import asyncio
from pathlib import Path
import json
from typing import List, Dict, Any, Optional, Type, Union, Callable
from dataclasses import dataclass, field
import numpy as np
from pydantic import BaseModel, Field
from rouge_score import rouge_scorer
from loguru import logger
from enum import Enum

# Import our configurable system
from .agent_config import EvaluationConfig, AgentConfig, ConfigLoader


class Agent(Enum):
    """Generic agent types - can be extended per project."""
    DATA_COLLECTION = 'data_collection'
    QA_AGENT = 'qa_agent'
    SUMMARIZATION = 'summarization'
    CLASSIFICATION = 'classification'


@dataclass
class EvaluationMetrics:
    """Metrics used for evaluating LLM response quality."""
    rouge_scores: Dict[str, float] = field(default_factory=dict)
    factual_accuracy: float = 0.0
    hallucination_score: float = 0.0
    format_accuracy: float = 0.0
    relevance_score: float = 0.0
    completeness_score: float = 0.0
    additional_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """Returns readable representation of metrics."""
        result = "\n=== Evaluation Results ===\n"
        
        if self.rouge_scores:
            result += f"ROUGE scores: {self.rouge_scores}\n"
        
        result += f"Factual Accuracy: {self.factual_accuracy:.2f}\n"
        result += f"Hallucination Score: {self.hallucination_score:.2f}\n"
        result += f"Format Accuracy: {self.format_accuracy:.2f}\n"
        result += f"Relevance Score: {self.relevance_score:.2f}\n"
        result += f"Completeness Score: {self.completeness_score:.2f}\n"
        
        if self.additional_metrics:
            result += "\nAdditional Metrics:\n"
            for name, value in self.additional_metrics.items():
                result += f"- {name}: {value:.2f}\n"
                
        # Calculate average score
        main_metrics = [
            self.factual_accuracy,
            1.0 - self.hallucination_score,  # Reverse so higher is better
            self.format_accuracy,
            self.relevance_score,
            self.completeness_score
        ]
        avg_score = sum(main_metrics) / len(main_metrics)
        result += f"\nAverage Score: {avg_score:.2f}/1.0\n"
        
        return result


@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    model_name: str
    evaluator_name: str
    test_case: Dict[str, Any]
    response: Any
    metrics: EvaluationMetrics
    raw_evaluator_responses: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        response_dict = self.response
        if hasattr(self.response, 'model_dump'):
            response_dict = self.response.model_dump()
            
        return {
            "model_name": self.model_name,
            "evaluator_name": self.evaluator_name,
            "test_case": self.test_case,
            "response": response_dict,
            "metrics": {
                "rouge_scores": self.metrics.rouge_scores,
                "factual_accuracy": self.metrics.factual_accuracy,
                "hallucination_score": self.metrics.hallucination_score,
                "format_accuracy": self.metrics.format_accuracy,
                "relevance_score": self.metrics.relevance_score,
                "completeness_score": self.metrics.completeness_score,
                "additional_metrics": self.metrics.additional_metrics
            },
            "raw_evaluator_responses": self.raw_evaluator_responses
        }
        
    def save_to_file(self, file_path: str) -> None:
        """Save result to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class BaseEvaluation:
    """Configurable base class for LLM evaluation."""

    def __init__(self, model_name: str, agent_type: str, config: EvaluationConfig):
        """Initialize BaseEvaluation with model name, agent type, and configuration."""
        self.model_name = model_name
        self.model_name_norm = self.model_name.replace(':', '_').replace('/', '_')
        self.agent_type = agent_type
        self.config = config
        
        # Get agent configuration
        if agent_type not in config.agents:
            raise ValueError(f"Agent type '{agent_type}' not found in configuration")
        
        self.agent_config = config.agents[agent_type]
        
        # Set up paths
        self.BASE_PATH = Path(config.source_path)
        self.SOURCE_PATH = self.BASE_PATH
        if not self.SOURCE_PATH.exists():
            self.SOURCE_PATH.mkdir(parents=True, exist_ok=True)

        # Load inference parameters
        self._load_inference_params()
        
        # Initialize clients
        self._initialize_clients()
        
        # Device monitoring
        self.device_info = self._get_device_info()

    def _load_inference_params(self):
        """Load inference parameters from config."""
        import yaml
        
        with open(self.config.inference_params_path, 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)
        
        # Extract parameters (adapt based on your config structure)
        self.TEMPERATURE = params.get('temperature', 0.7)
        self.TOP_P = params.get('top_p', 0.9)
        self.MAX_TOKENS = params.get('max_tokens', 1000)
        self.CONTEXT_SIZE = params.get('context_size', 4000)
        self.SEED = 42

    def _initialize_clients(self):
        """Initialize API clients."""
        try:
            import openai
            self.OPENAI_CLIENT = openai.OpenAI(api_key=self.config.evaluator_model)
        except ImportError:
            self.OPENAI_CLIENT = None
            
        try:
            import ollama
            self.OLLAMA_CLIENT = ollama.Client(host=self.config.ollama_host)
        except ImportError:
            self.OLLAMA_CLIENT = None

    def _get_device_info(self) -> Dict[str, Any]:
        """Get device information for logging."""
        import platform
        import psutil
        
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "machine": platform.machine(),
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "memory_total_gb": psutil.virtual_memory().total / (1024 ** 3),
        }

    def monitor_resources(self) -> Dict[str, Any]:
        """Monitor system resources during evaluation."""
        import psutil
        
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        memory_info = {
            "ram_total_gb": memory.total / (1024 ** 3),
            "ram_used_gb": memory.used / (1024 ** 3),
            "swap_used_gb": swap.used / (1024 ** 3)
        }

        return {
            "memory": memory_info, 
            "device": self.device_info,
            "timestamp": asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
        }

    def create_tool_for_agent(self) -> List[Dict]:
        """Create tool schema for the configured agent."""
        if not self.agent_config.pydantic_model:
            return []
        
        return self._create_tool(
            self.agent_config.pydantic_model,
            self.agent_config.tool_name,
            self.agent_config.description
        )

    @staticmethod
    def _create_tool(pydantic_model: Type[BaseModel], name: str, description: str) -> List[Dict]:
        """Create tool schema from Pydantic model."""
        tool_schema = pydantic_model.model_json_schema()
        
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": tool_schema
                }
            }
        ]

    def load_prompt(self) -> str:
        """Load prompt for the configured agent."""
        with open(self.agent_config.prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def load_schema(self) -> Dict[str, Any]:
        """Load schema for the configured agent."""
        with open(self.agent_config.schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_openai_response(self, messages: List[Dict], **kwargs) -> Union[str, Dict]:
        """Get response from OpenAI API."""
        if not self.OPENAI_CLIENT:
            raise RuntimeError("OpenAI client not initialized")
        
        response = self.OPENAI_CLIENT.chat.completions.create(
            model=self.config.evaluator_model,
            messages=messages,
            temperature=self.TEMPERATURE,
            max_tokens=self.MAX_TOKENS,
            top_p=self.TOP_P,
            **kwargs
        )
        
        return response.choices[0].message.content

    def get_ollama_response(self, messages: List[Dict], **kwargs) -> Union[str, Dict]:
        """Get response from Ollama API."""
        if not self.OLLAMA_CLIENT:
            raise RuntimeError("Ollama client not initialized")
        
        response = self.OLLAMA_CLIENT.chat(
            model=self.model_name,
            messages=messages,
            options={
                'temperature': self.TEMPERATURE,
                'top_p': self.TOP_P,
                'max_tokens': self.MAX_TOKENS,
            },
            **kwargs
        )
        
        return response['message']['content']

    def evaluate_response(self, response: Any, user_input: str, 
                         ground_truth: Optional[Dict[str, Any]] = None) -> EvaluationMetrics:
        """Evaluate LLM response quality."""
        metrics = EvaluationMetrics()
        
        # Calculate ROUGE scores if ground truth provided
        if ground_truth:
            response_text = str(response)
            ground_truth_text = json.dumps(ground_truth, ensure_ascii=False)
            metrics.rouge_scores = self._calculate_rouge_scores(response_text, ground_truth_text)
        
        # Here you would implement specific evaluation logic
        # For now, return placeholder metrics
        metrics.factual_accuracy = 0.8
        metrics.hallucination_score = 0.1
        metrics.format_accuracy = 0.9
        metrics.relevance_score = 0.85
        metrics.completeness_score = 0.75
        
        return metrics

    def _calculate_rouge_scores(self, response: str, ground_truth: str) -> Dict[str, float]:
        """Calculate ROUGE scores between response and ground truth."""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(ground_truth, response)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure,
        }

    def save_evaluation_log(self, log_data: Dict[str, Any], filename: str) -> None:
        """Save evaluation data to log file."""
        log_path = self.SOURCE_PATH / "output" / "logs" / f"{filename}.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Evaluation data saved to: {log_path}")