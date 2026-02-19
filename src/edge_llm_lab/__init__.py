"""
Edge LLM Lab - Framework for evaluating LLM models on edge devices.

This framework provides tools for:
- Evaluating LLM models on desktop and mobile devices
- Comparing model performance across different hardware configurations
- Generating comprehensive evaluation reports and visualizations
- Supporting both referenced and unreferenced evaluation strategies
"""

from .utils.base_eval import BaseEvaluation, Agent
from .evaluation.referenced_evaluator import EvalModelsReferenced
from .evaluation.unreferenced_evaluator import EvalModelsUnreferenced

__version__ = "0.1.0"
__author__ = "Maria Jastrzebska"

__all__ = [
    "BaseEvaluation",
    "Agent", 
    "EvalModelsReferenced",
    "EvalModelsUnreferenced",
]



