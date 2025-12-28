"""
Edge LLM Lab - Framework for evaluating LLM models on edge devices.

This framework provides tools for:
- Evaluating LLM models on desktop and mobile devices
- Comparing model performance across different hardware configurations
- Generating comprehensive evaluation reports and visualizations
- Supporting both referenced and unreferenced evaluation strategies
"""

from .core.base_evaluation import BaseEvaluation, Agent
from .evaluation.referenced_evaluator import ReferencedEvaluator
from .evaluation.unreferenced_evaluator import UnreferencedEvaluator

__version__ = "0.1.0"
__author__ = "Maria Jastrzebska"

__all__ = [
    "BaseEvaluation",
    "Agent", 
    "ReferencedEvaluator",
    "UnreferencedEvaluator",
]



