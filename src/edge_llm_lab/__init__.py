"""
Edge LLM Lab - Framework for evaluating LLM models on edge devices.

This framework provides tools for:
- Evaluating LLM models on desktop and mobile devices
- Comparing model performance across different hardware configurations
- Generating comprehensive evaluation reports and visualizations
- Supporting both referenced and unreferenced evaluation strategies

The legacy evaluator classes (BaseEvaluation, EvalModelsReferenced,
EvalModelsUnreferenced) are imported LAZILY: they pull in the full
pandas/matplotlib/openai stack, and eager imports made even the lightweight
instrument subpackages (edge_llm_lab.runtime, edge_llm_lab.telemetry)
unimportable without the whole legacy dependency tree.
"""

__version__ = "0.2.0"
__author__ = "Maria Jastrzebska"

__all__ = [
    "BaseEvaluation",
    "Agent",
    "EvalModelsReferenced",
    "EvalModelsUnreferenced",
]

_LAZY = {
    "BaseEvaluation": ("edge_llm_lab.utils.base_eval", "BaseEvaluation"),
    "Agent": ("edge_llm_lab.utils.base_eval", "Agent"),
    "EvalModelsReferenced": (
        "edge_llm_lab.evaluation.referenced_evaluator",
        "EvalModelsReferenced",
    ),
    "EvalModelsUnreferenced": (
        "edge_llm_lab.evaluation.unreferenced_evaluator",
        "EvalModelsUnreferenced",
    ),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
