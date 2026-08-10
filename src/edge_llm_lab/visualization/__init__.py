"""Visualization modules for edge-llm-lab framework.

NOTE: an earlier version imported a nonexistent `.plots` module here, which
made the whole package unimportable (the default `logs_and_viz` pipeline mode
crashed on import). Only re-export functions that actually exist.
"""

from .optimization_comparison import (
    plot_optimization_comparison,
    plot_gartner_style_energy_analysis,
)

__all__ = [
    'plot_optimization_comparison',
    'plot_gartner_style_energy_analysis',
]
