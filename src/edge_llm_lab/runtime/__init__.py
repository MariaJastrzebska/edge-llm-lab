"""Runtime instruments: llama-server orchestration and timing extraction.

Start of the restructure described in CLAUDE.md — measurement code ported out
of the BaseEvaluation god-class into small, pure, unit-tested functions.
"""

from .timings import parse_llama_timings

__all__ = ["parse_llama_timings"]
