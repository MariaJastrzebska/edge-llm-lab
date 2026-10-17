"""Parse llama-server timing fields into a consistent latency breakdown.

Ported verbatim (minus class coupling and prints) from
``utils/base_eval.py::BaseEvaluation._parse_llama_timings`` — the single most
valuable instrument in the original harness: a clean prefill/decode split with
derived throughput and wall-clock fallback.
"""

from __future__ import annotations

from typing import Any, Optional


def parse_llama_timings(
    result: dict,
    start_time: float,
    end_time: float,
    optimisations: Optional[dict] = None,
) -> dict:
    """Parse llama-server timing fields into a consistent latency_breakdown dict.

    Supports both top-level ``result['timings']`` and per-choice timings under
    ``result['choices'][0]['timings']``. When fields are missing, falls back to
    wall-clock total time (``end_time - start_time`` in seconds).
    """
    total_time_ms = round((end_time - start_time) * 1000, 2)
    timings = result.get("timings") or {}
    choice_timings: Any = None
    try:
        choice_timings = result.get("choices", [{}])[0].get("timings")
    except Exception:
        choice_timings = None

    # Prefer top-level timings; fallback to choice timings
    src = (
        timings
        if isinstance(timings, dict) and timings
        else (choice_timings if isinstance(choice_timings, dict) else {})
    )

    prompt_ms = src.get("prompt_ms") if isinstance(src, dict) else None
    predicted_ms = src.get("predicted_ms") if isinstance(src, dict) else None
    prompt_n = src.get("prompt_n") if isinstance(src, dict) else None
    predicted_n = src.get("predicted_n") if isinstance(src, dict) else None

    # Compute total from parts when available
    if isinstance(prompt_ms, (int, float)) and isinstance(predicted_ms, (int, float)):
        computed_total = round(prompt_ms + predicted_ms, 2)
    else:
        computed_total = total_time_ms

    throughput = None
    if (
        isinstance(predicted_ms, (int, float))
        and predicted_ms > 0
        and isinstance(predicted_n, (int, float))
    ):
        throughput = predicted_n / (predicted_ms / 1000.0)

    # Additional llama.cpp statistics
    prompt_per_token_ms = src.get("prompt_per_token_ms") if isinstance(src, dict) else None
    prompt_per_second = src.get("prompt_per_second") if isinstance(src, dict) else None
    predicted_per_token_ms = (
        src.get("predicted_per_token_ms") if isinstance(src, dict) else None
    )
    predicted_per_second = (
        src.get("predicted_per_second") if isinstance(src, dict) else None
    )

    breakdown = {
        "total_ms": computed_total,
        "model_loading_ms": None,  # llama-server API does not expose model load time here
        "prompt_eval_ms": float(prompt_ms) if isinstance(prompt_ms, (int, float)) else None,
        "token_generation_ms": float(predicted_ms)
        if isinstance(predicted_ms, (int, float))
        else None,
        "optimizations_details": optimisations if optimisations else {},
        "breakdown_percentage": {
            "loading": None,
            "prompt_eval": (float(prompt_ms) / computed_total * 100.0)
            if isinstance(prompt_ms, (int, float)) and computed_total
            else None,
            "generation": (float(predicted_ms) / computed_total * 100.0)
            if isinstance(predicted_ms, (int, float)) and computed_total
            else None,
        },
        "tokens": {
            "prompt_count": int(prompt_n) if isinstance(prompt_n, (int, float)) else None,
            "generated_count": int(predicted_n)
            if isinstance(predicted_n, (int, float))
            else None,
            "throughput_tokens_per_sec": throughput,
            "prompt_per_token_ms": float(prompt_per_token_ms)
            if isinstance(prompt_per_token_ms, (int, float))
            else None,
            "prompt_per_second": float(prompt_per_second)
            if isinstance(prompt_per_second, (int, float))
            else None,
            "predicted_per_token_ms": float(predicted_per_token_ms)
            if isinstance(predicted_per_token_ms, (int, float))
            else None,
            "predicted_per_second": float(predicted_per_second)
            if isinstance(predicted_per_second, (int, float))
            else None,
        },
    }

    # Startup occurs before the completion request and cannot be inferred here.
    breakdown['wall_ms'] = total_time_ms
    return breakdown
