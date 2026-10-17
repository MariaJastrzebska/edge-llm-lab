"""Tests for the ported llama-server timings parser, against a RECORDED
llama-server response fixture (shape as returned by /v1/chat/completions
with --jinja on llama.cpp)."""

from edge_llm_lab.runtime import parse_llama_timings

# Recorded llama-server response (trimmed to the fields the parser reads).
LLAMA_SERVER_RESPONSE = {
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {"content": '{"tool_calls": []}', "role": "assistant"},
        }
    ],
    "timings": {
        "prompt_n": 512,
        "prompt_ms": 831.5,
        "prompt_per_token_ms": 1.624,
        "prompt_per_second": 615.75,
        "predicted_n": 128,
        "predicted_ms": 2168.4,
        "predicted_per_token_ms": 16.94,
        "predicted_per_second": 59.03,
    },
}


def test_prefill_decode_split_from_top_level_timings():
    breakdown = parse_llama_timings(LLAMA_SERVER_RESPONSE, 0.0, 3.5)

    assert breakdown["prompt_eval_ms"] == 831.5
    assert breakdown["token_generation_ms"] == 2168.4
    # total computed from parts, not wall clock, when parts are present
    assert breakdown["total_ms"] == round(831.5 + 2168.4, 2)
    assert breakdown["tokens"]["prompt_count"] == 512
    assert breakdown["tokens"]["generated_count"] == 128
    # throughput derived: predicted_n / (predicted_ms/1000)
    assert abs(breakdown["tokens"]["throughput_tokens_per_sec"] - 128 / 2.1684) < 0.01
    assert breakdown["model_loading_ms"] is None
    assert breakdown["wall_ms"] == 3500.0
    # Measured server phases sum to approximately 100 percent.
    pct = breakdown["breakdown_percentage"]
    assert abs(pct["prompt_eval"] + pct["generation"] - 100.0) < 0.01


def test_per_choice_timings_fallback():
    response = {
        "choices": [
            {
                "timings": {
                    "prompt_ms": 100.0,
                    "predicted_ms": 400.0,
                    "prompt_n": 10,
                    "predicted_n": 40,
                }
            }
        ]
    }
    breakdown = parse_llama_timings(response, 0.0, 1.0)
    assert breakdown["prompt_eval_ms"] == 100.0
    assert breakdown["token_generation_ms"] == 400.0
    assert breakdown["total_ms"] == 500.0


def test_wall_clock_fallback_when_no_timings():
    breakdown = parse_llama_timings({}, 10.0, 12.5)
    assert breakdown["total_ms"] == 2500.0
    assert breakdown["prompt_eval_ms"] is None
    assert breakdown["token_generation_ms"] is None
    assert breakdown["tokens"]["throughput_tokens_per_sec"] is None


def test_optimisations_provenance_is_carried_through():
    opts = {"--cache-type-k": "q8_0", "--flash-attn": True}
    breakdown = parse_llama_timings(LLAMA_SERVER_RESPONSE, 0.0, 3.5, optimisations=opts)
    assert breakdown["optimizations_details"] == opts
