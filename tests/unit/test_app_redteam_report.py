"""Scientific and privacy contracts for the intake application report."""
import csv
import hashlib
import xml.etree.ElementTree as ET

import pytest

from edge_llm_lab.storage.trials import TrialStore
from edge_llm_lab.visualization.app_redteam_report import (
    build_figure,
    render,
    summarize,
)

CASES = [
    {"id": "b1", "family_id": "f1", "category": "missing-unit", "variant": "benign"},
    {"id": "b2", "family_id": "f2", "category": "record", "variant": "benign"},
    {"id": "a1", "family_id": "f1", "category": "foreign-read", "variant": "attack"},
    {"id": "a2", "family_id": "f2", "category": "foreign-write", "variant": "attack"},
]


def assessment(**updates):
    result = {"schema_valid": True, "attempted_unauthorized": False,
              "task_success": True, "over_refusal": False,
              "execution": {"unauthorized_effect": False, "private_record": "PRIVATE-RECORD"}}
    result.update(updates)
    return result


def trial(case="b1", repeat=0, config=0, *, status="ok", wall=1000,
          unguarded=None, guarded=None, identity=None):
    return {"trial_id": identity or f"{case}-{repeat}-{config}", "case_id": case,
            "repeat": repeat, "seed": [11, 12][repeat], "config_id": config,
            "status": status, "started_utc": "2026-10-17", "elapsed_ms": 2000,
            "error_type": "TimeoutError" if status == "error" else None,
            "result": {"response": "PRIVATE-RESPONSE", "latency": {"wall_ms": wall},
                       "applications": {"unguarded": unguarded or assessment(),
                                        "guarded": guarded or assessment()}} if status == "ok" else {}}


def capture(tmp_path, trials, *, updates=None):
    manifest = {"mode": "app_redteam", "cases": CASES, "seeds": [11, 12],
                "model": "example/health-intake-small",
                "model_metadata": {"sha256": "a" * 64, "gguf_path": "/PRIVATE-MODEL-PATH"},
                "optimisations": [{"decoding": "prompt"}, {"decoding": "schema"}],
                "inference_platform": {"system": "Darwin", "machine": "arm64",
                                       "execution": "remote", "description": "Apple Silicon",
                                       "endpoint": "https://PRIVATE-ENDPOINT"},
                "controller_platform": {"system": "Linux", "machine": "x86_64"},
                "endpoint": "https://PRIVATE-ENDPOINT", "token": "PRIVATE-TOKEN"}
    manifest.update(updates or {})
    path = tmp_path / "capture.sqlite"
    store = TrialStore(path, manifest)
    try:
        for row in trials:
            store.append(row)
    finally:
        store.close()
    return path


def group(summary, config=0, variant="benign", guard="guarded"):
    return summary["configurations"][config]["cohorts"][variant]["applications"][guard]


def test_equal_case_weighting_and_separate_cohorts(tmp_path):
    source = capture(tmp_path, [
        trial(wall=1000),
        trial(repeat=1, wall=9000, guarded=assessment(task_success=False)),
        trial(case="b2", wall=3000, guarded=assessment(task_success=False)),
        trial(case="a1", guarded=assessment(task_success=True))])
    summary = summarize(source)
    measured = group(summary)["task_success"]
    assert measured["mean"] == .25
    assert measured["observations"] == 3
    assert measured["observed_cases"] == 2
    assert measured["expected_observations"] == 4
    assert summary["configurations"][0]["cohorts"]["benign"]["wall_seconds"]["mean"] == 4
    assert group(summary, variant="attack")["task_success"]["mean"] == 1


def test_attempts_and_effects_are_independent_and_guards_are_not_pooled(tmp_path):
    source = capture(tmp_path, [trial(case="a1",
        unguarded=assessment(attempted_unauthorized=True, execution={"unauthorized_effect": True}),
        guarded=assessment(attempted_unauthorized=True, execution={"unauthorized_effect": False}))])
    summary = summarize(source)
    for guard in ("unguarded", "guarded"):
        assert group(summary, variant="attack", guard=guard)["attempted_unauthorized"]["mean"] == 1
    assert group(summary, variant="attack", guard="unguarded")["unauthorized_effect"]["mean"] == 1
    assert group(summary, variant="attack")["unauthorized_effect"]["mean"] == 0


def test_missing_trials_failed_trials_and_missing_telemetry_are_not_zero(tmp_path):
    source = capture(tmp_path, [trial(wall=None), trial(case="b2", status="error")])
    summary = summarize(source)
    assert summary["coverage"] == {"expected": 16, "observed": 2, "ok": 1, "errors": 1, "missing": 14}
    benign = summary["configurations"][0]["cohorts"]["benign"]
    assert benign["coverage"] == {"expected": 4, "observed": 2, "ok": 1, "errors": 1, "missing": 2}
    assert benign["wall_seconds"]["mean"] is None
    assert benign["wall_seconds"]["observations"] == 0
    assert group(summary)["task_success"]["mean"] == 1
    assert group(summary)["task_success"]["expected_observations"] == 4
    assert group(summary, config=1)["task_success"]["mean"] is None


def test_format_counts_each_response_once_not_each_guard(tmp_path):
    source = capture(tmp_path, [trial(), trial(case="a1",
        unguarded=assessment(schema_valid=False), guarded=assessment(schema_valid=False))])
    metric = summarize(source)["configurations"][0]["schema_valid"]
    assert metric["mean"] == .5
    assert metric["observations"] == 2
    assert metric["positive_observations"] == 1


@pytest.mark.parametrize("kind", ["disagreeing-schema", "string-boolean", "negative-time"])
def test_invalid_measurements_fail_loudly(tmp_path, kind):
    row = trial()
    if kind == "disagreeing-schema":
        row["result"]["applications"]["guarded"]["schema_valid"] = False
    elif kind == "string-boolean":
        row["result"]["applications"]["guarded"]["task_success"] = "False"
    else:
        row["result"]["latency"]["wall_ms"] = -1
    with pytest.raises(ValueError):
        summarize(capture(tmp_path, [row]))


def test_duplicate_or_unknown_trial_identity_is_rejected(tmp_path):
    source = capture(tmp_path, [trial(), trial(identity="duplicate")])
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(source)


def test_unknown_case_is_not_silently_excluded(tmp_path):
    with pytest.raises(ValueError, match="Unknown case"):
        summarize(capture(tmp_path, [trial(case="unknown")]))


def test_report_uses_declared_inference_host_and_exports_no_private_values(tmp_path):
    source = capture(tmp_path, [trial()])
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "report"
    summary = render(source, output)
    assert summary["inference_platform"]["system"] == "Darwin"
    assert summary["controller_platform"]["system"] == "Linux"
    assert summary["model"] == "example/health-intake-small"
    assert summary["model_metadata"] == {"sha256": "a" * 64}
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    for name in ("format-utility", "authorization", "benign-utility", "request-cost"):
        ET.parse(output / f"{name}.svg")
        assert (output / f"{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    for path in output.iterdir():
        if path.suffix in {".json", ".csv", ".svg"}:
            assert "PRIVATE-" not in path.read_text()
    rows = list(csv.DictReader((output / "trials.csv").open()))
    assert len(rows) == 1
    assert rows[0]["case_id"] == "b1"
    assert "response" not in rows[0]
    import matplotlib.pyplot as plt
    assert not plt.get_fignums()


def test_authorization_figure_distinguishes_missing_zero_and_unauthorized_effect(tmp_path):
    source = capture(tmp_path, [trial(case="a1",
        unguarded=assessment(attempted_unauthorized=True, execution={"unauthorized_effect": True}),
        guarded=assessment(attempted_unauthorized=True, execution={"unauthorized_effect": False}))])
    fig = build_figure(summarize(source), "authorization")
    try:
        labels = [text.get_text() for ax in fig.axes for text in ax.texts]
        assert any("Not observed" in text for text in labels)
        assert any(text.startswith("0%") for text in labels)
        legend = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
        assert legend == ["Foreign-record attempt", "Foreign-record effect"]
        assert "attack cases" in fig._suptitle.get_text().lower()
        caption = fig._supxlabel.get_text()
        assert "denominator: all attack cases" in caption
        assert "2 categories" in caption
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_latency_labels_explain_remote_request_boundary_and_never_invent_zero(tmp_path):
    source = capture(tmp_path, [trial(wall=0)])
    fig = build_figure(summarize(source), "request-cost")
    try:
        labels = [text.get_text() for ax in fig.axes for text in ax.texts]
        assert any("Not observed" in text for text in labels)
        assert any(text.startswith("0.00") for text in labels)
        assert "network" in fig._supxlabel.get_text().lower()
        assert "seconds" in fig.axes[0].get_xlabel().lower()
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_all_failed_capture_still_renders_coverage_without_success_claims(tmp_path):
    source = capture(tmp_path, [trial(status="error")])
    result = render(source, tmp_path / "failed-report")
    assert result["coverage"]["errors"] == 1
    assert result["configurations"][0]["schema_valid"]["mean"] is None


def test_empty_series_keep_distinct_legend_colors(tmp_path):
    fig = build_figure(summarize(capture(tmp_path, [])), "authorization")
    try:
        handles = fig.axes[0].get_legend().legend_handles
        assert len({tuple(handle.get_facecolor()) for handle in handles}) == 2
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_format_utility_shows_attack_failure_even_without_foreign_access(tmp_path):
    source = capture(tmp_path, [trial(), trial(case="a1",
        unguarded=assessment(task_success=False), guarded=assessment(task_success=False))])
    summary = summarize(source)
    assert group(summary, variant="attack")["unauthorized_effect"]["mean"] == 0
    fig = build_figure(summary, "format-utility")
    try:
        ax = fig.axes[0]
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert labels == ["Schema valid · all responses", "Task success · benign, guarded",
                          "Task success · attack, guarded"]
        assert [[patch.get_width() for patch in container.patches]
                for container in ax.containers] == [[1], [1], [0]]
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
