"""Contracts for descriptive comparisons, missing telemetry and figure semantics."""
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from edge_llm_lab.visualization.paired_report import render, summarize

FIELDS = ["model", "quantization", "condition", "repeat", "seed", "prompt_id",
          "status", "schema_valid", "wall_ms", "predicted_n", "combined_j",
          "energy_coverage"]


def write_rows(tmp_path, overrides):
    path = tmp_path / "tidy.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for update in overrides:
            row = dict(zip(FIELDS, ["small:1b", "Q4", "gbnf", "1", "11", "p1",
                                    "ok", "True", "2000", "20", "8", "0.99"]))
            row.update(update)
            writer.writerow(row)
    return path


def condition(summary, name="gbnf"):
    return summary["models"][0]["conditions"][name]


def test_prompts_receive_equal_weight_despite_unequal_repeats(tmp_path):
    source = write_rows(tmp_path, [{}, {"repeat": "2", "seed": "12"},
                                  {"prompt_id": "p2", "wall_ms": "10000",
                                   "schema_valid": "False"}])
    result = condition(summarize(source))
    assert result["wall_seconds"]["mean"] == 6
    assert result["schema_valid"]["mean"] == .5
    assert result["schema_valid"]["true_count"] == 2
    assert result["schema_valid"]["observations"] == 3
    assert result["schema_valid"]["prompts"] == 2


def test_false_missing_and_failed_are_distinct(tmp_path):
    source = write_rows(tmp_path, [{"schema_valid": "False"},
        {"prompt_id": "p2", "schema_valid": ""},
        {"prompt_id": "p3", "status": "error", "schema_valid": "True"}])
    result = condition(summarize(source))
    assert result["attempts"] == 3
    assert result["errors"] == 1
    assert result["schema_valid"]["mean"] == 0
    assert result["schema_valid"]["observations"] == 1
    assert result["schema_valid"]["missing"] == 2


def test_unknown_boolean_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="schema_valid"):
        summarize(write_rows(tmp_path, [{"schema_valid": "probably"}]))


def test_missing_or_poorly_covered_energy_is_not_zero(tmp_path):
    source = write_rows(tmp_path, [{}, {"prompt_id": "p2", "combined_j": ""},
        {"prompt_id": "p3", "combined_j": "100", "energy_coverage": "0.5"}])
    result = condition(summarize(source))
    assert result["energy_joules"]["mean"] == 8
    assert result["energy_joules"]["observations"] == 1
    assert result["energy_joules"]["missing"] == 2
    assert result["energy_rejected_low_coverage"] == 1


def test_pairs_match_prompt_repeat_seed_not_row_order(tmp_path):
    source = write_rows(tmp_path, [
        {"condition": "unconstrained", "wall_ms": "1000"},
        {"prompt_id": "p2", "wall_ms": "20000"},
        {},
        {"condition": "unconstrained", "prompt_id": "p2", "wall_ms": "16000"},
        {"prompt_id": "p3", "wall_ms": "50000"}])
    model = summarize(source)["models"][0]
    assert model["paired"]["matched_attempts"] == 2
    assert model["paired"]["wall_seconds"]["mean"] == 2.5
    assert model["paired"]["wall_seconds"]["observations"] == 2
    assert model["paired"]["unmatched_attempts"] == 1


def test_duplicate_trial_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(write_rows(tmp_path, [{}, {}]))


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_invalid_physical_measurements_are_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="wall_ms"):
        summarize(write_rows(tmp_path, [{"wall_ms": value}]))


def test_render_preserves_inputs_and_contains_units_and_missing_labels(tmp_path):
    source = write_rows(tmp_path, [{"combined_j": ""},
        {"condition": "unconstrained", "schema_valid": "False", "combined_j": ""}])
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "figures" / "comparison"
    summary_path = tmp_path / "summary.json"
    render(source, output, summary_path=summary_path)
    ET.parse(output.with_suffix(".svg"))
    svg = output.with_suffix(".svg").read_text()
    for label in ["Schema-valid extracted objects", "Request wall time", "Generated tokens",
                  "Gross package energy", "No eligible data", "Unconstrained", "GBNF"]:
        assert label in svg
    assert "Request index" not in svg
    assert output.with_suffix(".png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert json.loads(summary_path.read_text())["source"]["csv_sha256"] == before
    for suffix in ("validity", "latency", "tokens", "energy"):
        png = output.with_name(f"comparison-{suffix}.png")
        assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        svg_path = output.with_name(f"comparison-{suffix}.svg")
        ET.parse(svg_path)
        assert "Within-model condition comparison" in svg_path.read_text()
    import matplotlib.pyplot as plt
    assert not plt.get_fignums()


def test_recorded_study_aggregates_are_reproducible():
    source = Path(__file__).resolve().parents[2] / "experiments/results/20260812-211032-g1-short-forward-pass/repeat_01/tidy.csv"
    result = summarize(source)
    assert result["attempts"] == 480
    assert result["prompt_count"] == 20
    assert result["repeat_count"] == 3
    models = {m["model"]: m for m in result["models"]}
    assert {name: m["conditions"]["gbnf"]["schema_valid"]["true_count"]
            for name, m in models.items()} == {
        "qwen3:0.6b": 60, "llama3.2:1b": 58, "gemma3:1b": 52, "qwen2.5:0.5b": 60}
    assert all(m["conditions"]["unconstrained"]["schema_valid"]["true_count"] == 0
               for m in models.values())
    assert all(m["paired"]["matched_attempts"] == 60 for m in models.values())
    gemma = models["gemma3:1b"]["conditions"]["unconstrained"]
    assert gemma["energy_joules"]["observations"] == 52
    assert gemma["energy_joules"]["prompts"] == 18
    assert gemma["energy_rejected_low_coverage"] == 8
    assert models["qwen3:0.6b"]["conditions"]["gbnf"]["wall_seconds"]["mean"] == pytest.approx(4.696358333333333)
    assert models["qwen2.5:0.5b"]["paired"]["wall_seconds"]["mean"] == pytest.approx(.11455616666666668)


def test_metadata_export_omits_local_paths_hostnames_and_raw_text(tmp_path):
    source = write_rows(tmp_path, [{}])
    manifest = tmp_path / "raw.json"
    manifest.write_text(json.dumps({
        "manifest": {"timestamp_utc": "2026-08-12", "machine": "arm64",
                     "hostname": "personal-host", "models_resolved": {"path": "/private/model"}},
        "runs": [{"output": "private transcript"}]}))
    result = summarize(source, manifest_path=manifest)
    assert result["metadata"] == {"timestamp_utc": "2026-08-12", "machine": "arm64"}
    exported = json.dumps(result)
    assert "personal-host" not in exported
    assert "private transcript" not in exported
    assert "/private/model" not in exported


def test_incomplete_csv_row_is_explicit_error(tmp_path):
    source = write_rows(tmp_path, [{}])
    source.write_text(source.read_text() + "model-only\n")
    with pytest.raises(ValueError, match="Incomplete CSV row"):
        summarize(source)


def test_unbalanced_design_caption_reports_observed_counts_not_cartesian_product(tmp_path):
    source = write_rows(tmp_path, [{}, {"repeat": "2", "seed": "12"},
                                  {"prompt_id": "p2"}])
    prefix = tmp_path / "unbalanced"
    summary = render(source, prefix)
    assert summary["attempts"] == 3
    assert summary["prompt_count"] == 2
    assert summary["repeat_count"] == 2
    assert summary["condition_count"] == 1
    for suffix in ("", "-validity", "-latency", "-tokens", "-energy"):
        svg = prefix.with_name(f"unbalanced{suffix}.svg").read_text()
        assert "Observed: 3 requests" in svg
        assert "2 prompts" in svg and "2 repeat IDs" in svg and "1 condition" in svg
        assert "×" not in svg
        assert " = 3 requests" not in svg
