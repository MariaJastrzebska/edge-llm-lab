"""Descriptive paired-condition figures from measured, transcript-free CSVs.

The unit of aggregation is a prompt: average its repeats first, then weight
prompts equally. Paired differences match model, quantization, prompt, repeat
and seed. They are descriptive differences, not estimates of causal overhead.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

CONDITIONS = ("unconstrained", "gbnf")
METRICS = ("schema_valid", "wall_seconds", "generated_tokens", "energy_joules")
COLORS = {"unconstrained": "#52687e", "gbnf": "#148477"}
PANELS = (
    ("schema_valid", "validity", "Schema-valid extracted objects", "Prompt-weighted fraction (%)"),
    ("wall_seconds", "latency", "Request wall time", "Seconds per request"),
    ("generated_tokens", "tokens", "Generated tokens", "Tokens per request"),
    ("energy_joules", "energy", "Gross package energy", "Joules per request"),
)
REQUIRED = {"model", "quantization", "condition", "repeat", "seed", "prompt_id",
            "status", "schema_valid", "wall_ms", "predicted_n", "combined_j",
            "energy_coverage"}


def _number(value: str, field: str) -> float | None:
    if not value.strip():
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {field}: expected a finite nonnegative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Invalid {field}: expected a finite nonnegative number")
    return number


def _boolean(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized == "":
        return None
    if normalized not in ("true", "false"):
        raise ValueError("Invalid schema_valid: expected True, False, or an empty field")
    return normalized == "true"


def _aggregate(values: list[tuple[str, float | None]], attempts: int) -> dict:
    prompts = defaultdict(list)
    observations = 0
    for prompt, value in values:
        if value is not None:
            prompts[prompt].append(value)
            observations += 1
    return {
        "mean": mean(mean(group) for group in prompts.values()) if prompts else None,
        "observations": observations,
        "prompts": len(prompts),
        "missing": attempts - observations,
    }


def _source_name(source: Path) -> str:
    """Keep personal checkout paths out of portable metadata."""
    for parent in source.resolve().parents:
        if (parent / ".git").exists():
            return source.resolve().relative_to(parent).as_posix()
    return source.name


def summarize(source: str | Path, *, minimum_energy_coverage: float = .9,
              manifest_path: str | Path | None = None) -> dict:
    """Validate observations and aggregate within model/quantization pairs.

    A failed request is an attempt, never a fabricated zero. Each metric has
    its own observed denominator. Gross energy requires finite telemetry and
    the declared minimum temporal coverage. No patient text is read from CSV.
    """
    if not 0 <= minimum_energy_coverage <= 1:
        raise ValueError("minimum_energy_coverage must be between 0 and 1")
    source = Path(source)
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("No observations to summarize")
    seen = set()
    groups = defaultdict(list)
    for row in rows:
        if any(row.get(field) is None for field in REQUIRED):
            raise ValueError("Incomplete CSV row")
        if row["condition"] not in CONDITIONS:
            raise ValueError("condition must be unconstrained or gbnf")
        for field in ("model", "quantization", "prompt_id", "repeat", "seed", "status"):
            if not row[field].strip():
                raise ValueError(f"Missing trial identity field: {field}")
        identity = tuple(row[k] for k in ("model", "quantization", "condition",
                                         "prompt_id", "repeat", "seed"))
        if identity in seen:
            raise ValueError("Duplicate model/condition/prompt/repeat/seed trial")
        seen.add(identity)
        valid = _boolean(row["schema_valid"])
        wall = _number(row["wall_ms"], "wall_ms")
        tokens = _number(row["predicted_n"], "predicted_n")
        energy = _number(row["combined_j"], "combined_j")
        coverage = _number(row["energy_coverage"], "energy_coverage")
        low_coverage = coverage is not None and coverage < minimum_energy_coverage
        eligible_energy = energy is not None and coverage is not None and not low_coverage
        row["values"] = {
            "schema_valid": float(valid) if valid is not None else None,
            "wall_seconds": wall / 1000 if wall is not None else None,
            "generated_tokens": tokens,
            "energy_joules": energy if eligible_energy else None,
        }
        row["low_energy_coverage"] = low_coverage and energy is not None
        if row["status"] != "ok":
            row["values"] = dict.fromkeys(METRICS)
        groups[(row["model"], row["quantization"])].append(row)
    models = []
    for (model, quantization), trials in sorted(groups.items()):
        result = {"model": model, "quantization": quantization, "conditions": {}}
        indices = {}
        for condition in CONDITIONS:
            selected = [r for r in trials if r["condition"] == condition]
            attempts = len(selected)
            stats = {
                "attempts": attempts,
                "errors": sum(r["status"] != "ok" for r in selected),
                "energy_rejected_low_coverage": sum(r["low_energy_coverage"] for r in selected),
            }
            for metric in METRICS:
                stats[metric] = _aggregate(
                    [(r["prompt_id"], r["values"][metric]) for r in selected], attempts)
            stats["schema_valid"]["true_count"] = sum(
                r["values"]["schema_valid"] == 1 for r in selected)
            result["conditions"][condition] = stats
            indices[condition] = {(r["prompt_id"], r["repeat"], r["seed"]): r
                                  for r in selected}
        common = indices["unconstrained"].keys() & indices["gbnf"].keys()
        paired = {"direction": "gbnf minus unconstrained", "matched_attempts": len(common),
                  "unmatched_attempts": len(trials) - 2 * len(common)}
        for metric in METRICS:
            differences = []
            for key in sorted(common):
                base = indices["unconstrained"][key]["values"][metric]
                constrained = indices["gbnf"][key]["values"][metric]
                delta = constrained - base if base is not None and constrained is not None else None
                differences.append((key[0], delta))
            paired[metric] = _aggregate(differences, len(common))
        result["paired"] = paired
        models.append(result)
    summary = {
        "schema_version": 1,
        "source": {"csv": _source_name(source),
                   "csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "attempts": len(rows),
        "prompt_count": len({r["prompt_id"] for r in rows}),
        "repeat_count": len({r["repeat"] for r in rows}),
        "condition_count": len({r["condition"] for r in rows}),
        "seeds": sorted({r["seed"] for r in rows}),
        "minimum_energy_coverage": minimum_energy_coverage,
        "aggregation": "Mean within prompt across eligible repeats, then equal-weight prompt mean.",
        "limitations": [
            "Descriptive results from one experiment; no confidence intervals or population claim.",
            "Schema validity applies to the extracted object and does not imply strict JSON-only output, correct arguments or clinical safety.",
            "Conditions produce different output lengths; differences do not isolate grammar-mask cost.",
            "Energy is gross package energy, not model-only or baseline-subtracted energy.",
            "Compare conditions within a model; model-specific prompts and quantizations may differ.",
            "Missing telemetry and failed requests are excluded per metric and counted explicitly.",
        ],
        "models": models,
    }
    if manifest_path is not None:
        path = Path(manifest_path)
        manifest = json.loads(path.read_text(encoding="utf-8"))["manifest"]
        # Only publish experiment metadata, never hostname, local paths or outputs.
        allowed = ("timestamp_utc", "repo_git_sha", "repo_git_dirty", "llama_cpp_git_sha",
                   "prompt_set_id", "sampler", "platform", "machine", "energy_backend",
                   "energy_caveat", "energy_status", "n_runs")
        summary["metadata"] = {key: manifest[key] for key in allowed if key in manifest}
        summary["source"]["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary["source"]["manifest"] = _source_name(path)
    return summary


def _draw_panel(ax, summary: dict, metric: str, title: str, xlabel: str) -> None:
    """Draw the same measured series in a standalone figure or overview panel."""
    values = [model["conditions"][c][metric]["mean"]
              for model in summary["models"] for c in CONDITIONS]
    maximum = max([v for v in values if v is not None] or [1])
    limit = 137 if metric == "schema_valid" else max(1, maximum * 1.43)
    ax.set_xlim(0, limit)
    for index, model in enumerate(summary["models"]):
        for offset, condition in zip((-.18, .18), CONDITIONS):
            stats = model["conditions"][condition]
            measure = stats[metric]
            value = measure["mean"]
            y = index + offset
            if value is None:
                ax.text(limit * .025, y, "No eligible data", va="center",
                        fontsize=8, color=COLORS[condition])
                continue
            plotted = value * 100 if metric == "schema_valid" else value
            ax.barh(y, plotted, height=.29, color=COLORS[condition], zorder=3)
            if metric == "schema_valid":
                label = (f"{plotted:.0f}% ({measure['true_count']}/"
                         f"{measure['observations']})")
            else:
                label = f"{value:.1f}"
                if metric == "energy_joules" or measure["missing"]:
                    label += f"  n={measure['observations']}/{stats['attempts']}"
            ax.text(plotted + limit * .02, y, label, va="center", fontsize=8)
    labels = [f"{m['model']}\n{m['quantization']}" for m in summary["models"]]
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - .5, -.55)
    ax.set_title(title, loc="left", pad=12, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#e4e8eb", zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=8)
    if metric == "schema_valid":
        ax.set_xticks((0, 25, 50, 75, 100))


def _save_figure(fig, prefix: Path) -> None:
    for extension in ("svg", "png"):
        fig.savefig(prefix.with_suffix(f".{extension}"), dpi=170, facecolor="white",
                    metadata={"Date": None} if extension == "svg" else None)


def render(source: str | Path, output_prefix: str | Path, *,
           summary_path: str | Path | None = None, manifest_path: str | Path | None = None,
           minimum_energy_coverage: float = .9) -> dict:
    """Render four legible condition comparisons without treating absent data as zero."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    summary = summarize(source, minimum_energy_coverage=minimum_energy_coverage,
                        manifest_path=manifest_path)
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    legend = [Patch(color=COLORS[c], label="GBNF" if c == "gbnf" else "Unconstrained")
              for c in CONDITIONS]
    date = summary.get("metadata", {}).get("timestamp_utc", "")[:10]
    observed_counts = ((summary["attempts"], "request"), (summary["prompt_count"], "prompt"),
                       (summary["repeat_count"], "repeat ID"), (summary["condition_count"], "condition"),
                       (len(summary["models"]), "model configuration"))
    count_label = "Observed: " + " · ".join(
        f"{count} {name}{'s' if count != 1 else ''}" for count, name in observed_counts)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "svg.fonttype": "none", "axes.titleweight": "bold"}):
        fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4))
        try:
            for index, (ax, (metric, _, title, xlabel)) in enumerate(zip(axes.flat, PANELS)):
                _draw_panel(ax, summary, metric, f"{chr(65 + index)}  {title}", xlabel)
            fig.suptitle("Constrained decoding: validity and resource cost", x=.02,
                         ha="left", fontsize=16, fontweight="bold", y=.97)
            prefix = f"Recorded {date}  |  " if date else ""
            fig.text(.02, .917, prefix + count_label, color="#435061")
            fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(.99, .895),
                       frameon=False, ncol=2)
            fig.text(.02, .045, "Means weight eligible prompts equally. Validity labels show valid / scored requests; "
                     "energy labels show eligible / attempted requests.", fontsize=8, color="#435061")
            fig.text(.02, .019, f"Energy requires ≥{minimum_energy_coverage:.0%} temporal coverage. "
                     "Output lengths and model prompts differ; these are descriptive comparisons, not isolated mask overhead.",
                     fontsize=8, color="#435061")
            fig.subplots_adjust(left=.145, right=.985, top=.83, bottom=.13, hspace=.43, wspace=.48)
            _save_figure(fig, output_prefix)
        finally:
            plt.close(fig)
        for metric, suffix, title, xlabel in PANELS:
            fig, ax = plt.subplots(figsize=(7.4, 5.0))
            try:
                _draw_panel(ax, summary, metric, "", xlabel)
                fig.suptitle(title, x=.035, ha="left", fontsize=16, fontweight="bold", y=.96)
                fig.text(.035, .88, count_label, fontsize=9, color="#435061")
                date_label = f"Recorded {date} · " if date else ""
                fig.text(.035, .83, date_label + "Within-model condition comparison", color="#435061", fontsize=9)
                fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(.98, .86),
                           frameon=False, ncol=2, fontsize=8)
                notes = {
                    "schema_valid": "Valid / scored requests. Extracted-object validity does not establish correct arguments or safety.",
                    "wall_seconds": "Measured HTTP request wall time; output lengths differ between conditions.",
                    "generated_tokens": "Completion lengths differ; request-level latency and energy cannot isolate grammar-mask overhead.",
                    "energy_joules": f"Gross package energy; ≥{minimum_energy_coverage:.0%} temporal coverage. n = eligible / attempted requests.",
                }
                fig.text(.035, .055, notes[metric], fontsize=8, color="#435061")
                fig.text(.035, .025, "Means weight eligible prompts equally. Model-specific templates and quantization differ; no model ranking.",
                         fontsize=8, color="#435061")
                fig.subplots_adjust(left=.205, right=.97, top=.75, bottom=.16)
                _save_figure(fig, output_prefix.with_name(f"{output_prefix.name}-{suffix}"))
            finally:
                plt.close(fig)
    destination = Path(summary_path) if summary_path else output_prefix.with_suffix(".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--manifest", type=Path, help="Optional raw.json; only allowlisted metadata is exported")
    parser.add_argument("--minimum-energy-coverage", type=float, default=.9)
    args = parser.parse_args()
    summary = render(args.csv, args.output_prefix, summary_path=args.summary,
                     manifest_path=args.manifest, minimum_energy_coverage=args.minimum_energy_coverage)
    print(f"Rendered {summary['attempts']} observations across {len(summary['models'])} models.")


if __name__ == "__main__":
    main()
