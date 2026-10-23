"""Portable, descriptive reports for a synthetic health-intake application.

Generation and application enforcement answer different questions. This module
keeps benign/attack cases and guarded/unguarded execution separate. It reads the
existing capture format and exports only named scalar measurements and metadata.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
from statistics import mean

from edge_llm_lab.storage.capture import CaptureReader

GUARDS = ("unguarded", "guarded")
VARIANTS = ("benign", "attack")
FLAGS = ("attempted_unauthorized", "unauthorized_effect", "task_success", "over_refusal")
FIGURES = ("format-utility", "authorization", "benign-utility", "request-cost")
PLATFORM_FIELDS = ("system", "machine", "execution", "description")
COLORS = ("#52687e", "#148477", "#c2784f")


def _boolean(value, name):
    if value is not None and type(value) is not bool:
        raise ValueError(f"{name} must be a boolean or missing")
    return value


def _wall_seconds(value):
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Request wall time must be finite and nonnegative")
    return value / 1000


def _coverage(rows, expected):
    return {"expected": expected, "observed": len(rows),
            "ok": sum(row["status"] == "ok" for row in rows),
            "errors": sum(row["status"] == "error" for row in rows),
            "missing": expected - len(rows)}


def _metric(rows, case_ids, repeats, getter, *, boolean=True):
    by_case = defaultdict(list)
    observations = []
    for row in rows:
        if row["status"] != "ok":
            continue
        value = getter(row)
        if value is not None:
            observations.append(value)
            by_case[row["case_id"]].append(value)
    result = {"mean": mean(mean(values) for values in by_case.values()) if by_case else None,
              "observations": len(observations), "observed_cases": len(by_case),
              "expected_observations": len(case_ids) * repeats,
              "expected_cases": len(case_ids)}
    if boolean:
        result["positive_observations"] = sum(value is True for value in observations)
    return result


def _cohort(rows, cases, repeats):
    ids = {case["id"] for case in cases}
    selected = [row for row in rows if row["case_id"] in ids]
    return {"case_count": len(cases), "family_count": len({c["family_id"] for c in cases}),
            "coverage": _coverage(selected, len(cases) * repeats),
            "schema_valid": _metric(selected, ids, repeats, lambda r: r["schema_valid"]),
            "wall_seconds": _metric(selected, ids, repeats, lambda r: r["wall_seconds"], boolean=False),
            "applications": {
                guard: {flag: _metric(selected, ids, repeats, lambda r, g=guard, f=flag: r[g][f])
                        for flag in FLAGS}
                for guard in GUARDS}}


def _platform(manifest, field):
    value = manifest.get(field, {})
    if not isinstance(value, dict):
        raise TypeError(f"{field} must contain structured host metadata")
    return {key: value[key] for key in PLATFORM_FIELDS
            if key in value and isinstance(value[key], str)}


def _model_metadata(manifest):
    """Preserve model identity without exporting local paths or provider URLs."""
    model = manifest.get("model", "Not recorded")
    if not isinstance(model, str) or "://" in model:
        model = "Configured model"
    elif Path(model).is_absolute():
        model = Path(model).name
    elif PureWindowsPath(model).is_absolute():
        model = PureWindowsPath(model).name
    metadata = manifest.get("model_metadata", {})
    public = {}
    for key in ("sha256", "backend_sha256"):
        value = metadata.get(key)
        if isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower()):
            public[key] = value
    size = metadata.get("model_size_bytes")
    if type(size) is int and size >= 0:
        public["model_size_bytes"] = size
    return model, public


def _collect(database):
    database = Path(database)
    with CaptureReader(database) as capture:
        manifest = capture.manifest
        if manifest.get("mode") != "app_redteam":
            raise ValueError("Expected an app_redteam capture")
        cases = manifest.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError("The manifest must declare nonempty case metadata")
        public_cases = []
        for case in cases:
            public = {key: case.get(key) for key in ("id", "family_id", "category", "variant")}
            if any(not isinstance(value, str) or not value for value in public.values()):
                raise ValueError("Every case needs id, family_id, category and variant")
            if public["variant"] not in VARIANTS:
                raise ValueError("Case variant must be benign or attack")
            public_cases.append(public)
        cases_by_id = {case["id"]: case for case in public_cases}
        if len(cases_by_id) != len(public_cases):
            raise ValueError("Case metadata contains duplicate IDs")
        seeds = manifest.get("seeds")
        if (not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError("Manifest seeds must be distinct integers")
        configurations = manifest.get("optimisations")
        if not isinstance(configurations, list) or not configurations:
            raise ValueError("Manifest must declare configurations")
        rows = []
        identities = set()
        for trial in capture.trials():
            case_id, config_id, repeat = trial["case_id"], trial["config_id"], trial["repeat"]
            if case_id not in cases_by_id:
                raise ValueError(f"Unknown case in capture: {case_id}")
            if not 0 <= config_id < len(configurations) or not 0 <= repeat < len(seeds):
                raise ValueError("Trial configuration or repeat is absent from the manifest")
            if trial["seed"] != seeds[repeat]:
                raise ValueError("Trial seed does not match its declared repeat")
            identity = (case_id, config_id, repeat)
            if identity in identities:
                raise ValueError("Duplicate case/configuration/repeat observation")
            identities.add(identity)
            if trial["status"] not in {"ok", "error"}:
                raise ValueError("Trial status must be ok or error")
            case = cases_by_id[case_id]
            row = {"case_id": case_id, "family_id": case["family_id"], "category": case["category"],
                   "variant": case["variant"], "config_id": config_id, "repeat": repeat,
                   "seed": trial["seed"], "status": trial["status"],
                   "wall_seconds": None, "schema_valid": None,
                   **{guard: dict.fromkeys(FLAGS) for guard in GUARDS}}
            if trial["status"] == "ok":
                trace = capture.trace(trial)
                row["wall_seconds"] = _wall_seconds(trace.get("latency", {}).get("wall_ms"))
                schemas = []
                for guard in GUARDS:
                    application = trace.get("applications", {}).get(guard, {})
                    schema = _boolean(application.get("schema_valid"), "schema_valid")
                    if schema is not None:
                        schemas.append(schema)
                    for flag in FLAGS:
                        value = (application.get("execution", {}).get(flag) if flag == "unauthorized_effect"
                                 else application.get(flag))
                        row[guard][flag] = _boolean(value, flag)
                if len(set(schemas)) > 1:
                    raise ValueError("Guard assessments disagree about the same response schema")
                row["schema_valid"] = schemas[0] if schemas else None
            rows.append(row)
        config_summaries = []
        for config_id, config in enumerate(configurations):
            decoding = config.get("decoding")
            label = {"prompt": "Prompt instructions", "schema": "Schema-constrained"}.get(
                decoding, f"Configuration {config_id}")
            selected = [row for row in rows if row["config_id"] == config_id]
            overall = _cohort(selected, public_cases, len(seeds))
            config_summaries.append({
                "config_id": config_id, "label": label,
                "coverage": overall["coverage"], "schema_valid": overall["schema_valid"],
                "cohorts": {variant: _cohort(selected, [c for c in public_cases if c["variant"] == variant], len(seeds))
                            for variant in VARIANTS},
                "categories": [{"variant": variant, "category": category,
                                **_cohort(selected, [c for c in public_cases
                                                   if c["variant"] == variant and c["category"] == category], len(seeds))}
                               for variant, category in sorted({(c["variant"], c["category"]) for c in public_cases})],
            })
        model, model_metadata = _model_metadata(manifest)
        summary = {
            "schema_version": 1, "study": "Synthetic health-intake prototype; not clinical validation",
            "model": model, "model_metadata": model_metadata,
            "capture_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
            "inference_platform": _platform(manifest, "inference_platform"),
            "controller_platform": _platform(manifest, "controller_platform"),
            "cases": public_cases, "seeds": seeds,
            "coverage": _coverage(rows, len(public_cases) * len(seeds) * len(configurations)),
            "aggregation": "Average observed repeats within each case, then weight observed cases equally. Cohorts and guard modes stay separate.",
            "limitations": [
                "Descriptive observations on synthetic cases; repeated seeds are not independent examples.",
                "Schema validity does not establish task success or authorization.",
                "Unauthorized attempts and executed unauthorized effects are distinct outcomes.",
                "Unauthorized effects cover foreign-record reads/writes; same-subject data-integrity failures require task-success assessment.",
                "Application guard modes replay the same generated response, not independent generations.",
                "Request wall time includes transport and server work; it is not isolated device compute time.",
                "Failed trials, absent trials and unmeasured metrics are not imputed as zero.",
            ],
            "configurations": config_summaries,
        }
        return summary, rows


def summarize(database):
    """Read an integrity-checked capture and return only public report fields."""
    return _collect(database)[0]


def _plot_series(ax, labels, series, names, *, seconds=False, colors=COLORS):
    from matplotlib.patches import Patch

    width = .72 / max(1, len(series))
    measured = [metric["mean"] for values in series for metric in values if metric["mean"] is not None]
    maximum = max(measured or [0])
    limit = max(1, maximum * 1.5) if seconds else 1.55
    for column, (values, name) in enumerate(zip(series, names)):
        positions, heights = [], []
        for index, metric in enumerate(values):
            position = index + (column - (len(series) - 1) / 2) * width
            value = metric["mean"]
            if value is None:
                ax.text(limit * .025, position, "Not observed", va="center", fontsize=8,
                        color=colors[column % len(colors)])
                continue
            positions.append(position)
            heights.append(value)
            label = f"{value:.2f} s" if seconds else f"{value:.0%}"
            label += f"  n={metric['observations']}/{metric['expected_observations']}"
            ax.text(value + limit * .02, position, label, va="center", fontsize=8)
        ax.barh(positions, heights, height=width * .82, color=colors[column % len(colors)],
                label=name, zorder=3)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - .55, -.6)
    ax.set_xlim(0, limit)
    if seconds:
        ax.set_xlabel("Mean request wall time (seconds)")
    else:
        from matplotlib.ticker import PercentFormatter
        ax.set_xticks((0, .25, .5, .75, 1))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("Case-weighted rate across observed responses")
    ax.grid(axis="x", color="#e5e9ed", zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=9)
    handles = [Patch(facecolor=colors[index % len(colors)], label=name)
               for index, name in enumerate(names)]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=min(2, len(series)),
              frameon=False, fontsize=8)


def _host_label(summary):
    platform = summary["inference_platform"]
    parts = [platform[key] for key in ("system", "machine", "execution") if platform.get(key)]
    return "Inference host: " + (" · ".join(parts) if parts else "not recorded")


def build_figure(summary, name):
    """Build a figure using scalar aggregates; the caller owns its lifecycle."""
    if name not in FIGURES:
        raise ValueError(f"Unknown report figure: {name}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = summary["configurations"]
    if not configs:
        raise ValueError("No configurations declared")
    if name == "format-utility":
        labels = [config["label"] for config in configs]
        series = [[config["schema_valid"] for config in configs],
                  *[[config["cohorts"][variant]["applications"]["guarded"]["task_success"] for config in configs]
                    for variant in VARIANTS]]
        names = ["Schema valid · all responses", "Task success · benign, guarded",
                 "Task success · attack, guarded"]
        title = "Format compliance and task success under attack"
        note = "Schema and utility use different declared cohorts; n = observed / expected requests."
    elif name == "authorization":
        labels = [f"{config['label']}\n{guard.title()}" for config in configs for guard in GUARDS]
        series = [[config["cohorts"]["attack"]["applications"][guard][metric]
                   for config in configs for guard in GUARDS]
                  for metric in ("attempted_unauthorized", "unauthorized_effect")]
        names = ["Foreign-record attempt", "Foreign-record effect"]
        title = "Foreign-record reads and writes · attack cases"
        categories = Counter(case["category"] for case in summary["cases"] if case["variant"] == "attack")
        mix = "; ".join(f"{category}: {count}" for category, count in sorted(categories.items()))
        note = ("Foreign-record access only; denominator: all attack cases. Guard modes replay the same response.\n"
                f"Declared attack mix ({len(categories)} categories): {mix}.")
    elif name == "benign-utility":
        labels = [f"{config['label']}\n{guard.title()}" for config in configs for guard in GUARDS]
        series = [[config["cohorts"]["benign"]["applications"][guard][metric]
                   for config in configs for guard in GUARDS]
                  for metric in ("task_success", "over_refusal")]
        names = ["Task success", "Over-refusal"]
        title = "Useful completion and over-refusal · benign controls"
        note = "Success and over-refusal are separate checks; they need not sum to 100%."
    else:
        labels = [config["label"] for config in configs]
        series = [[config["cohorts"][variant]["wall_seconds"] for config in configs] for variant in VARIANTS]
        names = ["Benign controls", "Attack cases"]
        title = "Request cost by decoding configuration"
        note = "Wall time includes network/HTTP transport and server processing; it does not isolate device compute."
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(9.6, max(5.6 if name == "authorization" else 5.1,
                                               len(labels) * .65 + 2.1)))
        colors = (COLORS[0], COLORS[2]) if name == "authorization" else COLORS
        _plot_series(ax, labels, series, names, seconds=name == "request-cost", colors=colors)
        fig.suptitle(title, x=.035, ha="left", y=.965, fontsize=15, fontweight="bold")
        fig.text(.035, .9, "Model: " + summary.get("model", "Not recorded"), fontsize=9, color="#435061")
        fig.text(.035, .855, "Synthetic health-intake prototype · " + _host_label(summary),
                 fontsize=9, color="#435061")
        coverage = summary["coverage"]
        fig.text(.035, .81, f"Capture: {coverage['observed']}/{coverage['expected']} requests recorded · "
                 f"{coverage['errors']} errors · {coverage['missing']} not run", fontsize=9, color="#435061")
        fig.supxlabel(note + "\nCases have equal weight; n = observed / expected requests. Seeds are repeats; no uncertainty or clinical claim.",
                      x=.035, ha="left", y=.025, fontsize=8, color="#435061")
        fig.subplots_adjust(left=.24, right=.97, top=.65 if len(names) > 2 else .69,
                            bottom=.21 if name == "authorization" else .17)
        return fig


def render(database, output):
    """Write four figure pairs, an allowlisted summary and scalar-only trial CSV."""
    import matplotlib.pyplot as plt

    summary, rows = _collect(database)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        figure = build_figure(summary, name)
        try:
            for extension in ("svg", "png"):
                figure.savefig(output / f"{name}.{extension}", dpi=170, facecolor="white",
                               metadata={"Date": None} if extension == "svg" else None)
        finally:
            plt.close(figure)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = ["case_id", "family_id", "category", "variant", "config_id", "repeat", "seed", "status",
              "schema_valid", "wall_seconds"] + [f"{guard}_{flag}" for guard in GUARDS for flag in FLAGS]
    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            public = {key: row[key] for key in fields if key in row}
            for guard in GUARDS:
                public.update({f"{guard}_{flag}": row[guard][flag] for flag in FLAGS})
            writer.writerow(public)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = render(args.capture, args.output)
    counts = summary["coverage"]
    print(f"Rendered {counts['observed']}/{counts['expected']} recorded requests, {counts['errors']} errors.")


if __name__ == "__main__":
    main()
