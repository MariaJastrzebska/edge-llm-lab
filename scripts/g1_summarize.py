#!/usr/bin/env python3
"""First-glance descriptives for a G1 pilot result directory.

DESCRIPTIVE ONLY, by design. This prints per-condition means, medians and
validity rates so NIGHT-REPORT.md can state what was observed. It deliberately
does NOT run significance tests, fit models, or compare conditions with a
verdict — the analysis happens on the Linux side (see experiments/RUNBOOK.md:
the Mac is the instrument, not the analyst).

Every statistic is computed only over rows that actually carry the value; the
count behind each number is printed next to it, so a mean over 3 surviving runs
can never be mistaken for a mean over 100.

Usage:  python3 scripts/g1_summarize.py experiments/results/<dir>
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path


def _f(row: dict, key: str):
    v = row.get(key)
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _b(row: dict, key: str):
    v = (row.get(key) or "").strip().lower()
    if v in ("true", "1"):
        return True
    if v in ("false", "0"):
        return False
    return None


def describe(values: list[float]) -> str:
    if not values:
        return "n=0  —"
    mean = statistics.mean(values)
    med = statistics.median(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"n={len(values):<4} mean={mean:10.2f}  median={med:10.2f}  sd={sd:9.2f}"


def rate(flags: list[bool]) -> str:
    if not flags:
        return "n=0  —"
    ok = sum(1 for f in flags if f)
    return f"n={len(flags):<4} {ok}/{len(flags)} = {100.0 * ok / len(flags):5.1f}%"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    run_dir = Path(sys.argv[1])
    csv_path = run_dir / "tidy.csv"
    if not csv_path.exists():
        print(f"no tidy.csv in {run_dir}", file=sys.stderr)
        return 2

    all_rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))

    print(f"# G1 pilot descriptives — {run_dir.name}")
    print(f"total rows: {len(all_rows)}")

    models = sorted({(r.get("model") or "") for r in all_rows})
    if len(models) > 1:
        # A multi-model run must never be pooled: models differ in tokenizer,
        # template and speed, so a pooled mean is a mean of different things.
        for m in models:
            print("\n" + "=" * 72)
            print(f"= MODEL: {m or '(unspecified)'}")
            print("=" * 72)
            report([r for r in all_rows if (r.get("model") or "") == m])
    else:
        report(all_rows)
    return 0


def report(rows: list[dict]) -> None:
    conditions = sorted({r["condition"] for r in rows})
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    print(f"rows: {len(rows)}   status ok: {len(ok_rows)}   "
          f"non-ok: {len(rows) - len(ok_rows)}")
    print()

    metrics = [
        ("predicted_ms", "decode wall time (ms)"),
        ("predicted_per_token_ms", "ms per generated token"),
        ("predicted_n", "tokens generated"),
        ("prompt_ms", "prefill time (ms)"),
        ("combined_j", "energy, gross (J)"),
        ("combined_j_baseline_subtracted", "energy, baseline-subtracted (J)"),
        ("mean_combined_mw", "mean SoC power (mW)"),
    ]

    for key, label in metrics:
        print(f"## {label}  [{key}]")
        for cond in conditions:
            vals = [v for r in ok_rows if r["condition"] == cond
                    and (v := _f(r, key)) is not None]
            print(f"  {cond:<15} {describe(vals)}")
        print()

    print("## output validity")
    for key, label in [
        ("parses_json_strict", "parses as JSON (strict)"),
        ("parses_json_lenient", "parses as JSON (lenient)"),
        ("schema_valid", "validates against schema"),
    ]:
        print(f"  {label}")
        for cond in conditions:
            flags = [b for r in ok_rows if r["condition"] == cond
                     and (b := _b(r, key)) is not None]
            print(f"    {cond:<15} {rate(flags)}")
    print()

    print("## truncation (hit n_predict cap)")
    for cond in conditions:
        flags = [(r.get("stop_type") == "limit") for r in ok_rows
                 if r["condition"] == cond]
        print(f"  {cond:<15} {rate(flags)}")
    print()

    print("## energy sampling quality (coverage of each generation window)")
    for cond in conditions:
        vals = [v for r in ok_rows if r["condition"] == cond
                and (v := _f(r, "energy_coverage")) is not None]
        low = sum(1 for v in vals if v < 0.95)
        print(f"  {cond:<15} {describe(vals)}   windows below 0.95 coverage: {low}")
    print()

    print("## Joules per VALID output (G2) — gross energy / count of schema-valid runs")
    print("   (sum of energy over ALL runs divided by the number that validated:")
    print("    invalid generations burned energy and produced no usable tool call)")
    for cond in conditions:
        sub = [r for r in ok_rows if r["condition"] == cond]
        energies = [v for r in sub if (v := _f(r, "combined_j")) is not None]
        n_valid = sum(1 for r in sub if _b(r, "schema_valid") is True)
        if energies and n_valid:
            print(f"  {cond:<15} {sum(energies) / n_valid:8.2f} J/valid   "
                  f"({len(energies)} runs, {n_valid} valid)")
        else:
            print(f"  {cond:<15} undefined ({len(energies)} runs, {n_valid} valid)")


if __name__ == "__main__":
    sys.exit(main())
