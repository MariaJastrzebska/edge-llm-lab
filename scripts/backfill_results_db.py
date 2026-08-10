#!/usr/bin/env python3
"""Backfill experiments/results.db from existing result.json artifacts.

Idempotent (INSERT OR REPLACE). Run after pulling results produced by an
older runner version, or to rebuild the DB from scratch — raw JSON files
are always the source of truth.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import RESULTS_DIR, db_record  # noqa: E402


def main() -> int:
    count = 0
    for result_file in sorted(RESULTS_DIR.glob("*/result.json")):
        result = json.loads(result_file.read_text(encoding="utf-8"))
        db_record(result_file.parent, result)
        count += 1
    print(f"backfilled {count} run(s) into experiments/results.db")
    return 0


if __name__ == "__main__":
    sys.exit(main())
