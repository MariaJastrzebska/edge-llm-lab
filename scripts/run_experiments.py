#!/usr/bin/env python3
"""One-command experiment queue runner (see experiments/RUNBOOK.md).

Executes every YAML config in experiments/queue/ in filename order; each run is
wrapped with a hardware/git manifest and before/after resource snapshots; results
land in experiments/results/<timestamp>-<name>/; finished configs move to
experiments/done/. Failures are recorded and the queue CONTINUES.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUEUE_DIR = REPO_ROOT / "experiments" / "queue"
DONE_DIR = REPO_ROOT / "experiments" / "done"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
DB_PATH = REPO_ROOT / "experiments" / "results.db"

sys.path.insert(0, str(REPO_ROOT / "src"))


# ── SQLite results store ─────────────────────────────────────────────────────
# Queryable metadata (the social-media repo pattern: DB for queries, raw JSON
# files remain the source-of-truth artifacts). One row per run in `runs`, one
# per repeat in `repeats`; analysis-level metrics land in `metrics` as tidy
# long format (run × repeat × metric name × value).

def db_connect():
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
          run_dir TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          started_utc TEXT NOT NULL,
          status TEXT NOT NULL,
          git_sha TEXT,
          hostname TEXT,
          platform TEXT,
          config_json TEXT NOT NULL,
          manifest_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS repeats (
          run_dir TEXT NOT NULL REFERENCES runs(run_dir),
          repeat INTEGER NOT NULL,
          status TEXT NOT NULL,
          exit_code INTEGER,
          duration_s REAL,
          ram_delta_gb REAL,
          PRIMARY KEY (run_dir, repeat)
        );
        CREATE TABLE IF NOT EXISTS metrics (
          run_dir TEXT NOT NULL,
          repeat INTEGER,
          name TEXT NOT NULL,
          value REAL,
          unit TEXT
        );
        """
    )
    return conn


def db_record(run_dir: Path, result: dict) -> None:
    try:
        conn = db_connect()
        manifest = result.get("manifest", {})
        conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
            (
                run_dir.name,
                result["config"]["name"],
                manifest.get("timestamp_utc", ""),
                result.get("status", "unknown"),
                manifest.get("git_sha"),
                manifest.get("hostname"),
                manifest.get("platform"),
                json.dumps(result["config"]),
                json.dumps(manifest, default=str),
            ),
        )
        for rep in result.get("repeats", []):
            before = (rep.get("resources_before") or {}).get("memory", {})
            after = (rep.get("resources_after") or {}).get("memory", {})
            ram_delta = None
            if "ram_used_gb" in before and "ram_used_gb" in after:
                ram_delta = after["ram_used_gb"] - before["ram_used_gb"]
            conn.execute(
                "INSERT OR REPLACE INTO repeats VALUES (?,?,?,?,?,?)",
                (
                    run_dir.name,
                    rep.get("repeat"),
                    rep.get("status", "unknown"),
                    rep.get("exit_code"),
                    rep.get("duration_s"),
                    ram_delta,
                ),
            )
        conn.commit()
        conn.close()
    except Exception as e:  # the DB is a convenience index, never a run-killer
        print(f"⚠ results.db write failed (raw JSON is intact): {e}")


def sh(cmd: list[str], **kw) -> str:
    return subprocess.check_output(cmd, text=True, cwd=REPO_ROOT, **kw).strip()


def build_manifest() -> dict:
    """Everything a reviewer will ask about 'what exactly ran'."""
    manifest = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_sha": sh(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(sh(["git", "status", "--porcelain"])),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
    }
    try:
        manifest["ollama_version"] = sh(["ollama", "--version"])
    except Exception:
        manifest["ollama_version"] = None
    return manifest


def resource_snapshot() -> dict:
    try:
        from edge_llm_lab.telemetry import monitor_resources

        return monitor_resources()
    except Exception as e:  # telemetry must never kill an experiment run
        return {"error": str(e)}


def check_requirements(requires: list[str]) -> list[str]:
    missing = []
    for tool in requires or []:
        if shutil.which(tool) is None:
            missing.append(tool)
    return missing


def load_config(path: Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for key in ("name", "command"):
        if key not in config:
            raise ValueError(f"{path.name}: missing required key '{key}'")
    return config


def run_one(config_path: Path, no_commit: bool) -> bool:
    config = load_config(config_path)
    name = config["name"]
    repeats = int(config.get("repeats", 1))
    timeout_s = int(config.get("timeout_minutes", 60)) * 60

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS_DIR / f"{stamp}-{name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    missing = check_requirements(config.get("requires", []))
    result: dict = {
        "config": config,
        "manifest": build_manifest(),
        "repeats": [],
    }

    if missing:
        result["status"] = "failed"
        result["error"] = f"missing required tools: {missing}"
        print(f"✗ {name}: missing tools {missing} — skipping")
    else:
        print(f"▶ {name}: {repeats} repeat(s)")
        statuses = []
        for repeat in range(1, repeats + 1):
            rep_dir = run_dir / f"repeat_{repeat:02d}"
            rep_dir.mkdir(exist_ok=True)
            env = os.environ.copy()
            env.update({"RUN_DIR": str(rep_dir), "REPEAT": str(repeat)})

            rep: dict = {"repeat": repeat, "resources_before": resource_snapshot()}
            t0 = _dt.datetime.now()
            try:
                proc = subprocess.run(
                    config["command"],
                    shell=True,
                    cwd=REPO_ROOT,
                    env=env,
                    timeout=timeout_s,
                    capture_output=True,
                    text=True,
                )
                rep["exit_code"] = proc.returncode
                rep["status"] = "ok" if proc.returncode == 0 else "failed"
                (rep_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
                (rep_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
            except subprocess.TimeoutExpired:
                rep["status"] = "timeout"
            except Exception as e:
                rep["status"] = "error"
                rep["error"] = str(e)
            rep["duration_s"] = (_dt.datetime.now() - t0).total_seconds()
            rep["resources_after"] = resource_snapshot()
            statuses.append(rep["status"])
            result["repeats"].append(rep)
            print(f"  repeat {repeat}/{repeats}: {rep['status']} "
                  f"({rep['duration_s']:.0f}s)")

        result["status"] = "ok" if all(s == "ok" for s in statuses) else "partial"

    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    db_record(run_dir, result)

    DONE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.move(str(config_path), DONE_DIR / config_path.name)

    if not no_commit:
        subprocess.run(["git", "add", "-A", "experiments/"], cwd=REPO_ROOT)
        subprocess.run(
            ["git", "commit", "-q", "-m",
             f"experiment: {name} [{result['status']}] ({stamp})"],
            cwd=REPO_ROOT,
        )
    return result["status"] == "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", metavar="NAME")
    parser.add_argument("--no-commit", action="store_true")
    args = parser.parse_args()

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    queue = sorted(QUEUE_DIR.glob("*.yaml"))
    if args.only:
        queue = [p for p in queue if load_config(p)["name"] == args.only]

    if not queue:
        print("Queue is empty — nothing to run. (experiments/queue/)")
        return 0

    print(f"{len(queue)} experiment(s) queued:")
    for p in queue:
        print(f"  • {load_config(p)['name']}  ({p.name})")
    if args.dry_run:
        return 0

    ok = all([run_one(p, args.no_commit) for p in queue])

    if not args.no_commit:
        push = subprocess.run(["git", "push"], cwd=REPO_ROOT)
        if push.returncode != 0:
            print("⚠ push failed — run `git push` manually so Claude can analyze.")
    print("Done." if ok else "Done with failures — details in experiments/results/.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
