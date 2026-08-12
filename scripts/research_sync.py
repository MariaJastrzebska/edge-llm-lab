#!/usr/bin/env python3
"""research_sync.py — import Claude Science artifacts into this repo.

Claude Science (the local desktop research app, daemon on :8000) has no public
API — but it writes to the same disk. This reads its SQLite catalog, resolves a
project by name, and copies that project's latest text artifacts into a repo
research directory, so findings live in git per the repo's reporting contract
instead of only inside Science.

Mapping (project-name substring, case-insensitive → destination, relative to
the repo this script lives in):

    bigos, landscape        -> research/digests/from-science/
    edge-lab, g1, energy,    docs/research/from-science/   (edge-llm-lab layout)
    constrained, paper

Usage:
    tools/research_sync.py --list                 # show Science projects
    tools/research_sync.py --project bigos-landscape [--dest DIR] [--commit]
    tools/research_sync.py --all [--commit]       # every mapped project

Only .md/.txt/.csv/.json/.tsv artifacts are copied (research outputs, not
binaries); user uploads and ephemeral scratch files are skipped.
"""

import argparse
import datetime as dt
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

CS_ROOT = Path(os.environ.get("CLAUDE_SCIENCE_DIR",
                              str(Path.home() / ".claude-science")))
REPO = Path(__file__).resolve().parent.parent
TEXT_EXT = {".md", ".txt", ".csv", ".json", ".tsv", ".yaml", ".yml"}

# project-name keyword -> destination dir relative to REPO. First match wins.
ROUTES = [
    (("bigos", "landscape", "startup"), "research/digests/from-science"),
    (("edge-lab", "edge lab", "g1", "g2", "energy", "constrained",
      "decoding", "paper", "mlsys"), "docs/research/from-science"),
]


def _org_db():
    """Find the single org's operon-cli.db under ~/.claude-science/orgs/*."""
    orgs = CS_ROOT / "orgs"
    dbs = list(orgs.glob("*/operon-cli.db")) if orgs.is_dir() else []
    if not dbs:
        sys.exit(f"No Claude Science catalog under {orgs} — is the app set up?")
    if len(dbs) > 1:
        # Newest-touched org wins; print a note so it's never silent.
        dbs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"[note] {len(dbs)} orgs found; using most recent: {dbs[0].parent.name}")
    return dbs[0]


def _conn():
    c = sqlite3.connect(_org_db())
    c.row_factory = sqlite3.Row
    return c


def list_projects():
    c = _conn()
    rows = c.execute(
        "SELECT id, name FROM projects WHERE archived_at IS NULL ORDER BY updated_at DESC"
    ).fetchall()
    for r in rows:
        print(f"  {r['name']}")
    return rows


def _dest_for(name: str):
    low = name.lower()
    for keys, dest in ROUTES:
        if any(k in low for k in keys):
            return REPO / dest
    return None


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60]


def sync_project(name: str, dest: Path = None) -> int:
    """Copy a project's latest text artifacts into dest. Returns file count."""
    c = _conn()
    row = c.execute(
        "SELECT id, name FROM projects WHERE lower(name) = lower(?)", (name,)
    ).fetchone()
    if not row:
        # substring fallback so `--project bigos` matches `bigos-landscape`
        row = c.execute(
            "SELECT id, name FROM projects WHERE lower(name) LIKE ? "
            "ORDER BY updated_at DESC LIMIT 1", (f"%{name.lower()}%",)
        ).fetchone()
    if not row:
        sys.exit(f"No Science project matching '{name}'. Try --list.")

    pid, pname = row["id"], row["name"]
    dest = dest or _dest_for(pname)
    if dest is None:
        sys.exit(f"Project '{pname}' matches no route — pass --dest explicitly.")
    dest.mkdir(parents=True, exist_ok=True)

    org_dir = _org_db().parent
    arts = c.execute(
        """SELECT a.filename, v.storage_path, v.content_type, v.created_at
           FROM artifacts a JOIN artifact_versions v ON v.id = a.latest_version_id
           WHERE a.project_id = ?
             AND COALESCE(a.is_user_upload, 0) = 0
             AND COALESCE(a.is_ephemeral, 0) = 0""",
        (pid,),
    ).fetchall()

    today = dt.date.today().isoformat()
    n = 0
    for a in arts:
        fn = a["filename"] or "artifact"
        if Path(fn).suffix.lower() not in TEXT_EXT:
            continue
        src = org_dir / "artifacts" / a["storage_path"] if a["storage_path"] else None
        if not src or not src.exists():
            # storage_path may already be absolute or repo-relative
            src = Path(a["storage_path"]) if a["storage_path"] else None
            if not src or not src.exists():
                print(f"  [skip] missing on disk: {fn}")
                continue
        out = dest / f"{today}_{_slug(pname)}_{fn}"
        shutil.copyfile(src, out)
        n += 1
        print(f"  [ok] {fn} -> {out.relative_to(REPO)}")

    if n == 0:
        print(f"  (project '{pname}' has no text artifacts yet)")
    else:
        # Stamp a small index so the digest folder reads as a record, not a dump.
        idx = dest / "INDEX.md"
        with idx.open("a") as f:
            f.write(f"- {today}: {n} artefakt(y) z projektu Claude Science "
                    f"'{pname}'\n")
    return n


def git_commit(paths):
    subprocess.run(["git", "-C", str(REPO), "add", *map(str, paths)], check=True)
    r = subprocess.run(
        ["git", "-C", str(REPO), "commit", "-m",
         "research: import Claude Science artifacts (research_sync)"],
        capture_output=True, text=True)
    print(r.stdout or r.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--project")
    ap.add_argument("--dest")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    if args.list:
        list_projects()
        return

    touched = []
    if args.all:
        for r in list_projects():
            if _dest_for(r["name"]):
                if sync_project(r["name"]):
                    touched.append(_dest_for(r["name"]))
    elif args.project:
        dest = Path(args.dest).resolve() if args.dest else None
        if sync_project(args.project, dest):
            touched.append(dest or _dest_for(args.project))
    else:
        ap.error("give --list, --project NAME, or --all")

    if args.commit and touched:
        git_commit(set(touched))


if __name__ == "__main__":
    main()
