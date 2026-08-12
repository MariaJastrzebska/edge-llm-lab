"""Telegram notifications — sweep progress and change reports out of the lab.

Token and chat ID resolve through ``edge_llm_lab.utils.gcp_secrets`` (secrets
``Edge_lab_fed_bot`` and ``EDGE_LAB_CHAT_ID``), so the order is the same as for
every other credential: environment variable, then Secret Manager.

Stdlib only — this must run on a Pi or Jetson with nothing extra installed, and
``requests`` is not even declared in pyproject.toml.

Usage::

    python -m edge_llm_lab.utils.notify --report    # change summary -> Telegram
    python -m edge_llm_lab.utils.notify --dry-run   # print, do not send
    python -m edge_llm_lab.utils.notify "text"      # arbitrary message

Discipline: this module never kills its caller. Every failure is a False plus a
log line — an overnight sweep must not die at hour six because Telegram was
unreachable. Same contract as ``db_record`` in scripts/run_experiments.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

from .gcp_secrets import resolve

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
API_URL = "https://api.telegram.org/bot"
REPO_NAME = "edge-llm-lab"


def _redact(text: str, token: str) -> str:
    """Strip the token out of text headed for a log.

    The token lives in the URL, so urllib error messages can carry it verbatim —
    without this it would leak into the logs on the first 404.
    """
    if token and token in text:
        text = text.replace(token, "<token>")
    return text


def send(text: str, timeout: int = 10) -> bool:
    """Send a Telegram message. Returns True/False and never raises."""
    token = resolve("TELEGRAM_BOT_TOKEN", default="") or ""
    chat_id = resolve("TELEGRAM_CHAT_ID", default="") or ""

    if not token or not chat_id:
        logger.error(
            "Telegram not configured: token=%s chat_id=%s", bool(token), bool(chat_id)
        )
        return False

    payload = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
    request = urllib.request.Request(
        f"{API_URL}{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status == 200:
                logger.info("Telegram message sent (chat_id=%s)", chat_id)
                return True
            logger.error("Telegram API returned %s", response.status)
            return False
    except urllib.error.HTTPError as exc:
        # 403 "bot can't initiate conversation" = Start was never pressed.
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")[:200]
        except Exception:
            pass
        logger.error("Telegram API %s: %s", exc.code, _redact(body, token))
        return False
    except Exception as exc:
        logger.error("Telegram send failed: %s", _redact(str(exc), token))
        return False


def _git(*args: str) -> str:
    """Output of a git command, or an empty string. Never raises."""
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def report() -> str:
    """Summary of this branch against origin/main (or the latest commits)."""
    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "?"

    base = "origin/main"
    commits = _git("log", "--oneline", f"{base}..HEAD")
    if commits:
        stat = _git("diff", "--shortstat", f"{base}..HEAD")
        header = f"{len(commits.splitlines())} commits vs {base}"
    else:
        # No origin/main, or the branch does not diverge — show the last five.
        commits = _git("log", "--oneline", "-5")
        stat = _git("diff", "--shortstat", "HEAD~1..HEAD")
        header = "latest commits"

    lines = [f"{REPO_NAME} · {branch}", "", header]
    if stat:
        lines.append(stat)
    if commits:
        lines.append("")
        for line in commits.splitlines():
            _, _, title = line.partition(" ")
            lines.append(f"• {title or line}")

    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Send a Telegram message.")
    parser.add_argument("text", nargs="?", help="body; defaults to the git report")
    parser.add_argument("--report", action="store_true", help="summarise repo changes")
    parser.add_argument("--dry-run", action="store_true", help="print instead of sending")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    text = args.text if args.text and not args.report else report()

    if args.dry_run:
        print(text)
        return 0

    return 0 if send(text) else 1


if __name__ == "__main__":
    raise SystemExit(main())
