"""GCP Secret Manager — single resolution point for the lab's credentials.

Every secret is looked up in this order:

1. **environment variable** (including a local ``.env``) — existing runs and CI
   keep working exactly as before;
2. **GCP Secret Manager** — ``projects/<project>/secrets/<name>/versions/latest``;
3. the supplied ``default``, or ``MissingSecret`` when ``required=True``.

Environment first is deliberate: turning Secret Manager on cannot break a
machine that already has the variable exported, and any run can override a
credential locally without touching the cloud project.

Paper-grade note: nothing here is recorded in a run manifest. Credentials are
resolved lazily at call time and never logged — only the *source* they came
from, so a manifest can state "judge key: secret-manager" without leaking it.

Setup
-----
    poetry install                            # google-cloud-secret-manager included
    export GCP_PROJECT_ID=llm-apps-...        # project holding the secrets
    gcloud auth application-default login     # local credentials

Verify the wiring without printing any secret values:

    python -m edge_llm_lab.utils.gcp_secrets
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# Environment variable -> Secret Manager secret name.
#
# Override any single mapping without a code change by setting
# ``<ENV_VAR>_SECRET``, e.g. ``NEPTUNE_API_TOKEN_SECRET=NEPTUNE_API_TOKEN_v2``.
SECRETS: Dict[str, str] = {
    # Experiment tracking (RunTracker / NeptuneManager).
    "NEPTUNE_API_TOKEN": "NEPTUNE_API_TOKEN",
    # LLM-judge and reference generation.
    "OPENAI_API_KEY": "OPENAI_API_KEY",
    # Telegram bot reporting sweep progress out of the lab.
    "TELEGRAM_BOT_TOKEN": "Edge_lab_fed_bot",
    # Numeric chat ID the bot writes to.
    "TELEGRAM_CHAT_ID": "TELEGRAM_CHAT_ID",
}

# Checked in order; the first non-empty one wins.
PROJECT_ENV_VARS = ("GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT")


class MissingSecret(RuntimeError):
    """A secret marked ``required`` was not found in any source."""


_cache: Dict[Tuple[str, str], Optional[str]] = {}
_lock = threading.Lock()
_client = None
_client_unavailable = False


def project_id() -> Optional[str]:
    """Project holding the secrets, or None if it cannot be determined."""
    for var in PROJECT_ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            return value
    # Fall back to whatever project the ambient credentials default to.
    try:
        import google.auth

        return google.auth.default()[1]
    except Exception:
        return None


def _get_client():
    """Secret Manager client, or None when the library/credentials are absent.

    A missing client is not an error: the caller falls back to environment
    variables, which is the normal state on a Pi or Jetson with no GCP set up.
    """
    global _client, _client_unavailable
    if _client is not None or _client_unavailable:
        return _client

    try:
        from google.cloud import secretmanager
    except ImportError:
        logger.debug("google-cloud-secret-manager not installed — using env vars only")
        _client_unavailable = True
        return None

    try:
        _client = secretmanager.SecretManagerServiceClient()
    except Exception as exc:  # no ADC, malformed credentials, ...
        logger.warning("Secret Manager unavailable (%s) — using env vars only", exc)
        _client_unavailable = True
        return None

    return _client


def _fetch(secret_name: str, version: str = "latest") -> Optional[str]:
    """Read one secret version. Returns None on any failure (never raises).

    Failing soft matters here: an overnight sweep must not die at hour six
    because a credential lookup hiccuped.
    """
    client = _get_client()
    if client is None:
        return None

    project = project_id()
    if not project:
        logger.warning(
            "No GCP project configured — set GCP_PROJECT_ID to read %s from Secret Manager",
            secret_name,
        )
        return None

    path = f"projects/{project}/secrets/{secret_name}/versions/{version}"
    try:
        response = client.access_secret_version(request={"name": path})
    except Exception as exc:
        # NotFound / PermissionDenied carry the resource path, never the payload.
        logger.warning("Secret Manager lookup failed for %s: %s", secret_name, exc)
        return None

    # Pasting a token into the console adds a trailing newline more often than
    # not, and a stray "\n" in an Authorization header fails with an opaque 401.
    return response.payload.data.decode("utf-8").strip()


def _fetch_cached(secret_name: str) -> Optional[str]:
    key = (project_id() or "", secret_name)
    with _lock:
        if key in _cache:
            return _cache[key]
    # Fetched outside the lock: a slow RPC must not block other secrets.
    value = _fetch(secret_name)
    with _lock:
        _cache.setdefault(key, value)
        return _cache[key]


def clear_cache() -> None:
    """Drop cached values so the next resolve() re-reads Secret Manager."""
    global _client, _client_unavailable
    with _lock:
        _cache.clear()
        _client = None
        _client_unavailable = False


def secret_name_for(env_var: str) -> Optional[str]:
    """Secret name backing ``env_var``, honouring a ``<ENV_VAR>_SECRET`` override."""
    override = os.environ.get(f"{env_var}_SECRET", "").strip()
    return override or SECRETS.get(env_var)


def resolve(
    env_var: str,
    secret_name: Optional[str] = None,
    *,
    default: Optional[str] = None,
    required: bool = False,
) -> Optional[str]:
    """Resolve one secret: env var, then Secret Manager, then default.

    Args:
        env_var: Environment variable to check first, e.g. "NEPTUNE_API_TOKEN".
        secret_name: Secret Manager name. Defaults to the SECRETS mapping.
        default: Returned when nothing is found and required is False.
        required: Raise MissingSecret instead of returning default.

    Returns:
        The secret value, or default.
    """
    value = os.environ.get(env_var, "").strip()
    if value:
        return value

    name = secret_name or secret_name_for(env_var)
    if name:
        value = _fetch_cached(name)
        if value:
            logger.info("Resolved %s from Secret Manager (%s)", env_var, name)
            return value

    if required:
        raise MissingSecret(
            f"{env_var} is not set and could not be read from Secret Manager"
            + (f" (secret: {name})" if name else " (no secret name mapped)")
        )
    return default


def source_of(env_var: str) -> str:
    """Where ``env_var`` resolves from: "env", "secret-manager" or "missing".

    Safe to record in a run manifest — it names the source, not the secret.
    """
    if os.environ.get(env_var, "").strip():
        return "env"
    name = secret_name_for(env_var)
    if name and _fetch_cached(name):
        return "secret-manager"
    return "missing"


def status() -> Dict[str, Dict[str, object]]:
    """Where each registered secret resolves from — without exposing values."""
    report: Dict[str, Dict[str, object]] = {}
    for env_var in SECRETS:
        name = secret_name_for(env_var)
        if os.environ.get(env_var, "").strip():
            source, value = "env", os.environ[env_var].strip()
        else:
            value = _fetch_cached(name) if name else None
            source = "secret-manager" if value else "missing"
        report[env_var] = {
            "source": source,
            "secret": name,
            "length": len(value) if value else 0,
        }
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    project = project_id()
    print(f"GCP project: {project or '(not configured — set GCP_PROJECT_ID)'}")
    print()

    report = status()
    width = max(len(k) for k in report)
    for env_var, info in report.items():
        mark = "-" if info["source"] == "missing" else "+"
        print(
            f"  {mark} {env_var:<{width}}  {str(info['secret'] or '(unmapped)'):<24}"
            f"  {info['source']} ({info['length']} chars)"
        )

    missing = [k for k, v in report.items() if v["source"] == "missing"]
    if missing:
        print(f"\nUnresolved: {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
