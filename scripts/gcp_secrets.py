#!/usr/bin/env python3
"""Jedno źródło sekretów: GCP Secret Manager, dostępny z Maca i z chmury.

Zero zależności poza stdlib + binarką `openssl` (jest na macOS i w kontenerze
chmurowym). Kolejność uwierzytelnienia:

  1. GCP_ACCESS_TOKEN        — gotowy token (ucieczka awaryjna / CI),
  2. gcloud CLI              — ścieżka na Macu (`gcloud auth login` raz),
  3. klucz konta serwisowego — ścieżka w chmurze:
       GCP_SA_KEY            = pełny JSON klucza (wklejony w environment
                               variables środowiska Claude), albo
       GOOGLE_APPLICATION_CREDENTIALS = ścieżka do pliku JSON.
     Z klucza budowany jest JWT RS256 (podpis przez openssl), wymieniany
     na access token pod https://oauth2.googleapis.com/token.

UŻYCIE
  jako CLI:     python3 scripts/gcp_secrets.py NAZWA [--project PROJEKT]
  jako moduł:   from gcp_secrets import get_secret
                token = get_secret("edgelab-telegram-bot-token")

Projekt: --project / GCP_PROJECT / pole project_id z klucza SA — pierwszy
niepusty wygrywa. Sekret NIGDY nie jest logowany; CLI wypisuje go na stdout
(po to jest), diagnostyka idzie na stderr.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_TOKEN_CACHE: dict[str, tuple[str, float]] = {}


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _sign_rs256(signing_input: bytes, private_key_pem: str) -> bytes:
    """Podpis SHA256/RSA przez binarkę openssl (bez bibliotek crypto)."""
    with tempfile.NamedTemporaryFile("w", suffix=".pem", delete=False) as f:
        key_path = f.name
        os.chmod(key_path, 0o600)
        f.write(private_key_pem)
    try:
        out = subprocess.run(
            ["openssl", "dgst", "-sha256", "-sign", key_path],
            input=signing_input,
            capture_output=True,
            timeout=30,
            check=True,
        )
        return out.stdout
    finally:
        os.unlink(key_path)


def _load_sa_key() -> dict | None:
    raw = os.environ.get("GCP_SA_KEY", "").strip()
    if not raw:
        path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        if path and Path(path).exists():
            raw = Path(path).read_text()
    if not raw:
        return None
    try:
        key = json.loads(raw)
    except json.JSONDecodeError:
        print("[gcp_secrets] GCP_SA_KEY nie jest poprawnym JSON-em", file=sys.stderr)
        return None
    if key.get("type") != "service_account":
        print("[gcp_secrets] klucz nie jest kluczem konta serwisowego", file=sys.stderr)
        return None
    return key


def _token_from_sa(key: dict) -> str:
    """JWT RS256 podpisany kluczem SA → access token (cache do wygaśnięcia)."""
    email = key["client_email"]
    cached = _TOKEN_CACHE.get(email)
    if cached and cached[1] > time.time() + 60:
        return cached[0]

    now = int(time.time())
    header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    claims = _b64url(
        json.dumps(
            {
                "iss": email,
                "scope": "https://www.googleapis.com/auth/cloud-platform",
                "aud": "https://oauth2.googleapis.com/token",
                "iat": now,
                "exp": now + 3600,
            }
        ).encode()
    )
    signing_input = f"{header}.{claims}".encode()
    jwt = f"{header}.{claims}.{_b64url(_sign_rs256(signing_input, key['private_key']))}"

    body = urllib.parse.urlencode(
        {
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": jwt,
        }
    ).encode()
    req = urllib.request.Request("https://oauth2.googleapis.com/token", data=body)
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read().decode())
    token = resp["access_token"]
    _TOKEN_CACHE[email] = (token, time.time() + int(resp.get("expires_in", 3600)))
    return token


def _gcloud(args: list[str]) -> str:
    try:
        out = subprocess.run(
            ["gcloud", *args], capture_output=True, text=True, timeout=60, check=False
        )
    except FileNotFoundError:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _access_token() -> tuple[str, str]:
    """Zwraca (token, źródło). Pusty token = brak uwierzytelnienia."""
    tok = os.environ.get("GCP_ACCESS_TOKEN", "").strip()
    if tok:
        return tok, "GCP_ACCESS_TOKEN"
    key = _load_sa_key()
    if key is not None:
        return _token_from_sa(key), f"konto serwisowe {key['client_email']}"
    tok = _gcloud(["auth", "print-access-token"])
    if tok:
        return tok, "gcloud"
    return "", ""


def _project(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("GCP_PROJECT", "").strip()
    if env:
        return env
    key = _load_sa_key()
    if key and key.get("project_id"):
        return key["project_id"]
    return _gcloud(["config", "get-value", "project"])


def get_secret(name: str, project: str | None = None, version: str = "latest") -> str:
    """Czyta sekret z GCP Secret Managera. Rzuca RuntimeError z powodem."""
    token, source = _access_token()
    if not token:
        raise RuntimeError(
            "brak uwierzytelnienia do GCP: ustaw GCP_SA_KEY (chmura) albo "
            "zaloguj gcloud (`gcloud auth login`, Mac); patrz docs/SECRETS.md"
        )
    proj = _project(project)
    if not proj:
        raise RuntimeError("brak projektu GCP: podaj --project albo ustaw GCP_PROJECT")
    url = (
        f"https://secretmanager.googleapis.com/v1/projects/{proj}"
        f"/secrets/{name}/versions/{version}:access"
    )
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:300]
        raise RuntimeError(
            f"Secret Manager odmówił ({e.code}) dla '{name}' w projekcie '{proj}' "
            f"[auth: {source}]: {detail}"
        ) from e
    return base64.b64decode(payload["payload"]["data"]).decode()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("name", help="nazwa sekretu, np. edgelab-telegram-bot-token")
    p.add_argument("--project", default=None)
    p.add_argument("--version", default="latest")
    a = p.parse_args()
    try:
        sys.stdout.write(get_secret(a.name, a.project, a.version))
    except RuntimeError as e:
        print(f"[gcp_secrets] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
