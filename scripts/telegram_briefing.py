#!/usr/bin/env python3
"""Wysyła briefing na Telegrama: tekst + wiadomość głosowa.

DLACZEGO TO ISTNIEJE: Routine `paper-machine` chodzi w chmurze, a tam
`api.telegram.org` jest zablokowane przez politykę sieci (403 na CONNECT).
Więc chmura *przygotowuje* briefing w repo, a wysyła go maszyna, która ma
sieć i token — czyli Mac.

UŻYCIE (na Macu):
    export TELEGRAM_BOT_TOKEN=...        # albo ~/.secrets/telegram.env,
    export TELEGRAM_CHAT_ID=...          # albo GCP Secret Manager (niżej)

    # z GCP Secret Managera (wymaga `gcloud auth login` zrobionego raz):
    export GCP_PROJECT=twoj-projekt
    export GCP_SECRET_PREFIX=edgelab     # sekrety: edgelab-telegram-bot-token,
                                         #          edgelab-telegram-chat-id
    python3 scripts/telegram_briefing.py \
        --spoken docs/research/BRIEFING-2026-08-11.md \
        --report docs/reports/2026-08-11-go-no-go-G1.md

    # tylko sprawdzenie, bez wysyłki:
    python3 scripts/telegram_briefing.py --spoken ... --dry-run

CO ROBI
  1. czyta skrypt mówiony (markdown; komentarze HTML i nagłówki są wycinane),
  2. syntezuje mowę:
       - macOS `say -v Zosia` (domyślnie, zero zależności i zero kosztu), albo
       - ElevenLabs, jeśli ustawiony ELEVENLABS_API_KEY i --tts elevenlabs,
  3. konwertuje do OGG/Opus, jeśli jest ffmpeg → wychodzi prawdziwa
     wiadomość głosowa (sendVoice). Bez ffmpeg wysyła m4a przez sendAudio,
  4. wysyła podsumowanie tekstowe (sendMessage) i plik audio.

Nic nie usuwa i nie nadpisuje; pliki tymczasowe lądują w katalogu tymczasowym.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.telegram.org"
TG_LIMIT = 4096


def _gcp_secret(name: str) -> str:
    """Czyta sekret z GCP SM przez scripts/gcp_secrets.py (gcloud na Macu,
    klucz konta serwisowego w chmurze). Cicho degraduje się do ""."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from gcp_secrets import get_secret

        return get_secret(name).strip()
    except Exception as e:  # noqa: BLE001 — każde niepowodzenie = brak źródła
        print(f"[i] GCP SM niedostępny ({e})", file=sys.stderr)
        return ""


def load_token_and_chat() -> tuple[str, str]:
    """Kolejność źródeł: zmienne środowiskowe → ~/.secrets/telegram.env →
    GCP Secret Manager (`<prefix>-telegram-bot-token` / `-chat-id`)."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    env_file = Path.home() / ".secrets" / "telegram.env"
    if (not token or not chat) and env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if k.strip() == "TELEGRAM_BOT_TOKEN" and not token:
                token = v
            elif k.strip() == "TELEGRAM_CHAT_ID" and not chat:
                chat = v

    if not token or not chat:
        # Nazwy sekretów: jawnie przez TELEGRAM_BOT_TOKEN_SECRET /
        # TELEGRAM_CHAT_ID_SECRET (np. Edge_lab_fed_bot), albo konwencja
        # <GCP_SECRET_PREFIX>-telegram-bot-token / -chat-id.
        prefix = os.environ.get("GCP_SECRET_PREFIX", "edgelab").strip()
        tok_name = os.environ.get(
            "TELEGRAM_BOT_TOKEN_SECRET", f"{prefix}-telegram-bot-token"
        )
        chat_name = os.environ.get(
            "TELEGRAM_CHAT_ID_SECRET", f"{prefix}-telegram-chat-id"
        )
        token = token or _gcp_secret(tok_name)
        chat = chat or _gcp_secret(chat_name)

    if not token or not chat:
        sys.exit(
            "Brak TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID. Źródła sprawdzane po kolei:\n"
            "  1. zmienne środowiskowe TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,\n"
            "  2. ~/.secrets/telegram.env,\n"
            "  3. GCP Secret Manager: sekrety <GCP_SECRET_PREFIX>-telegram-bot-token\n"
            "     i -telegram-chat-id (wymaga gcloud + `gcloud auth login`;\n"
            "     projekt z GCP_PROJECT albo domyślny gcloud).\n"
            "Instrukcja konfiguracji: docs/SECRETS.md"
        )
    return token, chat


def spoken_text(path: Path) -> str:
    """Markdown → czysty tekst do przeczytania na głos."""
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL)  # komentarze HTML
    out = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith(("#", "|", "```")):
            continue
        s = re.sub(r"[*_`]", "", s)  # emfaza
        s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)  # linki
        out.append(s)
    return "\n".join(out).strip()


def synth_say(text: str, out_m4a: Path, voice: str) -> None:
    if sys.platform != "darwin":
        sys.exit(
            f"`say` jest tylko na macOS (ten host: {sys.platform}). "
            "Uruchom skrypt na Macu albo użyj --tts elevenlabs."
        )
    subprocess.run(["say", "-v", voice, "-o", str(out_m4a), text], check=True)


def synth_elevenlabs(text: str, out_mp3: Path, voice_id: str) -> None:
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        sys.exit("--tts elevenlabs wymaga ELEVENLABS_API_KEY.")
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        data=json.dumps({"text": text, "model_id": "eleven_multilingual_v2"}).encode(),
        headers={"xi-api-key": key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        out_mp3.write_bytes(r.read())


def to_opus(src: Path, dst: Path) -> bool:
    """m4a/mp3 → ogg/opus. True, jeśli się udało (jest ffmpeg)."""
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-c:a",
                "libopus",
                "-b:a",
                "32k",
                "-ar",
                "48000",
                "-ac",
                "1",
                str(dst),
            ],
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def post(
    token: str,
    method: str,
    fields: dict,
    file_field: str | None = None,
    file_path: Path | None = None,
) -> dict:
    url = f"{API}/bot{token}/{method}"
    if file_path is None:
        data = urllib.parse.urlencode(fields).encode()
        req = urllib.request.Request(url, data=data)
    else:
        boundary = "----briefing-boundary-7f3a"
        body = bytearray()
        for k, v in fields.items():
            body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        body += (
            f"--{boundary}\r\nContent-Disposition: form-data; "
            f'name="{file_field}"; filename="{file_path.name}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        body += file_path.read_bytes() + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        req = urllib.request.Request(url, data=bytes(body))
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())


def chunks(text: str, size: int = TG_LIMIT):
    while text:
        if len(text) <= size:
            yield text
            return
        cut = text.rfind("\n", 0, size)
        cut = cut if cut > size // 2 else size
        yield text[:cut]
        text = text[cut:].lstrip("\n")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--spoken", required=True, type=Path, help="skrypt mówiony (markdown)"
    )
    p.add_argument(
        "--report", type=Path, help="raport tekstowy do wysłania jako wiadomość"
    )
    p.add_argument("--tts", choices=["say", "elevenlabs", "none"], default="say")
    p.add_argument(
        "--voice", default="Zosia", help="głos `say` (macOS), domyślnie Zosia (pl)"
    )
    p.add_argument("--elevenlabs-voice-id", default="")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="nic nie wysyła; pokazuje, co poszłoby i ile ma znaków",
    )
    a = p.parse_args()

    if not a.spoken.exists():
        sys.exit(f"Nie ma pliku: {a.spoken}")
    text = spoken_text(a.spoken)
    words = len(text.split())
    print(
        f"[i] skrypt mówiony: {words} słów (~{words / 150:.1f} min mowy), {len(text)} znaków"
    )

    body = (
        a.report.read_text(encoding="utf-8") if a.report and a.report.exists() else text
    )
    parts = list(chunks(body))
    print(f"[i] wiadomość tekstowa: {len(body)} znaków → {len(parts)} wiadomość(i)")

    if a.dry_run:
        print(
            "[i] --dry-run: nie wysyłam, nie syntezuję. Podgląd pierwszych 300 znaków mowy:"
        )
        print("    " + text[:300].replace("\n", "\n    "))
        return 0

    token, chat = load_token_and_chat()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        audio = None
        if a.tts == "say":
            audio = td / "briefing.m4a"
            synth_say(text, audio, a.voice)
        elif a.tts == "elevenlabs":
            if not a.elevenlabs_voice_id:
                sys.exit("--tts elevenlabs wymaga --elevenlabs-voice-id")
            audio = td / "briefing.mp3"
            synth_elevenlabs(text, audio, a.elevenlabs_voice_id)

        for i, part in enumerate(parts, 1):
            r = post(
                token,
                "sendMessage",
                {"chat_id": chat, "text": part, "disable_web_page_preview": "true"},
            )
            print(f"[+] tekst {i}/{len(parts)}: ok={r.get('ok')}")

        if audio is not None:
            ogg = td / "briefing.ogg"
            if to_opus(audio, ogg):
                r = post(token, "sendVoice", {"chat_id": chat}, "voice", ogg)
                print(f"[+] wiadomość głosowa (ogg/opus): ok={r.get('ok')}")
            else:
                print(
                    "[!] brak ffmpeg — wysyłam jako audio zamiast wiadomości głosowej"
                )
                r = post(
                    token,
                    "sendAudio",
                    {"chat_id": chat, "title": "Briefing edge-llm-lab"},
                    "audio",
                    audio,
                )
                print(f"[+] audio: ok={r.get('ok')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
