#!/bin/bash
# telegram_report.sh — wysyłka raportów tego repo na Telegram Marii.
# Bot: Edge_lab_fed_bot → chat USER_ID (Maria)
# Sekrety: GCP Secret Manager, projekt llm-apps-425420.
#
# Kolejność pobierania sekretów:
#   1. zmienne środowiskowe TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
#   2. plik ~/.telegram.env (linie NAZWA=wartość, nazwy jak w Secret Managerze)
#   3. gcloud secrets versions access (wymaga zalogowanego gcloud)
#
# Użycie:
#   scripts/telegram_report.sh "tekst wiadomości"
#   scripts/telegram_report.sh -f raport.md [-f kolejny.md] ["podpis"]
set -uo pipefail

BOT_SECRET="Edge_lab_fed_bot"
CHAT_SECRET="USER_ID"
TTS_VOICE="${TTS_VOICE:-pl-PL-ZofiaNeural}"
GCP_PROJECT="llm-apps-425420"

GCLOUD="${GCLOUD:-$(command -v gcloud || echo /usr/local/google-cloud-sdk/bin/gcloud)}"
ENV_FILE="${TELEGRAM_ENV_FILE:-$HOME/.telegram.env}"

get_secret() { # nazwa-sekretu
  local name="$1" val=""
  if [ -f "$ENV_FILE" ]; then
    val=$(grep -m1 "^${name}=" "$ENV_FILE" | cut -d= -f2-)
  fi
  if [ -z "$val" ] && [ -x "$GCLOUD" ]; then
    val=$("$GCLOUD" secrets versions access latest --secret="$name" \
      --project="$GCP_PROJECT" 2>/dev/null)
  fi
  printf '%s' "$val"
}

TOKEN="${TELEGRAM_BOT_TOKEN:-$(get_secret "$BOT_SECRET")}"
CHAT="${TELEGRAM_CHAT_ID:-$(get_secret "$CHAT_SECRET")}"

if [ -z "$TOKEN" ] || [ -z "$CHAT" ]; then
  echo "Brak tokena lub chat ID ($BOT_SECRET / $CHAT_SECRET)." >&2
  echo "Zaloguj gcloud albo uzupełnij $ENV_FILE." >&2
  exit 1
fi

API="https://api.telegram.org/bot${TOKEN}"
TEXT=""
VOICE_TEXT=""
FILES=()
while [ $# -gt 0 ]; do
  case "$1" in
    -f) FILES+=("$2"); shift 2 ;;
    -v) VOICE_TEXT="$2"; shift 2 ;;
    *)  TEXT="$1"; shift ;;
  esac
done

fail=0
if [ -n "$TEXT" ]; then
  r=$(curl -sS -X POST "$API/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode text="$TEXT")
  echo "$r" | grep -q '"ok":true' && echo "OK  tekst" \
    || { echo "FAIL tekst: $r"; fail=1; }
fi
for f in ${FILES[@]+"${FILES[@]}"}; do
  # Podgląd Telegrama zgaduje kodowanie i psuje polskie znaki w plikach
  # tekstowych — BOM UTF-8 wymusza poprawną dekodację. Wysyłamy kopię z BOM
  # pod oryginalną nazwą pliku.
  sendf="$f"; tmpd=""
  case "$f" in *.md|*.txt)
    if ! head -c 3 "$f" | grep -q "$(printf '\xef\xbb\xbf')"; then
      tmpd=$(mktemp)
      { printf '\xef\xbb\xbf'; cat "$f"; } > "$tmpd"
      sendf="$tmpd"
    fi ;;
  esac
  r=$(curl -sS -X POST "$API/sendDocument" \
    -F chat_id="$CHAT" -F "document=@${sendf};filename=$(basename "$f")")
  [ -n "$tmpd" ] && rm -f "$tmpd"
  echo "$r" | grep -q '"ok":true' && echo "OK  $f" \
    || { echo "FAIL $f: $r"; fail=1; }
done

# ── Głosówka: -v "tekst" → edge-tts (darmowe) → Telegram sendVoice ────────
# Celowo NIE ElevenLabs — ten jest zarezerwowany do generacji scenariuszy
# w social-media; raporty idą darmowym edge-tts (pip install edge-tts).
if [ -n "$VOICE_TEXT" ]; then
  if ! command -v edge-tts >/dev/null 2>&1; then
    echo "FAIL głosówka: brak edge-tts (zainstaluj: pip install edge-tts)"; fail=1
  else
    tmpv=$(mktemp /tmp/tg_voice_XXXXXX.mp3)
    if edge-tts --voice "$TTS_VOICE" --text "$VOICE_TEXT" --write-media "$tmpv" \
        && [ -s "$tmpv" ]; then
      r=$(curl -sS -X POST "$API/sendVoice" -F chat_id="$CHAT" -F voice=@"$tmpv")
      echo "$r" | grep -q '"ok":true' && echo "OK  głosówka" \
        || { echo "FAIL głosówka: $r"; fail=1; }
    else
      echo "FAIL głosówka: edge-tts nie wygenerował audio"; fail=1
    fi
    rm -f "$tmpv"
  fi
fi
exit $fail
