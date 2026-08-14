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
  # Podgląd Telegrama to zwykły tekst: BOM UTF-8 wymusza poprawne polskie
  # znaki, tabele markdown zamieniamy na wyrównane kolumny, a surowe **
  # zdejmujemy — inaczej pliku nie da się czytać na telefonie.
  sendf="$f"; tmpd=""
  case "$f" in *.md|*.txt)
    tmpd=$(mktemp)
    python3 - "$f" > "$tmpd" <<'PYEOF'
import re
import sys

text = open(sys.argv[1], encoding="utf-8-sig").read()
# ** musi zniknąć PRZED wyrównaniem tabel — inaczej komórki z pogrubieniem
# liczą 4 znaki za dużo i kolumny schodkują.
text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.S)
lines = text.split("\n")
out, buf = [], []

def flush():
    rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in buf]
    rows = [r for r in rows if not all(re.fullmatch(r"[-: ]*", c) for c in r)]
    if rows:
        ncol = max(len(r) for r in rows)
        w = [max(len(r[i]) if i < len(r) else 0 for r in rows) for i in range(ncol)]
        for r in rows:
            out.append("  ".join((r[i] if i < len(r) else "").ljust(w[i])
                                 for i in range(ncol)).rstrip())
    buf.clear()

for ln in lines:
    s = ln.strip()
    if s.startswith("|") and s.endswith("|") and s.count("|") >= 2:
        buf.append(ln)
    else:
        flush()
        out.append(ln)
flush()
sys.stdout.write("\ufeff" + "\n".join(out))
PYEOF
    sendf="$tmpd" ;;
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
    # Auto-pick voice by language: Polish diacritics => PL voice, else EN.
    vvoice="$TTS_VOICE"
    if ! printf '%s' "$VOICE_TEXT" | grep -q '[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]'; then
      vvoice="${TTS_VOICE_EN:-en-US-JennyNeural}"
    fi
    # Split long narration into <=2400-char segments at sentence/space
    # boundaries and send each as its own voice note IN ORDER — the whole
    # message is heard, never cut off mid-sentence.
    segdir=$(mktemp -d)
    python3 - "$VOICE_TEXT" "$segdir" <<'PYSEG'
import re, sys
text, outdir = sys.argv[1], sys.argv[2]
limit = 2400
segs, cur = [], ""
for s in re.split(r"(?<=[.!?…])\s+", text):
    while len(s) > limit:
        cut = s[:limit].rsplit(" ", 1)[0] or s[:limit]
        segs.append(cut); s = s[len(cut):].lstrip()
    if len(cur) + len(s) + 1 > limit:
        if cur: segs.append(cur)
        cur = s
    else:
        cur = (cur + " " + s).strip()
if cur: segs.append(cur)
import os
for i, seg in enumerate(segs):
    open(os.path.join(outdir, f"{i:03d}.txt"), "w", encoding="utf-8").write(seg)
PYSEG
    vok=1
    for segfile in "$segdir"/*.txt; do
      [ -f "$segfile" ] || continue
      tmpv=$(mktemp /tmp/tg_voice_XXXXXX.mp3)
      if edge-tts --voice "$vvoice" --file "$segfile" --write-media "$tmpv" \
          && [ -s "$tmpv" ]; then
        r=$(curl -sS -X POST "$API/sendVoice" -F chat_id="$CHAT" -F voice=@"$tmpv")
        echo "$r" | grep -q '"ok":true' || { echo "FAIL głosówka: $r"; vok=0; fail=1; }
      else
        echo "FAIL głosówka: edge-tts nie wygenerował audio ($segfile)"; vok=0; fail=1
      fi
      rm -f "$tmpv"
    done
    rm -rf "$segdir"
    [ "$vok" = 1 ] && echo "OK  głosówka"
  fi
fi
exit $fail
