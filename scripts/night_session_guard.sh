#!/usr/bin/env bash
# night_session_guard.sh — make the nightly experiment run reliable, and ping
# Maria when it can't happen. The Mac is a Remote-Control peer that only runs
# when connected; this Linux box cannot reach it directly, so it manages the
# HUMAN side: remind to connect in the evening, and if experiments plainly did
# not run by morning, say so instead of staying silent.
#
#   night_session_guard.sh evening   — remind to connect the Mac (if queue non-empty)
#   night_session_guard.sh morning   — if queue non-empty AND nothing ran overnight, ping
#
# "Ran overnight" = a results dir or NIGHT-REPORT dated today exists on origin/main.
set -u

REPO=/root/repos/edge-llm-lab
SECRETS=/root/.secrets/telegram-bots.json
mode="${1:-}"

ping() { # message
  local token chat
  token=$(python3 -c "import json;print(json.load(open('$SECRETS'))['edge-llm-lab']['bot_token'])")
  chat=$(python3 -c "import json;print(json.load(open('$SECRETS'))['edge-llm-lab']['chat_id'])")
  curl -s -X POST "https://api.telegram.org/bot${token}/sendMessage" \
    -d chat_id="$chat" -d disable_web_page_preview=true \
    --data-urlencode text="$1" > /dev/null
}

cd "$REPO" || exit 1
git fetch -q origin main 2>/dev/null || exit 0

queue_n=$(git ls-tree -r --name-only origin/main experiments/queue/ 2>/dev/null \
  | grep -c '\.yaml$')
[ "${queue_n:-0}" -eq 0 ] && exit 0   # nothing queued → nothing to run, stay quiet

today=$(date +%Y-%m-%d)
ran_today=$(git ls-tree -r --name-only origin/main experiments/results/ 2>/dev/null \
  | grep -E "(${today}|NIGHT-REPORT-${today})" | head -1)

case "$mode" in
  evening)
    ping "🌙 Wieczorne przypomnienie: podłącz MacBooka na nocną sesję — w kolejce ${queue_n} eksperyment(y). Jak podłączysz, ruszą same. Bez tego dziś nic się nie policzy."
    ;;
  morning)
    if [ -z "$ran_today" ]; then
      ping "⚠️ Nocna sesja NIE ruszyła — brak wyników z dziś, a w kolejce ${queue_n} eksperyment(y). Najpewniej Mac nie był podłączony. Podłącz go, żeby policzyły się dziś."
    fi
    ;;
  *)
    echo "usage: $0 evening|morning" >&2; exit 2 ;;
esac
