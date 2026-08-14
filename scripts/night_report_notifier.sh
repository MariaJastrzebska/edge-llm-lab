#!/usr/bin/env bash
# night_report_notifier.sh — the durable "report after every night" watcher.
#
# Runs on the Linux box (system cron), NOT the Mac: the Mac is only a compute
# node and lacks gcloud/edge-tts/secrets, so it cannot send. This box has all
# three. It watches origin/main for a new NIGHT-REPORT-*.md and, when one
# lands, sends the FULL report file plus a detailed spoken narration of the
# whole report (Maria listens instead of reading) via telegram_report.sh.
#
# Independent of whatever the Mac does — the Mac just commits results; this
# box notices and reports. State: /root/.secrets/notifier-state/night-report.last
set -u

REPO=/root/repos/edge-llm-lab
STATE_DIR=/root/.secrets/notifier-state
STATE="$STATE_DIR/night-report.last"
mkdir -p "$STATE_DIR"

cd "$REPO" || exit 1
git fetch -q origin main 2>/dev/null || exit 0
git rev-parse --verify -q origin/main >/dev/null || exit 0

# Newest night report on origin/main (lexical sort → newest date last).
newest=$(git ls-tree -r --name-only origin/main experiments/results/ 2>/dev/null \
  | grep -E 'NIGHT-REPORT-[0-9]{4}-[0-9]{2}-[0-9]{2}\.md$' | sort | tail -1)
[ -z "$newest" ] && exit 0

last=""; [ -f "$STATE" ] && last=$(cat "$STATE")
if [ -z "$last" ]; then          # first run: baseline only, no backfill spam
  echo "$newest" > "$STATE"; exit 0
fi
[ "$newest" = "$last" ] && exit 0   # nothing new

git pull -q --ff-only origin main 2>/dev/null || git pull -q --rebase origin main 2>/dev/null
[ -f "$newest" ] || exit 0

# Build a long, listenable narration of the WHOLE report: keep every prose
# line, turn headers into spoken breaks, DROP raw numeric tables (they read
# terribly aloud — the prose beside them says the same in words). No length
# cap; edge-tts handles multi-minute narration.
narration=$(python3 - "$newest" <<'PYEOF'
import re, sys
out = []
for raw in open(sys.argv[1], encoding="utf-8", errors="ignore"):
    s = raw.strip()
    if not s or s.startswith("|"): continue
    if re.fullmatch(r"[-=_*]{3,}", s): continue
    h = re.match(r"^#{1,6}\s*(.+)$", s)
    out.append(". " + h.group(1).rstrip(".") + "." if h else s)
t = " ".join(out)
t = re.sub(r"`([^`]*)`", r"\1", t)
t = re.sub(r"https?://\S+", "", t)
t = re.sub(r"[*_>#\[\]]+", "", t)
print(re.sub(r"\s+", " ", t).strip())
PYEOF
)
voice="Report from the overnight lab session. $narration"
caption="🌙 Nowy raport nocny edge-llm-lab: $(basename "$newest"). Pełny raport w załączniku; głosówka czyta go w całości."

bash "$REPO/scripts/telegram_report.sh" -f "$newest" -v "$voice" "$caption" \
  && echo "$newest" > "$STATE"
