#!/usr/bin/env bash
# Nocna sesja edge-llm-lab: kolejka eksperymentów → analiza → raport na Telegrama.
#
#   ./scripts/night_session.sh
#
# Podział ról jest celowy i wynika z RUNBOOK-a:
#   1. POMIAR biegnie deterministycznie (scripts/run_experiments.py). Żadnego
#      LLM-a w pętli pomiarowej — liczby do pracy naukowej nie mogą być
#      mediowane przez model, który improwizuje.
#   2. ANALIZA idzie przez `claude -p` — tu model faktycznie pomaga: czyta
#      tidy.csv, liczy statystyki opisowe, pisze raport.
#   3. RAPORT na Telegrama przez utils/notify.py (sekrety Edge_lab_fed_bot
#      i EDGE_LAB_CHAT_ID z GCP Secret Managera).
#
# Mac nie uśnie w trakcie — `caffeinate -i` trzyma go wybudzonym na czas runu.
set -u

cd "$(dirname "$0")/.."
mkdir -p experiments/results
LOG="experiments/results/night-$(date -u +%Y%m%d-%H%M%S).log"

say() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$LOG"; }

# Powiadomienie nigdy nie może zabić sesji — stąd `|| true`.
notify() {
  PYTHONPATH=src python3 -m edge_llm_lab.utils.notify "$1" >>"$LOG" 2>&1 || true
}

say "START"
git pull --rebase >>"$LOG" 2>&1 || say "git pull nieudany — lecę na tym, co lokalnie"

QUEUED=$(ls experiments/queue/*.yaml 2>/dev/null | wc -l | tr -d ' ')
if [ "$QUEUED" -eq 0 ]; then
  say "Kolejka pusta — nie ma czego uruchomić."
  notify "🌙 edge-llm-lab: kolejka pusta, sesja nocna nie ma czego uruchomić."
  exit 0
fi

say "W kolejce: $QUEUED eksperymentów"
notify "🌙 edge-llm-lab — sesja nocna START, $QUEUED eksperymentów w kolejce."

# ── 1. Pomiar (deterministyczny) ────────────────────────────────────────────
BEFORE=$(ls experiments/results 2>/dev/null | wc -l | tr -d ' ')
caffeinate -i python3 scripts/run_experiments.py >>"$LOG" 2>&1
RC=$?
AFTER=$(ls experiments/results 2>/dev/null | wc -l | tr -d ' ')
say "runner zakończył rc=$RC (nowych katalogów wyników: $((AFTER - BEFORE)))"

if [ "$RC" -ne 0 ]; then
  notify "⚠️ edge-llm-lab: runner zakończył się kodem $RC. Log: $LOG"
fi

# ── 2. Analiza (tu LLM ma sens) ─────────────────────────────────────────────
if command -v claude >/dev/null 2>&1; then
  say "analiza przez claude -p"
  printf 'Jesteś analitykiem w edge-llm-lab. Nocny runner właśnie skończył
kolejkę eksperymentów (log: %s, kod wyjścia %s).

Zrób dokładnie to:
1. Znajdź NOWE katalogi w experiments/results/ (te z dzisiejszą datą UTC).
2. Dla każdego uruchom: python3 scripts/g1_summarize.py <katalog>
3. Napisz experiments/results/NIGHT-REPORT-%s.md wzorowany na istniejącym
   NIGHT-REPORT.md — ta sama dyscyplina: liczby opisowe, ŻADNYCH wniosków
   o G1/G2, jawnie oznaczone artefakty i konfaundy.
4. KLUCZOWE dla 004: enum "INCOMPLETE"/"incomplete" został naprawiony
   (commit b64b4ed). Poprzedni pomiar validity (1%% unconstrained) mierzył
   ten błąd assetu, nie model. Porównaj nową validity unconstrained ze starą
   i napisz WPROST, ile z tamtych 99 porażek to był asset, a ile model.
   Wyniki po naprawie NIE są porównywalne z 002/003 w kolumnie validity —
   napisz to wyraźnie.
5. Dla 005 sprawdź, czy któryś model zszedł poniżej 36.8 ms/token. Jeśli nie,
   G1 dalej NIE został wystawiony na próbę — napisz to zamiast udawać wynik.
6. Dla 006 sprawdź, czy qwen3 przestał się ucinać przy 3072 tokenach.
7. git add experiments/ && git commit -m "night: analiza %s" && git push
8. Na koniec wyślij podsumowanie (5-8 linii, po polsku, konkrety i liczby):
   PYTHONPATH=src python3 -m edge_llm_lab.utils.notify "<treść>"

Zasada: fakty z danych, zero zmyślania. Jeśli czegoś nie da się policzyć,
napisz że się nie da i dlaczego.' \
    "$LOG" "$RC" "$(date -u +%Y%m%d)" "$(date -u +%Y-%m-%d)" \
  | claude -p --permission-mode acceptEdits \
      --allowedTools "Read,Write,Edit,Glob,Grep,Bash(python3:*),Bash(git:*)" \
      >>"$LOG" 2>&1 || say "claude -p zakończył się błędem — wyniki i tak są zapisane"
else
  say "brak binarki claude — pomijam analizę, pomiary są zapisane"
  notify "✅ edge-llm-lab: pomiary gotowe ($((AFTER - BEFORE)) nowych katalogów). Brak claude na PATH, analizy nie zrobiłam."
fi

say "DONE"
