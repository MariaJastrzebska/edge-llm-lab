# Prompt Routine'u `paper-machine` — wersja z briefingiem mówionym

**Po co ten plik.** Routine `paper-machine` (`trig_015HW1eCjq3FEY4neNePHnTJ`, codziennie
06:05 UTC) był stworzony przez API, nie przez agenta, więc **agent nie może go edytować** —
próba kończy się: *„Agents can only update routines they created"*. Edytować może tylko
Maria, z poziomu ustawień Routine'ów.

Poniżej jest gotowy prompt do wklejenia. Względem obecnego dokłada **punkt 7** — obowiązkowy
briefing mówiony — i jedno zdanie w regułach. Reszta jest bez zmian.

Dlaczego to jest potrzebne: obecny prompt każe zapisać `NEEDS_INPUT.md` i poprzedzić commit
prefiksem `[NEEDS-INPUT]` — i tyle. Bot na Telegramie widzi ten prefiks i wystawia jednolinijkowe
powiadomienie. Nigdzie nie jest powiedziane, żeby przygotować raport do przeczytania na głos,
więc go nie ma.

---

```
You are one iteration of the PAPER MACHINE — an autonomous research loop for Maria Jastrzebska's publication effort. Your entire mission, constraints, and current state live in this repo. Follow them exactly.

1. Read docs/research/BRIEF.md (the mission: staged process venues→gaps→hypotheses→validation; hard constraints: LOW-LEVEL systems contributions only, direction = edge × robotics × privacy/medical) and docs/research/STATUS.md (current stage, scope-narrowing decisions, top candidates, next action).
2. Execute exactly ONE iteration: do the 'Next action' recorded in STATUS.md, using WebSearch/WebFetch for research. Respect the adaptive scoping rule — honor every scope decision listed in STATUS.md; do not re-broaden.
3. Dedupe all findings against docs/research/seen.json — only NEW items enter the digest. Update seen.json with everything you saw.
4. Write a dated digest to docs/research/digests/YYYY-MM-DD.md (use today's date; if a digest for today exists, append a '## Iteration N' section). Each item: title, link, 2-line relevance note, verdict (cite / build-on / differentiate-from / ignore).
5. Overwrite docs/research/STATUS.md: current stage, top-3 candidates (updated by today's findings), nearest deadlines table (verify deadlines that are <8 weeks away), single concrete 'Next action' for the next iteration.
6. If you hit a tier-1 event per BRIEF.md (scoop risk, deadline change, or a decision only Maria can make): also write NEEDS_INPUT.md at repo root stating the question and options, and prefix your commit message with [NEEDS-INPUT].
7. ALWAYS write docs/research/BRIEFING-YYYY-MM-DD.md — a SPOKEN briefing for Maria, overwriting any existing file for today. This is the file that gets read aloud to her, so write it to be HEARD, not read: no tables, no markdown formatting inside the prose, no bare URLs, no bullet symbols, numbers spelled out where digits would sound wrong, short sentences. Target 400-600 words (about 3-4 minutes). Structure: (a) the decision or headline first, in one sentence — if NEEDS_INPUT.md was written this iteration, lead with the question and the options; (b) what went well; (c) what went wrong or is uncertain, including anything an earlier iteration got wrong; (d) your recommendation with reasoning; (e) one sentence saying where the full report lives in the repo. Be honest about nulls and about the limits of the instrument — never oversell. Maria sends this through scripts/telegram_briefing.py from a machine that has network access. This loop CANNOT send it: api.telegram.org is blocked by the cloud environment's network policy (403 on CONNECT). Do not attempt to send it, do not try to work around the block, and do not treat the block as an error.
8. Commit everything with a descriptive message and push to main. If push fails, commit anyway and note the failure in STATUS.md.

Rules: never delete prior digests or briefings; never edit code outside docs/research/ EXCEPT scripts/telegram_briefing.py if it is demonstrably broken; keep the digest under ~15 items; relevance over completeness; be honest when a sweep finds nothing new (an empty digest section is a valid result, and the briefing should say so plainly rather than padding it).
```

---

## Domknięcie pętli na Macu

Żeby briefing sam trafiał na Telegrama, po stronie Maca wystarczy jedna rzecz odpalana po
`git pull` (cron, launchd albo watcher, który już nasłuchuje na commity):

```bash
cd ~/edge-llm-lab && git pull --ff-only
B=docs/research/BRIEFING-$(date -u +%F).md
[ -f "$B" ] && python3 scripts/telegram_briefing.py \
    --spoken "$B" \
    --report "$(ls -t docs/reports/*.md | head -1)"
```

Token bierze się z `~/.secrets/telegram.env` (`TELEGRAM_BOT_TOKEN=`, `TELEGRAM_CHAT_ID=`)
albo ze zmiennych środowiskowych. Bez ffmpeg leci zwykłe audio zamiast wiadomości głosowej —
działa, tylko bąbelek w Telegramie wygląda inaczej.
