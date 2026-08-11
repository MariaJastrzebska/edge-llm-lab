# Sekrety — GCP Secret Manager jako jedno źródło prawdy

**Zasada nadrzędna: żadna wartość sekretu nigdy nie trafia do gita ani do czatu.**
Ten plik dokumentuje wyłącznie NAZWY i procedurę.

## Dlaczego GCP Secret Manager

Maria ma kilka repo (edge-llm-lab, social-media, Bigos), kilka maszyn (Mac, sesje
w chmurze, w przyszłości Pi/Jetson) i rosnący zestaw sekretów (dwa boty Telegrama,
ElevenLabs, Apify…). Jedno centralne miejsce z wersjonowaniem i rotacją ma sens.
Sprawdzone 2026-08-11: `secretmanager.googleapis.com` jest osiągalne także
z chmurowego środowiska Claude (w przeciwieństwie do `api.telegram.org`).

## Konwencja nazw

| Sekret w GCP SM | Co to |
|---|---|
| `edgelab-telegram-bot-token` | token bota @Edge_lab_fed_bot |
| `edgelab-telegram-chat-id` | id czatu Marii z tym botem |
| `bigos-telegram-bot-token` | token bota @Bigos_lab_bot |
| `bigos-telegram-chat-id` | id czatu z tym botem |
| `elevenlabs-api-key` | klucz ElevenLabs (opcjonalny, do TTS) |

`scripts/telegram_briefing.py` czyta sekrety w kolejności: zmienne środowiskowe →
`~/.secrets/telegram.env` → GCP SM (`$GCP_SECRET_PREFIX-telegram-bot-token`,
domyślny prefiks `edgelab`; projekt z `$GCP_PROJECT` albo domyślny gcloud).

## Jednorazowa konfiguracja (na Macu, ~5 minut)

```bash
# 0. gcloud, jeśli brak:  brew install google-cloud-sdk
gcloud auth login
gcloud config set project TWOJ_PROJEKT          # albo: gcloud projects create ...
gcloud services enable secretmanager.googleapis.com

# 1. NAJPIERW zrewokuj oba tokeny u BotFathera (Revoke current token)
#    — dotychczasowe były pokazane na screenshotach, więc traktujemy je
#    jako spalone. Dopiero NOWE wartości wkładamy do SM.

# 2. Włóż sekrety (printf %s, żeby nie dokleić \n):
printf '%s' 'NOWY_TOKEN_EDGELAB'  | gcloud secrets create edgelab-telegram-bot-token --data-file=-
printf '%s' 'NOWY_TOKEN_BIGOS'    | gcloud secrets create bigos-telegram-bot-token   --data-file=-

# 3. chat_id (napisz najpierw /start do bota):
TOKEN=$(gcloud secrets versions access latest --secret=edgelab-telegram-bot-token)
curl -s "https://api.telegram.org/bot$TOKEN/getUpdates" | python3 -m json.tool | grep -m1 '"id"'
printf '%s' 'TWOJ_CHAT_ID' | gcloud secrets create edgelab-telegram-chat-id --data-file=-
# analogicznie dla bigos-telegram-chat-id
```

## Użycie

**Mac** — nic więcej nie trzeba; skrypt sam sięgnie po sekrety przez gcloud:
```bash
export GCP_PROJECT=TWOJ_PROJEKT   # raz, np. w ~/.zshrc
python3 scripts/telegram_briefing.py --spoken docs/research/BRIEFING-$(date +%F).md \
    --report "$(ls -t docs/reports/*.md | head -1)"
```

**Chmura (sesje Claude / Routine)** — dwa osobne warunki, oba po stronie Marii:
1. *uwierzytelnienie do GCP*: konto serwisowe z rolą `roles/secretmanager.secretAccessor`,
   a jego klucz JSON w zmiennych środowiskowych środowiska Claude
   (claude.ai/code → środowisko → environment variables). Uwaga na uczciwość tego
   układu: klucz SA to też sekret — środowisko Claude i tak przechowuje jeden
   sekret startowy; SM centralizuje całą resztę.
2. *wysyłka na Telegrama*: *network policy* środowiska musi dopuścić
   `api.telegram.org` — *Secret Manager tego nie obchodzi ani nie omija*;
   dziś proxy odrzuca tunel (403 na CONNECT) niezależnie od tokena.

## Rotacja

```bash
# BotFather → Revoke current token → potem:
printf '%s' 'SWIEZY_TOKEN' | gcloud secrets versions add edgelab-telegram-bot-token --data-file=-
```
Stare wersje zostają w historii SM (można je wyłączyć: `versions disable`).
