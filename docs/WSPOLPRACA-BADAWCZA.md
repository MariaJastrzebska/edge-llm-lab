# Protokół współpracy badawczej — człowiek w pętli, AI jako rozszerzenie

> Od Marii (19.08.2026): „mój osąd jest konieczny; nie uczę się, gdy wszystko
> robisz i każdą decyzję podejmujesz; skrót raportu nie wystarcza do decyzji —
> nie wiedziałam, skąd jest 3% ani co robi maska". Ten protokół to naprawia.

## Zasady twarde

1. **Żadna hipoteza nie jest wybierana po cichu.** Claude przedstawia
   przestrzeń hipotez (min. 2–3 opcje) z uzasadnieniem i rekomendacją;
   wybiera Maria. Nocne sesje realizują TYLKO hipotezy wybrane przez Marię
   w poniedziałkowym bloku (wyjątek: kontynuacja już zatwierdzonej linii).
2. **Każda liczba identyfikowalna.** Jeśli w raporcie pada „3%", obok jest
   ścieżka: 3% = 3 ucięte generacje / 100 przebiegów, warunek unconstrained,
   eksperyment 002, plik `.../repeat_XX`. Zero liczb-sierot.
3. **Warstwa ucząca w każdym raporcie** (sekcja obowiązkowa nocnego raportu):
   - „Pojęcia tej nocy" — 1–3 koncepty użyte w eksperymencie, wyjaśnione
     od zera (poziom: bystra osoba spoza tematu), z odwołaniem do MAPA-WIEDZY;
   - „Skąd są kluczowe liczby" — wyliczenia pokazane jawnie (dzielna/dzielnik);
   - „3 pytania do Twojego osądu" — decyzje, które NALEŻĄ do Marii, z tym,
     co trzeba rozumieć, żeby je podjąć.
4. **Mapa wiedzy jako program nauki.** `docs/MAPA-WIEDZY.md` — Maria raz
   zaznacza znam/mgliście/nie znam; lekcje (LEKCJA-XX) idą wyłącznie w luki,
   które dotykają bieżących eksperymentów. Jedna lekcja ≤ 10 min czytania.
5. **Wkład Marii jest nazwany, nie dorozumiany.** Obszary, gdzie jej research
   (ewaluacje, CoT-judges, serving) daje przewagę: projektowanie metryk
   validity/J-per-valid, mosty LLM↔VLM, sanity-checks metodologiczne.
   Brainstormy hipotez robimy w parze: Claude generuje szeroko, Maria tnie
   i dokłada z własnej wiedzy — na piśmie, w `docs/brainstorms/`.
6. **Test zrozumienia zamiast akceptu.** Przy decyzji Claude pyta „czy wiesz,
   skąd ta liczba / co robi ten mechanizm?" — jeśli nie, najpierw lekcja,
   potem decyzja. Decyzja bez zrozumienia się nie liczy.

## Rytm

- **Niedziela wieczór:** Claude pakuje poniedziałkową paczkę: wyniki tygodnia
  z warstwą uczącą + przestrzeń hipotez na kolejne noce + 1 lekcja do luki.
- **Poniedziałek 9–13 (blok Marii):** czyta, uczy się, wybiera hipotezy,
  dopisuje własne — to jest jej wkład badawczy, nie tylko akcept.
- **Noce (cron):** realizacja wybranych hipotez; raporty z warstwą uczącą.
- **Raz na 6–12 mies.:** benchmark/warsztat jako produkt uboczny (papier).

## Artefakty

- `docs/MAPA-WIEDZY.md` — samoocena + program nauki (start: dziś).
- `docs/lekcje/LEKCJA-01-maska-i-3-procent.md` — wzorzec lekcji (do iteracji
  z Marią, jak karta tematu w prażankach: najpierw jedna, potem reszta).
- `docs/brainstorms/` — zapisy wspólnych burz hipotez.
