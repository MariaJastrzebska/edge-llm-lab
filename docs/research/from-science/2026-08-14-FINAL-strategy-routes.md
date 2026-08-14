# FINALNY RAPORT STRATEGICZNY — trasy na topowy paper (do decyzji Marii)

Synteza czterech analiz (2026-08-14): `hypothesis-slate`, `landscape-systems`
(+oś VLM), `landscape-robotics-medical`, `optimization-methods`,
`mechanism-and-gaps`. Repo i cała maszyneria G1 są zachowane w KAŻDEJ trasie.

## Kontekst w dwóch zdaniach

Eksperyment 005 potwierdził mechanizm G1: poniżej ~25 ms/token podatek za
dekodowanie z gramatyką jest realny na wszystkich czterech małych modelach
(+1.8% do +30%), a zaprojektowana metoda (shadow-mask pipelining) potrafi go
odzyskać. Pytanie strategiczne brzmiało: w jakiej ramie ten silnik da
NAJWAŻNIEJSZY i NIEZAJĘTY paper — i przegląd literatury dał zaskakująco
zgodną odpowiedź.

---

## TRASA A (REKOMENDOWANA): Podatek dekodowania strukturalnego na EDGE VLM-ach
### — wersja wizyjna, medyczna („G1 podniesione do vision")

**Use case:** endoskop / kamera chirurgiczna / aparat USG podłączone do
urządzenia brzegowego, które NA MIEJSCU (bez chmury — prywatność pacjenta)
patrzy na obraz i wypełnia ustrukturyzowany raport kliniczny: synoptyczny
raport operacyjny, szablon opisu endoskopii. Schemat NIE jest wymyślony przez
nas — to zewnętrzne standardy kliniczne. Gramatyka pilnuje formatu, bo szpital
nie może dostać „prawie-JSONa".

**Hipoteza do potwierdzenia/obalenia:** (1) podatek constrained-decoding
istnieje i rośnie też na VLM-ach w fazie dekodowania; (2) teza „Seeing is
Free, Speaking is Not" (dekodowanie, nie enkoder wizji, dominuje energię edge
VLM — każdy token wyjścia kosztuje 11–39× token wejścia) przenosi się (lub
nie!) z dyskretnych GPU na unified-memory Apple Silicon; (3) shadow-mask
pipelining odzyskuje podatek bez zmiany wyników; (4) poprawność zadaniowa
nie zapada się pod gramatyką (oś correctness).

**Dlaczego blue ocean (zweryfikowane 4 puste cele):** koszt energetyczny
constrained-decoding — niezmierzony NIGDZIE; constrained decoding na VLM-ach
— n≈1 prac (tylko accuracy); systemowe liczby dla medycznych VLM-ów na
realnym edge — ZERO papers; energia VLM na Apple Silicon + powermetrics —
zero trafień.

**Impact:** • BIZNES/bigos: rdzeń tezy „agenci na edge" — ile kosztuje
pilnowanie struktury wyjścia agenta wizyjnego i jak to tanieć; metryka
J-per-poprawne-zadanie (Gap 2) staje się prymitywem schedulera bigosa.
• NAUKA: pierwsza charakteryzacja + mechanizm + optymalizacja tej warstwy na
VLM; replikacja/obalenie 2607.09520 na unified memory. • OPTYMALIZACJA:
gotowy hero-method (pipelining maski przez granicę CPU/GPU, ~200-400 LOC,
bit-exact) + kontrolowana macierz sterowników (vocab/rozmiar/architektura).
• CHIRURGIA/MEDYCYNA: dane pacjenta nie opuszczają sali; wiadomo, ile watów
i milisekund kosztuje wiarygodny raport przy stole.

**Koszt/ryzyko:** smoke-test llama.cpp mtmd (obraz+gramatyka) na M1 —
1 wieczór, PIERWSZY krok; cytować 2607.09520 jako właściciela generycznego
profilu (nasz wkład: constrained-tax + unified-memory + medical + metoda);
trzymać się VLM/medical, NIE VLA/robot (tam tłok i zarzut „M1 bez robota").
**Venue:** MLSys (~30.10, wciąż niepotwierdzone) / SenSys / RA-L; medyczne
warsztaty jako drugi dom.

---

## TRASA B: Gramatyki akcji robota pod deadline (G3 — zwycięzca sprzed osi VLM)

**Use case:** robot (asystent domowy/chirurgiczny) generuje komendy przez
LLM z gramatyką bezpieczeństwa (STL/CBF — SafeDec-style); pętla sterowania ma
twardy deadline — podatek maski to przekroczone deadline'y, pipelining to ich
odzysk. **Hipoteza:** SafeDec/VLSA twierdzą „minimal overhead" nie mierząc
edge-dżuli/latency — potwierdzić/obalić i naprawić. **Impact:** robotyka-
native (ICRA/CoRL + MLSys), bigos-blisko; chirurgicznie: warstwa planowania,
nie pętla 10–50 Hz (uczciwie: LLM tam nie siedzi). **Ryzyko:** ICRA deadline
15.09 — bardzo ciasno; workload trzeba dopiero zbudować (gramatyka akcji).
Punktacja slate: 23/25 (bita przez A po dojściu osi VLM: A zamyka te same
słabości G1, ale na 4 pustych celach zamiast 1 i bez wyścigu z ICRA).

## TRASA C: Energetyczny koszt prywatności (zero-egress medical)

**Use case:** szpital wybiera: ekstrakcja/opis on-device vs sanitize-then-
cloud; nikt nie zmierzył premii J/ms/termiki za „dane nie wychodzą".
**Hipoteza:** premia zero-egress jest mierzalna, spada z tierem modelu/kwant.
i bywa mniejsza niż koszt sanitizacji. Literatura zero-egress WPROST odsyła
energię do future work. **Impact:** najczystsza historia prywatności; polityka
szpitalna dostaje liczby; bigos: cennik „prywatność za waty". **Venue:**
SenSys/RA-L — więcej runway. Świetny paper #2 albo trasa, jeśli chcesz
maksymalnie medycznie a mniej systemowo.

## TRASA D: G1 klasyczne (tekst, medyczny JSON) — fallback

Najbardziej „shovel-ready" (005/006 + specy gotowe), ale rama najsłabsza:
schemat arbitralny (potwierdzone przez przegląd robotyczny), wąskość, którą
sama wytknęłaś. Zostaje jako komórka generalizacji wewnątrz A.

---

## REKOMENDACJA

**Trasa A**, z wchłoniętymi: kontrolowaną macierzą sterowników (mechanism-and-
gaps), osią poprawności (re-score offline + gold p06–p20) i hero-optymalizacją
(007). B jako paper #2 w sezonie robotycznym; C jako paper #2 alternatywny
lub rozdział. Pierwsze kroki po decyzji: (1) smoke-test mtmd na Macu,
(2) kolejka: VLM-owa wersja 005 na 3–4 modelach (moondream/LLaVA/Qwen2-VL/
MiniCPM-V) ze schematem synoptycznym, (3) fork --grammar-pipeline.

*Decyzja należy do Marii — worker czeka w trybie literature-only.*
