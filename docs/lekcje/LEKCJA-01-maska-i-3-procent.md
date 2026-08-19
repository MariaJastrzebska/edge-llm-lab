# LEKCJA 01: Co naprawdę robi „maska" — i skąd wzięło się 3%

> Wzorcowa mini-lekcja (≤10 min). Wszystko poniżej jest ugruntowane w TWOICH
> danych: eksperyment `002-g1-mask-cost-pilot`, raporty nocne 10–13.08.
> Na końcu: 3 pytania sprawdzające i 2 decyzje, które czekają na Ciebie.

## 1. Problem, który maska rozwiązuje

Model językowy na końcu każdego kroku produkuje **logity** — po jednej liczbie
dla każdego tokenu ze słownika (w Twoich modelach: dziesiątki tysięcy liczb).
Logit = „jak bardzo chcę teraz powiedzieć ten token". Sampler zamienia logity
na wybór jednego tokenu. I to jest cała generacja: krok po kroku, token po
tokenie.

Kłopot: gdy prosisz model o JSON zgodny ze schemą, model „chce" czasem
powiedzieć rzeczy, które JSON-a psują. W Twoim eksperymencie, PO naprawieniu
błędu z enumami, model granite 2B **bez żadnego przymusu** wyprodukował
poprawny JSON w 18 na 100 prób. 82 razy chciał źle.

## 2. Co robi maska (dosłownie)

**Dekodowanie ograniczone** (u Ciebie: GBNF i json_schema w llama.cpp) dokłada
przed samplerem jeden krok: bierze gramatykę i na KAŻDYM kroku generacji
wylicza zbiór tokenów **legalnych w tym miejscu**. Potem robi rzecz brutalnie
prostą — logitom wszystkich nielegalnych tokenów wpisuje minus nieskończoność.
To jest cała „maska": **wektor zer i minus-nieskończoności nałożony na logity**.
Po masce sampler fizycznie nie może wybrać złego tokenu — nielegalne tokeny
mają prawdopodobieństwo zero.

Dlatego w Twojej tabeli gbnf i json_schema mają **100/100 poprawnych**: to nie
model „nauczył się" schemy — to maska nie zostawiła mu wyboru.

Koszt? Maskę trzeba ZBUDOWAĆ na każdym kroku: przejść gramatykę i oznaczyć,
które z ~50 tys. tokenów są teraz legalne. To jest praca na CPU, poza samą
siecią. Twoja hipoteza G1 pyta dokładnie o to: **czy budowa maski kosztuje
zauważalny czas/energię, czy chowa się w cieniu forwardu sieci**. Nocne
pomiary z 13.08 mówią: na Twoim sprzęcie ~1,3% RÓŻNICY na token, w kierunku
przeciwnym do przewidywania G1 — czyli maska u Ciebie chowa się w cieniu
(dlaczego tak może być — to osobna lekcja o tym, co jeszcze dzieje się
między krokami).

## 3. Skąd DOKŁADNIE wzięło się „3%"

Z tej komórki tabeli (raport 12.08, eksperyment 002, model granite3.1-dense:2b):

> truncation, unconstrained: 5% (10.08) → **3% (12.08)**

Wyliczenie, bez magii: w warunku „unconstrained" było **n=100 przebiegów**;
w **3** z nich generacja urwała się na limicie tokenów, zanim JSON się domknął.
3/100 = 3%. Tyle. „Truncation" liczymy osobno od innych błędów, bo ucięcie to
inna choroba niż zły typ pola: ucięty JSON bywa POPRAWNY aż do miejsca cięcia —
model nie kłamał, tylko nie zdążył.

Przy okazji — najważniejsza liczba tamtej nocy, też z jawnym wyliczeniem:
**J per valid** = energia całego warunku ÷ liczba POPRAWNYCH wyników.
Unconstrained: cała energia 100 prób podzielona przez 18 sukcesów = 702 J.
GBNF: podobna energia, ale ÷100 sukcesów = 123 J. Stąd „~5,7× taniej za
poprawny wynik" (702/123 ≈ 5,7) — nie dlatego, że maska oszczędza prąd na
tokenie, tylko dlatego, że **nie marnuje całych przebiegów**.

## 4. Most do Twojego researchu (LLM ↔ VLM)

Maska logitów działa na WYJŚCIU modelu — więc jest ślepa na to, co model ma
w środku. To znaczy: ta sama technika działa identycznie dla VLM-ów (model
patrzy na obraz, ale MÓWI tokenami — a maska siedzi na tokenach). Pytanie
badawcze, które z tego wyrasta i którego mało kto dotyka na edge'u: czy
J-per-valid dla VLM-ów na urządzeniach zachowuje się jak u LLM-ów, skoro
prompt-obraz dramatycznie zmienia proporcję prefill/decode? To jest miejsce,
gdzie Twoje doświadczenie z ewaluacjami jest realnym wkładem, nie kibicowaniem.

## 5. Trzy pytania sprawdzające (odpowiedz sobie, nie mnie)

1. Dlaczego maska daje 100/100 poprawności, choć model „umie" schemę tylko
   w 18%? (Jedno zdanie.)
2. Gdyby limit tokenów był 2× większy, która komórka tabeli MUSI się poprawić,
   a która NIE MOŻE się pogorszyć?
3. Dlaczego J-per-valid spadł 100× po naprawie enumów, choć energia na token
   prawie się nie zmieniła?

## 6. Dwie decyzje, które należą do Ciebie (paczka poniedziałkowa)

A. G1 wyszło „maska chowa się w cieniu" na obecnym sprzęcie. Kierunki:
   (1) powtórzyć na słabszym urządzeniu, gdzie forward jest krótszy i cień
   mniejszy; (2) uznać G1 za zamknięte i iść w residualne 82 błędy
   (28× zły typ!); (3) most VLM z §4. Co wybierasz i DLACZEGO?
B. Czy „truncation" raportujemy dalej osobno, czy wliczamy do invalid?
   (Podpowiedź: zależy, czy metryka ma karać model, czy konfigurację limitu.)
