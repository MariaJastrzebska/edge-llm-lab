# Przewodnik po retuszu wizualnym: Optymalizacja LLM

Wykresy generowane przez framework są bogate w dane, co jest zaletą przy analizie inżynierskiej, ale może utrudniać czytelność w pracy magisterskiej. Poniżej znajdują się wskazówki, jak przygotować finalne ryciny do druku.

## 1. Wykres: "Optimization Comparison" (Główny)
To najbardziej złożony wykres (zawiera Latencję, Throughput, GPT Score i Memory). 

**Co wyciąć (Crop):**
- Skup się na dwóch górnych sekcjach: **Total Latency (ms)** oraz **Throughput (tokens/sec)**.
- Sekcja **GPT Judge Score** jest istotna tylko jako dowód na to, że linie idą "płasko" (brak degradacji jakości). Możesz ją pomniejszyć lub umieścić w aneksie.
- Sekcja **Memory Usage** (jeśli jest zerowa) – całkowicie usuń z ryciny głównej.

**Rekomendacja Legendy:**
- Jeśli legenda zasłania dane, przesuń ją pod wykres lub zostaw tylko 4-5 kluczowych linii:
  1. `Baseline` (linia referencyjna - czerwona).
  2. `Flash Attention + Cache f16` (Twój zwycięzca).
  3. `Speculative Decoding` (najszybszy throughput).
  4. `no-kv-offload` (przestroga - najgorszy wynik).

## 2. Wykresy Grupowe (Group Impact)
Wykresy takie jak `group_pure_speculative_decoding` są świetne do pokazania różnic w "nachyleniu" (slope).

- **Interpretacja nachylenia**: Zwróć uwagę, że latencja rośnie wraz z rundami (context window). Szukaj metody, która ma najbardziej "płaską" linię – to oznacza omijanie wąskiego gardła pamięci.

---

## Finałowa Hipoteza: "The Golden Trace" (Granite 3.2 2B)

Aby "domknąć" historię w pracy mgr, rekomenduję przeprowadzenie jednego, ostatecznego testu. Wcześniej optymalizowaliśmy Granite 3.1, ale selekcję wygrał **Granite 3.2 2B**.

**Proponowany test na desktopie (MacBook Air):**
Zastosuj "Złoty Zestaw" parametrów do modelu Granite 3.2 2B:
```bash
poetry run python examples/desktop/run_evaluation_pipeline.py \
  --model granite-3.2-2b-instruct \
  --mode logs_and_viz \
  --optimisations "[{'--flash-attn': None, '--cache-type-k': 'f16', '--cache-type-v': 'f16', '--threads': 4}]"
```

**Dlaczego to ważne?**
Pokażesz w ten sposób, że Twój proces budowy asystenta GlowAI był kompletny:
1. Etap 1: Wybór najlepszego "mózgu" (**Selection** -> Granite 3.2 2B).
2. Etap 2: Wybór najlepszych "mięśni" (**Optimization** -> Flash Attention + Cache f16).
3. Etap 3: Finalna symbioza (The Golden Trace).

Ten ostatni wykres (z jedną linią "Optimized" vs "Default") będzie idealnym zakończeniem części praktycznej.
