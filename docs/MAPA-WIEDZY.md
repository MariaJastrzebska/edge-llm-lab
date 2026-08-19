# Mapa wiedzy — samoocena Marii (program nauki bierze się z luk)

> Zaznacz przy każdym: ✅ znam (umiem wyjaśnić komuś) / 🌫 mgliście (kojarzę,
> nie wyjaśnię) / ❌ nie znam. Lekcje powstają TYLKO dla 🌫/❌, które dotykają
> bieżących eksperymentów. Szczerość > ambicja — mapa jest prywatna.

## A. Fundamenty sieci (DNN)
- [ ] tensor, warstwa liniowa, funkcja aktywacji
- [ ] gradient i backprop (po co, niekoniecznie jak liczyć ręcznie)
- [ ] overfitting / regularyzacja / dropout
- [ ] batch norm vs layer norm (i czemu transformery wolą layer norm)
- [ ] embedding — co to znaczy, że token jest wektorem

## B. Transformer / LLM
- [ ] attention: Q, K, V — co się z czym mnoży i po co
- [ ] **maska przyczynowa** (causal) vs maska paddingu vs **maska logitów
      w dekodowaniu ograniczonym** (→ LEKCJA-01) — trzy różne „maski"!
- [ ] KV-cache — czemu drugi token jest tańszy niż pierwszy
- [ ] prefill vs decode — dwie fazy o innym profilu obliczeń (kluczowe dla
      Twoich pomiarów energii!)
- [ ] tokenizacja (BPE) — czemu „zaręczynach" bywa 3 tokenami
- [ ] sampling: temperatura, top-p, greedy — gdzie w tym siedzi maska logitów
- [ ] MoE — czemu model 8×7B nie kosztuje jak 56B
- [ ] RLHF/DPO — skąd model „wie", czego nie mówić (z grubsza)

## C. VLM (najciekawszy dla Ciebie most)
- [ ] ViT: obraz jako patche-tokeny — dlaczego to TEN SAM transformer
- [ ] CLIP / kontrastywne uczenie — wspólna przestrzeń obrazu i tekstu
- [ ] projektor/adapter (LLaVA-style): obraz → tokeny „udające" tekst
- [ ] cross-attention vs early-fusion — dwa sposoby wpięcia obrazu
- [ ] czemu maska logitów działa dla VLM identycznie jak dla LLM (→ LEKCJA-01 §4)
- [ ] halucynacje wizualne i jak się je EWALUUJE (Twoja działka!)

## D. Edge / inference (rdzeń labu)
- [ ] kwantyzacja: int8/int4, GGUF — co tracimy i jak to MIERZYĆ
- [ ] llama.cpp: co robi, czym różni się od servingu GPU (Twój RAMP-owy świat)
- [ ] dekodowanie ograniczone: GBNF vs json_schema (→ LEKCJA-01)
- [ ] pomiar energii: J na token vs J per VALID — czemu to drugie jest metryką
      produktu, nie sprzętu
- [ ] batching/spekulatywne dekodowanie — czemu na edge'u zwykle ich nie ma
- [ ] destylacja — jak z dużego modelu robi się mały

## E. Metodologia (tu masz przewagę — do świadomego użycia)
- [ ] korelacja vs eksperyment; confound (Twój enum-bug to piękny case!)
- [ ] przedział ufności — co znaczy [−2.227, +0.349] w Twojej tabeli
- [ ] moc testu / wielkość próby — kiedy 100 przebiegów wystarcza
- [ ] LLM-as-judge: kiedy wolno, jak kalibrować (Twoje CoT-judges z RAMP)

---
Po wypełnieniu: zaznaczone 🌫/❌ z sekcji B i D idą pierwsze (dotykają nocnych
eksperymentów), C planujemy pod brainstorm mostu VLM, E odhaczamy szybko —
podejrzewam, że większość masz ✅ i to jest Twój wkład do par z Claude'em.
