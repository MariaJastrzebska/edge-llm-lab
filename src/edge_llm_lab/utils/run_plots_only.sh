#!/bin/bash
# Skrypt do uruchomienia generowania wykresów z zapisem outputu

cd /Users/mariamalycha/Documents/fed-mobile

# Przekaż odpowiedzi i zapisz output
(
    echo "1"  # Modele: Wszystkie

    echo "y"  # Auto-pobieranie modeli: nie

    echo "1"  # Język wykresów: Polski
    echo "3"  # Tryb: Viz (tylko wykresy)
    echo "n"  # Generować wykresy per-round: nie

    echo "n"  # Generować wykresy aggr_over_rounds: nie

    echo "n"  # Generować wykresy per-model: nie

    echo "y"  # Generować wykresy all-models: tak
) | poetry run python thesis_generators/referenced_clean.py

echo ""
echo "Output zapisany do output_log.txt"


