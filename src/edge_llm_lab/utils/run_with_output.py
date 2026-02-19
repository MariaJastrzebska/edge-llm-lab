#!/usr/bin/env python3
import subprocess
import sys
import os

# Przekaż odpowiedzi do skryptu
input_responses = [
    "1\n",  # Modele: Wszystkie
    "n\n",  # Auto-pobieranie modeli: nie
    "1\n",  # Język wykresów: Polski
    "3\n",  # Tryb: Viz (tylko wykresy)
    "2\n",  # Optymalizacje: Selected (nie będzie używane w trybie Viz)
    "1\n",  # Parametry inferencji: Standardowe (nie będzie używane w trybie Viz)
    "n\n",  # Generować wykresy per-round: nie
    "n\n",  # Generować wykresy aggr_over_rounds: nie
    "n\n",  # Generować wykresy per-model: nie
    "y\n",  # Generować wykresy all-models: tak
]

# Uruchom skrypt z przekazaniem inputu
process = subprocess.Popen(
    ["poetry", "run", "python", "thesis_generators/referenced_clean.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    cwd="/Users/mariamalycha/Documents/fed-mobile"
)

# Przekaż wszystkie odpowiedzi
input_data = "".join(input_responses)
stdout, stderr = process.communicate(input=input_data)

# Zapisz output do pliku
with open("output_log.txt", "w", encoding="utf-8") as f:
    f.write("STDOUT:\n")
    f.write(stdout)
    if stderr:
        f.write("\nSTDERR:\n")
        f.write(stderr)

print(f"Output zapisany do output_log.txt")
print(f"Exit code: {process.returncode}")
