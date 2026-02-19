#!/usr/bin/env python3
"""
Konwersja logów JSON do TinyDB (lekka embedded NoSQL).
Użycie: python convert_to_tinydb.py
"""
import json
import os
from tinydb import TinyDB, Query

# Ustal ścieżkę bazową (katalog gdzie jest skrypt)
script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, 'output/agents/constant_data_en/referenced/log/constant_data_en_evaluation_results.json')

# 1. Wczytaj JSON
with open(json_path) as f:
    data = json.load(f)

# 2. Utwórz bazę TinyDB
db = TinyDB('evaluation_logs.json', indent=2)

# 3. Rozplątaj dane i zapisz jako płaskie dokumenty (łatwiejsze do query)
rounds_table = db.table('rounds')
rounds_table.truncate()  # Wyczyść starą zawartość

for eval_session in data['evaluations']:
    model_name = eval_session['model_name']
    timestamp = eval_session['session_timestamp']
    
    for round_data in eval_session.get('rounds', []):
        # Stwórz płaski dokument dla każdej rundy
        doc = {
            'model_name': model_name,
            'session_timestamp': timestamp,
            'round': round_data['round'],
        }
        
        # GPT Judge Score
        llm_resp = round_data.get('llm_response', {})
        if isinstance(llm_resp, dict) and 'gpt_judge' in llm_resp:
            doc['gpt_score'] = llm_resp['gpt_judge'].get('score')
            doc['gpt_categories'] = llm_resp['gpt_judge'].get('category_scores')
        
        # Latency & Performance
        lat = round_data.get('latency_breakdown', {})
        if lat:
            doc['total_ms'] = lat.get('total_ms')
            doc['prompt_eval_ms'] = lat.get('prompt_eval_ms')
            doc['token_generation_ms'] = lat.get('token_generation_ms')
            
            tokens = lat.get('tokens', {})
            doc['tps'] = tokens.get('throughput_tokens_per_sec')
            doc['prompt_count'] = tokens.get('prompt_count')
            doc['generated_count'] = tokens.get('generated_count')
            
            # Energy
            energy = lat.get('start_resources', {}).get('energy', {})
            doc['cpu_power_mw'] = energy.get('cpu_power_mw')
            doc['gpu_power_mw'] = energy.get('gpu_power_mw')
            doc['total_power_mw'] = energy.get('total_power_mw')
        
        rounds_table.insert(doc)

print(f"Zapisano {len(rounds_table)} rund do evaluation_logs.json")
print("\nPrzykładowe query:")
print("------------------")

# Przykład 1: Wszystkie rundy dla Granite 3.2
Round = Query()
granite_rounds = rounds_table.search(Round.model_name == 'granite3.2:2b-instruct-q8_0')
print(f"\n1. Granite 3.2 Q8: {len(granite_rounds)} rund")
if granite_rounds:
    scores = [r.get('gpt_score') for r in granite_rounds if r.get('gpt_score')]
    if scores:
        print(f"   GPT Scores: {scores}")
        print(f"   Mean: {sum(scores)/len(scores):.2f}")

# Przykład 2: Porównanie TPS między modelami
models_to_compare = ['granite3.2:2b-instruct-q4_K_M', 'qwen2.5:3b-instruct-q4_K_M']
print(f"\n2. Porównanie TPS:")
for model in models_to_compare:
    rounds = rounds_table.search(Round.model_name == model)
    if rounds:
        tps_values = [r.get('tps') for r in rounds if r.get('tps')]
        if tps_values:
            print(f"   {model}: {sum(tps_values)/len(tps_values):.2f} tokens/s (n={len(tps_values)})")

print("\n✨ Gotowe! Teraz możesz używać TinyDB do query.")
