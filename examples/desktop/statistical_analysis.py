#!/usr/bin/env python3
"""
Wieloczynnikowa Analiza Statystyczna Logów Ewaluacyjnych
==========================================================
UŻYWA TinyDB (NoSQL embedded database) do zapytań!

Porównuje modele LLM na podstawie wielu metryk używając:
- ANOVA (jednoczynnikowa i dwuczynnikowa)
- Testy post-hoc (Tukey HSD)
- Testy dla par (Mann-Whitney U)
- Wizualizacje box plots z istotnością statystyczną

Każda runda dialogu = 1 punkt danych (round-based analysis).
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway, kruskal, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from tinydb import TinyDB, Query
import os
import subprocess

# Konfiguracja
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("📊 ANALIZA STATYSTYCZNA LOGÓW EWALUACYJNYCH (Round-Based, Multi-Factor)")
print("="*80)

# ============================================================================
# 1. PRZYGOTOWANIE BAZY DANYCH TINYDB
# ============================================================================

db_path = 'evaluation_logs.json'

if not os.path.exists(db_path):
    print("\n🔄 Baza danych TinyDB nie istnieje. Konwertuję JSON...")
    result = subprocess.run(['python3', 'examples/desktop/convert_to_tinydb.py'], 
                          capture_output=True, text=True, cwd='.')
    print(result.stdout)
    if result.returncode != 0:
        print("❌ Błąd konwersji:", result.stderr)
        exit(1)
else:
    print(f"\n✓ Znaleziono bazę TinyDB: {db_path}")

# ============================================================================
# 2. WCZYTANIE DANYCH Z TINYDB DO PANDAS
# ============================================================================

print("\n📂 Ładowanie danych z TinyDB...")
db = TinyDB(db_path)
rounds_table = db.table('rounds')

# Załaduj wszystkie rundy
all_rounds = rounds_table.all()
df = pd.DataFrame(all_rounds)

# Usuń kolumnę _default (TinyDB internal)
if '_default' in df.columns:
    df = df.drop(columns=['_default'])

print(f"✓ Załadowano {len(df)} rund z {df['model_name'].nunique()} modeli")
print(f"\nModele w bazie:")
for model in sorted(df['model_name'].unique()):
    count = len(df[df['model_name'] == model])
    print(f"  - {model}: {count} rund")

# Skrócenie nazw dla czytelności wykresów
df['model_short'] = (df['model_name']
                      .str.replace(':2b-instruct-', ' 2B ')
                      .str.replace(':3b-instruct-', ' 3B ')
                      .str.replace(':4b-instruct-', ' 4B ')
                      .str.replace('granite3.1-dense', 'G3.1')
                      .str.replace('granite3.2', 'G3.2')
                      .str.replace('granite4', 'G4')
                      .str.replace('qwen2.5', 'Qwen')
                      .str.replace('llama3.2', 'Llama')
                      .str.replace('nemotron-mini', 'Nemotron'))

# ============================================================================
# 3. PRZYGOTOWANIE METRYK DO ANALIZY
# ============================================================================

metrics = {
    'gpt_score': 'GPT Score (jakość)',
    'tps': 'Throughput (tokens/s)',
    'total_power_mw': 'Zużycie energii (mW)',
    'total_ms': 'Latencja (ms)'
}

# Usuń wiersze bez kluczowych metryk
df_clean = df.dropna(subset=list(metrics.keys()), how='all')
print(f"\n✓ Po czyszczeniu: {len(df_clean)} rund z kompletnymi danymi\n")

# ============================================================================
# 4. STATYSTYKI OPISOWE
# ============================================================================

print("="*80)
print("� STATYSTYKI OPISOWE (Mean ± SD)")
print("="*80)

summary_stats = df_clean.groupby('model_short')[list(metrics.keys())].agg(['mean', 'std', 'count'])
summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
print("\n" + summary_stats.to_string())

# ============================================================================
# 5. ANOVA JEDNOCZYNNIKOWA (czy modele różnią się istotnie?)
# ============================================================================

print("\n" + "="*80)
print("📈 ANOVA JEDNOCZYNNIKOWA: Porównanie modeli")
print("="*80)

anova_results = []

for metric_key, metric_name in metrics.items():
    # Przygotuj grupy dla każdego modelu
    groups = [group[metric_key].dropna().values 
              for name, group in df_clean.groupby('model_short') 
              if len(group[metric_key].dropna()) > 0]
    
    if len(groups) < 2:
        continue
    
    # ANOVA F-test
    f_stat, p_value = f_oneway(*groups)
    
    # Kruskal-Wallis (nieparametryczna alternatywa)
    h_stat, p_value_kw = kruskal(*groups)
    
    anova_results.append({
        'Metryka': metric_name,
        'Liczba_grup': len(groups),
        'F_statistic': f'{f_stat:.3f}',
        'p_value_ANOVA': f'{p_value:.4f}',
        'H_statistic': f'{h_stat:.3f}',
        'p_value_KW': f'{p_value_kw:.4f}',
        'Istotność': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    })
    
    print(f"\n{metric_name}:")
    print(f"  ANOVA: F = {f_stat:.3f}, p = {p_value:.4f} {'✓ ISTOTNE STATYSTYCZNIE' if p_value < 0.05 else '✗ nie istotne'}")
    print(f"  Kruskal-Wallis: H = {h_stat:.3f}, p = {p_value_kw:.4f}")

anova_df = pd.DataFrame(anova_results)
print("\n" + "="*80)
print(anova_df.to_string(index=False))
print("="*80)

# ============================================================================
# 6. TESTY POST-HOC (które pary modeli się różnią?)
# ============================================================================

print("\n" + "="*80)
print("🔍 TESTY POST-HOC (Tukey HSD): Pairwise Comparisons")
print("="*80)

for metric_key, metric_name in metrics.items():
    df_metric = df_clean[['model_short', metric_key]].dropna()
    
    if len(df_metric) > 10 and df_metric['model_short'].nunique() > 1:
        print(f"\n{metric_name}:")
        tukey = pairwise_tukeyhsd(endog=df_metric[metric_key], 
                                   groups=df_metric['model_short'], 
                                   alpha=0.05)
        print(tukey)

# ============================================================================
# 7. PORÓWNANIA KLUCZOWE (wybrane pary)
# ============================================================================

print("\n" + "="*80)
print("⚔️  PORÓWNANIA PAR MODELI (Mann-Whitney U)")
print("="*80)

# Kluczowe pary do porównania
comparisons = [
    ('granite3.2:2b-instruct-q8_0', 'granite3.2:2b-instruct-q4_K_M', 'Granite 3.2: Q8 vs Q4'),
    ('granite3.1-dense:2b-instruct-q8_0', 'granite3.2:2b-instruct-q8_0', 'Granite 3.1 vs 3.2 (Q8)'),
    ('granite3.2:2b-instruct-q4_K_M', 'qwen2.5:3b-instruct-q4_K_M', 'Granite 3.2 vs Qwen 2.5'),
    ('granite3.2:2b-instruct-q4_K_M', 'llama3.2:3b-instruct-q4_K_M', 'Granite 3.2 vs Llama 3.2'),
]

for model_a, model_b, label in comparisons:
    data_a = df_clean[df_clean['model_name'] == model_a]
    data_b = df_clean[df_clean['model_name'] == model_b]
    
    if len(data_a) > 0 and len(data_b) > 0:
        print(f"\n{label}:")
        
        for metric_key, metric_name in metrics.items():
            vals_a = data_a[metric_key].dropna()
            vals_b = data_b[metric_key].dropna()
            
            if len(vals_a) > 0 and len(vals_b) > 0:
                u_stat, p_value = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
                
                sig = ' ✓ ISTOTNE' if p_value < 0.05 else ''
                print(f"  {metric_name}:")
                print(f"    Δ = {vals_b.mean() - vals_a.mean():+.2f}, p = {p_value:.4f}{sig}")
                print(f"    {model_a.split(':')[0]}: {vals_a.mean():.2f} ± {vals_a.std():.2f} (n={len(vals_a)})")
                print(f"    {model_b.split(':')[0]}: {vals_b.mean():.2f} ± {vals_b.std():.2f} (n={len(vals_b)})")

# ============================================================================
# 8. WIZUALIZACJE
# ============================================================================

print("\n📊 Generowanie wykresów...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Analiza Statystyczna Modeli (Round-Based)', fontsize=16, fontweight='bold')

for idx, (metric_key, metric_name) in enumerate(metrics.items()):
    ax = axes[idx // 2, idx % 2]
    
    # Box plot
    df_plot = df_clean[[metric_key, 'model_short']].dropna()
    df_plot.boxplot(column=metric_key, by='model_short', ax=ax, grid=False, patch_artist=True)
    
    ax.set_title(metric_name, fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel(metric_key.replace('_', ' ').title())
    ax.tick_params(axis='x', rotation=45)
    
    # Dodaj p-value z ANOVA
    result = [r for r in anova_results if r['Metryka'] == metric_name]
    if result:
        p_val = result[0]['p_value_ANOVA']
        sig = result[0]['Istotność']
        ax.text(0.02, 0.98, f'ANOVA p={p_val} {sig}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                fontsize=9)
    
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()
output_path = 'output/agents/constant_data_en/referenced/statistical_analysis_boxplots.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Zapisano: {output_path}")

# ============================================================================
# 9. EXPORT WYNIKÓW
# ============================================================================

print("\n� Eksport wyników...")

# CSV z statystykami
summary_stats.to_csv('output/agents/constant_data_en/referenced/summary_statistics.csv')
anova_df.to_csv('output/agents/constant_data_en/referenced/anova_results.csv', index=False)

print("✓ Zapisano:")
print("  - summary_statistics.csv")
print("  - anova_results.csv")
print("  - statistical_analysis_boxplots.png")

# ============================================================================
# 10. PODSUMOWANIE
# ============================================================================

print("\n" + "="*80)
print("✅ ANALIZA STATYSTYCZNA ZAKOŃCZONA POMYŚLNIE")
print("="*80)
print("\n📝 WNIOSKI DO PRACY MAGISTERSKIEJ:")
print("-" * 80)
print("1. Sprawdź tabele ANOVA - jeśli p < 0.05, modele różnią się statystycznie")
print("2. Testy post-hoc (Tukey) pokazują które konkretne pary są różne")
print("3. Mann-Whitney U porównuje kluczowe pary (np. Granite 3.1 vs 3.2)")
print("4. Effect size można obliczyć jako: r = Z / √N")
print("\n💡 Interpretacja p-wartości:")
print("   p < 0.001: *** (bardzo istotne statystycznie)")
print("   p < 0.01:  **  (istotne statystycznie)")
print("   p < 0.05:  *   (istotne statystycznie)")
print("   p ≥ 0.05:  ns  (nieistotne)")
print("\n� Pliki gotowe do włączenia do pracy:")
print("   - summary_statistics.csv → Tabela X")
print("   - anova_results.csv → Tabela Y")
print("   - statistical_analysis_boxplots.png → Rysunek Z")
print("="*80)

db.close()
