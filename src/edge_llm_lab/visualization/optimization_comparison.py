"""
 OPTIMIZATION COMPARISON: Compare latency across different optimization methods for a single model
"""
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import re
from datetime import datetime
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import matplotlib.ticker as mtick
from typing import Dict, List, Optional, Tuple, Any
import hashlib

# Language translations for bilingual support
TRANSLATIONS = {
    'en': {
        'title': 'Optimization Comparison - {}',
        'total_latency': 'Total Latency',
        'latency_breakdown': 'Latency Breakdown', 
        'throughput': 'Throughput',
        'optimization_reference': 'Optimization Reference',
        'time_ms': 'Time (ms)',
        'tokens_per_second': 'Tokens per Second',
        'model_loading': 'Model Loading',
        'prompt_eval': 'Prompt Eval',
        'token_generation': 'Token Generation',
        'optimization_col': 'Optimization',
        'number_col': '#',
        'table_title': 'Optimization Performance Comparison - {}',
        'params_col': 'Params',
        'latency_col': 'Latency (ms)',
        'latency_delta_col': 'Latency Δ%',
        'throughput_col': 'Throughput (tok/s)',
        'throughput_delta_col': 'Throughput Δ%'
    },
    'pl': {
        'title': 'Porównanie Optymalizacji - {}',
        'total_latency': 'Całkowita Latencja',
        'latency_breakdown': 'Podział Latencji',
        'throughput': 'Przepustowość', 
        'optimization_reference': 'Wykaz Optymalizacji',
        'time_ms': 'Czas (ms)',
        'tokens_per_second': 'Tokeny na Sekundę',
        'model_loading': 'Ładowanie Modelu',
        'prompt_eval': 'Ewaluacja Promptu',
        'token_generation': 'Generacja Tokenów',
        'optimization_col': 'Optymalizacja',
        'number_col': 'Nr',
        'table_title': 'Porównanie Wydajności Optymalizacji - {}',
        'params_col': 'Parametry',
        'latency_col': 'Latencja (ms)',
        'latency_delta_col': 'Latencja Δ%',
        'throughput_col': 'Przepustowość (tok/s)',
        'throughput_delta_col': 'Przepustowość Δ%'
    }
}

def _canonicalize_optimization(optimization: Dict[str, Any]) -> Tuple[str, str]:
    """Return canonical key and human label for an optimisation dict.

    - Canonical key is order-independent and stable across formatting changes
    - Label is a readable short version for plots
    """
    if not optimization:
        return ('baseline', 'Baseline')

    # Normalize keys and values, sort by key for stability
    tokens = []
    for k, v in sorted(optimization.items(), key=lambda kv: str(kv[0])):
        key_norm = str(k).strip()
        if key_norm.startswith('--'):
            key_norm = key_norm[2:]
        key_norm = key_norm.replace('_', '-').lower()
        if v is None or v is True or v == '':
            tokens.append(f"{key_norm}")
        else:
            tokens.append(f"{key_norm}={v}")

    canon_key = '|'.join(tokens)

    # Human label
    def prettify(tok: str) -> str:
        if '=' in tok:
            k, v = tok.split('=', 1)
            return f"{k.replace('-', ' ').title()}: {v}"
        return tok.replace('-', ' ').title()

    label = ' + '.join(prettify(t) for t in tokens)
    return (canon_key, label[:25])


def _color_for_canon_key(canon_key: str) -> str:
    """Map canonical key to a stable color from a fixed palette."""
    palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    digest = hashlib.md5(canon_key.encode('utf-8')).hexdigest()
    idx = int(digest[:8], 16) % len(palette)
    return palette[idx]


def _color_for_performance(baseline_latency, current_latency, is_baseline=False):
    """
    Generuje kolor na podstawie wydajności względem baseline.
    
    Args:
        baseline_latency: Latencja baseline
        current_latency: Latencja bieżącej optymalizacji
        is_baseline: Czy to jest baseline
    
    Returns:
        str: Kolor hex
    """
    if is_baseline:
        return 'black'  # Baseline zawsze czarne
    
    # Oblicz różnicę procentową
    if baseline_latency == 0:
        return 'gray'
    
    improvement_ratio = (baseline_latency - current_latency) / baseline_latency
    
    if improvement_ratio > 0:  # Poprawa (zielony)
        # Im większa poprawa, tym bardziej zielony
        intensity = min(abs(improvement_ratio), 0.5) / 0.5  # Normalizuj do 0-1
        green_value = int(100 + 155 * intensity)  # 100-255
        return f'#{0:02x}{green_value:02x}{0:02x}'  # RGB: (0, green, 0)
    else:  # Pogorszenie (czerwony)
        # Im większe pogorszenie, tym bardziej czerwony
        intensity = min(abs(improvement_ratio), 0.5) / 0.5  # Normalizuj do 0-1
        red_value = int(100 + 155 * intensity)  # 100-255
        return f'#{red_value:02x}{0:02x}{0:02x}'  # RGB: (red, 0, 0)


def _generate_optimization_table_image_single_language(session_data, model_name, plotting_session_timestamp, output_dir, output_file_name, language='en'):
    """
    Generuje tabelę porównawczą optymalizacji jako osobny obrazek w określonym języku.
    
    Args:
        session_data: Dict z optymalizacjami
        model_name: Nazwa modelu
        plotting_session_timestamp: Timestamp
        output_dir: Katalog wyjściowy
        output_file_name: Nazwa pliku
        language: Kod języka ('en' lub 'pl')
        
    Returns:
        str: Ścieżka do wygenerowanej tabeli
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get translations for the specified language
    t = TRANSLATIONS[language]
    
    # Przygotuj dane dla tabeli
    table_data = []
    
    for opt_key, session in session_data.items():
        opt_name = str(session.get('optimisation', opt_key))
        if opt_name == '{}':
            opt_name = 'Baseline'
        
        rounds_data = session.get('rounds', [])
        if not rounds_data:
            continue
        
        # Oblicz średnie wartości
        latencies = []
        throughputs = []
        
        for round_data in rounds_data:
            latency_breakdown = round_data.get('latency_breakdown', {})
            total_ms = latency_breakdown.get('total_ms', 0)
            tokens = latency_breakdown.get('tokens', {})
            throughput = tokens.get('throughput_tokens_per_sec', 0)
            
            latencies.append(total_ms)
            throughputs.append(throughput)
        
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0
        num_params = len(session.get('optimisation', {})) if session.get('optimisation') else 0
        
        table_data.append({
            'optimization': opt_name,
            'avg_latency': avg_latency,
            'avg_throughput': avg_throughput,
            'num_params': num_params
        })
    
    # Sortuj zawsze według latencji (najszybsze pierwsze)
    table_data.sort(key=lambda x: x['avg_latency'])
    
    # Znajdź baseline dla obliczenia popraw
    baseline_latency = None
    baseline_throughput = None
    for row in table_data:
        if row['optimization'] == 'Baseline':
            baseline_latency = row['avg_latency']
            baseline_throughput = row['avg_throughput']
            break
    
    # Przygotuj dane dla matplotlib table
    headers = [t['optimization_col'], t['params_col'], t['latency_col'], t['latency_delta_col'], t['throughput_col'], t['throughput_delta_col']]
    table_rows = []
    
    for row in table_data:
        latency_delta = 0.0
        throughput_delta = 0.0
        
        if baseline_latency and baseline_latency > 0 and row['optimization'] != 'Baseline':
            # Dla latencji: ujemny % = poprawa (szybciej), dodatni % = pogorszenie (wolniej)
            latency_delta = ((row['avg_latency'] - baseline_latency) / baseline_latency) * 100
        
        if baseline_throughput and baseline_throughput > 0 and row['optimization'] != 'Baseline':
            # Dla throughput: dodatni % = poprawa (więcej tok/s), ujemny % = pogorszenie
            throughput_delta = ((row['avg_throughput'] - baseline_throughput) / baseline_throughput) * 100
        
        table_rows.append([
            row['optimization'],  # Pełna nazwa bez skracania
            str(row['num_params']),
            f"{row['avg_latency']:.0f}",
            f"{latency_delta:+.1f}%" if row['optimization'] != 'Baseline' else "0.0%",
            f"{row['avg_throughput']:.1f}",
            f"{throughput_delta:+.1f}%" if row['optimization'] != 'Baseline' else "0.0%"
        ])
    
    # Stwórz obrazek z tabelą - bardzo duży rozmiar dla pełnych nazw
    fig, ax = plt.subplots(figsize=(20, max(12, len(table_rows) * 1.0)))
    ax.axis('off')
    
    # Stwórz tabelę z bardzo szeroką kolumną dla nazw
    table = ax.table(cellText=table_rows, colLabels=headers, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.08, 0.12, 0.1, 0.12, 0.08])
    
    # Stylizuj tabelę
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 2.5)
    
    # Stylizuj nagłówki
    for i in range(len(headers)):
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_facecolor('#2c3e50')
    
    # Koloruj wiersze na przemian
    for i in range(1, len(table_rows) + 1):
        color = '#f8f9fa' if i % 2 == 0 else '#ffffff'
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color)
    
    plt.title(t['table_title'].format(model_name), 
              fontsize=14, fontweight='bold', pad=20)
    
    # Zapisz tabelę
    table_path = os.path.join(output_dir, f'{output_file_name}_table_{language}.png')
    plt.savefig(table_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f" Table image ({language}) saved: {table_path}")
    return table_path


def generate_optimization_table_image(session_data, model_name, plotting_session_timestamp, output_dir, output_file_name):
    """
    Generuje tabele porównawcze optymalizacji w obu językach (Polski i Angielski).
    
    Args:
        session_data: Dict z optymalizacjami
        model_name: Nazwa modelu
        plotting_session_timestamp: Timestamp
        output_dir: Katalog wyjściowy
        output_file_name: Nazwa pliku
        
    Returns:
        dict: Ścieżki do wygenerowanych tabel {'en': path, 'pl': path}, lub None jeśli błąd
    """
    
    generated_tables = {}
    
    # Generate English version
    en_table = _generate_optimization_table_image_single_language(
        session_data, model_name, plotting_session_timestamp, 
        output_dir, output_file_name, 'en'
    )
    if en_table:
        generated_tables['en'] = en_table
    
    # Generate Polish version
    pl_table = _generate_optimization_table_image_single_language(
        session_data, model_name, plotting_session_timestamp, 
        output_dir, output_file_name, 'pl'
    )
    if pl_table:
        generated_tables['pl'] = pl_table
    
    return generated_tables if generated_tables else None


def extract_optimization_metrics(optimization: dict) -> Dict[str, Any]:
    """Extract and normalize optimization metrics from optimization dict."""
    if not optimization:
        return {
            'name': 'baseline',
            'label': 'Baseline',
            'color': 'black',  # Baseline czarne
            'priority': 0  # Baseline pierwsza
        }
    
    # Canonical key and readable label
    canon_key, short_label = _canonicalize_optimization(optimization)
    
    # Create a sorting key based on optimization type
    priority_map = {
        'baseline': 0,
        'flash': 1,
        'kv': 2,
        'cache': 3,
        'threads': 4,
        'batch': 5,
        'draft': 6,
        'cont': 7,
        'no': 8
    }
    
    # Determine priority based on optimization keys
    priority = 999  # Default high priority
    for key_pattern, prio in priority_map.items():
        if key_pattern in canon_key.lower():
            priority = prio
            break
    
    return {
        'name': canon_key,
        'label': short_label,
        'color': _color_for_canon_key(canon_key),
        'priority': priority
    }




def parse_optimization_key(opt_key: str) -> Dict[str, Any]:
    """Parse optimization key into a dictionary of parameters."""
    if not opt_key or opt_key.lower() == 'baseline':
        return {}
    
    params = {}
    # Try to parse dict-like strings first
    try:
        import ast
        parsed = ast.literal_eval(opt_key)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, (list, tuple)):
            try:
                return dict(parsed)
            except Exception:
                pass
    except Exception:
        pass

    # Fallback simple parser like "--flash-attn, --kv-cache=f16"
    parts = [p.strip() for p in re.split(r'[,+]', opt_key) if p.strip()]
    for part in parts:
        if '=' in part or ':' in part:
            sep = '=' if '=' in part else ':'
            k, v = part.split(sep, 1)
            params[k.strip()] = v.strip()
        else:
            params[part.strip()] = True
    return params


def _plot_optimization_comparison_single_language(
   session_data,
    model_name,
    plotting_session_timestamp,
    metadata,
    output_dir,
    output_file_name,
    language='en'
):
    """
     Generate optimization comparison plot in a specific language.
    
    Args:
        session_data: Dict mapping optimisation key -> session dict (with 'optimisation' and 'rounds')
        model_name: Name of the model being evaluated
        plotting_session_timestamp: Timestamp for the plot title
        metadata: Optional metadata for the model
        output_dir: Directory to save the output plot
        output_file_name: Base name for the output file
        language: Language code ('en' or 'pl')
        
    Returns:
        str: Path to the generated plot file, or None if generation failed
    """
    
    # Get translations for the specified language
    t = TRANSLATIONS[language]

    # Group data by optimization configuration for this specific model
    optimization_data = {}
    model_versions = set()


    if not isinstance(session_data, dict):
        print("❌ session_data must be a dict of {optimisation_key: session_dict}")
        return None

    for opt_key, sess in session_data.items():
        if not isinstance(sess, dict):
            continue
        # Model name for this session
        eval_model_name = sess.get('model_name') or sess.get('model_info', {}).get('name', 'Unknown')
        if eval_model_name != model_name:
            continue

        # Derive optimisation dict
        opt_cfg = sess.get('optimisation')
        if isinstance(opt_cfg, dict):
            opt_cfg_dict = opt_cfg
        else:
            # Try to parse from key or fallback to baseline
            if isinstance(opt_key, dict):
                opt_cfg_dict = opt_key
            elif isinstance(opt_key, (list, tuple)):
                # Convert list/tuple to dict if possible, else baseline
                try:
                    opt_cfg_dict = dict(opt_key)
                except Exception:
                    opt_cfg_dict = {}
            else:
                # Assume string like "--flash-attn, --kv-cache" and parse
                opt_cfg_dict = parse_optimization_key(str(opt_key)) if opt_key else {}
        
        # Canonical optimization key for grouping (stable across label changes)
        canon_opt_key, short_label = _canonicalize_optimization(opt_cfg_dict)
        # Prepare container for this optimization
        if canon_opt_key not in optimization_data:
            meta_info = extract_optimization_metrics(opt_cfg_dict)
            meta_info['label'] = short_label
            optimization_data[canon_opt_key] = {
                'total_times': [],
                'prompt_eval_times': [],
                'generation_times': [],
                'loading_times': [],
                'throughput': [],
                'meta': meta_info,
                'quality_metrics': {},
                'original_optimization': opt_cfg_dict  # Zapisz oryginalne dane
            }
        
        # Collect round metrics
        for round_data in sess.get('rounds', []) or []:
            if not isinstance(round_data, dict):
                continue
            latency = round_data.get('latency_breakdown', {}) or {}
            total_ms = latency.get('total_ms', 0)
            # Support both naming variants
            prompt_ms = latency.get('prompt_evaluation_ms', latency.get('prompt_eval_ms', 0))
            generation_ms = latency.get('token_generation_ms', 0)
            loading_ms = latency.get('model_loading_ms', 0)
            
            if isinstance(total_ms, (int, float)) and total_ms > 0:
                optimization_data[canon_opt_key]['total_times'].append(total_ms)
                optimization_data[canon_opt_key]['prompt_eval_times'].append(prompt_ms or 0)
                optimization_data[canon_opt_key]['generation_times'].append(generation_ms or 0)
                optimization_data[canon_opt_key]['loading_times'].append(loading_ms or 0)
                # Throughput
                tokens = latency.get('tokens', {}) or {}
                throughput = tokens.get('throughput_tokens_per_sec', 0)
                if isinstance(throughput, (int, float)) and throughput > 0:
                    optimization_data[canon_opt_key]['throughput'].append(throughput)

    if not optimization_data:
        print("❌ No optimization data found for this model")
        return None
    
    # Sort optimizations by average latency (baseline first, then by performance)
    def sort_key(item):
        _, data = item
        if data['meta']['name'] == 'baseline' or 'Baseline' in data['meta']['label']:
            return (0, 0)  # Baseline zawsze pierwsze
        avg_latency = np.mean(data['total_times']) if data['total_times'] else float('inf')
        return (1, avg_latency)  # Pozostałe według latencji
    
    sorted_opts = sorted(optimization_data.items(), key=sort_key)
    
    # Create a 2x2 layout z większymi odstępami - szerszy format
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Total latency
    ax2 = fig.add_subplot(gs[0, 1])  # Latency breakdown
    ax3 = fig.add_subplot(gs[1, 0])  # Throughput
    ax4 = fig.add_subplot(gs[1, 1])  # Memory usage
    
    # Set main title with model info
    model_version = f" ({list(model_versions)[0]})" if model_versions else ""
    fig.suptitle(
        t['title'].format(f'{model_name}{model_version}'), 
        fontsize=18, 
        fontweight='bold',
        y=0.99
    )
    
    # Znajdź baseline latency dla kolorowania względem wydajności
    baseline_latency = None
    for opt_name, data in sorted_opts:
        if opt_name == 'Baseline' or data['meta']['name'] == 'baseline':
            baseline_latency = np.mean(data['total_times']) if data['total_times'] else 0
            break
    
    # Set colors based on performance relative to baseline
    colors = []
    for opt_name, data in sorted_opts:
        is_baseline = (opt_name == 'Baseline' or data['meta']['name'] == 'baseline')
        current_latency = np.mean(data['total_times']) if data['total_times'] else 0
        color = _color_for_performance(baseline_latency or 0, current_latency, is_baseline)
        colors.append(color)
    
    # Helper function to format large numbers
    def human_format(num, pos=None):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
    
    # Helper function to add value labels on bars - dynamically sized
    def add_bar_labels(ax, bars, values, fmt='.0f', offset=0.05, color='black', fontsize=9):
        # Skip labels if too many optimizations
        if len(sorted_opts) > 12:
            return
        
        # Adjust font size based on number of optimizations
        if len(sorted_opts) > 8:
            fontsize = 6
        elif len(sorted_opts) > 6:
            fontsize = 7
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height * (1 + offset),
                f"{val:{fmt}}",
                ha='center',
                va='bottom',
                color=color,
                fontsize=fontsize,
                fontweight='bold'
            )
    
    # 1. Total Latency Comparison - używamy numerów zamiast długich nazw
    opt_labels = [f"{i+1}" for i in range(len(sorted_opts))]  # Numery 1, 2, 3...
    opt_names = [data['meta']['label'] for _, data in sorted_opts]  # Pełne nazwy dla tabeli
    total_means, total_stds, total_mins, total_maxs = [], [], [], []
    for _, data in sorted_opts:
        times = data['total_times']
        if times:
            total_means.append(np.mean(times))
            total_stds.append(np.std(times))
            total_mins.append(np.min(times))
            total_maxs.append(np.max(times))
        else:
            total_means.append(0); total_stds.append(0); total_mins.append(0); total_maxs.append(0)
    x = np.arange(len(opt_labels))
    bars1 = ax1.bar(x, total_means, color=colors, alpha=0.8, 
                   yerr=[np.array(total_means) - np.array(total_mins), 
                         np.array(total_maxs) - np.array(total_means)],
                   error_kw=dict(lw=1.5, capsize=5, capthick=1.5))
    add_bar_labels(ax1, bars1, total_means, '.0f', 0.05, 'black')
    ax1.set_title(t['total_latency'], fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel(t['time_ms'], fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(opt_labels, fontsize=8)  # Proste numery, bez rotacji
    ax1.grid(True, alpha=0.2, axis='y')
    if total_maxs and max(total_maxs) > 1000:
        ax1.get_yaxis().set_major_formatter(FuncFormatter(lambda v, p: human_format(v)))
    ax1.set_facecolor('#f8f9fa')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#dddddd')
    
    # 2. Latency Breakdown (Stacked Bar)
    prompt_means, generation_means, loading_means = [], [], []
    for _, data in sorted_opts:
        prompt_means.append(np.mean(data['prompt_eval_times']) if data['prompt_eval_times'] else 0)
        generation_means.append(np.mean(data['generation_times']) if data['generation_times'] else 0)
        loading_means.append(np.mean(data['loading_times']) if data['loading_times'] else 0)
    width = 0.85
    loading_bars = ax2.bar(x, loading_means, width, label=t['model_loading'], color='#e74c3c', alpha=0.8)
    prompt_bars = ax2.bar(x, prompt_means, width, bottom=loading_means, label=t['prompt_eval'], color='#3498db', alpha=0.8)
    gen_bars = ax2.bar(x, generation_means, width, bottom=np.array(loading_means) + np.array(prompt_means), label=t['token_generation'], color='#2ecc71', alpha=0.8)
    # annotate percentages - dynamically sized
    def annotate_pct(bars, pct_vals):
        # Skip percentages if too many optimizations
        if len(sorted_opts) > 10:
            return
            
        # Adjust font size based on number of optimizations
        fontsize = 6
        if len(sorted_opts) > 6:
            fontsize = 5
        elif len(sorted_opts) > 8:
            fontsize = 5
            
        for bar, pct in zip(bars, pct_vals):
            if pct > 5:
                h = bar.get_height(); ax2.text(bar.get_x()+bar.get_width()/2., bar.get_y()+h/2, f'{pct:.0f}%', ha='center', va='center', color='white' if pct>20 else 'black', fontsize=fontsize, fontweight='bold')
    total_times_arr = np.maximum(np.array(total_means), 1)  # avoid div by 0
    annotate_pct(loading_bars, (np.array(loading_means)/total_times_arr)*100)
    annotate_pct(prompt_bars, (np.array(prompt_means)/total_times_arr)*100)
    annotate_pct(gen_bars, (np.array(generation_means)/total_times_arr)*100)
    ax2.set_title(t['latency_breakdown'], fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel(t['time_ms'], fontweight='bold')
    ax2.set_xticks(x); ax2.set_xticklabels(opt_labels, fontsize=8)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True, framealpha=0.9)
    ax2.grid(True, alpha=0.2, axis='y'); ax2.set_facecolor('#f8f9fa')
    for spine in ax2.spines.values(): spine.set_edgecolor('#dddddd')
    
    # 3. Throughput Comparison
    throughput_means, throughput_stds, throughput_mins, throughput_maxs = [], [], [], []
    for _, data in sorted_opts:
        th = data['throughput']
        throughput_means.append(np.mean(th) if th else 0)
        throughput_stds.append(np.std(th) if th else 0)
        throughput_mins.append(np.min(th) if th else 0)
        throughput_maxs.append(np.max(th) if th else 0)
    bars3 = ax3.bar(x, throughput_means, color=colors, alpha=0.8,
                   yerr=[np.array(throughput_means) - np.array(throughput_mins),
                         np.array(throughput_maxs) - np.array(throughput_means)],
                   error_kw=dict(lw=1.5, capsize=5, capthick=1.5))
    add_bar_labels(ax3, bars3, throughput_means, '.1f', 0.05, 'black')
    ax3.set_title(t['throughput'], fontsize=14, fontweight='bold', pad=15)
    ax3.set_ylabel(t['tokens_per_second'], fontweight='bold')
    ax3.set_xticks(x); ax3.set_xticklabels(opt_labels, fontsize=8)
    ax3.grid(True, alpha=0.2, axis='y'); ax3.set_facecolor('#f8f9fa')
    for spine in ax3.spines.values(): spine.set_edgecolor('#dddddd')
    
    # 4. Optimization Reference Table - dynamically sized
    ax4.axis('off')
    
    # Przygotuj dane dla tabeli referencyjnej (numer -> nazwa)
    ref_table_data = []
    for i, (_, data) in enumerate(sorted_opts):
        # Użyj pełnej nazwy bez obcinania z oryginalnych danych
        optimization = data.get('original_optimization', {})
        if not optimization:
            full_name = 'Baseline'
        else:
            # Generuj pełną nazwę bez ograniczenia długości
            tokens = []
            for k, v in sorted(optimization.items(), key=lambda kv: str(kv[0])):
                key_norm = str(k).strip()
                if key_norm.startswith('--'):
                    key_norm = key_norm[2:]
                key_norm = key_norm.replace('_', '-').lower()
                if v is None or v is True or v == '':
                    tokens.append(f"{key_norm}")
                else:
                    tokens.append(f"{key_norm}={v}")
            
            def prettify(tok: str) -> str:
                if '=' in tok:
                    k, v = tok.split('=', 1)
                    return f"{k.replace('-', ' ').title()}: {v}"
                return tok.replace('-', ' ').title()
            
            full_name = ' + '.join(prettify(t) for t in tokens)  # BEZ [:25] !
        
        ref_table_data.append([f"{i+1}", full_name])
    
    # Adjust table parameters based on number of optimizations
    num_opts = len(ref_table_data)
    if num_opts <= 6:
        fontsize = 7
        row_height = 1.5
    elif num_opts <= 10:
        fontsize = 6
        row_height = 1.3
    elif num_opts <= 15:
        fontsize = 5
        row_height = 1.1
    else:
        fontsize = 4
        row_height = 0.9
    
    # Stwórz tabelę referencyjną z lepszym pozycjonowaniem
    ref_table = ax4.table(cellText=ref_table_data, 
                         colLabels=[t['number_col'], t['optimization_col']], 
                         cellLoc='left', 
                         loc='upper left',
                         colWidths=[0.08, 0.92],
                         bbox=[0, 0, 1, 1])  # Pełny obszar axes
    
    # Stylizuj tabelę referencyjną
    ref_table.auto_set_font_size(False)
    ref_table.set_fontsize(fontsize)
    ref_table.scale(1.0, row_height)
    
    # Ustaw granice axes żeby nic nie było obcięte
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    
    # Stylizuj nagłówki
    ref_table[(0, 0)].set_text_props(weight='bold', color='white')
    ref_table[(0, 0)].set_facecolor('#2c3e50')
    ref_table[(0, 1)].set_text_props(weight='bold', color='white')
    ref_table[(0, 1)].set_facecolor('#2c3e50')
    
    # Koloruj wiersze na przemian i ustaw kolor tekstu
    for i in range(1, len(ref_table_data) + 1):
        bg_color = '#f8f9fa' if i % 2 == 0 else '#ffffff'
        ref_table[(i, 0)].set_facecolor(bg_color)
        ref_table[(i, 1)].set_facecolor(bg_color)
        # Ustaw czarny tekst dla czytelności
        ref_table[(i, 0)].set_text_props(color='black')
        ref_table[(i, 1)].set_text_props(color='black')
    
    ax4.set_title(t['optimization_reference'], fontsize=14, fontweight='bold', pad=15)
    
    # Layout and save - lepsze spacing
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    safe_model_name = re.sub(r'[^\w\-]', '_', model_name)[:50]
    plot_path = os.path.join(output_dir, f'{output_file_name}_{safe_model_name}_{language}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    # Export JSON
    data_export = {
        'model': model_name,
        'timestamp': plotting_session_timestamp,
        'optimizations': {k: {
            'name': v['meta']['name'],
            'latency_ms': {
                'mean': float(np.mean(v['total_times'])) if v['total_times'] else 0,
                'prompt_eval': float(np.mean(v['prompt_eval_times'])) if v['prompt_eval_times'] else 0,
                'generation': float(np.mean(v['generation_times'])) if v['generation_times'] else 0,
                'loading': float(np.mean(v['loading_times'])) if v['loading_times'] else 0
            },
            'throughput_tokens_per_sec': float(np.mean(v['throughput'])) if v['throughput'] else 0
        } for k, v in optimization_data.items()}
    }
    with open(plot_path.replace('.png', '.json'), 'w') as f:
        json.dump(data_export, f, indent=2)
    plt.close()
    print(f" Optimization comparison plot ({language}) saved: {plot_path}")
    return plot_path


def plot_optimization_comparison(
   session_data,
    model_name,
    plotting_session_timestamp,
    metadata,
    output_dir,
    output_file_name,
):
    """
     Compare performance and quality metrics across different optimization methods for a single model.
    Generates both Polish and English versions.
    
    Args:
        session_data: Dict mapping optimisation key -> session dict (with 'optimisation' and 'rounds')
        model_name: Name of the model being evaluated
        plotting_session_timestamp: Timestamp for the plot title
        metadata: Optional metadata for the model
        output_dir: Directory to save the output plot
        output_file_name: Base name for the output file
        
    Returns:
        dict: Paths to generated plot files {'en': path, 'pl': path}, or None if generation failed
    """
    
    generated_plots = {}
    
    # Generate English version
    en_plot = _plot_optimization_comparison_single_language(
        session_data, model_name, plotting_session_timestamp, metadata,
        output_dir, output_file_name, 'en'
    )
    if en_plot:
        generated_plots['en'] = en_plot
    
    # Generate Polish version
    pl_plot = _plot_optimization_comparison_single_language(
        session_data, model_name, plotting_session_timestamp, metadata,
        output_dir, output_file_name, 'pl'
    )
    if pl_plot:
        generated_plots['pl'] = pl_plot
    
    return generated_plots if generated_plots else None


def plot_gartner_style_energy_analysis(session_data, model_name, plotting_session_timestamp, metadata, output_dir, output_file_name, language='en'):
    """
    Tworzy wykres w stylu Gartner Magic Quadrant z profilem energetycznym.
    
    Layout:
    - Główny wykres: Magic Quadrant (Performance vs Energy Efficiency)
    - Dodatkowe wykresy: Energy per Token, Energy per Second
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    
    # Pobierz dane
    rounds_data = session_data.get('rounds', [])
    if not rounds_data:
        print("❌ No rounds data available")
        return None
    
    # Przygotuj dane optymalizacji
    optimization_data = {}
    baseline_data = None
    
    for round_data in rounds_data:
        optimization = round_data.get('optimisation', {})
        opt_name = _format_optimization_name(optimization)
        
        if opt_name == 'Baseline':
            baseline_data = round_data
            continue
            
        if opt_name not in optimization_data:
            optimization_data[opt_name] = {
                'latencies': [],
                'throughputs': [],
                'cpu_powers': [],
                'gpu_powers': [],
                'memory_usage': [],
                'optimization': optimization
            }
        
        # Pobierz metryki
        latency_breakdown = round_data.get('latency_breakdown', {})
        total_ms = latency_breakdown.get('total_ms', 0)
        tokens = latency_breakdown.get('tokens', {})
        throughput = tokens.get('throughput_tokens_per_sec', 0)
        
        # Pobierz dane energetyczne
        resource_differences = latency_breakdown.get('resource_differences', {})
        if resource_differences:
            energy_diff = resource_differences.get('energy', {})
            cpu_power = energy_diff.get('cpu_power_delta_mw', 0) or 0
            gpu_power = energy_diff.get('gpu_power_delta_mw', 0) or 0
            memory_diff = resource_differences.get('memory', {})
            memory_usage = memory_diff.get('ram_delta_gb', 0)
        else:
            cpu_power = gpu_power = memory_usage = 0
        
        optimization_data[opt_name]['latencies'].append(total_ms)
        optimization_data[opt_name]['throughputs'].append(throughput)
        optimization_data[opt_name]['cpu_powers'].append(cpu_power)
        optimization_data[opt_name]['gpu_powers'].append(gpu_power)
        optimization_data[opt_name]['memory_usage'].append(memory_usage)
    
    if not optimization_data:
        print("❌ No optimization data available")
        return None
    
    # Oblicz średnie i metryki energetyczne
    energy_metrics = {}
    for opt_name, data in optimization_data.items():
        avg_latency = np.mean(data['latencies'])
        avg_throughput = np.mean(data['throughputs'])
        avg_cpu_power = np.mean(data['cpu_powers'])
        avg_gpu_power = np.mean(data['gpu_powers'])
        avg_memory = np.mean(data['memory_usage'])
        
        # Oblicz metryki energetyczne
        total_power = avg_cpu_power + avg_gpu_power  # mW
        energy_per_token = total_power / max(avg_throughput, 0.1) if avg_throughput > 0 else float('inf')  # mW per token/s
        energy_per_second = total_power  # mW
        
        # Performance score (wyższa throughput, niższa latencja = lepsze)
        performance_score = avg_throughput / max(avg_latency / 1000, 0.1)  # tokens/s per second of latency
        
        # Energy efficiency score (wyższa throughput, niższa energia = lepsze)
        energy_efficiency = avg_throughput / max(total_power, 1)  # tokens/s per mW
        
        energy_metrics[opt_name] = {
            'avg_latency': avg_latency,
            'avg_throughput': avg_throughput,
            'total_power': total_power,
            'energy_per_token': energy_per_token,
            'energy_per_second': energy_per_second,
            'performance_score': performance_score,
            'energy_efficiency': energy_efficiency,
            'cpu_power': avg_cpu_power,
            'gpu_power': avg_gpu_power,
            'memory_usage': avg_memory
        }
    
    # Stwórz wykres 2x2
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Tytuł główny
    fig.suptitle(f'Gartner-Style Energy Analysis\n{model_name}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Magic Quadrant: Performance vs Energy Efficiency
    ax1.set_title('Magic Quadrant: Performance vs Energy Efficiency', fontweight='bold')
    
    # Oblicz granice kwadrantów
    perf_scores = [data['performance_score'] for data in energy_metrics.values()]
    eff_scores = [data['energy_efficiency'] for data in energy_metrics.values()]
    
    perf_median = np.median(perf_scores)
    eff_median = np.median(eff_scores)
    
    # Narysuj kwadranty
    ax1.axhline(y=eff_median, color='gray', linestyle='--', alpha=0.5, label='Energy Efficiency Median')
    ax1.axvline(x=perf_median, color='gray', linestyle='--', alpha=0.5, label='Performance Median')
    
    # Dodaj etykiety kwadrantów
    ax1.text(0.02, 0.98, 'Challengers\n(High Performance\nLow Efficiency)', 
             transform=ax1.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax1.text(0.98, 0.98, 'Leaders\n(High Performance\nHigh Efficiency)', 
             transform=ax1.transAxes, fontsize=8, verticalalignment='top', ha='right',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    ax1.text(0.02, 0.02, 'Niche Players\n(Low Performance\nLow Efficiency)', 
             transform=ax1.transAxes, fontsize=8, verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
    ax1.text(0.98, 0.02, 'Visionaries\n(Low Performance\nHigh Efficiency)', 
             transform=ax1.transAxes, fontsize=8, verticalalignment='bottom', ha='right',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    # Narysuj punkty z kolorami według grup optymalizacji
    for opt_name, data in energy_metrics.items():
        color = _get_optimization_group_color(opt_name)
        size = 100 + data['avg_throughput'] * 10  # Rozmiar zależy od throughput
        
        ax1.scatter(data['performance_score'], data['energy_efficiency'], 
                   c=[color], s=size, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Dodaj etykiety (skrócone nazwy)
        short_name = _get_short_optimization_name(opt_name)
        ax1.annotate(short_name, (data['performance_score'], data['energy_efficiency']),
                    xytext=(5, 5), textcoords='offset points', fontsize=7)
    
    ax1.set_xlabel('Performance Score (Throughput/Latency)')
    ax1.set_ylabel('Energy Efficiency (Throughput/Power)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Energy per Token vs Latency
    ax2.set_title('Energy per Token vs Latency', fontweight='bold')
    
    for opt_name, data in energy_metrics.items():
        color = _get_optimization_group_color(opt_name)
        size = 100 + data['avg_throughput'] * 10
        
        ax2.scatter(data['avg_latency'], data['energy_per_token'], 
                   c=[color], s=size, alpha=0.7, edgecolors='black', linewidth=1)
        
        short_name = _get_short_optimization_name(opt_name)
        ax2.annotate(short_name, (data['avg_latency'], data['energy_per_token']),
                    xytext=(5, 5), textcoords='offset points', fontsize=7)
    
    ax2.set_xlabel('Average Latency (ms)')
    ax2.set_ylabel('Energy per Token (mW/token/s)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Power Breakdown (Stacked Bar)
    ax3.set_title('Power Breakdown by Optimization', fontweight='bold')
    
    opt_names = list(energy_metrics.keys())
    cpu_powers = [energy_metrics[opt]['cpu_power'] for opt in opt_names]
    gpu_powers = [energy_metrics[opt]['gpu_power'] for opt in opt_names]
    
    x_pos = np.arange(len(opt_names))
    width = 0.6
    
    ax3.bar(x_pos, cpu_powers, width, label='CPU Power (mW)', alpha=0.8)
    ax3.bar(x_pos, gpu_powers, width, bottom=cpu_powers, label='GPU Power (mW)', alpha=0.8)
    
    ax3.set_xlabel('Optimization')
    ax3.set_ylabel('Power (mW)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([_get_short_optimization_name(name) for name in opt_names], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Energy Efficiency Ranking
    ax4.set_title('Energy Efficiency Ranking', fontweight='bold')
    
    # Posortuj według energy efficiency
    sorted_opts = sorted(energy_metrics.items(), key=lambda x: x[1]['energy_efficiency'], reverse=True)
    opt_names_sorted = [opt[0] for opt in sorted_opts]
    eff_scores_sorted = [opt[1]['energy_efficiency'] for opt in sorted_opts]
    
    colors = [_get_optimization_group_color(name) for name in opt_names_sorted]
    
    bars = ax4.barh(range(len(opt_names_sorted)), eff_scores_sorted, color=colors, alpha=0.7)
    ax4.set_yticks(range(len(opt_names_sorted)))
    ax4.set_yticklabels([_get_short_optimization_name(name) for name in opt_names_sorted])
    ax4.set_xlabel('Energy Efficiency Score')
    ax4.grid(True, alpha=0.3)
    
    # Dodaj wartości na słupkach
    for i, (bar, score) in enumerate(zip(bars, eff_scores_sorted)):
        ax4.text(bar.get_width() + max(eff_scores_sorted) * 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.2f}', va='center', fontsize=8)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Dodaj timestamp
    fig.text(0.99, 0.01, f'Generated: {plotting_session_timestamp}', 
            ha='right', fontsize=8, style='italic')
    
    # Zapisz wykres
    plot_filename = f"{output_file_name}_gartner_energy_{language}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Gartner-style energy analysis plot saved: {plot_path}")
    return plot_path


def _get_optimization_group_color(opt_name):
    """Zwraca kolor na podstawie grupy optymalizacji"""
    import matplotlib.pyplot as plt
    
    # Definicja kolorów dla grup
    group_colors = {
        'Cache Optimization': '#1f77b4',      # niebieski
        'Context & Attention': '#ff7f0e',     # pomarańczowy  
        'Batch Processing': '#2ca02c',        # zielony
        'Speculative Decoding': '#d62728',    # czerwony
        'Memory & Storage': '#9467bd',        # fioletowy
        'Hardware Optimization': '#8c564b',   # brązowy
        'Other': '#7f7f7f'                    # szary
    }
    
    # Wykryj grupę optymalizacji
    from edge_llm_lab.evaluation.referenced_evaluator import EvalModelsReferenced
    group_name, _ = EvalModelsReferenced.categorize_optimization(opt_name)
    
    return group_colors.get(group_name, '#7f7f7f')


def _get_short_optimization_name(opt_name):
    """Zwraca skróconą nazwę optymalizacji"""
    if opt_name == 'Baseline':
        return 'Baseline'
    
    # Mapowanie długich nazw na krótkie
    short_names = {
        'Cache Type K: f16': 'K:f16',
        'Cache Type V: f16': 'V:f16', 
        'Cache Type K: f16 + Cache Type V: f16': 'KV:f16',
        'Flash Attn': 'Flash',
        'Cont Batching': 'Batching',
        'Draft Max: 2': 'Draft:2',
        'Draft Max: 3': 'Draft:3',
        'Draft Max: 4': 'Draft:4',
        'Threads: 2': 'T:2',
        'Threads: 4': 'T:4', 
        'Threads: 8': 'T:8',
        'Batch Size: 16': 'B:16',
        'Batch Size: 32': 'B:32',
        'Ubatch Size: 16': 'U:16',
        'Ubatch Size: 32': 'U:32',
        'No Mmap': 'NoMMap',
        'No Kv Offload': 'NoKV'
    }
    
    return short_names.get(opt_name, opt_name[:10])  # Fallback: pierwsze 10 znaków
