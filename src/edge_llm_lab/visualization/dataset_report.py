"""Offline SQL aggregation; transcripts are never read to draw a figure."""
import csv
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean

from edge_llm_lab.storage.json_log import atomic_write_json

_CONFIG_COLORS = ('#4b958b', '#376b8c', '#b47a43', '#796394')


def summarize(database, output):
    """Read a capture snapshot and export aggregate JSON plus a trial table."""
    summary, columns, rows = _read_summary(database)
    _export_summary(summary, columns, rows, output)
    return summary


def _read_summary(database):
    """Read one consistent SQLite snapshot without creating report files."""
    with sqlite3.connect(f'{Path(database).resolve().as_uri()}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute('BEGIN')
        manifest = json.loads(db.execute('SELECT value FROM manifest WHERE id=1').fetchone()[0])
        trials = [dict(r) for r in db.execute('SELECT config_id, COUNT(*) AS attempts, COUNT(DISTINCT case_id) AS cases, SUM(status="error") AS errors, AVG(elapsed_ms) AS mean_trial_ms FROM trials GROUP BY config_id')]
        # First average rounds within a trial, then repeats within a case.
        # Cases, not turns or repeated seeds, receive equal weight.
        scores = [dict(r) for r in db.execute('''
            SELECT config_id, case_id, AVG(score) AS score FROM (
                SELECT t.id, t.config_id, t.case_id, AVG(m.value) AS score
                FROM trials t JOIN metrics m ON t.id=m.trial_id
                WHERE t.status='ok' AND m.name GLOB 'rounds.*.metrics.exact_match.score'
                GROUP BY t.id) GROUP BY config_id, case_id''')]
        for trial in trials:
            values = [r['score'] for r in scores if r['config_id'] == trial['config_id']]
            trial['mean_case_exact_match'] = sum(values) / len(values) if values else None
            trial['scored_cases'] = len(values)
        latencies = [dict(row) for row in db.execute("""
            SELECT t.id AS trial_id, t.case_id, t.config_id, m.name, m.value AS wall_ms
            FROM trials t JOIN metrics m ON m.trial_id=t.id
            WHERE m.name GLOB 'rounds.*.latency_breakdown.wall_ms'
            ORDER BY t.rowid, m.name
        """)]
        coverage = []
        expected_per_config = manifest['case_count'] * len(manifest['seeds'])
        for config_id in range(len(manifest['optimisations'])):
            counts = db.execute("""
                SELECT COUNT(*), COALESCE(SUM(status='error'),0),
                    COALESCE(SUM(status='ok' AND NOT EXISTS (
                        SELECT 1 FROM metrics m WHERE m.trial_id=t.id
                        AND m.name GLOB 'rounds.*.metrics.exact_match.score')),0)
                FROM trials t WHERE config_id=?
            """, (config_id,)).fetchone()
            coverage.append({'config_id': config_id, 'attempted_trials': counts[0],
                             'error_trials': counts[1], 'unscored_trials': counts[2],
                             'scored_trials': counts[0] - counts[1] - counts[2],
                             'missing_trials': max(0, expected_per_config - counts[0])})
        result = {'manifest': manifest, 'configurations': trials,
                  'case_ids': [row[0] for row in db.execute('SELECT DISTINCT case_id FROM trials ORDER BY case_id')],
                  'case_scores': scores, 'latencies': latencies, 'coverage': coverage,
                  'expected_trials': manifest['case_count'] * len(manifest['seeds']) * len(manifest['optimisations']),
                  'note': 'Descriptive case means. Errors reported separately; missing scores are not zero. No confidence interval from a tiny demo.'}
        result['complete'] = sum(t['attempts'] for t in trials) == result['expected_trials']
        cursor = db.execute('SELECT id,case_id,repeat,seed,config_id,status,elapsed_ms,error_type FROM trials ORDER BY rowid')
        columns = [column[0] for column in cursor.description]
        rows = [tuple(row) for row in cursor]
    return result, columns, rows


def _export_summary(summary, columns, rows, output):
    """Export an already-read snapshot; callers decide whether it is publishable."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary, output / 'summary.json')
    with (output / 'trials.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def _validate_report_output(summary, output):
    """Reject incompatible report reuse before writing data or figures."""
    output = Path(output)
    if not summary['case_scores']:
        raise ValueError('No exact-match scores to plot; call summarize() to inspect coverage and errors')
    if _paired_request_data(summary) is None and any(
            (output / f'paired-request-cost.{extension}').exists() for extension in ('png', 'svg')):
        raise ValueError('Cannot regenerate existing paired-request-cost output from this summary; '
                         'use a fresh output directory to avoid mixing reports')


def render(database, output):
    """Aggregate a capture and render both research and diagnostic figures."""
    summary, columns, rows = _read_summary(database)
    _validate_report_output(summary, output)
    _export_summary(summary, columns, rows, output)
    return render_summary(summary, output)


def render_summary(summary, output):
    """Render an aggregate summary without needing its database or private traces.

    Existing ``case-scores`` and ``diagnostics`` outputs are retained. The
    ``case-outcomes`` figure aligns reference agreement and measured request
    latency by named case and configuration; ``request-cost`` also provides
    request timing as a separate figure. ``coverage`` separates unscored and
    failed trials. With exactly two measured configurations, a paired timing
    figure reports differences of case means. No uncertainty is inferred.
    Use a dedicated output directory for each summary. Existing paired figures
    that cannot be regenerated cause an error before any figure is overwritten.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    _validate_report_output(summary, output)
    output.mkdir(parents=True, exist_ok=True)
    render_diagnostics(summary, output)
    for name, builder in [('case-scores', build_case_scores_figure),
                          ('case-outcomes', build_case_outcomes_figure),
                          ('request-cost', build_request_cost_figure),
                          ('paired-request-cost', build_paired_request_cost_figure),
                          ('coverage', build_coverage_figure)]:
        fig = builder(summary)
        if fig is None:
            continue
        try:
            for extension in ('svg', 'png'):
                fig.savefig(output / f'{name}.{extension}', dpi=160)
        finally:
            plt.close(fig)
    return summary


def _configuration_label(summary, config_id):
    options = summary['manifest']['optimisations']
    config = options[config_id] if 0 <= config_id < len(options) else {}
    threads = config.get('--threads')
    if isinstance(threads, int) and not isinstance(threads, bool) and threads > 0:
        label = f'{threads} thread' + ('s' if threads != 1 else '')
        return label if set(config) == {'--threads'} else f'Config {config_id} ({label})'
    return f'Config {config_id}'


def case_outcomes(summary):
    """Align observed cases and configurations without replacing missing values.

    Scores are already case means in the capture summary. For request times,
    first average measured turns within each trial, then average trials within
    each case. This keeps unequal turn counts from weighting repeated trials
    differently. Missing turns or trials are not imputed. The coverage report
    remains necessary: older manifests do not contain names of unrun cases.
    """
    case_ids = list(dict.fromkeys(summary.get('case_ids', [])))
    for row in summary['case_scores'] + summary['latencies']:
        if row['case_id'] not in case_ids:
            case_ids.append(row['case_id'])
    config_ids = sorted(set(range(len(summary['manifest']['optimisations']))) |
                        {row['config_id'] for row in summary['coverage']} |
                        {row['config_id'] for row in summary['case_scores'] + summary['latencies']})
    scores = {}
    for row in summary['case_scores']:
        value = row['score']
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('Exact-match scores must be finite values between 0 and 1')
        scores[row['case_id'], row['config_id']] = value
    trials = defaultdict(list)
    for row in summary['latencies']:
        value = row['wall_ms']
        if value is None:
            continue
        if not math.isfinite(value) or value < 0:
            raise ValueError('Request wall times must be finite, non-negative measurements')
        trials[row['case_id'], row['config_id'], row['trial_id']].append(value / 1000)
    case_times = defaultdict(list)
    for (case_id, config_id, _), values in trials.items():
        case_times[case_id, config_id].append(mean(values))
    return {
        'case_ids': case_ids, 'config_ids': config_ids,
        'config_labels': [_configuration_label(summary, config) for config in config_ids],
        'scores': [[scores.get((case, config)) for config in config_ids] for case in case_ids],
        'request_seconds': [[mean(case_times[case, config]) if (case, config) in case_times else None
                             for config in config_ids] for case in case_ids],
    }


def build_case_scores_figure(summary):
    """Group configurations within named case rows rather than duplicating rows."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outcomes = case_outcomes(summary)
    cases, configs = outcomes['case_ids'], outcomes['config_ids']
    if not cases or not configs:
        raise ValueError('No named cases and configurations to plot')
    fig, ax = plt.subplots(figsize=(8, max(3.8, len(cases) * .25 + 1.6)), layout='constrained')
    height = .7 / len(configs)
    for column, label in enumerate(outcomes['config_labels']):
        offset = (column - (len(configs) - 1) / 2) * height
        positions, values = [], []
        for row, scores in enumerate(outcomes['scores']):
            value = scores[column]
            y = row + offset
            if value is None:
                ax.text(.02, y, 'Not scored', va='center', color='#646b73', fontsize=8)
                continue
            positions.append(y)
            values.append(value)
            ax.annotate(f'{value:.0%}', (value, y), xytext=(-6 if value > .15 else 6, 0),
                        textcoords='offset points', ha='right' if value > .15 else 'left',
                        va='center', color='white' if value > .15 else '#18232b', fontsize=8, zorder=4)
        ax.barh(positions, values, color=_CONFIG_COLORS[column % len(_CONFIG_COLORS)], height=height * .85,
                label=label, zorder=3)
    ax.set_yticks(range(len(cases)), cases)
    ax.set_xlim(0, 1)
    ax.set_ylim(len(cases) - .5, -.5)
    ax.grid(axis='x', color='#e6e8eb', linewidth=.7, zorder=0)
    ax.tick_params(axis='y', length=0, labelsize=9)
    kind = 'text' if summary['manifest'].get('metrics_profile') == 'text' else 'JSON'
    ax.set_xlabel(f'Exact {kind} match, mean across repeats and reference turns', fontsize=9)
    ax.set_title('Per-case agreement with frozen references', loc='left',
                 pad=35 if len(configs) > 1 else 14, fontsize=11)
    if len(configs) > 1:
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1), ncols=min(len(configs), 4), frameon=False)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    return fig


def build_case_outcomes_figure(summary):
    """Build a figure from public aggregate data; the caller owns its lifecycle."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    outcomes = case_outcomes(summary)
    cases, configs = outcomes['case_ids'], outcomes['config_ids']
    if not cases or not configs:
        raise ValueError('No named cases and configurations to plot')
    fig, (score_ax, latency_ax) = plt.subplots(
        1, 2, figsize=(10.2 + .5 * max(0, len(configs) - 2), max(3.8, len(cases) * .25 + 1.8)),
        sharey=True, gridspec_kw={'width_ratios': [max(1.6, len(configs)), 4]},
        layout='constrained')
    score_values = np.ma.masked_invalid(np.asarray(outcomes['scores'], dtype=float))
    cmap = LinearSegmentedColormap.from_list('agreement', ['#e8aa84', '#f4efe3', '#4b958b']).with_extremes(bad='#e6e8eb')
    score_ax.imshow(score_values, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    for row, values in enumerate(outcomes['scores']):
        for column, value in enumerate(values):
            score_ax.text(column, row, 'Not scored' if value is None else f'{value:.0%}',
                          ha='center', va='center', fontsize=9, color='#18232b')
    score_ax.set_yticks(range(len(cases)), cases)
    score_ax.set_xticks(range(len(configs)), outcomes['config_labels'])
    kind = 'text' if summary['manifest'].get('metrics_profile') == 'text' else 'JSON'
    score_ax.set_title(f'Exact {kind} match', loc='left', fontsize=11, pad=12)
    score_ax.set_xlabel('0% = disagreement; 100% = agreement', fontsize=9)
    score_ax.tick_params(length=0, pad=8, labelsize=9)
    score_ax.spines[:].set_visible(False)

    _plot_request_times(latency_ax, outcomes)
    model = summary['manifest'].get('model')
    fig.suptitle('Case-level evaluation' + (f' · {model}' if model else ''), fontsize=12)
    fig.supxlabel('Descriptive means: turns within each trial, then trials within each case.\n'
                  'Request timing excludes model startup; no uncertainty interval is estimated.',
                  fontsize=8, color='#535b63')
    return fig


def _plot_request_times(latency_ax, outcomes):
    """Use the same measured values and missingness in both figure layouts."""
    cases, configs = outcomes['case_ids'], outcomes['config_ids']
    for column, config in enumerate(configs):
        config_label = outcomes['config_labels'][column]
        offset = (column - (len(configs) - 1) / 2) * .55 / max(1, len(configs) - 1)
        points = [(value[column], row + offset) for row, value in enumerate(outcomes['request_seconds'])
                  if value[column] is not None]
        if points:
            x, y = zip(*points)
            latency_ax.scatter(x, y, color=_CONFIG_COLORS[column % len(_CONFIG_COLORS)],
                               s=28, label=config_label, zorder=3)
            if len(configs) == 1:
                for value, row in points:
                    latency_ax.annotate(f'{value:.2f}', (value, row), xytext=(7, 0),
                                        textcoords='offset points', va='center', fontsize=9)
        for row, values in enumerate(outcomes['request_seconds']):
            if values[column] is None:
                label = 'Not recorded' if len(configs) == 1 else f'{config_label}: not recorded'
                latency_ax.text(.02, row + offset, label, va='center', fontsize=8,
                                color='#646b73', transform=latency_ax.get_yaxis_transform())
    latency_ax.set_title('Measured request time', loc='left', fontsize=11, pad=12)
    latency_ax.set_xlabel('Mean request wall time (seconds)', fontsize=9)
    latency_ax.set_yticks(range(len(cases)), cases)
    latency_ax.set_xlim(left=0)
    maximum = max((value for row in outcomes['request_seconds'] for value in row if value is not None), default=0)
    latency_ax.set_xlim(right=max(maximum * 1.22, 1))
    latency_ax.set_ylim(len(cases) - .5, -.5)
    latency_ax.grid(axis='x', color='#e6e8eb', linewidth=.7, zorder=0)
    latency_ax.tick_params(axis='y', length=0, labelsize=9)
    latency_ax.spines[['top', 'right', 'left']].set_visible(False)
    if len(configs) > 1 and latency_ax.collections:
        latency_ax.legend(loc='lower right', bbox_to_anchor=(1, 1), ncols=min(len(configs), 4), frameon=False)


def build_request_cost_figure(summary):
    """Build a standalone, named-case request-time figure from aggregate data.

    Times are mean HTTP request durations in seconds, averaged within a trial
    before averaging trials. They are not throughput, token-normalized costs,
    energy measurements, or end-to-end trial durations. Missing times stay absent.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outcomes = case_outcomes(summary)
    if not outcomes['case_ids'] or not outcomes['config_ids']:
        raise ValueError('No named cases and configurations to plot')
    fig, ax = plt.subplots(figsize=(8, max(3.8, len(outcomes['case_ids']) * .25 + 1.5)),
                           layout='constrained')
    _plot_request_times(ax, outcomes)
    ax.set_title('Measured request time by case', loc='left', pad=14)
    fig.supxlabel('Descriptive means: turns within a trial, then trials within a case.\n'
                  'Request timing excludes model startup; no uncertainty interval is estimated.',
                  fontsize=8, color='#535b63')
    return fig


def _paired_request_data(summary):
    outcomes = case_outcomes(summary)
    if len(outcomes['config_ids']) != 2:
        return None
    pairs = [(case, values[1] - values[0])
             for case, values in zip(outcomes['case_ids'], outcomes['request_seconds'])
             if all(value is not None for value in values)]
    return (outcomes['config_labels'], pairs) if pairs else None


def build_paired_request_cost_figure(summary):
    """Compare two configurations by named case, using their measured case means.

    Return ``None`` unless exactly two configurations have at least one shared
    measured case. Pairing is by case, not necessarily by seed; the plot makes
    no inferential claim. The second configuration minus the first defines the
    sign, with negative values indicating a faster second configuration.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = _paired_request_data(summary)
    if data is None:
        return None
    labels, pairs = data
    cases, differences = zip(*pairs)
    baseline, comparison = labels
    fig, ax = plt.subplots(figsize=(8, max(3.8, len(cases) * .25 + 1.8)), layout='constrained')
    ax.barh(range(len(cases)), differences, height=.58,
            color=['#4b958b' if value < 0 else '#c78b62' for value in differences], zorder=3)
    ax.axvline(0, color='#535b63', linewidth=1, zorder=2)
    for row, value in enumerate(differences):
        ax.annotate(f'{value:+.2f}', (value, row), xytext=(-5 if value < 0 else 5, 0),
                    textcoords='offset points', ha='right' if value < 0 else 'left',
                    va='center', fontsize=8, color='#18232b')
    maximum = max(max(abs(value) for value in differences) * 1.3, 1)
    ax.set_xlim(-maximum, maximum)
    ax.set_yticks(range(len(cases)), cases)
    ax.set_ylim(len(cases) - .5, -.5)
    ax.tick_params(axis='y', length=0, labelsize=9)
    ax.set_xlabel(f'{comparison} − {baseline}: mean request wall time (seconds)', fontsize=9)
    ax.set_title('Paired request-time difference by case', loc='left', pad=14, fontsize=11)
    ax.grid(axis='x', color='#e6e8eb', linewidth=.7, zorder=0)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    fig.supxlabel(f'Negative: {comparison} faster. Positive: {comparison} slower.\n'
                  'Cases measured in both settings only; differences of case means, no uncertainty interval.',
                  fontsize=8, color='#535b63')
    return fig


def _plot_coverage(ax, summary):
    from matplotlib.ticker import MaxNLocator
    bottom = [0] * len(summary['coverage'])
    for key, label, color in [('scored_trials', 'Scored', '#376b8c'),
                              ('unscored_trials', 'Unscored', '#d4a84b'),
                              ('error_trials', 'Error', '#b44b4b'),
                              ('missing_trials', 'Not run', '#cccccc')]:
        values = [row[key] for row in summary['coverage']]
        ax.bar(range(len(values)), values, bottom=bottom, label=label, color=color, width=.55)
        for index, value in enumerate(values):
            if value:
                ax.text(index, bottom[index] + value / 2, str(value), ha='center', va='center',
                        fontsize=9, color='white' if key in ('scored_trials', 'error_trials') else '#18232b')
        bottom = [a + b for a, b in zip(bottom, values)]
    ax.set_xticks(range(len(bottom)),
                   [_configuration_label(summary, row['config_id']) for row in summary['coverage']])
    ax.set(ylabel='Trials', title='Evaluation coverage')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
    ax.spines[['top', 'right']].set_visible(False)


def build_coverage_figure(summary):
    """Separate attempted, scored, unscored, failed, and missing trial counts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.8), layout='constrained')
    _plot_coverage(ax, summary)
    fig.supxlabel('Scored trials can agree or disagree with references. Missing scores are not failures.\n'
                  'Counts describe completion, not model capability or statistical confidence.',
                  fontsize=8, color='#535b63')
    return fig


def render_diagnostics(summary, output):
    """Show measured request latency and coverage without reading transcripts."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    try:
        latency_ax, coverage_ax = axes
        latencies = summary['latencies']
        for config in summary['coverage']:
            config_id = config['config_id']
            values = [r['wall_ms'] for r in latencies if r['config_id'] == config_id]
            if values:
                latency_ax.plot(range(1, len(values) + 1), values, 'o',
                                label=f'{_configuration_label(summary, config_id)} (n={len(values)})', alpha=.7)
        latency_ax.set(xlabel='Captured request index', ylabel='Request wall time (ms)',
                       title='Individual request measurements')
        latency_ax.set_ylim(bottom=0)
        latency_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if latencies:
            maximum = max(sum(r['config_id'] == c['config_id'] for r in latencies)
                          for c in summary['coverage'])
            latency_ax.set_xlim(.5, maximum + .5)
            latency_ax.legend()
        else:
            latency_ax.text(.5, .5, 'Request wall time not recorded',
                            ha='center', va='center', transform=latency_ax.transAxes)
        _plot_coverage(coverage_ax, summary)
        for ax in axes:
            ax.spines[['top', 'right']].set_visible(False)
        for extension in ('svg', 'png'):
            fig.savefig(Path(output) / f'diagnostics.{extension}', dpi=160)
    finally:
        plt.close(fig)
