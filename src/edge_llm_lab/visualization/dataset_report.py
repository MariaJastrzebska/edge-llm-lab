"""Offline SQL aggregation; transcripts are never read to draw a figure."""
import csv
import json
import sqlite3
from pathlib import Path

from edge_llm_lab.storage.json_log import atomic_write_json


def summarize(database, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f'{Path(database).resolve().as_uri()}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
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
                  'case_scores': scores, 'latencies': latencies, 'coverage': coverage,
                  'expected_trials': manifest['case_count'] * len(manifest['seeds']) * len(manifest['optimisations']),
                  'note': 'Descriptive case means. Errors reported separately; missing scores are not zero. No confidence interval from a tiny demo.'}
        result['complete'] = sum(t['attempts'] for t in trials) == result['expected_trials']
        atomic_write_json(result, output / 'summary.json')
        with (output / 'trials.csv').open('w', newline='') as stream:
            cursor = db.execute('SELECT id,case_id,repeat,seed,config_id,status,elapsed_ms,error_type FROM trials ORDER BY rowid')
            writer = csv.writer(stream)
            writer.writerow([column[0] for column in cursor.description])
            writer.writerows(cursor)
    return result


def render(database, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = summarize(database, output)
    cases = summary['case_scores']
    if not cases:
        raise ValueError('No exact-match scores to plot; inspect summary and errors')
    fig, ax = plt.subplots(figsize=(8, max(2.8, len(cases) * .42)), layout='constrained')
    ax.barh(range(len(cases)), [r['score'] for r in cases], color='#376b8c')
    ax.set_yticks(range(len(cases)), [f"{r['case_id']} / config {r['config_id']}" for r in cases])
    ax.set_xlim(0, 1)
    kind = 'text' if summary['manifest'].get('metrics_profile') == 'text' else 'JSON'
    ax.set_xlabel(f'Exact {kind} match, mean across repeats and reference turns')
    ax.set_title('Per-case agreement with frozen references')
    ax.spines[['top', 'right']].set_visible(False)
    for extension in ('svg', 'png'):
        fig.savefig(Path(output) / f'case-scores.{extension}', dpi=160)
    plt.close(fig)
    render_diagnostics(summary, output)
    return summary


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
                                label=f'Config {config_id} (n={len(values)})', alpha=.7)
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
        bottom = [0] * len(summary['coverage'])
        for key, label, color in [('scored_trials', 'Scored', '#376b8c'),
                                  ('unscored_trials', 'Unscored', '#d4a84b'),
                                  ('error_trials', 'Error', '#b44b4b'),
                                  ('missing_trials', 'Not run', '#cccccc')]:
            values = [r[key] for r in summary['coverage']]
            coverage_ax.bar(range(len(values)), values, bottom=bottom, label=label, color=color)
            bottom = [a + b for a, b in zip(bottom, values)]
        coverage_ax.set_xticks(range(len(bottom)),
                               [f"Config {r['config_id']}" for r in summary['coverage']])
        coverage_ax.set(ylabel='Trials', title='Evaluation coverage')
        coverage_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        coverage_ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
        for ax in axes:
            ax.spines[['top', 'right']].set_visible(False)
        for extension in ('svg', 'png'):
            fig.savefig(Path(output) / f'diagnostics.{extension}', dpi=160)
    finally:
        plt.close(fig)
