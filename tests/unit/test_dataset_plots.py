"""Plot contracts check scientific meaning before testing image rendering."""
import hashlib
from pathlib import Path
import sqlite3
import xml.etree.ElementTree as ET
import pytest
from edge_llm_lab.storage.trials import TrialStore
from edge_llm_lab.visualization.dataset_report import render, summarize


def outcome_summary():
    return {
        'manifest': {'metrics_profile': 'structured', 'optimisations': [{}, {}],
                     'seeds': [11, 12], 'case_count': 3, 'model': 'example-model'},
        'case_ids': ['record', 'privacy', 'not-scored'],
        'case_scores': [{'config_id': 0, 'case_id': 'record', 'score': 0.5},
                        {'config_id': 1, 'case_id': 'record', 'score': 1},
                        {'config_id': 0, 'case_id': 'privacy', 'score': 0}],
        'latencies': [
            {'config_id': 0, 'case_id': 'record', 'trial_id': 'a', 'wall_ms': 1000},
            {'config_id': 0, 'case_id': 'record', 'trial_id': 'a', 'wall_ms': 3000},
            {'config_id': 0, 'case_id': 'record', 'trial_id': 'b', 'wall_ms': 8000},
            {'config_id': 1, 'case_id': 'privacy', 'trial_id': 'c', 'wall_ms': 2500},
        ],
        'coverage': [{'config_id': i, 'scored_trials': 1, 'unscored_trials': 1,
                      'error_trials': 1, 'missing_trials': 3} for i in range(2)],
        'expected_trials': 12,
    }


def test_case_outcomes_align_cases_and_configs_without_zero_imputation():
    from edge_llm_lab.visualization.dataset_report import case_outcomes
    rows = case_outcomes(outcome_summary())
    assert rows['case_ids'] == ['record', 'privacy', 'not-scored']
    assert rows['config_ids'] == [0, 1]
    assert rows['scores'] == [[0.5, 1], [0, None], [None, None]]
    # Average turns within each trial, then trials; not (1 + 3 + 8) / 3.
    assert rows['request_seconds'] == [[5.0, None], [None, 2.5], [None, None]]


def test_outcome_figure_exposes_missingness_and_measured_units():
    from edge_llm_lab.visualization.dataset_report import build_case_outcomes_figure
    import matplotlib.pyplot as plt
    fig = build_case_outcomes_figure(outcome_summary())
    try:
        score_ax, latency_ax = fig.axes[:2]
        values = score_ax.images[0].get_array()
        assert values[1, 0] == 0
        assert values.mask[1, 1]
        assert 'Not scored' in [text.get_text() for text in score_ax.texts]
        assert [tick.get_text() for tick in score_ax.get_yticklabels()] == [
            'record', 'privacy', 'not-scored']
        assert 'seconds' in latency_ax.get_xlabel()
        assert 'Request index' not in latency_ax.get_xlabel()
        assert score_ax.get_ylim() == latency_ax.get_ylim()
        positions = [collection.get_offsets() for collection in latency_ax.collections]
        assert [float(offset[0, 0]) for offset in positions] == [5.0, 2.5]
        assert latency_ax.get_xlim()[0] == 0
        assert any('not recorded' in text.get_text().lower() for text in latency_ax.texts)
    finally:
        plt.close(fig)


def test_render_summary_needs_no_capture_or_transcripts(tmp_path):
    from edge_llm_lab.visualization.dataset_report import render_summary
    import copy
    summary = outcome_summary()
    before = copy.deepcopy(summary)
    output = tmp_path / 'portable-report'
    assert render_summary(summary, output) is summary
    assert summary == before
    for name in ('case-scores', 'diagnostics', 'case-outcomes', 'request-cost', 'coverage'):
        ET.parse(output / f'{name}.svg')
        assert (output / f'{name}.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    import matplotlib.pyplot as plt
    assert not plt.get_fignums()


def test_summary_keeps_unscored_case_names(tmp_path):
    summary = summarize(capture(tmp_path, scored=False), tmp_path / 'report')
    assert summary['case_ids'] == ['a', 'b']


@pytest.mark.parametrize('field,value', [('score', -0.1), ('score', 1.1),
                                         ('score', float('nan')), ('wall_ms', -1),
                                         ('wall_ms', float('inf'))])
def test_case_outcomes_rejects_invalid_measurements(field, value):
    from edge_llm_lab.visualization.dataset_report import case_outcomes
    summary = outcome_summary()
    summary['case_scores' if field == 'score' else 'latencies'][0][field] = value
    with pytest.raises(ValueError, match='must be finite'):
        case_outcomes(summary)


def test_older_summary_infers_named_cases_and_keeps_absent_latency_missing():
    from edge_llm_lab.visualization.dataset_report import case_outcomes
    summary = outcome_summary()
    del summary['case_ids']
    summary['latencies'] = []
    outcomes = case_outcomes(summary)
    assert outcomes['case_ids'] == ['record', 'privacy']
    assert outcomes['request_seconds'] == [[None, None], [None, None]]


def test_standalone_request_plot_shares_trial_balanced_measurements():
    from edge_llm_lab.visualization.dataset_report import build_request_cost_figure
    import matplotlib.pyplot as plt
    fig = build_request_cost_figure(outcome_summary())
    try:
        ax = fig.axes[0]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == ['record', 'privacy', 'not-scored']
        assert 'seconds' in ax.get_xlabel()
        assert ax.collections[0].get_offsets()[0, 0] == 5.0
        assert ax.collections[1].get_offsets()[0, 0] == 2.5
        assert 'excludes model startup' in fig._supxlabel.get_text()
    finally:
        plt.close(fig)


def paired_summary():
    summary = outcome_summary()
    summary['manifest']['optimisations'] = [{'--threads': 1}, {'--threads': 2}]
    summary['latencies'].append({'config_id': 1, 'case_id': 'record',
                                 'trial_id': 'd', 'wall_ms': 3000})
    return summary


def test_thread_config_labels_are_shared_across_plot_types():
    from edge_llm_lab.visualization.dataset_report import (
        build_case_outcomes_figure, build_case_scores_figure, case_outcomes)
    import matplotlib.pyplot as plt
    summary = paired_summary()
    assert case_outcomes(summary)['config_labels'] == ['1 thread', '2 threads']
    assert case_outcomes(outcome_summary())['config_labels'] == ['Config 0', 'Config 1']
    fig = build_case_outcomes_figure(summary)
    bars = build_case_scores_figure(summary)
    try:
        assert [tick.get_text() for tick in fig.axes[0].get_xticklabels()] == ['1 thread', '2 threads']
        assert [text.get_text() for text in fig.axes[1].get_legend().get_texts()] == ['1 thread', '2 threads']
        assert [text.get_text() for text in bars.axes[0].get_legend().get_texts()] == ['1 thread', '2 threads']
        # Two configurations still occupy three named rows, not six repeated labels.
        assert [tick.get_text() for tick in bars.axes[0].get_yticklabels()] == ['record', 'privacy', 'not-scored']
    finally:
        plt.close(fig)
        plt.close(bars)


def test_paired_request_cost_only_uses_shared_cases_and_states_direction():
    from edge_llm_lab.visualization.dataset_report import build_paired_request_cost_figure
    import matplotlib.pyplot as plt
    fig = build_paired_request_cost_figure(paired_summary())
    try:
        ax = fig.axes[0]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == ['record']
        assert [patch.get_width() for patch in ax.patches] == [-2.0]
        assert list(ax.lines[0].get_xdata()) == [0, 0]
        assert '2 threads' in ax.get_xlabel() and '1 thread' in ax.get_xlabel()
        assert 'seconds' in ax.get_xlabel()
        caption = fig._supxlabel.get_text()
        assert 'faster' in caption and 'slower' in caption
        assert 'no uncertainty interval' in caption
    finally:
        plt.close(fig)


def test_render_only_emits_pair_plot_with_two_shared_configurations(tmp_path):
    from edge_llm_lab.visualization.dataset_report import render_summary
    out = tmp_path / 'paired'
    render_summary(paired_summary(), out)
    ET.parse(out / 'paired-request-cost.svg')
    assert (out / 'paired-request-cost.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    unmatched = tmp_path / 'unmatched'
    render_summary(outcome_summary(), unmatched)
    assert not (unmatched / 'paired-request-cost.png').exists()
    single = tmp_path / 'single'
    render(capture(tmp_path), single)
    assert not (single / 'paired-request-cost.png').exists()


def test_coverage_figure_preserves_each_trial_status():
    from edge_llm_lab.visualization.dataset_report import build_coverage_figure
    import matplotlib.pyplot as plt
    fig = build_coverage_figure(paired_summary())
    try:
        ax = fig.axes[0]
        assert [patch.get_height() for patch in ax.patches] == [1, 1, 1, 1, 1, 1, 3, 3]
        assert [text.get_text() for text in ax.get_legend().get_texts()] == ['Scored', 'Unscored', 'Error', 'Not run']
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ['1 thread', '2 threads']
        assert 'not model capability' in fig._supxlabel.get_text()
    finally:
        plt.close(fig)


def test_twenty_case_scores_keep_twenty_readable_rows():
    from edge_llm_lab.visualization.dataset_report import build_case_scores_figure
    import matplotlib.pyplot as plt
    summary = paired_summary()
    summary['case_ids'] = [f'case-{index:02}' for index in range(20)]
    summary['case_scores'] = [{'case_id': case, 'config_id': config, 'score': .5}
                              for case in summary['case_ids'] for config in range(2)]
    summary['latencies'] = []
    fig = build_case_scores_figure(summary)
    try:
        assert len(fig.axes[0].get_yticks()) == 20
        assert len(fig.axes[0].patches) == 40
        assert fig.get_size_inches()[1] <= 7
    finally:
        plt.close(fig)


@pytest.mark.parametrize('extension', ['png', 'svg'])
@pytest.mark.parametrize('mode', ['unmatched', 'single'])
def test_render_rejects_stale_pair_artifact_before_writing_any_figures(tmp_path, extension, mode):
    from edge_llm_lab.visualization.dataset_report import render_summary
    output = tmp_path / 'existing-report'
    output.mkdir()
    stale = output / f'paired-request-cost.{extension}'
    stale.write_bytes(b'previous two-configuration figure')
    sentinel = output / 'case-scores.svg'
    sentinel.write_bytes(b'previous case scores')
    summary = outcome_summary()
    if mode == 'single':
        summary['manifest']['optimisations'] = [{}]
        for key in ('case_scores', 'latencies', 'coverage'):
            summary[key] = [row for row in summary[key] if row['config_id'] == 0]
    # No valid pair exists in either summary. The old paired figure cannot
    # be regenerated and must not silently survive the report update.
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(ValueError, match='existing paired-request-cost'):
        render_summary(summary, output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_public_render_preserves_previous_report_data_when_pair_becomes_invalid(tmp_path):
    paired_database = tmp_path / 'paired.sqlite'
    store = TrialStore(paired_database, {
        'mode': 'referenced', 'optimisations': [{'--threads': 1}, {'--threads': 2}],
        'case_count': 1, 'seeds': [11], 'metrics_profile': 'structured',
    })
    for config_id, wall_ms in enumerate([7000, 5000]):
        store.append({'trial_id': f'pair-{config_id}', 'case_id': 'record',
                      'repeat': 0, 'seed': 11, 'config_id': config_id, 'status': 'ok',
                      'started_utc': 'now', 'elapsed_ms': wall_ms,
                      'result': {'rounds': [{'metrics': {'exact_match': {'score': 1}},
                                             'latency_breakdown': {'wall_ms': wall_ms}}]}})
    store.close()
    output = tmp_path / 'report'
    render(paired_database, output)
    assert (output / 'paired-request-cost.png').exists()
    assert (output / 'summary.json').exists()
    assert (output / 'trials.csv').exists()
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(ValueError, match='existing paired-request-cost'):
        render(capture(tmp_path), output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def capture(tmp_path, profile='text', scored=True):
    path = tmp_path / 'capture.sqlite'
    store = TrialStore(path, {'mode': 'referenced', 'optimisations': [{}],
                            'case_count': 2, 'seeds': [1, 2], 'metrics_profile': profile})
    # Unequal turn counts must not cause a case with more turns to dominate.
    for index, (case, values) in enumerate([('a', [1, 1, 1]), ('a', [0]), ('b', [0])]):
        rounds = [{'metrics': {'exact_match': {'score': v}} if scored else {}} for v in values]
        store.append({'trial_id': str(index), 'case_id': case, 'repeat': index,
                      'seed': 1, 'config_id': 0, 'status': 'ok', 'started_utc': 'now',
                      'elapsed_ms': 10, 'result': {'rounds': rounds}})
    store.close()
    return path


def test_plot_means_weight_cases_not_turns(tmp_path):
    summary = summarize(capture(tmp_path), tmp_path / 'report')
    assert summary['configurations'][0]['mean_case_exact_match'] == .25
    assert summary['complete'] is False


def test_render_preserves_capture_and_labels_text_profile(tmp_path):
    source = capture(tmp_path)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    out = tmp_path / 'report'
    render(source, out)
    svg = out / 'case-scores.svg'
    ET.parse(svg)
    assert 'Exact text match' in svg.read_text()
    assert (out / 'case-scores.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    import matplotlib.pyplot as plt
    assert not plt.get_fignums()


def test_unscored_capture_is_not_plotted_as_zero(tmp_path):
    source = capture(tmp_path, scored=False)
    with pytest.raises(ValueError, match='No exact-match scores'):
        render(source, tmp_path / 'report')
    assert not (tmp_path / 'report' / 'case-scores.png').exists()


def test_report_separates_request_latency_from_trial_duration(tmp_path):
    source = capture(tmp_path)
    with sqlite3.connect(source) as db:
        db.execute("INSERT INTO metrics VALUES ('0','rounds.0.latency_breakdown.wall_ms',7.0)")
    summary = summarize(source, tmp_path / 'report')
    assert summary['latencies'][0]['wall_ms'] == 7.0
    assert summary['coverage'][0]['unscored_trials'] == 0
    assert summary['coverage'][0]['missing_trials'] == 1
