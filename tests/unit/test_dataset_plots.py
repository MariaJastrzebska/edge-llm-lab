"""Plot contracts check scientific meaning before testing image rendering."""
import hashlib
from pathlib import Path
import sqlite3
import xml.etree.ElementTree as ET
import pytest
from edge_llm_lab.storage.trials import TrialStore
from edge_llm_lab.visualization.dataset_report import render, summarize


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
