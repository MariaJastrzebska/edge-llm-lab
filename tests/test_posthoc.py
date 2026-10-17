"""Offline grading preserves capture evidence and visible failure coverage."""
import hashlib
import sqlite3

import pytest

from edge_llm_lab.evaluation.posthoc import score_capture
from edge_llm_lab.storage.trials import TrialStore


def make_capture(tmp_path):
    path = tmp_path / 'capture.sqlite'
    store = TrialStore(path, {'mode': 'referenced', 'optimisations': [{}]})
    for index, status in enumerate(('ok', 'error')):
        store.append({'trial_id': str(index), 'case_id': str(index), 'repeat': 0,
                      'seed': 11, 'config_id': 0, 'status': status,
                      'started_utc': '2026-10-17T00:00:00Z', 'elapsed_ms': 1,
                      'error_type': 'RuntimeError' if status == 'error' else None,
                      'result': {'rounds': [{'llm_response': '{"value":1}',
                                            'reference_response': '{"value":1}'}]}})
    store.close()
    return path


def test_offline_score_preserves_evidence_and_failures(tmp_path):
    capture = make_capture(tmp_path)
    before = hashlib.sha256(capture.read_bytes()).hexdigest()
    output = tmp_path / 'scores.sqlite'
    assert score_capture(capture, output) == {'capture_error': 1, 'scored': 1}
    assert hashlib.sha256(capture.read_bytes()).hexdigest() == before
    with sqlite3.connect(output) as db:
        assert db.execute("SELECT value FROM scores WHERE metric='exact_match'").fetchone() == (1.0,)
    with pytest.raises(ValueError):
        score_capture(capture, output)


def test_corrupt_trace_is_unscored_not_a_model_failure(tmp_path):
    capture = make_capture(tmp_path)
    for artifact in capture.with_suffix('.artifacts').glob('*.gz'):
        artifact.write_bytes(b'broken')
    assert score_capture(capture, tmp_path / 'scores.sqlite') == {
        'capture_error': 1, 'scoring_error': 1}
