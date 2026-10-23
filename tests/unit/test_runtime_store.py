"""Runtime ownership, persistence and optimization regression coverage."""
from unittest.mock import Mock
import pytest
from edge_llm_lab.runtime.server import LlamaServerRuntime
from edge_llm_lab.storage.trials import TrialStore
from edge_llm_lab.evaluation.optimization.optimization_engine import OptimizationEngine


def test_runtime_reuses_only_its_matching_live_process(tmp_path):
    runtime = LlamaServerRuntime('server', '127.0.0.1', 12345, 1)
    process = Mock()
    process.poll.return_value = None
    runtime.process = process
    import json
    runtime.identity = json.dumps([str((tmp_path / 'model').resolve()), 1024, 32, {}], sort_keys=True)
    assert runtime.ensure(str(tmp_path / 'model'), 1024, 32, {}) is process
    process.terminate.assert_not_called()
    runtime.close()
    process.terminate.assert_called_once()


def test_runtime_does_not_kill_an_external_server():
    runtime = LlamaServerRuntime('server', '127.0.0.1', 12345, 1)
    runtime.session.get = Mock(return_value=Mock(status_code=200))
    with pytest.raises(RuntimeError, match='occupied'):
        runtime.ensure('model', 1024, 32, {})
    assert runtime.process is None
    runtime.close()


def test_store_rejects_resume_with_changed_protocol(tmp_path):
    path = tmp_path / 'capture.sqlite'
    store = TrialStore(path, {'optimisations': [{}], 'seed': 1})
    store.close()
    with pytest.raises(ValueError, match='Cannot resume'):
        TrialStore(path, {'optimisations': [{}], 'seed': 2})


@pytest.mark.parametrize('large_integer', [10 ** 400, -(10 ** 400)], ids=['positive-overflow', 'negative-overflow'])
def test_store_preserves_large_integer_artifacts_without_overflowing_scalar_metrics(tmp_path, large_integer):
    store = TrialStore(tmp_path / 'capture.sqlite', {'optimisations': [{}]})
    result = {'trial_id': 'large-output', 'case_id': 'synthetic-case', 'repeat': 0,
              'seed': 11, 'config_id': 0, 'status': 'ok', 'started_utc': '2026-10-17',
              'elapsed_ms': 12.5,
              'result': {'application': {'generated_integer': large_integer},
                         'finite': [3, 2.5, True]}}
    try:
        store.append(result)
        assert store.contains('large-output')
        assert store.trace('large-output') == result['result']
        values = dict(store.db.execute('SELECT name, value FROM metrics WHERE trial_id=?',
                                       ('large-output',)))
        assert values == {'finite.0': 3.0, 'finite.1': 2.5, 'finite.2': 1.0}
        result['trial_id'] = 'next-output'
        result['result'] = {'finite': 7}
        store.append(result)
        assert store.contains('next-output')
    finally:
        store.close()


def test_optimizer_rejects_test_set_and_nonfinite_objective():
    with pytest.raises(ValueError, match='dev'):
        OptimizationEngine(dataset_split='test')
    optimizer = OptimizationEngine()
    with pytest.raises(ValueError, match='finite'):
        optimizer.run_optimization(lambda trial: float('nan'), n_trials=1)
