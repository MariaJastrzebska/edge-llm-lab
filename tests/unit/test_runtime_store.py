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


def test_optimizer_rejects_test_set_and_nonfinite_objective():
    with pytest.raises(ValueError, match='dev'):
        OptimizationEngine(dataset_split='test')
    optimizer = OptimizationEngine()
    with pytest.raises(ValueError, match='finite'):
        optimizer.run_optimization(lambda trial: float('nan'), n_trials=1)
