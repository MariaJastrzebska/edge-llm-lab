"""Exercise the real Inspect task without a remote model or credentials."""
import pytest
pytest.importorskip('inspect_ai')
from inspect_ai import eval
from inspect_ai.model import ModelOutput, ModelUsage
from edge_llm_lab.integrations.inspect_ai import referenced_task


def test_inspect_uses_shared_grader_and_frozen_prefix(tmp_path):
    task = referenced_task('examples/datasets/single-command.jsonl')
    sample = list(task.dataset)[0]
    assert sample.target not in [m.content for m in sample.input]
    output = ModelOutput.from_content('mock', sample.target)
    output.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
    logs = eval(task, model='mockllm/model',
                model_args={'custom_outputs': [output]},
                log_dir=str(tmp_path), display='none', log_format='json')
    assert logs[0].status == 'success', logs[0].error
    assert logs[0].samples[0].scores['response_exact_match'].value == 1.0


def test_capture_replay_never_calls_target_model(tmp_path):
    from edge_llm_lab.integrations.inspect_ai import captured_task
    from edge_llm_lab.storage.trials import TrialStore
    source = tmp_path / 'source.sqlite'
    store = TrialStore(source, {'mode': 'referenced', 'optimisations': [{}],
                                'metrics_profile': 'text'})
    store.append({'trial_id': 'one', 'case_id': 'one', 'repeat': 0, 'seed': 1,
                  'config_id': 0, 'status': 'ok', 'started_utc': 'now', 'elapsed_ms': 1,
                  'result': {'rounds': [{'context': [{'role': 'user', 'content': 'x'}],
                                         'llm_response': 'yes', 'reference_response': 'yes'}]}})
    store.close()
    def forbidden(*args):
        raise AssertionError('Posthoc replay must not call the target model')
    logs = eval(captured_task(str(source)), model='mockllm/model',
                model_args={'custom_outputs': forbidden}, log_dir=str(tmp_path / 'logs'), display='none', log_format='json')
    assert logs[0].status == 'success', logs[0].error
    assert logs[0].samples[0].scores['response_exact_match'].value == 1.0
