"""Regression tests at the original evaluator boundaries."""
import json
from unittest.mock import Mock
import pytest
from edge_llm_lab.evaluation.unreferenced_evaluator import EvalModelsUnreferenced
from edge_llm_lab.evaluation.referenced_evaluator import EvalModelsReferenced
from edge_llm_lab.runtime.timings import parse_llama_timings


@pytest.mark.parametrize('status', ['not complete', 'almost_complete', 'incomplete'])
def test_status_substrings_cannot_end_a_patient_trajectory(status):
    evaluator = EvalModelsUnreferenced.__new__(EvalModelsUnreferenced)
    assert evaluator.extract_status_and_question(json.dumps({'status': status}))[0] == 'INCOMPLETE'


def test_unknown_model_loading_time_is_not_reported_as_zero():
    assert parse_llama_timings({}, 0, 1)['model_loading_ms'] is None


def test_capture_does_not_call_graders_or_leak_target():
    evaluator = EvalModelsReferenced.__new__(EvalModelsReferenced)
    evaluator.defer_scoring = True
    evaluator.tools = []
    evaluator.run_inference = Mock(return_value=('answer', {'total_ms': 1}))
    evaluator.calculate_fast_metrics = Mock(side_effect=AssertionError('Grader called during capture'))
    case = {'conversation': [{'role': 'user', 'content': 'question'},
                             {'role': 'assistant', 'content': 'target'}]}
    result = evaluator.evaluate_case(case, {})
    assert result['rounds'][0]['metrics'] == {}
    assert evaluator.run_inference.call_args.args[1] == case['conversation'][:1]
