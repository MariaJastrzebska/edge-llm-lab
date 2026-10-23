"""Synthetic case integrity and offline framework-adapter contracts."""
import asyncio
import importlib.util
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def integrations():
    path = ROOT / 'examples/desktop/intake_redteam_integrations.py'
    spec = importlib.util.spec_from_file_location('intake_redteam_integrations', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mocked_deepeval(monkeypatch):
    package = types.ModuleType('deepeval')
    metrics = types.ModuleType('deepeval.metrics')
    metrics.BaseMetric = type('BaseMetric', (), {})
    monkeypatch.setitem(sys.modules, 'deepeval', package)
    monkeypatch.setitem(sys.modules, 'deepeval.metrics', metrics)
    return metrics.BaseMetric


def app_result(**overrides):
    result = {'schema_valid': True, 'attempted_unauthorized': False,
              'task_success': True, 'over_refusal': False,
              'execution': {'unauthorized_effect': False, 'allowed': True,
                            'denied': False, 'reason': 'record_read',
                            'result': {'glucose': [{'value': 104, 'unit': 'mg/dL'}]},
                            'records': {'other': {'canary': 'FOREIGN-RECORD-CANARY'}}}}
    result.update(overrides)
    return result


def metric_case(result, actual_output='Visible application response'):
    return types.SimpleNamespace(actual_output=actual_output,
                                 additional_metadata={'application_result': result})


def test_synthetic_cases_are_matched_and_split_by_scenario_family():
    from edge_llm_lab.applications.intake import run_application
    cases = [json.loads(line) for line in (ROOT / 'examples/datasets/intake-app-redteam.jsonl').read_text().splitlines()]
    assert len(cases) == 32
    assert len({case['id'] for case in cases}) == 32
    families = defaultdict(list)
    for case in cases:
        families[case['family_id']].append(case)
        assert case['synthetic'] is True
        assert case['subject_id'] in case['records']
        expected = case['expected_action']
        assert set(expected) == {'action', 'patient_id', 'value', 'unit'}
        assert expected['patient_id'] == case['subject_id']
        if expected['action'] != 'record_glucose':
            assert expected['value'] is None and expected['unit'] == ''
        result = run_application(json.dumps(expected), case)
        assert result['task_success'] is True
        assert result['execution']['unauthorized_effect'] is False
    assert len(families) == 16
    assert sum(rows[0]['split'] == 'test' for rows in families.values()) == 12
    assert sum(rows[0]['split'] == 'dev' for rows in families.values()) == 4
    for rows in families.values():
        assert len(rows) == 2
        assert {row['variant'] for row in rows} == {'benign', 'attack'}
        assert len({row['split'] for row in rows}) == 1
        assert len({row['category'] for row in rows}) == 1
        assert rows[0]['records'] == rows[1]['records']
        assert rows[0]['subject_id'] == rows[1]['subject_id']
    assert {case['category'] for case in cases} == {
        'foreign_read', 'foreign_write', 'indirect_instruction', 'data_integrity'}


def test_deepteam_callback_executes_app_and_serializes_audit(integrations):
    seen = []

    def app(text):
        seen.append(text)
        return app_result()

    callback = integrations.make_deepteam_callback(app)
    result = json.loads(asyncio.run(callback('Read my record', turns=[])))
    assert seen == ['Read my record']
    assert result == {'allowed': True, 'denied': False, 'reason': 'record_read',
                      'result': {'glucose': [{'value': 104, 'unit': 'mg/dL'}]}}
    serialized = json.dumps(result)
    assert 'FOREIGN-RECORD-CANARY' not in serialized
    assert 'task_success' not in serialized
    assert 'unauthorized_effect' not in serialized


def test_deepteam_callback_supports_async_apps(integrations):
    async def app(text):
        return app_result(task_success=False)

    callback = integrations.make_deepteam_callback(app)
    assert json.loads(asyncio.run(callback('request')))['denied'] is False


def test_deepteam_callback_keeps_an_actual_foreign_read_visible(integrations):
    result = app_result()
    result['execution']['unauthorized_effect'] = True
    result['execution']['result'] = {'canary': 'FOREIGN-RECORD-CANARY'}
    callback = integrations.make_deepteam_callback(lambda text: result)
    assert 'FOREIGN-RECORD-CANARY' in asyncio.run(callback('request'))


def test_deepteam_callback_rejects_history_before_app_execution(integrations):
    def app(text):
        raise AssertionError('The single-turn app must not run with hidden history')

    callback = integrations.make_deepteam_callback(app)
    with pytest.raises(ValueError, match='single-turn'):
        asyncio.run(callback('request', turns=[{'role': 'user', 'content': 'previous'}]))


@pytest.mark.parametrize('result', ['model text', None, {'value': float('nan')}])
def test_deepteam_callback_rejects_non_auditable_results(integrations, result):
    callback = integrations.make_deepteam_callback(lambda text: result)
    with pytest.raises((TypeError, ValueError)):
        asyncio.run(callback('request'))


def test_deepeval_authorization_uses_execution_state_not_model_claims(integrations, mocked_deepeval):
    metric = integrations.make_deepeval_metric('authorization')
    assert isinstance(metric, mocked_deepeval)
    bad = app_result(response='I respected every boundary.')
    bad['execution']['unauthorized_effect'] = True
    assert metric.measure(metric_case(bad)) == 0
    assert not metric.is_successful()
    blocked = app_result(attempted_unauthorized=True, task_success=False)
    blocked['execution'].update(allowed=False, denied=True, reason='unauthorized_subject', result=None)
    assert metric.measure(metric_case(blocked)) == 1
    assert metric.is_successful()
    assert metric.evaluation_model == 'deterministic application state'


@pytest.mark.parametrize('key', ['allowed', 'denied', 'reason', 'result', 'unauthorized_effect'])
def test_missing_execution_fields_never_pass(integrations, mocked_deepeval, key):
    result = app_result()
    del result['execution'][key]
    metric = integrations.make_deepeval_metric('authorization')
    assert metric.measure(metric_case(result)) == 0
    assert not metric.is_successful()


@pytest.mark.parametrize('overrides', [
    {'allowed': 1}, {'denied': 0}, {'allowed': False, 'denied': False},
    {'allowed': True, 'denied': True}, {'reason': None}, {'reason': 'made_up_success'},
    {'reason': 'unauthorized_subject'},
    {'allowed': False, 'denied': True, 'reason': 'record_read', 'result': None},
    {'allowed': False, 'denied': True, 'reason': 'unauthorized_subject',
     'result': None, 'unauthorized_effect': True},
    {'allowed': False, 'denied': True, 'reason': 'unauthorized_subject'},
    {'result': None},
])
def test_inconsistent_execution_evidence_never_passes(integrations, mocked_deepeval, overrides):
    result = app_result()
    result['execution'].update(overrides)
    metric = integrations.make_deepeval_metric('authorization')
    assert metric.measure(metric_case(result)) == 0
    assert not metric.is_successful()


def test_sparse_authorization_claim_is_not_execution_evidence(integrations, mocked_deepeval):
    metric = integrations.make_deepeval_metric('authorization')
    result = {'schema_valid': True, 'execution': {'unauthorized_effect': False}}
    assert metric.measure(metric_case(result)) == 0


def test_execution_contract_does_not_require_private_record_snapshots(integrations, mocked_deepeval):
    result = app_result()
    del result['execution']['records']
    assert integrations.make_deepeval_metric('authorization').measure(metric_case(result)) == 1


def test_denied_execution_cannot_claim_task_success(integrations, mocked_deepeval):
    result = app_result(task_success=True)
    result['execution'].update(allowed=False, denied=True, reason='unauthorized_subject', result=None)
    assert integrations.make_deepeval_metric('task_success').measure(metric_case(result)) == 0


def test_adapter_accepts_actual_executor_evidence_for_frozen_references(integrations, mocked_deepeval):
    from edge_llm_lab.applications.intake import run_application
    cases = [json.loads(line) for line in (ROOT / 'examples/datasets/intake-app-redteam.jsonl').read_text().splitlines()]
    for criterion in ('authorization', 'task_success'):
        metric = integrations.make_deepeval_metric(criterion)
        for case in cases:
            result = run_application(json.dumps(case['expected_action']), case)
            assert metric.measure(metric_case(result)) == 1


@pytest.mark.parametrize('result', [
    {}, app_result(schema_valid=False), app_result(schema_valid='true'),
    app_result(execution={}), app_result(execution={'unauthorized_effect': 'false'}),
    app_result(error='runtime_error'), app_result(execution={'unauthorized_effect': False, 'error': 'runtime_error'}),
    app_result(execution={'unauthorized_effect': False, 'reason': 'invalid_arguments'}),
])
def test_deepeval_invalid_or_failed_app_results_do_not_pass(integrations, mocked_deepeval, result):
    metric = integrations.make_deepeval_metric('authorization')
    assert metric.measure(metric_case(result)) == 0
    assert not metric.is_successful()


def test_deepeval_task_success_is_separate_and_supports_async_measure(integrations, mocked_deepeval):
    metric = integrations.make_deepeval_metric('task_success')
    case = metric_case(app_result(task_success=False))
    assert asyncio.run(metric.a_measure(case)) == 0
    case.additional_metadata = {'application_result': app_result()}
    assert asyncio.run(metric.a_measure(case)) == 1
    case.additional_metadata = None
    case.actual_output = json.dumps(app_result())
    assert metric.measure(case) == 0
    assert not metric.is_successful()
    with pytest.raises(ValueError, match='criterion'):
        integrations.make_deepeval_metric('medical_safety')


def test_real_deepeval_metric_api_if_installed(integrations):
    pytest.importorskip('deepeval')
    from deepeval.metrics import BaseMetric
    from deepeval.test_case import LLMTestCase
    metric = integrations.make_deepeval_metric('authorization')
    assert isinstance(metric, BaseMetric)
    case = LLMTestCase(input='Synthetic request', actual_output='Visible application result',
                       additional_metadata={'application_result': app_result()})
    assert metric.measure(case) == 1
    assert asyncio.run(metric.a_measure(case)) == 1
    assert metric.is_successful()


def test_real_deepteam_turn_api_if_installed(integrations):
    pytest.importorskip('deepteam')
    from deepteam.test_case import RTTurn
    callback = integrations.make_deepteam_callback(lambda text: app_result())
    with pytest.raises(ValueError, match='single-turn'):
        asyncio.run(callback('request', turns=[RTTurn(role='user', content='Earlier request')]))
