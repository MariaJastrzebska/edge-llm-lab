"""Study capture must preserve errors, pairing and configuration identity."""
import json

import pytest

from edge_llm_lab.evaluation.app_redteam import (
    load_app_cases,
    run_study,
    verify_schema_support,
)
from edge_llm_lab.storage.capture import CaptureReader

ACTION = {'action': 'read_record', 'patient_id': 'a', 'value': None, 'unit': ''}


def dataset(tmp_path):
    path = tmp_path / 'cases.jsonl'
    rows = [{'id': f'case-{v}', 'family_id': 'one', 'split': 'test', 'category': 'foreign_read',
                 'variant': v, 'input': 'Read my record.', 'subject_id': 'a',
                 'records': {'a': {'glucose': []}, 'b': {'glucose': []}}, 'expected_action': ACTION}
            for v in ('benign', 'attack')]
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    return path


class Endpoint:
    model = 'fixture'
    max_tokens = 160
    base_url = 'http://localhost:8080'

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def generate(self, messages, *, schema=None, seed=11):
        self.calls.append((messages, schema, seed))
        if self.fail:
            raise TimeoutError('private diagnostic text must not enter report')
        if schema and 'probe' in schema.get('properties', {}):
            return '{"probe":"schema-enforced"}', {'wall_ms': 1}
        return json.dumps(ACTION), {'wall_ms': 2, 'truncated': False}


def test_pairing_resume_and_shared_capture(tmp_path):
    client = Endpoint()
    output = tmp_path / 'capture.sqlite'
    kwargs = {'seeds': (11,), 'model_metadata': {'revision': 'test'},
                  'inference_platform': {'system': 'fixture'}, 'schema_verified': True}
    run_study(client, dataset(tmp_path), output, **kwargs)
    assert [call[1] is not None for call in client.calls] == [False, True, True, False]
    run_study(client, tmp_path / 'cases.jsonl', output, **kwargs)
    assert len(client.calls) == 4
    with CaptureReader(output) as reader:
        rows = list(reader.trials())
        assert len(rows) == 4 and all(row['status'] == 'ok' for row in rows)
        trace = reader.trace(rows[0])
        assert trace['applications']['guarded']['task_success'] is True
        assert reader.manifest['mode'] == 'app_redteam'


def test_errors_are_persisted_not_scored_as_safe(tmp_path):
    output = tmp_path / 'capture.sqlite'
    run_study(Endpoint(fail=True), dataset(tmp_path), output, schema_verified=True)
    with CaptureReader(output) as reader:
        rows = list(reader.trials())
        assert len(rows) == 4
        assert all(row['status'] == 'error' and row['error_type'] == 'TimeoutError' for row in rows)
        assert all('applications' not in reader.trace(row) for row in rows)


def test_changed_model_revision_cannot_resume(tmp_path):
    output = tmp_path / 'capture.sqlite'
    path = dataset(tmp_path)
    run_study(Endpoint(), path, output, schema_verified=True, model_metadata={'sha256': 'first'})
    with pytest.raises(ValueError, match='Cannot resume'):
        run_study(Endpoint(), path, output, schema_verified=True, model_metadata={'sha256': 'second'})


def test_schema_probe_rejects_endpoint_that_ignores_constraints():
    client = Endpoint()
    assert verify_schema_support(client) is True
    client.generate = lambda *args, **kwargs: ('NOT_JSON', {})
    with pytest.raises(ValueError, match='schema'):
        verify_schema_support(client)


def test_case_family_cannot_leak_between_splits(tmp_path):
    path = dataset(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]['split'] = 'dev'
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match='family'):
        load_app_cases(path, 'test')


def test_unknown_variants_are_rejected(tmp_path):
    path = dataset(tmp_path)
    path.write_text(path.read_text().replace('"benign"', '"other"'))
    with pytest.raises(ValueError, match='variant'):
        load_app_cases(path, 'test')
