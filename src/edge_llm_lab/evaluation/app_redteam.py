"""Capture application proposals once, then replay authorization ablations.

The existing SQLite store, capture reader, timing parser and dataset family
validation are shared with the referenced/unreferenced evaluation harness.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from edge_llm_lab.applications.intake import (
    ACTION_SCHEMA,
    SYSTEM_PROMPT,
    build_messages,
    parse_action,
    process_response,
    run_application,
)
from edge_llm_lab.evaluation.dataset import content_hash, validate_case_splits
from edge_llm_lab.storage.trials import TrialStore


def load_app_cases(path, split='test'):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    selected = validate_case_splits(rows, split)
    for row in rows:
        if row.get('variant') not in {'benign', 'attack'}:
            raise ValueError('Every case requires a benign or attack variant')
        if not isinstance(row.get('category'), str) or not row['category']:
            raise ValueError('Every case requires a category')
        build_messages(row)
        expected = parse_action(json.dumps(row['expected_action'], allow_nan=False))
        assessment = run_application(json.dumps(expected), row)
        if not assessment['task_success']:
            raise ValueError('The expected action must be executable under the authorized policy')
    return selected


def verify_schema_support(endpoint):
    """Reject an endpoint that does not honor a deliberately conflicting schema."""
    schema = {'type': 'object', 'properties': {
        'probe': {'type': 'string', 'enum': ['schema-enforced']}},
        'required': ['probe'], 'additionalProperties': False}
    text, timing = endpoint.generate(
        [{'role': 'user', 'content': 'Output only the exact plain text NOT_JSON.'}],
        schema=schema)
    try:
        valid = json.loads(text) == {'probe': 'schema-enforced'}
    except (ValueError, TypeError):
        valid = False
    if not valid or timing.get('truncated'):
        raise ValueError('Endpoint failed the schema enforcement preflight: '
                         f'finish={timing.get("finish_reason")}, '
                         f'preview={text[:100]!r}')
    return True


def run_study(endpoint, dataset, output, *, split='test', seeds=(11,),
              model_metadata=None, inference_platform=None, schema_verified=False):
    """One inference per case/config/seed; never rerun the model for a guard toggle."""
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Provide unique integer seeds')
    cases = load_app_cases(dataset, split)
    if not schema_verified:
        verify_schema_support(endpoint)
    source = Path(__file__).resolve().parents[1]
    implementation = hashlib.sha256(b''.join(
        path.read_bytes() for path in sorted(source.rglob('*.py')))).hexdigest()
    configurations = [{'decoding': 'prompt'}, {'decoding': 'schema'}]
    manifest = {
        'schema_version': 1, 'mode': 'app_redteam', 'split': split,
        'dataset_hash': content_hash(cases), 'case_count': len(cases),
        'cases': [{key: case[key] for key in ('id', 'family_id', 'category', 'variant')} for case in cases],
        'seeds': list(seeds), 'optimisations': configurations,
        'model': endpoint.model, 'model_metadata': model_metadata or {},
        'inference_platform': inference_platform or {'execution': 'remote', 'system': 'unknown'},
        'controller_platform': {'system': platform.system(), 'machine': platform.machine(),
                                'python': platform.python_version()},
        'endpoint_hash': content_hash(endpoint.base_url),
        'parameters': {'max_tokens': endpoint.max_tokens, 'temperature': 0, 'cache_prompt': False},
        'implementation_hash': implementation, 'prompt': SYSTEM_PROMPT,
        'action_schema': ACTION_SCHEMA, 'schema_preflight': True,
        'protocol': 'single_step_application_with_paired_offline_authorization_replay',
    }
    store = TrialStore(output, manifest)
    try:
        for repeat, seed in enumerate(seeds):
            for index, case in enumerate(cases):
                order = [0, 1] if (index + repeat) % 2 == 0 else [1, 0]
                for config_id in order:
                    trial_id = content_hash([case['id'], repeat, config_id])
                    if store.contains(trial_id):
                        continue
                    row = {'trial_id': trial_id, 'case_id': case['id'], 'repeat': repeat,
                           'seed': seed, 'config_id': config_id,
                           'started_utc': datetime.now(timezone.utc).isoformat()}  # noqa: UP017 -- Python 3.10
                    started = time.perf_counter()
                    try:
                        response, latency = endpoint.generate(
                            build_messages(case), seed=seed,
                            schema=ACTION_SCHEMA if config_id else None)
                        row['result'] = {'response': response, 'latency': latency,
                            'applications': {name: run_application(response, case, enabled)
                                             for name, enabled in [('unguarded', False), ('guarded', True)]}}
                        row['status'] = 'ok'
                    except Exception as error:  # noqa: BLE001 -- Preserve failed trials in the denominator.
                        row.update(status='error', error_type=type(error).__name__)
                    row['elapsed_ms'] = (time.perf_counter() - started) * 1000
                    store.append(row)
        return store.results()
    finally:
        store.close()


class IntakeApplication:
    """A live app callback with no reference answer or grader in its control flow."""

    def __init__(self, endpoint, subject_id, records, *, constrained=True):
        self.endpoint, self.subject_id, self.records = endpoint, subject_id, records
        self.constrained = constrained

    def __call__(self, text):
        case = {'input': text, 'subject_id': self.subject_id}
        response, latency = self.endpoint.generate(
            build_messages(case), schema=ACTION_SCHEMA if self.constrained else None)
        result = process_response(response, self.subject_id, self.records)
        result['latency'] = latency
        return result
