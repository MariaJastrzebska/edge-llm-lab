"""Dataset orchestration for the existing referenced/unreferenced evaluators."""
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from copy import deepcopy
from pathlib import Path

from edge_llm_lab.storage.trials import TrialStore


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def reference_turns(conversation):
    """Yield target index, strict prior history and frozen response."""
    for index, message in enumerate(conversation):
        if message['role'] == 'assistant':
            yield index, conversation[:index], message['content']


def validate_case_splits(cases, split):
    """Share identity and family-leakage checks across evaluation protocols."""
    if split not in {'dev', 'test'}:
        raise ValueError('Choose dev or test split')
    if not cases or any(not isinstance(c.get('id'), str) or not c['id'] for c in cases):
        raise ValueError('Dataset must contain unique, nonempty case IDs')
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Dataset must contain unique, nonempty case IDs')
    families = {}
    for case in cases:
        family = case.get('family_id', case['id'])
        if not isinstance(family, str) or not family:
            raise ValueError('Every case requires a nonempty family ID')
        if family in families and families[family] != case.get('split'):
            raise ValueError('A case family cannot span dev and test splits')
        families[family] = case.get('split')
        if case.get('split') not in {'dev', 'test'}:
            raise ValueError('Every case requires an explicit dev/test split')
    selected = [c for c in cases if c['split'] == split]
    if not selected:
        raise ValueError(f'No cases in split {split}')
    return selected


def load_cases(path, mode, split):
    if mode not in {'referenced', 'unreferenced'}:
        raise ValueError('Unknown evaluation mode')
    path = Path(path)
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    selected = validate_case_splits(cases, split)
    for case in cases:
        if mode == 'referenced':
            conversation = case.get('conversation')
            if not conversation or not any(m.get('role') == 'assistant' for m in conversation):
                raise ValueError('Referenced cases require a frozen conversation with assistant targets')
            for index, message in enumerate(conversation):
                if message.get('role') not in {'system', 'user', 'assistant', 'tool'} or not isinstance(message.get('content'), str):
                    raise ValueError('Conversation messages require role and text content')
                if message['role'] == 'assistant' and index == 0:
                    raise ValueError('An assistant target must have preceding context')
        else:
            for key in ('patient_file', 'expected_file'):
                resolved = (path.parent / case[key]).resolve()
                case[key] = str(resolved)
                case[key + '_sha256'] = hashlib.sha256(resolved.read_bytes()).hexdigest()
            expected = json.loads(Path(case['expected_file']).read_text())
            if expected.get('patient_type') not in {'successful', 'unsuccessful'}:
                raise ValueError('Expected file requires a valid patient_type')
    return selected


class DatasetEvaluation:
    def evaluate_dataset(self, dataset_path, output, *, split='test', seeds=(11, 23, 47),
                         optimisations=None, use_cache=False, defer_scoring=False):
        """Run complete cases sequentially; repeats never count as independent cases.

        Existing evaluator inference and graders are used unchanged. References
        are frozen input, not generated inside this evaluation. Exceptions are
        persisted as failed trials and never silently removed from denominators.
        """
        if split not in {'dev', 'test'} or not seeds or any(type(s) is not int for s in seeds):
            raise ValueError('Choose dev/test and at least one integer seed')
        if defer_scoring and self.eval_type != 'referenced':
            raise ValueError('Deferred scoring currently supports frozen referenced replay only')
        cases = load_cases(dataset_path, self.eval_type, split)
        configurations = [{}] if optimisations is None else optimisations
        if not configurations or len({content_hash(c) for c in configurations}) != len(configurations):
            raise ValueError('Optimization configurations must be nonempty and unique')
        sources = Path(__file__).resolve().parents[1]
        implementation = hashlib.sha256(b''.join(p.read_bytes() for p in sorted(sources.rglob('*.py')))).hexdigest()
        manifest = {'schema_version': 1, 'defer_scoring': defer_scoring, 'mode': self.eval_type, 'split': split,
                    'dataset_hash': content_hash(cases), 'case_count': len(cases),
                    'model': self.model_name, 'model_metadata': self.current_model_metadata,
                    'parameters': self.MULTI_TURN_GLOBAL_CONFIG, 'seeds': list(seeds),
                    'optimisations': configurations, 'judge_model': self.evaluator_model_name,
                    'grader_cache': use_cache, 'implementation_hash': implementation,
                    'platform': platform.platform(), 'python': platform.python_version(),
                    'tools': self.tools, 'prompt': self.cot_prompt,
                    'request_options': getattr(self, 'request_options', {}),
                    'metrics_profile': getattr(self, 'metrics_profile', 'full')}
        store = TrialStore(output, manifest)
        original_seed = self.SEED
        original_defer = getattr(self, 'defer_scoring', False)
        self.defer_scoring = defer_scoring
        try:
            for repeat, seed in enumerate(seeds):
                self.SEED = seed
                for case_index, case in enumerate(cases):
                    order = list(enumerate(configurations))
                    if (repeat + case_index) % 2:
                        order.reverse()
                    for config_index, config in order:
                        trial_id = content_hash([case['id'], repeat, config_index])
                        if store.contains(trial_id):
                            continue
                        result = {'trial_id': trial_id, 'case_id': case['id'], 'repeat': repeat,
                                  'seed': seed, 'config_id': config_index, 'optimisation': config,
                                  'started_utc': datetime.now(timezone.utc).isoformat()}
                        started = time.perf_counter()
                        try:
                            result['result'] = self.evaluate_case(deepcopy(case), deepcopy(config), use_cache)
                            result['status'] = 'ok'
                        except Exception as exc:
                            result.update(status='error', error_type=type(exc).__name__)
                        result['elapsed_ms'] = (time.perf_counter() - started) * 1000
                        store.append(result)
            return store.results()
        finally:
            self.SEED = original_seed
            self.defer_scoring = original_defer
            store.close()
