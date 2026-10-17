"""Inspect tasks reusing Edge-LLM-Lab datasets and deterministic graders."""
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ChatMessageAssistant
from inspect_ai.scorer import Score, accuracy, scorer
from inspect_ai.solver import generate

from edge_llm_lab.evaluation.dataset import load_cases, reference_turns
from edge_llm_lab.metrics.grading import PROFILES, grade_response


@scorer(metrics=[accuracy()])
def response_exact_match(profile='structured'):
    """Exact agreement is a task metric, not a clinical or general safety score."""
    if profile not in PROFILES:
        raise ValueError(f'Unknown grading profile: {profile}')

    async def score(state, target):
        metrics = grade_response(state.output.completion, target.text, profile)
        return Score(value=metrics['exact_match']['score'], answer=state.output.completion,
                     explanation=metrics['exact_match'].get('details', ''), metadata=metrics)
    return score


@task
def referenced_task(dataset: str, split: str = 'test', profile: str = 'structured'):
    """Replay frozen prefixes through an Inspect model provider.

    This evaluates independent turns under reference history, not recovery from
    earlier model errors. Case IDs remain available for grouped analysis.
    """
    cases = load_cases(dataset, 'referenced', split)
    classes = {'system': ChatMessageSystem, 'user': ChatMessageUser,
               'assistant': ChatMessageAssistant}
    samples = []
    for case in cases:
        conversation = case['conversation']
        for index, prefix, target in reference_turns(conversation):
            if any(message['role'] not in classes for message in prefix):
                raise ValueError('Tool-message replay requires call IDs; this adapter accepts text conversations')
            samples.append(Sample(
                id=f"{case['id']}:{index}",
                input=[classes[m['role']](content=m['content']) for m in prefix],
                target=target,
                metadata={'case_id': case['id'], 'split': split,
                          'protocol': 'frozen_reference_replay',
                          'category': case.get('category', 'unspecified'),
                          'family_id': case.get('family_id', case['id'])}))
    return Task(dataset=MemoryDataset(samples), solver=generate(),
                scorer=response_exact_match(profile), metadata={'profile': profile})


@task
def captured_task(capture: str):
    """Grade stored responses in Inspect without generating new target output.

    Refuse incomplete/error captures rather than silently excluding failures.
    Use the native offline report to inspect and resolve capture coverage first.
    """
    from inspect_ai.model import ModelOutput
    from inspect_ai.solver import solver
    from edge_llm_lab.storage.capture import CaptureReader

    @solver
    def replay():
        async def solve(state, generate):
            state.output = ModelOutput.from_content('captured-response', state.metadata['response'])
            state.completed = True
            return state
        return solve

    samples = []
    with CaptureReader(capture) as reader:
        manifest = reader.manifest
        if manifest['mode'] != 'referenced':
            raise ValueError('Only referenced captures are supported')
        for trial in reader.trials():
            if trial['status'] != 'ok':
                raise ValueError('Capture contains failed trials; inspect coverage before import')
            for index, turn in enumerate(reader.trace(trial)['rounds']):
                samples.append(Sample(id=f"{trial['id']}:{index}",
                    input=str(turn['context']), target=turn['reference_response'],
                    metadata={'response': turn['llm_response'], 'case_id': trial['case_id'],
                              'trial_id': trial['id'], 'config_id': trial['config_id']}))
    if not samples:
        raise ValueError('No captured responses')
    return Task(dataset=MemoryDataset(samples), solver=replay(),
                scorer=response_exact_match(manifest.get('metrics_profile', 'structured')),
                metadata={'source_manifest': manifest, 'protocol': 'captured_response_scoring'})
