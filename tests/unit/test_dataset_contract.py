"""Data splits and reference history are benchmark boundaries."""
import json
import pytest
from edge_llm_lab.evaluation.dataset import load_cases, reference_turns


def test_family_cannot_leak_across_splits(tmp_path):
    path = tmp_path / 'cases.jsonl'
    conversation = [{'role': 'user', 'content': 'x'}, {'role': 'assistant', 'content': 'y'}]
    path.write_text('\n'.join(json.dumps({'id': split, 'split': split,
        'family_id': 'same-patient', 'conversation': conversation}) for split in ['dev', 'test']))
    with pytest.raises(ValueError, match='family'):
        load_cases(path, 'referenced', 'test')


def test_reference_turns_never_include_current_target():
    conversation = [{'role': 'system', 'content': 'instruction'},
                    {'role': 'user', 'content': 'question'},
                    {'role': 'user', 'content': 'extra'},
                    {'role': 'assistant', 'content': 'target'}]
    turns = list(reference_turns(conversation))
    assert turns == [(3, conversation[:3], 'target')]


def test_unknown_mode_rejected_before_reading_files(tmp_path):
    with pytest.raises(ValueError, match='mode'):
        load_cases(tmp_path / 'missing', 'typo', 'test')
