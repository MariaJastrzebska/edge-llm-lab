"""Shared grading contract for local, offline and Inspect consumers."""
import pytest
from edge_llm_lab.metrics.grading import grade_response


def test_unstructured_text_does_not_require_json():
    assert grade_response('hello', 'hello', 'text')['exact_match']['score'] == 1


def test_text_equality_is_not_a_semantic_safety_claim():
    assert grade_response('Hello', 'hello', 'text')['exact_match']['score'] == 0


@pytest.mark.parametrize('response', ['NaN', '{"tool_calls":[{}]}', '{"tool_calls":null}'])
def test_malformed_structures_do_not_pass(response):
    assert grade_response(response, '{"tool_calls":[]}', 'structured')['exact_match']['score'] == 0


def test_tool_identifiers_do_not_affect_argument_match():
    response = '{"tool_calls":[{"id":"new","function":{"name":"record","arguments":"{\\"value\\":1}"}}]}'
    target = '{"tool_calls":[{"name":"record","arguments":{"value":1}}]}'
    assert grade_response(response, target, 'structured')['exact_match']['score'] == 1


def test_bad_reference_is_an_evaluation_error():
    with pytest.raises(ValueError):
        grade_response('{}', 'NaN', 'structured')


def test_unknown_profile_is_not_silently_accepted():
    with pytest.raises(ValueError):
        grade_response('a', 'a', 'unknown')
