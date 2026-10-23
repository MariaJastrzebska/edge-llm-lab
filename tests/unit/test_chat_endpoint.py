"""HTTP contracts for the same transport used by legacy and app evaluations."""
from unittest.mock import Mock

import pytest

from edge_llm_lab.runtime.chat import ChatEndpoint, request_completion


def session_with(content='{"ok":true}', finish='stop'):
    session = Mock()
    session.post.return_value.json.return_value = {
        'choices': [{'message': {'content': content}, 'finish_reason': finish}],
        'timings': {'prompt_ms': 1, 'predicted_ms': 2, 'predicted_n': 3},
    }
    return session


def test_schema_is_sent_to_decoder_and_normalized_endpoint():
    session = session_with()
    client = ChatEndpoint('http://localhost:8080/v1', 'medical-model', session=session)
    schema = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}}
    text, timing = client.generate([{'role': 'user', 'content': 'x'}], schema=schema, seed=17)
    args = session.post.call_args
    assert args.args[0] == 'http://localhost:8080/v1/chat/completions'
    assert args.kwargs['json']['response_format'] == {'type': 'json_schema', 'schema': schema}
    assert args.kwargs['json']['seed'] == 17
    assert args.kwargs['json']['cache_prompt'] is False
    assert text == '{"ok":true}'
    assert timing['wall_ms'] >= 0 and timing['model_loading_ms'] is None
    assert timing['tokens']['generated_count'] == 3


def test_prompt_condition_has_no_decoder_schema_and_records_truncation():
    session = session_with(finish='length')
    client = ChatEndpoint('http://localhost:8080', 'model', session=session)
    _, timing = client.generate([{'role': 'user', 'content': 'x'}])
    assert 'response_format' not in session.post.call_args.kwargs['json']
    assert timing['truncated'] is True


@pytest.mark.parametrize('url', ['file:///tmp/x', 'https://user:secret@host', 'http://host?token=x', 'http://host/#x'])
def test_endpoint_rejects_embedded_credentials_or_non_http(url):
    with pytest.raises(ValueError):
        ChatEndpoint(url, 'model')


def test_transport_does_not_hide_http_errors():
    session = session_with()
    session.post.return_value.raise_for_status.side_effect = RuntimeError('http error')
    with pytest.raises(RuntimeError):
        request_completion(session, 'http://localhost:8080', {}, 10)


@pytest.mark.parametrize('result', [{}, {'choices': []}, {'choices': [{'message': {}}]}])
def test_malformed_response_is_an_execution_error(result):
    session = session_with()
    session.post.return_value.json.return_value = result
    client = ChatEndpoint('http://localhost:8080', 'model', session=session)
    with pytest.raises(ValueError):
        client.generate([])


def test_request_messages_are_not_mutated():
    messages = [{'role': 'user', 'content': 'x'}]
    client = ChatEndpoint('http://localhost:8080', 'model', session=session_with())
    client.generate(messages)
    assert messages == [{'role': 'user', 'content': 'x'}]
