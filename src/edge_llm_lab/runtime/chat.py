"""Shared llama.cpp HTTP transport for evaluator and application callbacks."""
from __future__ import annotations

import time
from urllib.parse import urlsplit, urlunsplit

import requests

from edge_llm_lab.runtime.timings import parse_llama_timings


def endpoint_root(url: str) -> str:
    parts = urlsplit(url)
    if (parts.scheme not in {'http', 'https'} or not parts.hostname
            or parts.username or parts.password or parts.query or parts.fragment):
        raise ValueError('Use an HTTP(S) endpoint without embedded credentials, query or fragment')
    path = parts.path.rstrip('/')
    path = path.removesuffix('/v1')
    return urlunsplit((parts.scheme, parts.netloc, path, '', ''))


def request_completion(session, base_url, payload, timeout):
    """Preserve HTTP failures; callers record them as execution errors."""
    started = time.perf_counter()
    response = session.post(endpoint_root(base_url) + '/v1/chat/completions',
                            json=payload, timeout=timeout)
    ended = time.perf_counter()
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict) or not result.get('choices'):
        raise ValueError('Inference response contains no choices')
    return result, started, ended


class ChatEndpoint:
    """Connect to an existing server; never start or terminate its process.

    Wall time includes HTTP transport. Remote controller memory and power are
    deliberately not reported as target-device telemetry.
    """

    def __init__(self, base_url, model, *, max_tokens=160, timeout=120,
                 api_key=None, session=None):
        self.base_url = endpoint_root(base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.session = session if session is not None else requests.Session()
        self._owns_session = session is None
        if api_key:
            self.session.headers['Authorization'] = f'Bearer {api_key}'

    def generate(self, messages, *, schema=None, seed=11):
        payload = {'model': self.model, 'messages': messages, 'temperature': 0,
                   'top_p': 1, 'max_tokens': self.max_tokens, 'stream': False,
                   'seed': seed, 'cache_prompt': False,
                   'chat_template_kwargs': {'enable_thinking': False}}
        if schema is not None:
            payload['response_format'] = {'type': 'json_schema', 'schema': schema}
        result, started, ended = request_completion(
            self.session, self.base_url, payload, self.timeout)
        choice = result['choices'][0]
        content = choice.get('message', {}).get('content')
        if not isinstance(content, str):
            raise ValueError('The bounded action agent requires text completion content')  # noqa: TRY004
        timing = parse_llama_timings(result, started, ended)
        timing['finish_reason'] = choice.get('finish_reason')
        timing['truncated'] = choice.get('finish_reason') == 'length'
        return content, timing

    def close(self):
        if self._owns_session:
            self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
