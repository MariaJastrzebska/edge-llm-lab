"""Deterministic structured-response grading without embedding or judge calls."""
import json


def reject_constant(value):
    raise ValueError(f'Non-finite JSON constant: {value}')


def canonical_response(text):
    value = json.loads(text, parse_constant=reject_constant)
    if isinstance(value, dict) and 'tool_calls' in value:
        if not isinstance(value['tool_calls'], list):
            raise ValueError('tool_calls must be a list')
        tools = []
        for call in value['tool_calls']:
            if not isinstance(call, dict):
                raise ValueError('Each tool call must be an object')
            function = call.get('function', call)
            if not isinstance(function, dict) or not isinstance(function.get('name'), str) or not function['name']:
                raise ValueError('Each tool call requires a function name')
            arguments = function.get('arguments', {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments, parse_constant=reject_constant)
            if not isinstance(arguments, dict):
                raise ValueError('Tool arguments must be an object')
            tools.append({'name': function.get('name'), 'arguments': arguments})
        return {'tool_calls': tools}
    return value


def structured_metrics(response, reference):
    # Invalid reference is an evaluation error, not a model failure.
    expected = canonical_response(reference)
    try:
        actual = canonical_response(response)
    except (ValueError, TypeError, AttributeError, KeyError):
        return {'json_validity': {'score': 0.0}, 'exact_match': {'score': 0.0}}
    return {'json_validity': {'score': 1.0},
            'exact_match': {'score': float(json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)),
                            'details': 'Exact JSON equality; tool call IDs ignored, call order preserved'}}
