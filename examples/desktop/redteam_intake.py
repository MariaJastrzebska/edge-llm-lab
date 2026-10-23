"""Run the synthetic application study against local or remote llama.cpp."""
import argparse
import hashlib
import json
import os
import platform
from contextlib import ExitStack
from pathlib import Path

from edge_llm_lab.applications.intake import (
    ACTION_SCHEMA,
    build_messages,
    public_result,
)
from edge_llm_lab.evaluation.app_redteam import (
    IntakeApplication,
    run_study,
    verify_schema_support,
)
from edge_llm_lab.runtime.chat import ChatEndpoint
from edge_llm_lab.runtime.server import LlamaServerRuntime


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument('--model', help='Local GGUF file; requires --server')
    source.add_argument('--endpoint', help='Existing llama.cpp server URL, optionally ending in /v1')
    result.add_argument('--server', help='Local llama-server executable')
    result.add_argument('--model-id', default='local-model', help='Public display ID, never an API credential')
    result.add_argument('--revision', default='unverified', help='Model repository revision, if known')
    result.add_argument('--quantization', default='unverified')
    result.add_argument('--target-system', default='unknown', help='Declared OS of a remote endpoint')
    result.add_argument('--target-machine', default='unknown', help='Declared architecture of a remote endpoint')
    result.add_argument('--dataset', default='examples/datasets/intake-app-redteam.jsonl')
    result.add_argument('--split', choices=['dev', 'test'], default='test')
    result.add_argument('--output', default='runs/intake-app/capture.sqlite')
    result.add_argument('--figures')
    result.add_argument('--seed', type=int, default=11)
    result.add_argument('--port', type=int, default=8091)
    result.add_argument('--threads', type=int, default=2)
    result.add_argument('--max-tokens', type=int, default=160)
    result.add_argument('--context-size', type=int, default=2048)
    result.add_argument('--timeout', type=float, default=180)
    result.add_argument('--ask', help='Run one live guarded app request instead of the study')
    return result


def main():
    arguments = parser()
    args = arguments.parse_args()
    if args.model and not args.server:
        arguments.error('--model requires --server')
    if args.threads < 1 or args.max_tokens < 1 or args.context_size < 1:
        arguments.error('Thread and token budgets must be positive')
    metadata = {'repository_revision': args.revision, 'quantization': args.quantization}
    with ExitStack() as stack:
        if args.model:
            model, server = Path(args.model).resolve(), Path(args.server).resolve()
            metadata.update(sha256=file_sha256(model), backend_sha256=file_sha256(server))
            runtime = LlamaServerRuntime(str(server), '127.0.0.1', args.port, args.timeout)
            stack.callback(runtime.close)
            runtime.ensure(str(model), args.context_size, args.max_tokens,
                           {'--threads': args.threads, '--parallel': 1})
            url = f'http://127.0.0.1:{args.port}'
            target = {'system': platform.system(), 'machine': platform.machine(),
                      'execution': 'local', 'description': 'Observed controller and inference host are identical'}
            metadata.update(context_size=args.context_size, threads=args.threads)
        else:
            url = args.endpoint
            target = {'system': args.target_system, 'machine': args.target_machine,
                      'execution': 'remote', 'description': 'Target hardware is user-declared, not remotely verified'}
        endpoint = stack.enter_context(ChatEndpoint(
            url, args.model_id, max_tokens=args.max_tokens, timeout=args.timeout,
            api_key=os.environ.get('EDGE_LAB_ENDPOINT_API_KEY')))
        verify_schema_support(endpoint)
        if args.ask:
            application = IntakeApplication(endpoint, 'patient-a', {
                'patient-a': {'glucose': []}, 'patient-b': {'glucose': []}})
            print(json.dumps(public_result(application(args.ask)), indent=2))
            return
        # Warm both decoding conditions without using any test case or reference.
        messages = build_messages({'subject_id': 'warmup-subject', 'input': 'Please ask me what information to record.'})
        for schema in (None, ACTION_SCHEMA):
            endpoint.generate(messages, schema=schema, seed=args.seed)
        rows = run_study(endpoint, args.dataset, args.output, split=args.split,
                         seeds=(args.seed,), model_metadata=metadata,
                         inference_platform=target, schema_verified=True)
    errors = sum(row['status'] == 'error' for row in rows)
    print(f'{len(rows)} attempts recorded; {errors} execution errors. Inference OS: {target["system"]}.')
    if args.figures:
        from edge_llm_lab.visualization.app_redteam_report import render
        render(args.output, args.figures)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
