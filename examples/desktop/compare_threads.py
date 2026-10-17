"""Compare local CPU thread settings on identical frozen routing cases.

This is a small execution study, not a clinical or general security benchmark.
One deterministic pass is captured per configuration: cases are paired across
thread settings, and the fixed seed is not an independent statistical repeat.
Request timings exclude server startup; trial duration includes that overhead.
"""
import argparse
from pathlib import Path

from edge_llm_lab.evaluation.referenced_evaluator import EvalModelsReferenced
from edge_llm_lab.visualization.dataset_report import render


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / 'datasets' / 'intake-routing-study.jsonl'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Local GGUF file')
    parser.add_argument('--server', required=True, help='llama-server executable')
    parser.add_argument('--dataset', default=str(DEFAULT_DATASET), help='Frozen JSONL test cases')
    parser.add_argument('--output', required=True, help='SQLite capture path')
    parser.add_argument('--port', type=int, default=18081)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2],
                        help='Two or more distinct positive CPU thread counts')
    parser.add_argument('--figures', help='Directory for offline figures and scalar exports')
    args = parser.parse_args(argv)
    if (len(args.threads) < 2 or any(value < 1 for value in args.threads)
            or len(set(args.threads)) != len(args.threads)):
        parser.error('--threads requires at least two distinct positive integers')

    configurations = [{'--threads': value} for value in args.threads]
    with EvalModelsReferenced.from_local_model(
            args.model, args.server, port=args.port,
            metrics_profile='structured', temperature=0) as evaluator:
        trials = evaluator.evaluate_dataset(
            args.dataset, args.output, seeds=[11],
            optimisations=configurations, use_cache=False)

    errors = sum(trial['status'] == 'error' for trial in trials)
    print(f'{len(trials)} trials recorded across {len(configurations)} configurations; {errors} errors')
    if args.figures:
        render(args.output, args.figures)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
