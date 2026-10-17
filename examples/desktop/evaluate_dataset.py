"""Run the existing referenced evaluator on a frozen dataset with a local GGUF."""
import argparse
from edge_llm_lab.evaluation.referenced_evaluator import EvalModelsReferenced
from edge_llm_lab.visualization.dataset_report import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--server', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True, help='SQLite database path')
    parser.add_argument('--seed', type=int, action='append', dest='seeds')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--figures')
    parser.add_argument('--profile', choices=['structured', 'text'], default='structured')
    parser.add_argument('--defer-scoring', action='store_true',
                        help='Capture responses without calling graders')
    args = parser.parse_args()
    with EvalModelsReferenced.from_local_model(args.model, args.server, port=args.port, metrics_profile=args.profile) as evaluator:
        results = evaluator.evaluate_dataset(args.dataset, args.output, seeds=args.seeds or [11, 23, 47],
                                             defer_scoring=args.defer_scoring)
    if args.figures:
        render(args.output, args.figures)
    print(f'{len(results)} trials recorded; {sum(r["status"] == "error" for r in results)} errors')
    if any(r['status'] == 'error' for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
