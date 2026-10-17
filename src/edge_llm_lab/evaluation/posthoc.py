"""Score captured referenced trials offline without loading the target model.

The input database and its adjacent .artifacts directory form one bundle.
Scores are written to a new SQLite database; capture evidence stays immutable.
"""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from edge_llm_lab.metrics.grading import grade_response
from edge_llm_lab.storage.capture import CaptureReader


def score_capture(source, output):
    """Apply deterministic JSON grading and retain failures in coverage counts."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists():
        raise ValueError('Choose a new output database to preserve previous evidence')
    capture = CaptureReader(source)
    try:
        manifest = json.dumps(capture.manifest, sort_keys=True)
        profile = json.loads(manifest).get('metrics_profile', 'structured')
        if json.loads(manifest)['mode'] != 'referenced':
            raise ValueError('Only frozen referenced captures are supported')
        output.parent.mkdir(parents=True, exist_ok=True)
        scores = sqlite3.connect(output)
        try:
            scores.executescript('''
                CREATE TABLE provenance (source_manifest TEXT, grader_sha256 TEXT);
                CREATE TABLE trials (id TEXT PRIMARY KEY, case_id TEXT, status TEXT, error_type TEXT);
                CREATE TABLE scores (trial_id TEXT, round INTEGER, metric TEXT, value REAL,
                    PRIMARY KEY(trial_id, round, metric));
            ''')
            scorer_path = Path(__file__).resolve().parents[1] / 'metrics' / 'structured.py'
            grader_hash = hashlib.sha256(Path(__file__).read_bytes() + scorer_path.read_bytes()).hexdigest()
            scores.execute('INSERT INTO provenance VALUES (?,?)', (manifest, grader_hash))
            for trial in capture.trials():
                status, error, values = 'capture_error', trial['error_type'], []
                if trial['status'] == 'ok':
                    try:
                        rounds = capture.trace(trial)['rounds']
                        if not rounds:
                            raise ValueError('No captured rounds')
                        for index, turn in enumerate(rounds):
                            metrics = grade_response(turn['llm_response'], turn['reference_response'], profile)
                            values.extend((trial['id'], index, name, metric['score'])
                                          for name, metric in metrics.items())
                        status, error = 'scored', None
                    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                        status, error, values = 'scoring_error', type(exc).__name__, []
                with scores:
                    scores.execute('INSERT INTO trials VALUES (?,?,?,?)',
                                   (trial['id'], trial['case_id'], status, error))
                    scores.executemany('INSERT INTO scores VALUES (?,?,?,?)', values)
            return dict(scores.execute('SELECT status, count(*) FROM trials GROUP BY status'))
        finally:
            scores.close()
    finally:
        capture.db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(score_capture(args.capture, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
