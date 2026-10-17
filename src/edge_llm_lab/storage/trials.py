"""Indexed SQLite measurements with content-addressed, compressed trace artifacts."""
import gzip
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from pathlib import Path


def scalar_metrics(value, prefix=''):
    """Extract numeric leaves for SQL analysis without loading transcripts."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {'context', 'reference_response', 'llm_response', 'collected_data', 'expected_data'}:
                continue
            yield from scalar_metrics(child, f'{prefix}.{key}' if prefix else key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from scalar_metrics(child, f'{prefix}.{index}')
    elif isinstance(value, (int, float)) and math.isfinite(value):
        yield prefix, float(value)


class TrialStore:
    def __init__(self, path, manifest):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts = self.path.with_suffix('.artifacts')
        self.artifacts.mkdir(exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS manifest (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS configurations (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS trials (
                id TEXT PRIMARY KEY, case_id TEXT NOT NULL, repeat INTEGER NOT NULL,
                seed INTEGER NOT NULL, config_id INTEGER NOT NULL REFERENCES configurations(id),
                status TEXT NOT NULL, started_utc TEXT NOT NULL, elapsed_ms REAL NOT NULL,
                error_type TEXT, artifact TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS trials_by_config_status ON trials(config_id, status);
            CREATE INDEX IF NOT EXISTS trials_by_case ON trials(case_id);
            CREATE TABLE IF NOT EXISTS metrics (
                trial_id TEXT NOT NULL REFERENCES trials(id), name TEXT NOT NULL, value REAL NOT NULL,
                PRIMARY KEY(trial_id, name));
            CREATE INDEX IF NOT EXISTS metrics_by_name ON metrics(name, trial_id);
        ''')
        encoded = json.dumps(manifest, sort_keys=True, allow_nan=False)
        previous = self.db.execute('SELECT value FROM manifest WHERE id=1').fetchone()
        if previous and previous[0] != encoded:
            self.db.close()
            raise ValueError('Cannot resume: dataset, protocol, model or implementation changed')
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO manifest VALUES (1, ?)', (encoded,))
            for index, config in enumerate(manifest['optimisations']):
                self.db.execute('INSERT OR IGNORE INTO configurations VALUES (?,?)',
                                (index, json.dumps(config, sort_keys=True)))

    def contains(self, trial_id):
        return self.db.execute('SELECT 1 FROM trials WHERE id=?', (trial_id,)).fetchone() is not None

    def append(self, result):
        trace = json.dumps(result.get('result', {}), sort_keys=True, allow_nan=False).encode()
        name = hashlib.sha256(trace).hexdigest() + '.json.gz'
        target = self.artifacts / name
        if not target.exists():
            fd, temporary = tempfile.mkstemp(dir=self.artifacts)
            try:
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(gzip.compress(trace, mtime=0))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        with self.db:
            self.db.execute('INSERT INTO trials VALUES (?,?,?,?,?,?,?,?,?,?)', (
                result['trial_id'], result['case_id'], result['repeat'], result['seed'],
                result['config_id'], result['status'], result['started_utc'], result['elapsed_ms'],
                result.get('error_type'), name))
            self.db.executemany('INSERT INTO metrics VALUES (?,?,?)',
                                [(result['trial_id'], key, value) for key, value in scalar_metrics(result.get('result', {}))])

    def results(self):
        """Return small trial metadata only. Full traces are explicitly opt-in."""
        return [dict(row) for row in self.db.execute('SELECT * FROM trials ORDER BY rowid')]

    def trace(self, trial_id):
        row = self.db.execute('SELECT artifact FROM trials WHERE id=?', (trial_id,)).fetchone()
        if row is None:
            raise KeyError(trial_id)
        with gzip.open(self.artifacts / row[0], 'rt') as stream:
            return json.load(stream)

    def close(self):
        self.db.close()
