"""Read-only, integrity-checked access to portable capture bundles."""
import gzip
import hashlib
import json
import sqlite3
from pathlib import Path


class CaptureReader:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.db = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)
        self.db.row_factory = sqlite3.Row
        try:
            self.manifest = json.loads(self.db.execute('SELECT value FROM manifest WHERE id=1').fetchone()[0])
        except BaseException:
            self.db.close()
            raise

    def trials(self):
        yield from self.db.execute('SELECT * FROM trials ORDER BY rowid')

    def trace(self, trial):
        name = trial['artifact']
        if Path(name).name != name:
            raise ValueError('Invalid artifact name')
        blob = gzip.decompress((self.path.with_suffix('.artifacts') / name).read_bytes())
        if hashlib.sha256(blob).hexdigest() + '.json.gz' != name:
            raise ValueError('Artifact content hash mismatch')
        return json.loads(blob)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.db.close()
