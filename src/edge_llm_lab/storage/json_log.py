"""Atomic JSON snapshots. Never filter failed or unreferenced observations."""
import json
import os
import tempfile
from pathlib import Path


def atomic_write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize first: errors leave the previous snapshot untouched.
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
