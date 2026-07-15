"""One atomic JSON write for the whole project — tmp file + os.replace.

The pattern existed in 4 hand-rolled copies (namebook, namefix's queue writer, the API's
queue writer, the retired apply path) before the 2026-07-02 review consolidated it here.
The tmp name carries the pid so two writers can't corrupt each other's tmp bytes; the
read-merge-write races themselves are documented as out of scope for this single-user app
(last os.replace wins).
"""
import json
import os
from pathlib import Path


def atomic_write_json(path: Path, obj, indent: int = 2) -> None:
    """Write ``obj`` as JSON to ``path`` atomically, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(obj, indent=indent, ensure_ascii=False))
    os.replace(tmp, path)
