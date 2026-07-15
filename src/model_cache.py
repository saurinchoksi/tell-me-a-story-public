"""Per-session cache for expensive model outputs — stamp-and-skip.

A model output (the LLM's name corrections, the story segmentation) is a function of its
INPUT and its CONFIG (the model + prompt). Cache it keyed on fingerprints of both, so a
re-enrich that changes neither reuses the result instead of reloading the model. Same idea
as the failure-mode detectors' (transcript + config) freshness gate, applied to the
pipeline's costly steps. The cache lives in the session dir and is freely regenerable.

Re-running the cleanup passes is cheap; reloading Qwen/Gemma is not. The input to a pass
is stable across a re-enrich (it derives from the immutable transcript-raw), so the cache
recomputes ONLY when the transcription changes or the pass's own model/prompt changes.
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

CACHE_FILENAME = "model-cache.json"
ABOUT = ("Cached expensive model outputs keyed on (input + config) fingerprints. Reused "
         "across re-enrich when nothing relevant changed. Regenerable; safe to delete.")


def fingerprint(s: str) -> str:
    """Content stamp of a string — the model input, or its model+prompt config."""
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


def _cache_path(cache_dir) -> Path:
    return Path(cache_dir) / CACHE_FILENAME


def _load(cache_dir) -> dict:
    path = _cache_path(cache_dir)
    if path.exists():
        return json.loads(path.read_text())
    return {"_about": ABOUT, "stages": {}}


def _save(cache_dir, data: dict) -> None:
    path = _cache_path(cache_dir)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def cached_or_run(cache_dir, stage: str, input_fp: str, config_fp: str, compute):
    """Return (output, was_cached). If the saved `stage` entry's input + config
    fingerprints both match, reuse its output and DO NOT call compute. Otherwise call
    compute(), store {input_fp, config_fp, output}, and return it. compute()'s result
    must be JSON-serializable (it is persisted)."""
    data = _load(cache_dir)
    entry = data["stages"].get(stage)
    if (entry is not None
            and entry.get("input_fingerprint") == input_fp
            and entry.get("config_fingerprint") == config_fp):
        return entry["output"], True
    output = compute()
    data["stages"][stage] = {
        "input_fingerprint": input_fp,
        "config_fingerprint": config_fp,
        "output": output,
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }
    _save(cache_dir, data)
    return output, False
