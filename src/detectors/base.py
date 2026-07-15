"""Shared detector contract and detections.json persistence.

Detectors are read-only monitors: they scan a session's transcript and emit
flags. The transcript itself is never modified.

Each detections.json section carries a fingerprint of the transcript it
scanned. `ensure_fresh_detections()` is the freshness gate: a section whose
fingerprint no longer matches the current transcript (e.g. after --re-enrich)
is re-run automatically — results can be cached but never silently stale.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

DETECTIONS_FILENAME = "detections.json"
ABOUT = (
    "Failure-mode detector output. Detection only — the transcript is never "
    "modified. Sections keyed by detector id; each run overwrites only its own section."
)


class Detector:
    """Base contract for failure-mode detectors.

    Subclasses set the class attributes and implement
    run(session_dir) -> {"n_word_tokens": int, "flags": list[dict]}.
    """

    id: str
    label: str
    failure_mode: str
    version: str
    accepts_judge: bool = False  # True if run() takes an optional `judge` (M9b)
    offline_only: bool = False   # True = an expensive offline detector; scan_session
    #                              skips it unless the caller passes run_offline=True
    #                              (the CLI --story-names / process_inbox). Keeps a
    #                              multi-minute model load out of every web request.

    def run(self, session_dir: Path) -> dict:
        raise NotImplementedError

    def config_fingerprint(self) -> str | None:
        """Content hash of the detector's configuration (e.g. a roster file),
        or None for a configless detector. A detector's output is a function
        of (transcript, config) — both feed the staleness check, so config
        edits trigger re-runs just like transcript changes do."""
        return None


def load_transcript(session_dir: Path) -> dict:
    path = Path(session_dir) / "transcript-rich.json"
    if not path.exists():
        raise FileNotFoundError(f"No transcript-rich.json in {session_dir}")
    return json.loads(path.read_text())


def transcript_fingerprint(session_dir: Path) -> str:
    """Content hash of the transcript a detector run is tied to."""
    path = Path(session_dir) / "transcript-rich.json"
    if not path.exists():
        raise FileNotFoundError(f"No transcript-rich.json in {session_dir}")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def build_section(detector: Detector, result: dict, fingerprint: str,
                  config_fingerprint: str | None) -> dict:
    """Stamp detector metadata onto a run result — one place, so every
    detector's detections.json section has the same shape."""
    return {
        "label": detector.label,
        "failure_mode": detector.failure_mode,
        "detector_version": detector.version,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "transcript_fingerprint": fingerprint,
        "config_fingerprint": config_fingerprint,
        "n_word_tokens": result["n_word_tokens"],
        "n_flags": len(result["flags"]),
        "flags": result["flags"],
    }


def _load_detections(session_dir: Path) -> dict:
    path = Path(session_dir) / DETECTIONS_FILENAME
    if path.exists():
        return json.loads(path.read_text())
    return {"_about": ABOUT, "detectors": {}}


def _save_detections(session_dir: Path, data: dict) -> None:
    """Write via temp file + atomic rename so a crash mid-write can't leave
    corrupt JSON behind. There is deliberately no cross-process lock:
    concurrent writers (e.g. detect.py during a Monitor view) last-write-wins
    on the whole file — acceptable for a single-user local tool; revisit if
    the registry grows past a couple of detectors."""
    path = Path(session_dir) / DETECTIONS_FILENAME
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def write_detections(session_dir: Path, detector: Detector, result: dict) -> dict:
    """Read-merge-write detections.json, replacing only this detector's section."""
    data = _load_detections(session_dir)
    section = build_section(detector, result, transcript_fingerprint(session_dir),
                            detector.config_fingerprint())
    data["detectors"][detector.id] = section
    _save_detections(session_dir, data)
    return section


def section_is_stale(section: dict, session_dir: Path) -> bool:
    """True if a saved section was produced from a different transcript than the
    one on disk now (e.g. after --re-enrich). Read-only — used to surface a
    're-scan' prompt; viewing never re-runs anything."""
    return section.get("transcript_fingerprint") != transcript_fingerprint(session_dir)


def scan_session(session_dir: Path, detectors: list, *, force: bool = False,
                 judge=None, run_offline: bool = False) -> dict:
    """Run detectors and persist detections.json. THE scan entry point — used
    after transcription (process_inbox), by detect.py, and by the manual re-scan
    API. Viewing never calls this.

    Skips a detector whose section is already fresh (matching transcript + config
    fingerprints) unless `force`. `judge` (a callable) is passed only to detectors
    that accept it (the M9b LLM layer); a configless code detector ignores it.

    `run_offline` opts into the expensive `offline_only` detectors (the per-story
    name auditor). Without it they are skipped entirely — never even fingerprint-
    compared — so a slow model load can't enter a web request. Only the CLI
    (--story-names) and process_inbox pass it; the API scan routes never do.

    Sections from unregistered detectors are left untouched. Corrupt
    detections.json propagates json.JSONDecodeError (fail loud).
    """
    fingerprint = transcript_fingerprint(session_dir)
    data = _load_detections(session_dir)
    dirty = False
    for det in detectors:
        if det.offline_only and not run_offline:
            continue  # expensive offline detector — only when the caller opts in
        config_fp = det.config_fingerprint()
        section = data["detectors"].get(det.id)
        if (
            not force
            and section is not None
            and section.get("transcript_fingerprint") == fingerprint
            and section.get("config_fingerprint") == config_fp
        ):
            continue
        use_judge = judge if det.accepts_judge else None
        result = det.run(session_dir, judge=use_judge) if use_judge else det.run(session_dir)
        sec = build_section(det, result, fingerprint, config_fp)
        sec["judge_applied"] = bool(use_judge)
        data["detectors"][det.id] = sec
        dirty = True
    if dirty:
        _save_detections(session_dir, data)
    return data
