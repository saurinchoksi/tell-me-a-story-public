"""Detection (monitor) endpoints — failure-mode detector results.

Scan and view are separated. **Viewing is read-only** — the GET routes serve
whatever the last scan wrote and never run a detector (so a slow LLM judge can
live in a scan without ever blocking a page load). **Scanning** runs the
detectors and persists results, triggered after transcription, by detect.py, or
by the manual re-scan POST routes here. A section whose transcript changed since
its scan is marked `stale` so the UI can prompt a re-scan; it is never silently
recomputed. The transcript itself is never modified.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from api.helpers import (
    _derive_story_label,
    _read_transcript_facts,
    annotate_canon_tiers,
    apply_name_verdicts,
    get_session_dir,
    name_verdict_status,
    read_name_verdicts,
    validate_session_id,
)

bp = Blueprint("detections", __name__)


def _detectors():
    return current_app.config["DETECTORS"]  # always concrete — resolved in create_app


def _read_detections(session_dir: Path) -> dict | None:
    """Parsed detections.json, or None if never scanned. Corrupt file
    propagates json.JSONDecodeError (fail loud)."""
    path = session_dir / "detections.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _make_judge():
    """The M9b LLM judge for a full scan, or (None, reason) if its venv is absent
    — in which case the scan falls back to code-only rather than failing."""
    try:
        from detectors.name_consistency_judge import make_judge
        return make_judge(), None
    except FileNotFoundError as e:
        return None, str(e)


# --- canon de-dup (m9b defers to m9c) -----------------------------------------

M9B_ID = "m9b-name-consistency"
M9C_ID = "m9c-canon"


def _apply_canon_dedup(detector_sections: dict) -> None:
    """In-place, view-time: drop the whole m9b CLUSTER the canon reader (m9c) owns any token of,
    so a sourced-canon name surfaces ONLY under M9c, never split across both sections. M9b's unit
    is the cluster (a name spelled many ways, keyed by cluster_id); a token-level cut would leave
    the rest of the cluster stranded in M9b (e.g. 'james'/'jamis' after M9c takes the other James
    spellings), doubling the same name, mislabeled. Detectors stay independent on disk; this
    reconciles their overlap at read time, for both the rollup counts and the detail flags.

    Trade-off: a canon spelling M9c MISSED then shows in neither detector (an M9c recall gap, not
    a confusing split). Safe no-op when m9c found nothing (e.g. world unrecognized) — m9b then keeps
    those catches, the only thing flagging them."""
    m9c = detector_sections.get(M9C_ID)
    m9b = detector_sections.get(M9B_ID)
    if not m9c or not m9b:
        return
    canon = set()
    # `tier` is set by annotate_canon_tiers, which MUST run before this (both call sites order it so).
    # If it ever didn't, .get returns None != "low" — every flag is kept, i.e. the pre-tier behavior.
    for f in m9c.get("flags", []):
        if f.get("tier") == "low":  # a low-confidence canon claim must not hide a real m9b inconsistency
            continue
        if f.get("cleaned"):
            canon.add(f["cleaned"])
        canon.update(f.get("wrong_cleaned") or [])
    if not canon:
        return
    owned = {f["cluster_id"] for f in m9b["flags"] if f.get("cleaned") in canon}
    kept = [f for f in m9b["flags"] if f.get("cluster_id") not in owned]
    m9b["flags"] = kept
    m9b["n_flags"] = len(kept)


# --- shared read builders (used by both GET and POST-scan) --------------------

def _rollup(sessions_dir, detector_objs):
    from detectors.base import section_is_stale
    detectors = [
        {"id": d.id, "label": d.label, "failure_mode": d.failure_mode, "version": d.version}
        for d in detector_objs
    ]
    totals = {d["id"]: 0 for d in detectors}
    sessions = []
    if sessions_dir.exists():
        for entry in sorted(sessions_dir.iterdir(), reverse=True):
            if not entry.is_dir() or not validate_session_id(entry.name):
                continue
            if not (entry / "transcript-rich.json").exists():
                continue
            data = _read_detections(entry)  # may raise JSONDecodeError -> caller handles
            if data:
                apply_name_verdicts(data["detectors"], read_name_verdicts(entry))  # human corrections first
                annotate_canon_tiers(data["detectors"])  # tier m9c + count default-visible (confident+best_guess)
                _apply_canon_dedup(data["detectors"])  # then m9b defers to m9c on canon names
            results, stale = {}, False
            for det_id, section in (data["detectors"].items() if data else []):
                results[det_id] = {
                    "n_flags": section["n_flags"],
                    "run_at": section["run_at"],
                    "detector_version": section["detector_version"],
                }
                totals[det_id] = totals.get(det_id, 0) + section["n_flags"]
                if section_is_stale(section, entry):
                    stale = True
            facts = _read_transcript_facts(entry)
            sessions.append({
                "session_id": entry.name,
                "duration_seconds": facts["duration_seconds"],
                "stories": facts["stories"],
                "results": results,
                "stale": stale,
            })
    return {"detectors": detectors, "sessions": sessions, "totals": totals}


def _session_detail(session_dir, session_id):
    from detectors.base import section_is_stale
    has_audio = next(session_dir.glob("audio.*"), None) is not None

    # Read the transcript once: it feeds both the per-flag segment-text join and
    # the story summary shown in the page header. None when not transcribed.
    transcript_path = session_dir / "transcript-rich.json"
    seg_by_id = {}
    stories = None
    if transcript_path.exists():
        with open(transcript_path) as f:
            tj = json.load(f)
        seg_by_id = {seg["id"]: seg for seg in tj["segments"]}
        stories = _derive_story_label(tj.get("_stories") or [])

    verdicts_raw = read_name_verdicts(session_dir)
    data = _read_detections(session_dir)
    if data is None:
        return {"session_id": session_id, "has_audio": has_audio, "stories": stories,
                "name_verdicts": name_verdict_status({}, verdicts_raw), "detectors": {}}
    # Status is read off the RAW sections (before apply mutates them); then human
    # verdicts run BEFORE the dedup so a not_canon name releases back to m9b.
    verdicts = name_verdict_status(data["detectors"], verdicts_raw)
    apply_name_verdicts(data["detectors"], verdicts_raw)
    annotate_canon_tiers(data["detectors"])  # tier m9c flags + count default-visible (confident+best_guess)
    _apply_canon_dedup(data["detectors"])  # m9b defers to m9c on canon names (skips low-tier claims)

    detectors = {}
    for det_id, section in data["detectors"].items():
        flags = []
        for flag in section["flags"]:
            seg = seg_by_id.get(flag["segment_id"])
            speaker = (seg.get("_speaker") or {}) if seg else {}
            flags.append({
                **flag,
                "segment_text": seg["text"].strip() if seg else None,
                "segment_start": seg["start"] if seg else None,
                "segment_end": seg["end"] if seg else None,
                "segment_speaker": speaker.get("label"),
            })
        stale = transcript_path.exists() and section_is_stale(section, session_dir)
        detectors[det_id] = {**section, "flags": flags, "stale": stale}

    return {"session_id": session_id, "has_audio": has_audio, "stories": stories,
            "name_verdicts": verdicts, "detectors": detectors}


# --- GET (read-only) ----------------------------------------------------------

@bp.route("/detections")
def detections_rollup():
    """System-wide rollup — reads the last scan; never runs a detector."""
    try:
        return jsonify(_rollup(current_app.config["SESSIONS_DIR"], _detectors()))
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Corrupt detections.json: {e}"}), 500


@bp.route("/sessions/<session_id>/detections")
def session_detections(session_id: str):
    """Full flag detail for one session, joined to transcript segments. Read-only."""
    try:
        session_dir = get_session_dir(current_app.config["SESSIONS_DIR"], session_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    return jsonify(_session_detail(session_dir, session_id))


@bp.route("/sessions/<session_id>/name-verdicts", methods=["POST"])
def save_name_verdict(session_id: str):
    """Toggle a human 'set the record straight' verdict on a flagged name, then return
    the fresh detail. Re-sending an identical verdict removes it (undo). Read-merge-write
    of name-verdicts.json — a human-owned file the detectors never touch."""
    try:
        session_dir = get_session_dir(current_app.config["SESSIONS_DIR"], session_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    body = request.get_json(silent=True)
    vtype = (body or {}).get("type")
    if vtype == "not_canon" and not body.get("name"):
        return jsonify({"error": "not_canon verdict requires 'name'"}), 400
    if vtype == "correct" and not body.get("cleaned"):
        return jsonify({"error": "correct verdict requires 'cleaned'"}), 400
    if vtype not in ("not_canon", "correct"):
        return jsonify({"error": "verdict 'type' must be 'not_canon' or 'correct'"}), 400

    def same(v):  # identity is type + its key (canonical name, or cleaned spelling)
        return (v["type"] == vtype and v.get("name") == body.get("name")
                and v.get("cleaned") == body.get("cleaned"))

    path = session_dir / "name-verdicts.json"
    data = json.loads(path.read_text()) if path.exists() else {"verdicts": []}
    verdicts = data["verdicts"]
    if any(same(v) for v in verdicts):
        verdicts = [v for v in verdicts if not same(v)]  # toggle off
    else:
        entry = {k: v for k, v in body.items() if k != "type"}
        entry = {"type": vtype, **entry, "at": datetime.now(timezone.utc).isoformat()}
        verdicts.append(entry)
    data["verdicts"] = verdicts
    path.write_text(json.dumps(data, indent=2))

    return jsonify(_session_detail(session_dir, session_id))


# --- POST (scan) --------------------------------------------------------------

@bp.route("/sessions/<session_id>/detections/scan", methods=["POST"])
def scan_one(session_id: str):
    """Force a full re-scan of one session (code detectors + the M9b LLM judge),
    then return its fresh detail. Slow — the judge loads a model."""
    from detectors.base import scan_session

    try:
        session_dir = get_session_dir(current_app.config["SESSIONS_DIR"], session_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    if not (session_dir / "transcript-rich.json").exists():
        return jsonify({"error": "Session has no transcript to scan"}), 400

    judge, warning = _make_judge()
    try:
        scan_session(session_dir, _detectors(), force=True, judge=judge)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        return jsonify({"error": f"Scan failed on {session_id}: {e}"}), 500
    detail = _session_detail(session_dir, session_id)
    if warning:
        detail["warning"] = warning
    return jsonify(detail)


@bp.route("/detections/scan", methods=["POST"])
def scan_all():
    """Re-scan every session whose results are missing or stale (full pass).
    Fresh sessions are skipped, so this is usually cheap."""
    from detectors.base import scan_session

    sessions_dir = current_app.config["SESSIONS_DIR"]
    detector_objs = _detectors()
    judge, warning = _make_judge()
    if sessions_dir.exists():
        for entry in sorted(sessions_dir.iterdir()):
            if not entry.is_dir() or not validate_session_id(entry.name):
                continue
            if not (entry / "transcript-rich.json").exists():
                continue
            try:
                scan_session(entry, detector_objs, force=False, judge=judge)
            except (FileNotFoundError, ValueError, RuntimeError) as e:
                return jsonify({"error": f"Scan failed on {entry.name}: {e}"}), 500
    rollup = _rollup(sessions_dir, detector_objs)
    if warning:
        rollup["warning"] = warning
    return jsonify(rollup)
