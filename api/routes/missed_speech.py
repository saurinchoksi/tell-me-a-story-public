"""Missed-speech endpoints — the canonical EMP gap detection served live,
plus the hear-box sidecar that captures the ear's verdict on each candidate.

GET /api/sessions/<id>/missed-speech runs the canonical missed-speech logic
(src/gaps.py — the same single implementation behind the EMP sweep,
emp/src/gap_analysis.py) server-side from transcript-rich.json +
diarization.json, so the validator's candidate spans can never diverge from
the published EMP floor and never go stale on re-enrichment.

Read-only VIEW: computes from existing JSON on every request, never writes a
file, never modifies a transcript, never loads a model, never runs a detector.

A candidate is a stretch where diarization heard a voice but the transcript
has NO segment of any kind — no transcribed line, not even an
[unintelligible] placeholder (the coverage rule lives in src/gaps.py).

GET/POST /api/sessions/<id>/missed-speech/hearings (TMAS-75) read and write
missed-speech-verdicts.json — the per-session hear-box sidecar (the ear-check
convention: every listening probe gets a "what I hear" box + a verdict; named
like name-verdicts.json, the other human-rulings file — NOT to be confused
with emp/results/visuals/missed-speech-hearings.json, the EMP probe's own
sidecar with a different schema). A candidate has no transcript segment, so
the validator's segment-anchored notes can't hold an observation about one;
hearings are keyed by candidate span (start/end) instead, so a future re-run
of the sweep can compare its candidates against the recorded verdicts. The
POST writes ONLY this human-authored sidecar — never a transcript.
"""

import json
import math

from flask import Blueprint, current_app, jsonify, request

from api.helpers import get_session_dir
from gaps import HIGH_CONFIDENCE_MIN_GAP, find_gaps
from jsonio import atomic_write_json

bp = Blueprint("missed_speech", __name__)

# Default threshold = the EMP sweep's high-confidence floor, the count that
# justified the ticket's Option A volumes — imported from src/gaps.py, the
# threshold's one home. ?min_gap=0.3 reaches the sweep's broader candidate
# tier when a reviewer wants the slivers too.
DEFAULT_MIN_GAP = HIGH_CONFIDENCE_MIN_GAP


@bp.route("/sessions/<session_id>/missed-speech")
def get_missed_speech(session_id: str):
    """Missed-speech candidate spans for one session: {candidates, min_gap}."""
    try:
        session_dir = get_session_dir(
            current_app.config["SESSIONS_DIR"], session_id
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    raw_min_gap = request.args.get("min_gap")
    if raw_min_gap is None:
        min_gap = DEFAULT_MIN_GAP
    else:
        try:
            min_gap = float(raw_min_gap)
        except ValueError:
            return jsonify(
                {"error": f"min_gap must be a number, got {raw_min_gap!r}"}
            ), 400
        if not math.isfinite(min_gap) or min_gap <= 0:
            return jsonify(
                {"error": "min_gap must be a positive finite number"}
            ), 400

    # Missing inputs are NOT an empty result — returning {candidates: []}
    # would silently mask "can't compute". Say exactly what's missing.
    transcript_path = session_dir / "transcript-rich.json"
    diarization_path = session_dir / "diarization.json"
    if not transcript_path.exists():
        return jsonify(
            {"error": f"transcript-rich.json not found for session {session_id}"}
        ), 404
    if not diarization_path.exists():
        return jsonify(
            {"error": f"diarization.json not found for session {session_id}"}
        ), 404

    try:
        with open(transcript_path) as f:
            transcript = json.load(f)
        with open(diarization_path) as f:
            diarization = json.load(f)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Corrupt session JSON: {e}"}), 500

    # Same stance as the exists() checks above: a diarization file without its
    # segments is "can't compute", never a confident empty result.
    if "segments" not in diarization:
        return jsonify(
            {"error": f"diarization.json for session {session_id} has no "
                      "'segments' key — re-run diarization"}
        ), 500

    gaps = find_gaps(diarization, transcript, min_gap)
    # find_gaps attaches full prev/next segment dicts for the sweep's HTML
    # table — strip them here; the API contract is just the spans.
    candidates = [
        {"start": g["start"], "end": g["end"], "dur": g["dur"],
         "speaker": g["speaker"]}
        for g in gaps
    ]
    return jsonify({"candidates": candidates, "min_gap": min_gap})


# The four things an ear can conclude about a candidate span (TMAS-75):
# speech the transcript truly missed; noise adjacent to a hallucinated
# stretch; the uncovered tail of a partially captured word (Choksi's
# "Wowwwww" case); or nothing there at all. Mirrored in the HearingVerdict
# type (ui/src/types/index.ts) and the VERDICTS labels
# (ui/src/components/MissedHearBox.tsx) — the three move together.
HEARING_VERDICTS = {
    "real-speech", "hallucination-adjacent", "boundary-tail", "nothing"
}

HEARINGS_FILENAME = "missed-speech-verdicts.json"


@bp.route("/sessions/<session_id>/missed-speech/hearings")
def get_hearings(session_id: str):
    """Return the hear-box entries for a session, or empty list if none."""
    try:
        session_dir = get_session_dir(
            current_app.config["SESSIONS_DIR"], session_id
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    hearings_path = session_dir / HEARINGS_FILENAME
    if not hearings_path.exists():
        return jsonify({"hearings": []})

    try:
        with open(hearings_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Corrupt {HEARINGS_FILENAME}: {e}"}), 500
    # Shape-check before serving: a hand-edited sidecar that is valid JSON
    # but not {"hearings": [...]} must be a clean 500 here, not a TypeError
    # (non-dict) or a frontend crash on hearings.find (non-list).
    if not isinstance(data, dict) or not isinstance(data.get("hearings"), list):
        return jsonify(
            {"error": f"{HEARINGS_FILENAME} must be an object with a "
                      "'hearings' array"}
        ), 500
    return jsonify({"hearings": data["hearings"]})


@bp.route("/sessions/<session_id>/missed-speech/hearings", methods=["POST"])
def save_hearings(session_id: str):
    """Save hear-box entries (full array replacement, like validation notes).

    Each entry must carry a usable span key (finite numeric start/end) and a
    known verdict (or null while only text is typed) — a hearing that can't
    be joined back to a candidate is a wasted listen, so reject it loudly
    rather than store it.
    """
    try:
        session_dir = get_session_dir(
            current_app.config["SESSIONS_DIR"], session_id
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    body = request.get_json(silent=True)
    # isinstance, not a None-check: a bare JSON scalar (`5`, `true`) would
    # TypeError on the `in` test and escape as a raw 500.
    if not isinstance(body, dict) or "hearings" not in body:
        return jsonify(
            {"error": "Request body must contain 'hearings' array"}
        ), 400
    if not isinstance(body["hearings"], list):
        return jsonify({"error": "'hearings' must be an array"}), 400

    for i, h in enumerate(body["hearings"]):
        if not isinstance(h, dict):
            return jsonify({"error": f"hearings[{i}] must be an object"}), 400
        # start/end are the join key back to a candidate; min_gap is the
        # provenance tier. All three must be finite — json.dump would write
        # NaN as a bare token that the browser's strict JSON.parse rejects,
        # bricking every later GET for the session.
        for key in ("start", "end", "min_gap"):
            v = h.get(key)
            if not isinstance(v, (int, float)) or isinstance(v, bool) \
                    or not math.isfinite(v):
                return jsonify(
                    {"error": f"hearings[{i}].{key} must be a finite number"}
                ), 400
        if not isinstance(h.get("hear"), str):
            return jsonify(
                {"error": f"hearings[{i}].hear must be a string"}
            ), 400
        verdict = h.get("verdict")
        if verdict is not None and verdict not in HEARING_VERDICTS:
            return jsonify(
                {"error": f"hearings[{i}].verdict must be null or one of "
                          f"{sorted(HEARING_VERDICTS)}, got {verdict!r}"}
            ), 400

    hearings_path = session_dir / HEARINGS_FILENAME
    # Atomic (tmp + os.replace, the name_corrections.py precedent): a crash
    # mid-write must never truncate hand-authored verdicts into corrupt JSON.
    atomic_write_json(hearings_path, {"hearings": body["hearings"]})

    return jsonify({"saved": len(body["hearings"])})
