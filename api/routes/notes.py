"""Validation notes endpoints — read and write per-session notes."""

import json

from flask import Blueprint, current_app, jsonify, request

from api.helpers import get_session_dir

bp = Blueprint("notes", __name__)


@bp.route("/sessions/<session_id>/notes")
def get_notes(session_id: str):
    """Return validation notes for a session, or empty list if none exist."""
    try:
        session_dir = get_session_dir(
            current_app.config["SESSIONS_DIR"], session_id
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    notes_path = session_dir / "validation-notes.json"
    if not notes_path.exists():
        return jsonify({"notes": []})

    try:
        with open(notes_path) as f:
            data = json.load(f)
        return jsonify({"notes": data.get("notes", [])})
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Corrupt validation-notes.json: {e}"}), 500


@bp.route("/sessions/<session_id>/notes", methods=["POST"])
def save_notes(session_id: str):
    """Save validation notes for a session (full array replacement)."""
    try:
        session_dir = get_session_dir(
            current_app.config["SESSIONS_DIR"], session_id
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    body = request.get_json(silent=True)
    if body is None or "notes" not in body:
        return jsonify({"error": "Request body must contain 'notes' array"}), 400

    if not isinstance(body["notes"], list):
        return jsonify({"error": "'notes' must be an array"}), 400

    notes_path = session_dir / "validation-notes.json"
    with open(notes_path, "w") as f:
        json.dump({"notes": body["notes"]}, f, indent=2)

    return jsonify({"saved": len(body["notes"])})
