"""Axial label endpoints — read and write per-segment axial codes for a session.

A label carries one or more categorical codes from the EMP failure modes
(M1..M9) or NotA (none-of-the-above). Used to count failure-mode frequencies
for EMP step 5. Multi-code per label as of 2026-05-21; legacy single-code
files (one `code: str` field) are auto-backed-up on first write under the
new shape — see `_backup_if_legacy_shape`.
"""

import json
import shutil

from flask import Blueprint, current_app, jsonify, request

from api.helpers import get_session_dir

bp = Blueprint("axial_labels", __name__)

LEGACY_BACKUP_FILENAME = "axial-labels.legacy-pre-multicode.json"


def _has_legacy_entry(data: dict) -> bool:
    """True if any label in `data` has a `code` (str) field and no `codes` (list)."""
    labels = data.get("labels", [])
    for entry in labels:
        if isinstance(entry, dict) and "code" in entry and "codes" not in entry:
            return True
    return False


def _backup_if_legacy_shape(labels_path) -> None:
    """One-time-per-session backup: if the existing axial-labels.json is in the
    legacy single-code shape, copy it to LEGACY_BACKUP_FILENAME before any write.
    Idempotent — never overwrites an existing backup.
    """
    if not labels_path.exists():
        return
    backup_path = labels_path.parent / LEGACY_BACKUP_FILENAME
    if backup_path.exists():
        return
    try:
        with open(labels_path) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        # Don't lose a corrupt file silently — copy it aside before the POST
        # overwrites it. Idempotent like the legacy backup.
        corrupt_path = labels_path.parent / "axial-labels.corrupt.json"
        if not corrupt_path.exists():
            shutil.copy2(labels_path, corrupt_path)
        return
    if _has_legacy_entry(data):
        shutil.copy2(labels_path, backup_path)


@bp.route("/sessions/<session_id>/axial-labels")
def get_axial_labels(session_id: str):
    """Return axial labels for a session, or empty list if none exist."""
    try:
        session_dir = get_session_dir(
            current_app.config["SESSIONS_DIR"], session_id
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    labels_path = session_dir / "axial-labels.json"
    if not labels_path.exists():
        return jsonify({"labels": []})

    try:
        with open(labels_path) as f:
            data = json.load(f)
        return jsonify({"labels": data.get("labels", [])})
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Corrupt axial-labels.json: {e}"}), 500


@bp.route("/sessions/<session_id>/axial-labels", methods=["POST"])
def save_axial_labels(session_id: str):
    """Save axial labels for a session (full array replacement)."""
    try:
        session_dir = get_session_dir(
            current_app.config["SESSIONS_DIR"], session_id
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    body = request.get_json(silent=True)
    if body is None or "labels" not in body:
        return jsonify({"error": "Request body must contain 'labels' array"}), 400

    if not isinstance(body["labels"], list):
        return jsonify({"error": "'labels' must be an array"}), 400

    labels_path = session_dir / "axial-labels.json"
    _backup_if_legacy_shape(labels_path)
    with open(labels_path, "w") as f:
        json.dump({"labels": body["labels"]}, f, indent=2)

    return jsonify({"saved": len(body["labels"])})
