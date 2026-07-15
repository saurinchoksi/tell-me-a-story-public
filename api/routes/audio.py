"""Audio endpoint — stream session audio files."""

from flask import Blueprint, current_app, jsonify, send_file

from api.helpers import get_session_dir

bp = Blueprint("audio", __name__)


@bp.route("/sessions/<session_id>/audio")
def get_audio(session_id: str):
    """Stream the audio file for a session.

    Flask handles Content-Type detection and HTTP range requests automatically,
    which is what the HTML5 <audio> element needs for seeking.
    """
    try:
        session_dir = get_session_dir(
            current_app.config["SESSIONS_DIR"], session_id
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    audio_path = next(session_dir.glob("audio.*"), None)
    if audio_path is None:
        return jsonify({"error": "Audio not found"}), 404

    return send_file(audio_path)
