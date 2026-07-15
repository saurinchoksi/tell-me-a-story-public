"""Speaker confirmation endpoint — batch-confirm speaker assignments."""

import json

from flask import Blueprint, current_app, jsonify, request

from api.helpers import get_session_dir

bp = Blueprint("speakers", __name__)

VALID_ACTIONS = {"confirm", "confirm_variant", "create", "reassign", "skip"}
ACTIONS_REQUIRING_PROFILE = {"confirm", "confirm_variant", "reassign"}


@bp.route("/sessions/<session_id>/confirm-speakers", methods=["POST"])
def confirm_speakers(session_id: str):
    """Process a batch of speaker identity decisions for a session.

    Validates all decisions up front, applies changes atomically,
    then re-runs identification to return fresh results.
    """
    # --- Deferred imports (avoid loading torch/numpy at startup) ---
    from profiles import (
        load_profiles,
        save_profiles,
        create_profile,
        add_embedding,
        add_voice_variant,
    )
    from identify import identify_speakers, save_identifications

    # --- Validate session ---
    sessions_dir = current_app.config["SESSIONS_DIR"]
    try:
        session_dir = get_session_dir(sessions_dir, session_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    # --- Load embeddings ---
    embeddings_path = session_dir / "embeddings.json"
    if not embeddings_path.exists():
        return jsonify({"error": "No embeddings found for this session"}), 400

    with open(embeddings_path) as f:
        embeddings_data = json.load(f)
    speakers = embeddings_data.get("speakers", {})

    # --- Parse and validate request body ---
    body = request.get_json()
    if body is None or "decisions" not in body:
        return jsonify({"error": "Request body must contain 'decisions'"}), 400

    decisions = body["decisions"]

    for d in decisions:
        action = d.get("action")
        if action not in VALID_ACTIONS:
            return jsonify({"error": f"Unknown action: {action}"}), 400

        speaker_key = d.get("speaker_key")
        if action != "skip" and speaker_key not in speakers:
            return jsonify({"error": f"Speaker key not found in embeddings: {speaker_key}"}), 400

        if action in ACTIONS_REQUIRING_PROFILE and not d.get("profile_id"):
            return jsonify({"error": f"Action '{action}' requires 'profile_id'"}), 400

        if action == "create":
            if not d.get("new_name") or not d.get("new_role"):
                return jsonify({"error": "Action 'create' requires 'new_name' and 'new_role'"}), 400

    # --- Load profiles ---
    profiles_path = current_app.config["PROFILES_PATH"]
    profiles = load_profiles(profiles_path)

    # --- Validate profile IDs exist before applying any changes ---
    for d in decisions:
        if d.get("action") in ACTIONS_REQUIRING_PROFILE:
            profile_id = d["profile_id"]
            found = any(p["id"] == profile_id for p in profiles.get("profiles", []))
            if not found:
                return jsonify({"error": f"Profile not found: {profile_id}"}), 404

    # --- Apply decisions ---
    applied = 0
    skipped = 0
    created_profiles = []
    confirmed_map = {}  # speaker_key -> (action, profile_id)

    for d in decisions:
        action = d["action"]
        speaker_key = d.get("speaker_key")

        if action == "skip":
            skipped += 1
            continue

        speaker_embedding = speakers[speaker_key]
        embedding_data = {
            "vector": speaker_embedding["vector"],
            "session_id": session_id,
            "source_speaker_key": speaker_key,
        }

        if action in ("confirm", "reassign"):
            add_embedding(profiles, d["profile_id"], embedding_data)
            confirmed_map[speaker_key] = (action, d["profile_id"])
            applied += 1

        elif action == "confirm_variant":
            variant_data = {
                "session_id": session_id,
                "source_speaker_key": speaker_key,
                "vector": speaker_embedding["vector"],
            }
            add_voice_variant(profiles, d["profile_id"], variant_data)
            confirmed_map[speaker_key] = (action, d["profile_id"])
            applied += 1

        elif action == "create":
            new_id = create_profile(profiles, d["new_name"], d["new_role"])
            add_embedding(profiles, new_id, embedding_data)
            created_profiles.append(new_id)
            confirmed_map[speaker_key] = (action, new_id)
            applied += 1

    # --- Save profiles once ---
    save_profiles(profiles, profiles_path)

    # --- Re-run identification with updated profiles ---
    identifications = identify_speakers(
        str(embeddings_path), profiles_path=profiles_path
    )

    # --- Overlay confirmed human decisions onto algorithmic results ---
    for ident in identifications.get("identifications", []):
        if ident["speaker_key"] in confirmed_map:
            action, pid = confirmed_map[ident["speaker_key"]]
            ident["confirmed"] = True
            ident["confirmed_action"] = action
            ident["confirmed_profile_id"] = pid

    save_identifications(
        identifications, str(session_dir / "identifications.json")
    )

    return jsonify({
        "applied": applied,
        "skipped": skipped,
        "created_profiles": created_profiles,
        "identifications": identifications,
    })
