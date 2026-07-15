"""Profile endpoints — list, create, update, delete, and detail views."""

import json
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

bp = Blueprint("profiles", __name__)


def _get_profiles_path() -> str:
    return current_app.config["PROFILES_PATH"]


def _get_sessions_dir() -> Path:
    return Path(current_app.config["SESSIONS_DIR"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scan_latest_matches(sessions_dir: Path, profile_ids: set[str]) -> dict:
    """Scan identifications.json across sessions, find the most recent match per profile.

    Returns {profile_id: {"confidence": float, "session_id": str, "identified_at": str}}.
    """
    matches: dict[str, dict] = {}

    if not sessions_dir.exists():
        return matches

    for session_path in sessions_dir.iterdir():
        if not session_path.is_dir():
            continue
        ident_file = session_path / "identifications.json"
        if not ident_file.exists():
            continue

        try:
            ident_data = json.loads(ident_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        identified_at = ident_data.get("identified_at", "")
        session_id = ident_data.get("session_id", session_path.name)

        for entry in ident_data.get("identifications", []):
            pid = entry.get("profile_id")
            if pid not in profile_ids:
                continue
            confidence = entry.get("confidence")
            if confidence is None:
                continue

            existing = matches.get(pid)
            if existing is None or identified_at > existing["identified_at"]:
                matches[pid] = {
                    "confidence": confidence,
                    "session_id": session_id,
                    "identified_at": identified_at,
                }

    return matches


def _find_representative_segment(segments: list[dict], speaker_key: str) -> dict | None:
    """Pick a representative audio segment for the speaker.

    Ported from SpeakerCard.tsx: finds the temporal midpoint of the speaker's
    speech, then prefers a segment >= 3s near that midpoint.
    """
    speaker_segs = [s for s in segments if s.get("speaker") == speaker_key]
    if not speaker_segs:
        return None

    total_duration = sum(s["end"] - s["start"] for s in speaker_segs)
    mid_target = total_duration / 2

    cumulative = 0.0
    mid_segment = speaker_segs[0]
    for seg in speaker_segs:
        cumulative += seg["end"] - seg["start"]
        if cumulative >= mid_target:
            mid_segment = seg
            break

    # Prefer a segment >= 3s near the midpoint
    long_segments = [s for s in speaker_segs if (s["end"] - s["start"]) >= 3]
    if long_segments:
        long_segments.sort(
            key=lambda s: abs(s["start"] - mid_segment["start"])
        )
        return long_segments[0]

    return mid_segment


def _enrich_embedding(embedding: dict, sessions_dir: Path) -> dict:
    """Augment an embedding entry with duration/segment count from session data."""
    session_id = embedding.get("session_id")
    source_key = embedding.get("source_speaker_key")
    info = {
        "session_id": session_id,
        "source_speaker_key": source_key,
        "total_duration_s": None,
        "num_segments": None,
    }

    if not session_id:
        return info

    emb_file = sessions_dir / session_id / "embeddings.json"
    if not emb_file.exists():
        return info

    try:
        emb_data = json.loads(emb_file.read_text())
    except (json.JSONDecodeError, OSError):
        return info

    # Find this speaker in the session's embeddings
    speakers = emb_data.get("speakers", {})
    speaker_data = speakers.get(source_key) if source_key else None
    if speaker_data:
        info["total_duration_s"] = speaker_data.get("total_duration_s")
        info["num_segments"] = speaker_data.get("num_segments")

    return info


# ---------------------------------------------------------------------------
# GET /api/profiles — list (augmented with match scores)
# ---------------------------------------------------------------------------

@bp.route("/profiles")
def list_profiles():
    """Return all profiles with vectors stripped, augmented with match scores."""
    from profiles import load_profiles

    data = load_profiles(_get_profiles_path())
    profiles_list = data.get("profiles", [])

    # Collect profile IDs for match scanning
    profile_ids = {p["id"] for p in profiles_list}
    sessions_dir = _get_sessions_dir()
    matches = _scan_latest_matches(sessions_dir, profile_ids) if profile_ids else {}

    stripped = []
    for profile in profiles_list:
        p = {**profile}
        p["embeddings"] = len(profile.get("embeddings", []))
        p.pop("centroid", None)
        p["voice_variants"] = len(profile.get("voice_variants", []))

        # Match scores
        match = matches.get(profile["id"])
        p["latest_match_score"] = match["confidence"] if match else None
        p["latest_match_session"] = match["session_id"] if match else None

        # Last seen — most recent session_id from embeddings list
        emb_list = profile.get("embeddings", [])
        session_ids = [e.get("session_id") for e in emb_list if e.get("session_id")]
        p["last_seen"] = max(session_ids) if session_ids else None

        stripped.append(p)

    return jsonify({"profiles": stripped})


# ---------------------------------------------------------------------------
# GET /api/profiles/:id — detail
# ---------------------------------------------------------------------------

@bp.route("/profiles/<profile_id>")
def get_profile(profile_id: str):
    """Return full profile with enriched embeddings and audio sample."""
    from profiles import load_profiles

    data = load_profiles(_get_profiles_path())

    target = None
    for p in data.get("profiles", []):
        if p["id"] == profile_id:
            target = p
            break

    if target is None:
        return jsonify({"error": f"Profile not found: {profile_id}"}), 404

    sessions_dir = _get_sessions_dir()

    # Enrich embeddings with session metadata
    enriched_embeddings = [
        _enrich_embedding(e, sessions_dir)
        for e in target.get("embeddings", [])
    ]

    # Strip voice variant vectors, keep metadata
    voice_variants = []
    for v in target.get("voice_variants", []):
        voice_variants.append({
            "id": v.get("id"),
            "created": v.get("created"),
            "session_id": v.get("session_id"),
            "source_speaker_key": v.get("source_speaker_key"),
        })

    # Audio sample from most recent embedding's diarization
    audio_sample = None
    if enriched_embeddings:
        # Use the most recent embedding (last by session_id sort)
        by_session = sorted(
            enriched_embeddings,
            key=lambda e: e.get("session_id") or "",
            reverse=True,
        )
        recent = by_session[0]
        sid = recent.get("session_id")
        spk = recent.get("source_speaker_key")
        if sid and spk:
            diar_file = sessions_dir / sid / "diarization.json"
            if diar_file.exists():
                try:
                    diar_data = json.loads(diar_file.read_text())
                    seg = _find_representative_segment(
                        diar_data.get("segments", []), spk
                    )
                    if seg:
                        audio_sample = {
                            "session_id": sid,
                            "start": seg["start"],
                            "end": seg["end"],
                        }
                except (json.JSONDecodeError, OSError):
                    pass

    # Latest match
    matches = _scan_latest_matches(sessions_dir, {profile_id})
    match = matches.get(profile_id)
    latest_match = None
    if match:
        latest_match = {
            "session_id": match["session_id"],
            "confidence": match["confidence"],
            "identified_at": match["identified_at"],
        }

    return jsonify({
        "id": target["id"],
        "name": target["name"],
        "role": target["role"],
        "created": target.get("created"),
        "updated": target.get("updated"),
        "embeddings": enriched_embeddings,
        "voice_variants": voice_variants,
        "audio_sample": audio_sample,
        "latest_match": latest_match,
    })


# ---------------------------------------------------------------------------
# POST /api/profiles — create
# ---------------------------------------------------------------------------

@bp.route("/profiles", methods=["POST"])
def create_new_profile():
    """Create a new speaker profile. Body: {name, role}."""
    from profiles import load_profiles, save_profiles, create_profile

    body = request.get_json()
    if body is None:
        return jsonify({"error": "Missing JSON body"}), 400

    name = body.get("name")
    role = body.get("role")
    if not name or not role:
        return jsonify({"error": "Both 'name' and 'role' are required"}), 400

    path = _get_profiles_path()
    profiles = load_profiles(path)
    profile_id = create_profile(profiles, name, role)
    save_profiles(profiles, path)

    return jsonify({"profile_id": profile_id}), 201


# ---------------------------------------------------------------------------
# PUT /api/profiles/:id — update
# ---------------------------------------------------------------------------

@bp.route("/profiles/<profile_id>", methods=["PUT"])
def update_profile(profile_id: str):
    """Update a profile's name and/or role. Body: {name?, role?}."""
    from profiles import load_profiles, save_profiles

    body = request.get_json()
    if body is None:
        return jsonify({"error": "Missing JSON body"}), 400

    if "name" not in body and "role" not in body:
        return jsonify({"error": "Provide at least 'name' or 'role' to update"}), 400

    path = _get_profiles_path()
    profiles = load_profiles(path)

    target = None
    for p in profiles.get("profiles", []):
        if p["id"] == profile_id:
            target = p
            break

    if target is None:
        return jsonify({"error": f"Profile not found: {profile_id}"}), 404

    if "name" in body:
        target["name"] = body["name"]
    if "role" in body:
        target["role"] = body["role"]
    target["updated"] = datetime.now(timezone.utc).isoformat()

    save_profiles(profiles, path)

    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# DELETE /api/profiles/:id — delete with cascade
# ---------------------------------------------------------------------------

@bp.route("/profiles/<profile_id>", methods=["DELETE"])
def delete_profile(profile_id: str):
    """Delete a profile and reset its identifications across all sessions."""
    from profiles import load_profiles, save_profiles

    path = _get_profiles_path()
    profiles = load_profiles(path)

    # Find and remove profile
    profile_list = profiles.get("profiles", [])
    original_len = len(profile_list)
    profiles["profiles"] = [p for p in profile_list if p["id"] != profile_id]

    if len(profiles["profiles"]) == original_len:
        return jsonify({"error": f"Profile not found: {profile_id}"}), 404

    save_profiles(profiles, path)

    # Cascade: reset identifications referencing this profile
    sessions_dir = _get_sessions_dir()
    sessions_updated = 0

    if sessions_dir.exists():
        for session_path in sessions_dir.iterdir():
            if not session_path.is_dir():
                continue
            ident_file = session_path / "identifications.json"
            if not ident_file.exists():
                continue

            try:
                ident_data = json.loads(ident_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            modified = False
            for entry in ident_data.get("identifications", []):
                if entry.get("profile_id") == profile_id:
                    entry["profile_id"] = None
                    entry["profile_name"] = None
                    entry["status"] = "unknown"
                    entry["confidence"] = None
                    modified = True

            if modified:
                with open(ident_file, "w") as f:
                    json.dump(ident_data, f, indent=2)
                    f.write("\n")
                sessions_updated += 1

    return jsonify({"success": True, "sessions_updated": sessions_updated})


# ---------------------------------------------------------------------------
# POST /api/profiles/:id/refresh-centroid
# ---------------------------------------------------------------------------

@bp.route("/profiles/<profile_id>/refresh-centroid", methods=["POST"])
def refresh_centroid(profile_id: str):
    """Recompute the centroid from current embeddings."""
    from profiles import load_profiles, save_profiles, compute_centroid

    path = _get_profiles_path()
    profiles = load_profiles(path)

    target = None
    for p in profiles.get("profiles", []):
        if p["id"] == profile_id:
            target = p
            break

    if target is None:
        return jsonify({"error": f"Profile not found: {profile_id}"}), 404

    target["centroid"] = compute_centroid(target)
    target["updated"] = datetime.now(timezone.utc).isoformat()
    save_profiles(profiles, path)

    return jsonify({"success": True, "has_centroid": target["centroid"] is not None})


# ---------------------------------------------------------------------------
# DELETE /api/profiles/:id/embeddings/:session_id — remove enrollment
# ---------------------------------------------------------------------------

@bp.route("/profiles/<profile_id>/embeddings/<session_id>", methods=["DELETE"])
def remove_embedding(profile_id: str, session_id: str):
    """Remove an embedding by session_id, recompute centroid."""
    from profiles import load_profiles, save_profiles, compute_centroid

    path = _get_profiles_path()
    profiles = load_profiles(path)

    target = None
    for p in profiles.get("profiles", []):
        if p["id"] == profile_id:
            target = p
            break

    if target is None:
        return jsonify({"error": f"Profile not found: {profile_id}"}), 404

    emb_list = target.get("embeddings", [])
    original_len = len(emb_list)
    target["embeddings"] = [e for e in emb_list if e.get("session_id") != session_id]

    if len(target["embeddings"]) == original_len:
        return jsonify({"error": f"No embedding found for session: {session_id}"}), 404

    target["centroid"] = compute_centroid(target)
    target["updated"] = datetime.now(timezone.utc).isoformat()
    save_profiles(profiles, path)

    return jsonify({
        "success": True,
        "embeddings_remaining": len(target["embeddings"]),
    })
