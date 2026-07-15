"""Speaker profile storage — persist and manage cross-session speaker identities.

Profiles live at data/speaker_profiles.json as a single project-wide file.
Each profile stores a name, role, multiple embeddings from confirmed sessions,
a computed centroid, and voice variants. The centroid is computed from embeddings
only — voice variants are preserved but excluded from the centroid so character
voices don't pull it away from the speaker's natural voice.
"""

import copy
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_PROFILES_PATH = str(
    Path(__file__).parent.parent / "data" / "speaker_profiles.json"
)


def load_profiles(path: str = _DEFAULT_PROFILES_PATH) -> dict:
    """Load speaker profiles from JSON.

    Args:
        path: Path to the profiles JSON file

    Returns:
        Dict with a "profiles" list. Returns {"profiles": []} if file
        is missing or empty.

    Raises:
        json.JSONDecodeError: If file contains malformed JSON
    """
    p = Path(path)
    if not p.exists():
        return {"profiles": []}

    raw = p.read_text()
    if not raw.strip():
        return {"profiles": []}

    data = json.loads(raw)
    data.setdefault("profiles", [])
    return data


def save_profiles(profiles: dict, path: str = _DEFAULT_PROFILES_PATH) -> None:
    """Write speaker profiles to JSON.

    Deep-copies the data before writing to prevent serialization side effects.

    Args:
        profiles: Dict with a "profiles" list
        path: Destination path
    """
    data = copy.deepcopy(profiles)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _find_profile(profiles: dict, profile_id: str) -> dict:
    """Look up a profile by ID, returning a reference to it.

    Raises:
        KeyError: If no profile matches the given ID
    """
    for profile in profiles.get("profiles", []):
        if profile["id"] == profile_id:
            return profile
    raise KeyError(f"Profile not found: {profile_id}")


def compute_centroid(profile: dict) -> list[float] | None:
    """Compute the element-wise average of a profile's embedding vectors.

    Args:
        profile: A single profile dict (not the top-level container)

    Returns:
        List of floats (the averaged vector), or None if no embeddings
    """
    embeddings = profile.get("embeddings", [])
    if not embeddings:
        return None
    vectors = [e["vector"] for e in embeddings]
    n = len(vectors)
    return [sum(components) / n for components in zip(*vectors)]


def create_profile(profiles: dict, name: str, role: str) -> str:
    """Add a new speaker profile.

    Mutates the profiles dict in place.

    Args:
        profiles: Top-level profiles container
        name: Display name for the speaker
        role: Freeform role text ("parent", "child", etc.)

    Returns:
        The generated profile ID (e.g. "spk_a1b2c3")
    """
    profile_id = "spk_" + secrets.token_hex(3)
    now = datetime.now(timezone.utc).isoformat()
    profile = {
        "id": profile_id,
        "name": name,
        "role": role,
        "created": now,
        "updated": now,
        "embeddings": [],
        "centroid": None,
        "voice_variants": [],
    }
    profiles.setdefault("profiles", []).append(profile)
    return profile_id


def add_embedding(profiles: dict, profile_id: str, embedding_data: dict) -> None:
    """Add a confirmed embedding to a profile and recompute the centroid.

    Silently skips if an embedding with the same session_id already exists
    (idempotent for pipeline re-runs).

    Args:
        profiles: Top-level profiles container
        profile_id: ID of the target profile
        embedding_data: Dict with at least "vector" (list of floats).
            May also include "session_id", "source_speaker_key", etc.

    Raises:
        KeyError: If profile_id not found
    """
    profile = _find_profile(profiles, profile_id)

    session_id = embedding_data.get("session_id")
    if session_id is not None:
        for existing in profile["embeddings"]:
            if existing.get("session_id") == session_id:
                return

    profile["embeddings"].append(embedding_data)
    profile["centroid"] = compute_centroid(profile)
    profile["updated"] = datetime.now(timezone.utc).isoformat()


def add_voice_variant(profiles: dict, profile_id: str, variant_data: dict) -> None:
    """Add a voice variant (merged fragment) to a profile.

    Does NOT recompute the centroid — variants are acoustic fragments preserved
    for reference but excluded from the speaker's identity vector.

    Idempotent: if a variant with the same session_id already exists, the call
    is silently skipped (safe to retry).

    Args:
        profiles: Top-level profiles container
        profile_id: ID of the target profile
        variant_data: Dict describing the voice variant

    Raises:
        KeyError: If profile_id not found
    """
    profile = _find_profile(profiles, profile_id)

    session_id = variant_data.get("session_id")
    if session_id is not None:
        for existing in profile["voice_variants"]:
            if existing.get("session_id") == session_id:
                return

    variant_data.setdefault("id", "var_" + secrets.token_hex(3))
    variant_data.setdefault("created", datetime.now(timezone.utc).isoformat())
    profile["voice_variants"].append(variant_data)
    profile["updated"] = datetime.now(timezone.utc).isoformat()
