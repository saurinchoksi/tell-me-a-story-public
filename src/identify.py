"""Speaker identification — match session embeddings to known profiles.

Given a session's embeddings and the current speaker profiles, proposes
who each speaker is with a confidence score. Not a pipeline step — called
on demand by UI or CLI.

Confidence tiers:
  >= 0.75  "identified"  — high confidence, safe for automation
  >= 0.45  "suggested"   — plausible match, needs human confirmation
  <  0.45  "unknown"     — no confident match
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from profiles import load_profiles

logger = logging.getLogger(__name__)

THRESHOLD_IDENTIFIED = 0.75
THRESHOLD_SUGGESTED = 0.45


def cosine_similarity(vec_a, vec_b) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First vector (list or array, 256-dim).
        vec_b: Second vector (list or array, 256-dim).

    Returns:
        Similarity as a plain Python float in [-1, 1].

    Raises:
        ValueError: If vectors have different dimensions.
    """
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(
            f"Dimension mismatch: {a.shape[0]} vs {b.shape[0]}"
        )

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def identify_speakers(embeddings_path, profiles_path=None) -> dict:
    """Match session speaker embeddings against known profiles.

    Args:
        embeddings_path: Path to session's embeddings.json.
        profiles_path: Path to speaker_profiles.json. If None, uses
            the default path from profiles.load_profiles().

    Returns:
        Dict with identification results per speaker.
    """
    embeddings_path = Path(embeddings_path)
    session_id = embeddings_path.parent.name

    # Graceful handling of missing embeddings
    if not embeddings_path.exists():
        logger.warning(f"Embeddings file not found: {embeddings_path}")
        return {
            "session_id": session_id,
            "identified_at": datetime.now(timezone.utc).isoformat(),
            "profiles_used": 0,
            "identifications": [],
        }

    with open(embeddings_path) as f:
        embeddings = json.load(f)

    # Load profiles (handles missing/empty gracefully)
    if profiles_path is not None:
        profiles_data = load_profiles(profiles_path)
    else:
        profiles_data = load_profiles()

    # Build centroid lookup — only profiles with computed centroids
    centroid_lookup = {}
    for profile in profiles_data.get("profiles", []):
        if profile.get("centroid") is not None:
            centroid_lookup[profile["id"]] = profile

    if not centroid_lookup:
        return _cold_start_result(session_id, embeddings, profiles_used=0)

    # Match each speaker against all profile centroids
    identifications = []
    speakers = embeddings.get("speakers", {})

    for speaker_key in sorted(speakers.keys()):
        speaker_vec = speakers[speaker_key]["vector"]
        best_sim = -1.0
        best_profile = None

        for profile_id, profile in centroid_lookup.items():
            sim = cosine_similarity(speaker_vec, profile["centroid"])
            if sim > best_sim:
                best_sim = sim
                best_profile = profile

        confidence = round(best_sim, 4)

        if best_sim >= THRESHOLD_IDENTIFIED:
            status = "identified"
        elif best_sim >= THRESHOLD_SUGGESTED:
            status = "suggested"
        else:
            status = "unknown"

        if status == "unknown":
            identifications.append({
                "speaker_key": speaker_key,
                "status": status,
                "profile_id": None,
                "profile_name": None,
                "confidence": None,
            })
        else:
            identifications.append({
                "speaker_key": speaker_key,
                "status": status,
                "profile_id": best_profile["id"],
                "profile_name": best_profile["name"],
                "confidence": confidence,
            })

    return {
        "session_id": session_id,
        "identified_at": datetime.now(timezone.utc).isoformat(),
        "profiles_used": len(centroid_lookup),
        "identifications": identifications,
    }


def _cold_start_result(session_id, embeddings, profiles_used=0) -> dict:
    """Build all-unknown identifications when no centroids are available."""
    speakers = embeddings.get("speakers", {})
    identifications = []

    for speaker_key in sorted(speakers.keys()):
        identifications.append({
            "speaker_key": speaker_key,
            "status": "unknown",
            "profile_id": None,
            "profile_name": None,
            "confidence": None,
        })

    return {
        "session_id": session_id,
        "identified_at": datetime.now(timezone.utc).isoformat(),
        "profiles_used": profiles_used,
        "identifications": identifications,
    }


def save_identifications(results, output_path) -> None:
    """Write identification results to JSON.

    Args:
        results: Dict from identify_speakers().
        output_path: Destination file path.
    """
    dirname = Path(output_path).parent
    dirname.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
