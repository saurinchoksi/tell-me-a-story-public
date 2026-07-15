"""Tests for src/identify.py — speaker identification logic."""

import json
import math
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from identify import (
    THRESHOLD_IDENTIFIED,
    THRESHOLD_SUGGESTED,
    cosine_similarity,
    identify_speakers,
    save_identifications,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 256


def _unit_vector(dim=DIM):
    """A deterministic unit vector: [1/sqrt(dim), ...] * dim."""
    val = 1.0 / math.sqrt(dim)
    return [val] * dim


def _vector_with_similarity(base, target_sim):
    """Construct a vector with approximately target_sim cosine similarity to base.

    Strategy: blend base with an orthogonal vector. For unit base [v,v,...v]:
    - orthogonal = [-v, v, -v, v, ...] (alternating signs, same magnitude)
    - result = target_sim * base + sqrt(1 - target_sim^2) * orthogonal
    This gives exact cosine similarity when base is a unit vector.
    """
    base = np.asarray(base, dtype=np.float64)
    # Build orthogonal vector by alternating signs
    ortho = base.copy()
    ortho[::2] *= -1
    # Normalize orthogonal component
    ortho_norm = np.linalg.norm(ortho)
    if ortho_norm > 0:
        ortho = ortho / ortho_norm

    base_norm = np.linalg.norm(base)
    if base_norm > 0:
        base_unit = base / base_norm
    else:
        base_unit = base

    # Gram-Schmidt to ensure true orthogonality
    ortho = ortho - np.dot(ortho, base_unit) * base_unit
    ortho_norm = np.linalg.norm(ortho)
    if ortho_norm > 0:
        ortho = ortho / ortho_norm

    complement = math.sqrt(max(0, 1.0 - target_sim ** 2))
    result = target_sim * base_unit + complement * ortho
    return result.tolist()


def _embeddings_data(speakers_dict):
    """Build an embeddings.json-shaped dict."""
    return {
        "_generator_version": "test",
        "_dimension": DIM,
        "speakers": speakers_dict,
    }


def _speaker_entry(vector=None):
    """Build a speaker entry for embeddings data."""
    if vector is None:
        vector = _unit_vector()
    return {
        "vector": vector,
        "num_segments": 5,
        "total_duration_s": 60.0,
    }


def _profiles_with_centroids(entries):
    """Build a profiles dict.

    entries: list of (profile_id, name, centroid_vector_or_None)
    """
    profiles = []
    for pid, name, centroid in entries:
        profiles.append({
            "id": pid,
            "name": name,
            "role": "test",
            "created": "2026-01-01T00:00:00+00:00",
            "updated": "2026-01-01T00:00:00+00:00",
            "embeddings": [],
            "centroid": centroid,
            "voice_variants": [],
        })
    return {"profiles": profiles}


def _write_json(path, data):
    """Write data as JSON to path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    """Tests 1-6: cosine_similarity function."""

    def test_identical_vectors(self):
        """Test 1: identical vectors → 1.0."""
        vec = _unit_vector()
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Test 2: orthogonal vectors → 0.0."""
        a = [1.0, 0.0] + [0.0] * (DIM - 2)
        b = [0.0, 1.0] + [0.0] * (DIM - 2)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Test 3: opposite vectors → -1.0."""
        a = _unit_vector()
        b = [-x for x in a]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_known_similarity(self):
        """Test 4: constructed vector with known similarity ≈ 0.85."""
        base = _unit_vector()
        other = _vector_with_similarity(base, 0.85)
        result = cosine_similarity(base, other)
        assert result == pytest.approx(0.85, abs=1e-6)

    def test_dimension_mismatch(self):
        """Test 5: mismatched dimensions → ValueError."""
        a = [1.0] * 128
        b = [1.0] * DIM
        with pytest.raises(ValueError, match="Dimension mismatch"):
            cosine_similarity(a, b)

    def test_accepts_lists_returns_float(self):
        """Test 6: accepts plain lists, returns plain Python float."""
        a = [1.0] * DIM
        b = [2.0] * DIM
        result = cosine_similarity(a, b)
        assert isinstance(result, float)
        assert not isinstance(result, np.floating)


# ---------------------------------------------------------------------------
# identify_speakers
# ---------------------------------------------------------------------------

class TestIdentifySpeakers:
    """Tests 7-14: identify_speakers function."""

    def test_identified_zone(self):
        """Test 7: similarity > 0.75 → status 'identified'."""
        base = _unit_vector()
        # Build a speaker vector very close to the profile centroid
        speaker_vec = _vector_with_similarity(base, 0.90)

        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "sessions" / "20260218-185123"
            session_dir.mkdir(parents=True)

            emb_path = session_dir / "embeddings.json"
            _write_json(emb_path, _embeddings_data({
                "SPEAKER_00": _speaker_entry(speaker_vec),
            }))

            prof_path = Path(tmp) / "profiles.json"
            _write_json(prof_path, _profiles_with_centroids([
                ("spk_aaa", "Choksi", base),
            ]))

            result = identify_speakers(str(emb_path), str(prof_path))

        assert len(result["identifications"]) == 1
        ident = result["identifications"][0]
        assert ident["status"] == "identified"
        assert ident["profile_id"] == "spk_aaa"
        assert ident["profile_name"] == "Choksi"
        assert ident["confidence"] >= THRESHOLD_IDENTIFIED

    def test_suggested_zone(self):
        """Test 8: similarity 0.45–0.75 → status 'suggested'."""
        base = _unit_vector()
        speaker_vec = _vector_with_similarity(base, 0.60)

        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "sessions" / "20260218-185123"
            session_dir.mkdir(parents=True)

            emb_path = session_dir / "embeddings.json"
            _write_json(emb_path, _embeddings_data({
                "SPEAKER_00": _speaker_entry(speaker_vec),
            }))

            prof_path = Path(tmp) / "profiles.json"
            _write_json(prof_path, _profiles_with_centroids([
                ("spk_bbb", "Robin", base),
            ]))

            result = identify_speakers(str(emb_path), str(prof_path))

        ident = result["identifications"][0]
        assert ident["status"] == "suggested"
        assert ident["confidence"] >= THRESHOLD_SUGGESTED
        assert ident["confidence"] < THRESHOLD_IDENTIFIED

    def test_unknown_zone(self):
        """Test 9: similarity < 0.45 → status 'unknown', null fields."""
        base = _unit_vector()
        speaker_vec = _vector_with_similarity(base, 0.20)

        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "sessions" / "20260218-185123"
            session_dir.mkdir(parents=True)

            emb_path = session_dir / "embeddings.json"
            _write_json(emb_path, _embeddings_data({
                "SPEAKER_00": _speaker_entry(speaker_vec),
            }))

            prof_path = Path(tmp) / "profiles.json"
            _write_json(prof_path, _profiles_with_centroids([
                ("spk_ccc", "Guest", base),
            ]))

            result = identify_speakers(str(emb_path), str(prof_path))

        ident = result["identifications"][0]
        assert ident["status"] == "unknown"
        assert ident["profile_id"] is None
        assert ident["profile_name"] is None
        assert ident["confidence"] is None

    def test_cold_start_no_profiles_file(self):
        """Test 10: no profiles file → all unknown, profiles_used: 0."""
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "sessions" / "20260218-185123"
            session_dir.mkdir(parents=True)

            emb_path = session_dir / "embeddings.json"
            _write_json(emb_path, _embeddings_data({
                "SPEAKER_00": _speaker_entry(),
                "SPEAKER_01": _speaker_entry(),
            }))

            prof_path = str(Path(tmp) / "nonexistent_profiles.json")
            result = identify_speakers(str(emb_path), prof_path)

        assert result["profiles_used"] == 0
        assert len(result["identifications"]) == 2
        for ident in result["identifications"]:
            assert ident["status"] == "unknown"

    def test_missing_embeddings_file(self):
        """Test 11: missing embeddings file → empty identifications."""
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "sessions" / "20260218-185123"
            session_dir.mkdir(parents=True)
            emb_path = session_dir / "embeddings.json"
            # Don't create the file

            result = identify_speakers(str(emb_path))

        assert result["identifications"] == []
        assert result["profiles_used"] == 0

    def test_empty_profiles(self):
        """Test 12: empty profiles list → all unknown."""
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "sessions" / "20260218-185123"
            session_dir.mkdir(parents=True)

            emb_path = session_dir / "embeddings.json"
            _write_json(emb_path, _embeddings_data({
                "SPEAKER_00": _speaker_entry(),
            }))

            prof_path = Path(tmp) / "profiles.json"
            _write_json(prof_path, {"profiles": []})

            result = identify_speakers(str(emb_path), str(prof_path))

        assert result["profiles_used"] == 0
        assert result["identifications"][0]["status"] == "unknown"

    def test_null_centroid_skipped(self):
        """Test 13: profile with null centroid → skipped, excluded from profiles_used."""
        base = _unit_vector()

        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "sessions" / "20260218-185123"
            session_dir.mkdir(parents=True)

            emb_path = session_dir / "embeddings.json"
            _write_json(emb_path, _embeddings_data({
                "SPEAKER_00": _speaker_entry(base),
            }))

            prof_path = Path(tmp) / "profiles.json"
            _write_json(prof_path, _profiles_with_centroids([
                ("spk_new", "New Person", None),     # null centroid
                ("spk_known", "Known Person", base),  # valid centroid
            ]))

            result = identify_speakers(str(emb_path), str(prof_path))

        # Only the profile with a centroid is counted
        assert result["profiles_used"] == 1
        ident = result["identifications"][0]
        assert ident["profile_id"] == "spk_known"

    def test_two_speakers_same_profile(self):
        """Test 14: two speakers matching same profile → both reported."""
        base = _unit_vector()
        speaker_a = _vector_with_similarity(base, 0.95)
        speaker_b = _vector_with_similarity(base, 0.80)

        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "sessions" / "20260218-185123"
            session_dir.mkdir(parents=True)

            emb_path = session_dir / "embeddings.json"
            _write_json(emb_path, _embeddings_data({
                "SPEAKER_00": _speaker_entry(speaker_a),
                "SPEAKER_01": _speaker_entry(speaker_b),
            }))

            prof_path = Path(tmp) / "profiles.json"
            _write_json(prof_path, _profiles_with_centroids([
                ("spk_aaa", "Choksi", base),
            ]))

            result = identify_speakers(str(emb_path), str(prof_path))

        assert len(result["identifications"]) == 2
        # Both match the same profile but with different confidences
        ids = {i["speaker_key"]: i for i in result["identifications"]}
        assert ids["SPEAKER_00"]["profile_id"] == "spk_aaa"
        assert ids["SPEAKER_01"]["profile_id"] == "spk_aaa"
        assert ids["SPEAKER_00"]["confidence"] != ids["SPEAKER_01"]["confidence"]

    def test_boundary_at_identified_threshold(self):
        """Test 15b: similarity of exactly 0.75 → 'identified' (>= boundary)."""
        base = _unit_vector()
        speaker_vec = _vector_with_similarity(base, THRESHOLD_IDENTIFIED)

        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "sessions" / "20260218-185123"
            session_dir.mkdir(parents=True)

            emb_path = session_dir / "embeddings.json"
            _write_json(emb_path, _embeddings_data({
                "SPEAKER_00": _speaker_entry(speaker_vec),
            }))

            prof_path = Path(tmp) / "profiles.json"
            _write_json(prof_path, _profiles_with_centroids([
                ("spk_boundary", "Boundary Speaker", base),
            ]))

            result = identify_speakers(str(emb_path), str(prof_path))

        ident = result["identifications"][0]
        assert ident["status"] == "identified"

    def test_boundary_at_suggested_threshold(self):
        """Test 15c: similarity of exactly 0.45 → 'suggested' (>= boundary)."""
        base = _unit_vector()
        speaker_vec = _vector_with_similarity(base, THRESHOLD_SUGGESTED)

        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "sessions" / "20260218-185123"
            session_dir.mkdir(parents=True)

            emb_path = session_dir / "embeddings.json"
            _write_json(emb_path, _embeddings_data({
                "SPEAKER_00": _speaker_entry(speaker_vec),
            }))

            prof_path = Path(tmp) / "profiles.json"
            _write_json(prof_path, _profiles_with_centroids([
                ("spk_boundary", "Boundary Speaker", base),
            ]))

            result = identify_speakers(str(emb_path), str(prof_path))

        ident = result["identifications"][0]
        assert ident["status"] == "suggested"


# ---------------------------------------------------------------------------
# save_identifications
# ---------------------------------------------------------------------------

class TestSaveIdentifications:
    """Tests 15-16: save_identifications function."""

    def test_write_read_round_trip(self):
        """Test 15: write + read back → equal."""
        data = {
            "session_id": "20260218-185123",
            "identified_at": "2026-02-18T19:00:00+00:00",
            "profiles_used": 2,
            "identifications": [
                {
                    "speaker_key": "SPEAKER_00",
                    "status": "identified",
                    "profile_id": "spk_aaa",
                    "profile_name": "Choksi",
                    "confidence": 0.9123,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "identifications.json")
            save_identifications(data, path)

            with open(path) as f:
                loaded = json.load(f)

            assert loaded == data

    def test_creates_parent_directories(self):
        """Test 16: creates parent directories."""
        data = {"session_id": "test", "identifications": []}
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "nested" / "deep" / "identifications.json")
            save_identifications(data, path)
            assert Path(path).exists()


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Tests 17-18: output schema validation."""

    def test_session_id_from_path(self):
        """Test 17: session_id extracted from parent directory name."""
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "sessions" / "20260218-185123"
            session_dir.mkdir(parents=True)

            emb_path = session_dir / "embeddings.json"
            _write_json(emb_path, _embeddings_data({
                "SPEAKER_00": _speaker_entry(),
            }))

            prof_path = Path(tmp) / "profiles.json"
            _write_json(prof_path, {"profiles": []})

            result = identify_speakers(str(emb_path), str(prof_path))

        assert result["session_id"] == "20260218-185123"

    def test_identified_at_valid_iso(self):
        """Test 18: identified_at is a valid ISO timestamp."""
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "sessions" / "20260218-185123"
            session_dir.mkdir(parents=True)

            emb_path = session_dir / "embeddings.json"
            _write_json(emb_path, _embeddings_data({
                "SPEAKER_00": _speaker_entry(),
            }))

            prof_path = Path(tmp) / "profiles.json"
            _write_json(prof_path, {"profiles": []})

            result = identify_speakers(str(emb_path), str(prof_path))

        # Should parse without error
        ts = datetime.fromisoformat(result["identified_at"])
        assert ts.tzinfo is not None  # timezone-aware
