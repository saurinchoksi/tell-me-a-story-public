"""Tests for API profile endpoints."""

import json
from pathlib import Path

import pytest

from api.app import create_app


EMBEDDING_DIM = 256


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_profile(profile_id, name, role="parent", session_id="20260101-120000"):
    """Build a profile dict matching profiles.py's schema."""
    vec = [0.5] * EMBEDDING_DIM
    return {
        "id": profile_id,
        "name": name,
        "role": role,
        "created": "2026-01-01T00:00:00+00:00",
        "updated": "2026-01-01T00:00:00+00:00",
        "embeddings": [{
            "vector": vec,
            "session_id": session_id,
            "source_speaker_key": "SPEAKER_00",
        }],
        "centroid": vec,
        "voice_variants": [{
            "id": "var_abc",
            "created": "2026-01-01T00:00:00+00:00",
            "session_id": session_id,
            "source_speaker_key": "SPEAKER_01",
        }],
    }


def _make_identifications(session_id, entries):
    """Build identifications.json content."""
    return {
        "session_id": session_id,
        "identified_at": "2026-01-15T10:00:00+00:00",
        "profiles_used": 2,
        "identifications": entries,
    }


def _make_embeddings(speaker_keys):
    """Build a minimal embeddings.json dict."""
    return {
        "_generator_version": "test",
        "_dimension": EMBEDDING_DIM,
        "speakers": {
            key: {
                "vector": [0.1 * (i + 1)] * EMBEDDING_DIM,
                "num_segments": 5 + i,
                "total_duration_s": 30.0 + i * 10,
            }
            for i, key in enumerate(speaker_keys)
        },
    }


def _make_diarization(segments):
    """Build a minimal diarization.json dict."""
    return {"segments": segments}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path):
    """Flask test client with isolated profiles path (no sessions data)."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    profiles_path = str(tmp_path / "profiles.json")
    app = create_app(sessions_dir=sessions_dir, profiles_path=profiles_path)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def env(tmp_path):
    """Rich test environment with sessions, profiles, embeddings, diarization, identifications."""
    sessions_dir = tmp_path / "sessions"

    # --- Session 1: full artifacts ---
    s1_id = "20260101-120000"
    s1_dir = sessions_dir / s1_id
    s1_dir.mkdir(parents=True)
    (s1_dir / "audio.m4a").write_bytes(b"fake")
    (s1_dir / "embeddings.json").write_text(
        json.dumps(_make_embeddings(["SPEAKER_00", "SPEAKER_01"]))
    )
    (s1_dir / "diarization.json").write_text(json.dumps(_make_diarization([
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0},
        {"speaker": "SPEAKER_00", "start": 10.0, "end": 18.0},
        {"speaker": "SPEAKER_01", "start": 5.0, "end": 10.0},
        {"speaker": "SPEAKER_01", "start": 20.0, "end": 22.0},
    ])))
    (s1_dir / "identifications.json").write_text(json.dumps(_make_identifications(
        s1_id,
        [
            {"speaker_key": "SPEAKER_00", "status": "identified",
             "profile_id": "spk_aaa", "profile_name": "Saurin", "confidence": 0.85},
            {"speaker_key": "SPEAKER_01", "status": "suggested",
             "profile_id": "spk_bbb", "profile_name": "Robin", "confidence": 0.62},
        ],
    )))

    # --- Session 2: no identifications ---
    s2_id = "20260102-180000"
    s2_dir = sessions_dir / s2_id
    s2_dir.mkdir()
    (s2_dir / "audio.m4a").write_bytes(b"fake")
    (s2_dir / "embeddings.json").write_text(
        json.dumps(_make_embeddings(["SPEAKER_00"]))
    )

    # --- Profiles ---
    profiles_path = str(tmp_path / "profiles.json")
    profiles_data = {"profiles": [
        _make_profile("spk_aaa", "Saurin", "parent", s1_id),
        _make_profile("spk_bbb", "Robin", "child", s1_id),
    ]}
    Path(profiles_path).write_text(json.dumps(profiles_data))

    app = create_app(sessions_dir=str(sessions_dir), profiles_path=profiles_path)
    app.config["TESTING"] = True

    class Env:
        pass

    e = Env()
    e.tmp_path = tmp_path
    e.sessions_dir = sessions_dir
    e.s1_id = s1_id
    e.s2_id = s2_id
    e.s1_dir = s1_dir
    e.profiles_path = profiles_path
    e.app = app
    with app.test_client() as c:
        e.client = c
        yield e


# ---------------------------------------------------------------------------
# GET /api/profiles — basic tests
# ---------------------------------------------------------------------------

def test_list_profiles_empty(client):
    resp = client.get("/api/profiles")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["profiles"] == []


def test_list_profiles_strips_vectors(tmp_path):
    """Embedding vectors should be replaced with counts in list view."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text(json.dumps({"profiles": [{
        "id": "spk_abc123",
        "name": "Saurin",
        "role": "parent",
        "created": "2026-01-01T00:00:00+00:00",
        "updated": "2026-01-01T00:00:00+00:00",
        "embeddings": [
            {"vector": [0.1] * 256, "session_id": "s1"},
            {"vector": [0.2] * 256, "session_id": "s2"},
        ],
        "centroid": [0.15] * 256,
        "voice_variants": [{"id": "var_abc"}],
    }]}))

    app = create_app(sessions_dir=sessions_dir, profiles_path=str(profiles_path))
    app.config["TESTING"] = True
    with app.test_client() as c:
        resp = c.get("/api/profiles")
        profile = resp.get_json()["profiles"][0]
        # Vectors replaced with count
        assert profile["embeddings"] == 2
        assert profile["voice_variants"] == 1
        # Centroid stripped entirely
        assert "centroid" not in profile


# --- GET /api/profiles — match scores ---

def test_list_includes_match_scores(env):
    """Identifications present — verify scores returned."""
    resp = env.client.get("/api/profiles")
    profiles = resp.get_json()["profiles"]

    saurin = next(p for p in profiles if p["id"] == "spk_aaa")
    assert saurin["latest_match_score"] == 0.85
    assert saurin["latest_match_session"] == env.s1_id

    robin = next(p for p in profiles if p["id"] == "spk_bbb")
    assert robin["latest_match_score"] == 0.62


def test_list_no_matches_returns_null(client):
    """No identifications — scores are null."""
    # Create a profile via API
    client.post("/api/profiles", json={"name": "Test", "role": "parent"})
    resp = client.get("/api/profiles")
    profile = resp.get_json()["profiles"][0]
    assert profile["latest_match_score"] is None
    assert profile["latest_match_session"] is None


def test_list_last_seen_from_embeddings(env):
    """Last seen should be the most recent session_id from embeddings."""
    resp = env.client.get("/api/profiles")
    profiles = resp.get_json()["profiles"]
    saurin = next(p for p in profiles if p["id"] == "spk_aaa")
    assert saurin["last_seen"] == env.s1_id


# ---------------------------------------------------------------------------
# POST /api/profiles
# ---------------------------------------------------------------------------

def test_create_profile(client):
    resp = client.post("/api/profiles", json={"name": "Robin", "role": "child"})
    assert resp.status_code == 201
    data = resp.get_json()
    assert "profile_id" in data
    assert data["profile_id"].startswith("spk_")


def test_create_profile_persists(client):
    client.post("/api/profiles", json={"name": "Robin", "role": "child"})
    resp = client.get("/api/profiles")
    profiles = resp.get_json()["profiles"]
    assert len(profiles) == 1
    assert profiles[0]["name"] == "Robin"


def test_create_profile_missing_fields(client):
    resp = client.post("/api/profiles", json={"name": "Robin"})
    assert resp.status_code == 400

    resp = client.post("/api/profiles", json={"role": "child"})
    assert resp.status_code == 400


def test_create_profile_no_body(client):
    resp = client.post("/api/profiles", content_type="application/json")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# PUT /api/profiles/:id
# ---------------------------------------------------------------------------

def test_update_profile_name(client):
    resp = client.post("/api/profiles", json={"name": "Robin", "role": "child"})
    profile_id = resp.get_json()["profile_id"]

    resp = client.put(f"/api/profiles/{profile_id}", json={"name": "Riley"})
    assert resp.status_code == 200

    resp = client.get("/api/profiles")
    assert resp.get_json()["profiles"][0]["name"] == "Riley"


def test_update_profile_not_found(client):
    resp = client.put("/api/profiles/spk_nonexistent", json={"name": "X"})
    assert resp.status_code == 404


def test_update_profile_no_fields(client):
    resp = client.post("/api/profiles", json={"name": "Robin", "role": "child"})
    profile_id = resp.get_json()["profile_id"]

    resp = client.put(f"/api/profiles/{profile_id}", json={})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/profiles/:id — detail
# ---------------------------------------------------------------------------

def test_get_detail_happy_path(env):
    """All fields present in detail response."""
    resp = env.client.get("/api/profiles/spk_aaa")
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["id"] == "spk_aaa"
    assert data["name"] == "Saurin"
    assert data["role"] == "parent"
    assert data["created"] is not None
    assert data["updated"] is not None
    assert isinstance(data["embeddings"], list)
    assert isinstance(data["voice_variants"], list)


def test_get_detail_not_found(env):
    resp = env.client.get("/api/profiles/spk_nonexistent")
    assert resp.status_code == 404


def test_get_detail_embeddings_enriched(env):
    """Embeddings should include duration and segment count from session."""
    resp = env.client.get("/api/profiles/spk_aaa")
    data = resp.get_json()
    emb = data["embeddings"][0]
    assert emb["session_id"] == env.s1_id
    assert emb["source_speaker_key"] == "SPEAKER_00"
    assert emb["total_duration_s"] == 30.0
    assert emb["num_segments"] == 5


def test_get_detail_audio_sample(env):
    """Audio sample should be resolved from diarization."""
    resp = env.client.get("/api/profiles/spk_aaa")
    data = resp.get_json()
    sample = data["audio_sample"]
    assert sample is not None
    assert sample["session_id"] == env.s1_id
    assert isinstance(sample["start"], (int, float))
    assert isinstance(sample["end"], (int, float))
    # Should pick the 8s segment (10-18) as representative (>= 3s, near midpoint)
    assert sample["start"] == 10.0
    assert sample["end"] == 18.0


def test_get_detail_no_diarization(tmp_path):
    """Audio sample is null when no diarization exists."""
    sessions_dir = tmp_path / "sessions"
    s_id = "20260101-120000"
    s_dir = sessions_dir / s_id
    s_dir.mkdir(parents=True)
    (s_dir / "embeddings.json").write_text(
        json.dumps(_make_embeddings(["SPEAKER_00"]))
    )
    # No diarization.json

    profiles_path = str(tmp_path / "profiles.json")
    Path(profiles_path).write_text(json.dumps({
        "profiles": [_make_profile("spk_aaa", "Test", "parent", s_id)]
    }))

    app = create_app(sessions_dir=str(sessions_dir), profiles_path=profiles_path)
    app.config["TESTING"] = True
    with app.test_client() as c:
        resp = c.get("/api/profiles/spk_aaa")
        assert resp.get_json()["audio_sample"] is None


def test_get_detail_latest_match(env):
    """Match data from identifications should appear."""
    resp = env.client.get("/api/profiles/spk_aaa")
    data = resp.get_json()
    match = data["latest_match"]
    assert match is not None
    assert match["session_id"] == env.s1_id
    assert match["confidence"] == 0.85


def test_get_detail_voice_variants(env):
    """Voice variants should be returned with metadata, no vectors."""
    resp = env.client.get("/api/profiles/spk_aaa")
    data = resp.get_json()
    variants = data["voice_variants"]
    assert len(variants) == 1
    assert variants[0]["id"] == "var_abc"
    assert variants[0]["session_id"] == env.s1_id
    assert "vector" not in variants[0]


# ---------------------------------------------------------------------------
# DELETE /api/profiles/:id
# ---------------------------------------------------------------------------

def test_delete_profile(env):
    """Profile should be removed from file."""
    resp = env.client.delete("/api/profiles/spk_aaa")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True

    # Verify removed
    resp = env.client.get("/api/profiles")
    ids = [p["id"] for p in resp.get_json()["profiles"]]
    assert "spk_aaa" not in ids


def test_delete_not_found(env):
    resp = env.client.delete("/api/profiles/spk_nonexistent")
    assert resp.status_code == 404


def test_delete_cleans_identifications(env):
    """Identifications referencing the deleted profile should be reset."""
    env.client.delete("/api/profiles/spk_aaa")

    ident_file = env.s1_dir / "identifications.json"
    ident_data = json.loads(ident_file.read_text())
    speaker_00 = next(
        e for e in ident_data["identifications"]
        if e["speaker_key"] == "SPEAKER_00"
    )
    assert speaker_00["profile_id"] is None
    assert speaker_00["status"] == "unknown"
    assert speaker_00["confidence"] is None


def test_delete_no_identifications(env):
    """Sessions without identifications should be skipped gracefully."""
    # Session 2 has no identifications.json
    resp = env.client.delete("/api/profiles/spk_bbb")
    data = resp.get_json()
    assert data["success"] is True
    # Only session 1 had identifications referencing spk_bbb
    assert data["sessions_updated"] == 1


# ---------------------------------------------------------------------------
# POST /api/profiles/:id/refresh-centroid
# ---------------------------------------------------------------------------

def test_refresh_centroid(env):
    resp = env.client.post("/api/profiles/spk_aaa/refresh-centroid")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["has_centroid"] is True


def test_refresh_centroid_no_embeddings(env):
    """Profile with no embeddings should get null centroid."""
    # Remove all embeddings from profile first
    profiles = json.loads(Path(env.profiles_path).read_text())
    for p in profiles["profiles"]:
        if p["id"] == "spk_aaa":
            p["embeddings"] = []
            break
    Path(env.profiles_path).write_text(json.dumps(profiles))

    resp = env.client.post("/api/profiles/spk_aaa/refresh-centroid")
    data = resp.get_json()
    assert data["has_centroid"] is False


def test_refresh_centroid_not_found(env):
    resp = env.client.post("/api/profiles/spk_nonexistent/refresh-centroid")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/profiles/:id/embeddings/:session_id
# ---------------------------------------------------------------------------

def test_remove_embedding(env):
    """Embedding should be removed and centroid recomputed."""
    resp = env.client.delete(f"/api/profiles/spk_aaa/embeddings/{env.s1_id}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["embeddings_remaining"] == 0


def test_remove_embedding_not_found(env):
    resp = env.client.delete("/api/profiles/spk_aaa/embeddings/20260199-000000")
    assert resp.status_code == 404
    assert "embedding" in resp.get_json()["error"].lower()


def test_remove_embedding_profile_not_found(env):
    resp = env.client.delete(f"/api/profiles/spk_nonexistent/embeddings/{env.s1_id}")
    assert resp.status_code == 404
    assert "profile" in resp.get_json()["error"].lower()


def test_remove_last_embedding_clears_centroid(env):
    """After removing the last embedding, centroid should be null."""
    env.client.delete(f"/api/profiles/spk_aaa/embeddings/{env.s1_id}")

    profiles = json.loads(Path(env.profiles_path).read_text())
    saurin = next(p for p in profiles["profiles"] if p["id"] == "spk_aaa")
    assert saurin["centroid"] is None
    assert len(saurin["embeddings"]) == 0
