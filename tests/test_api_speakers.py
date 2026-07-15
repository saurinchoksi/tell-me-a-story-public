"""Tests for the batch speaker confirmation endpoint."""

import json
from pathlib import Path

import pytest

from api.app import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 256


def _make_embeddings(speaker_keys):
    """Build a minimal embeddings.json dict for the given speaker keys."""
    return {
        "_generator_version": "test",
        "_dimension": EMBEDDING_DIM,
        "speakers": {
            key: {"vector": [0.1 * (i + 1)] * EMBEDDING_DIM, "num_segments": 3}
            for i, key in enumerate(speaker_keys)
        },
    }


def _make_profile(profile_id, name, role="parent"):
    """Build a profile dict matching profiles.py's schema."""
    embedding_vec = [0.5] * EMBEDDING_DIM
    return {
        "id": profile_id,
        "name": name,
        "role": role,
        "created": "2026-01-01T00:00:00+00:00",
        "updated": "2026-01-01T00:00:00+00:00",
        "embeddings": [{"vector": embedding_vec, "session_id": "seed"}],
        "centroid": embedding_vec,
        "voice_variants": [],
    }


@pytest.fixture
def env(tmp_path):
    """Set up a session directory with embeddings and a profiles file.

    Returns a namespace-like dict with paths and the test client.
    """
    # Session with embeddings
    session_id = "20260101-120000"
    session_dir = tmp_path / "sessions" / session_id
    session_dir.mkdir(parents=True)
    (session_dir / "audio.m4a").write_bytes(b"fake")
    (session_dir / "embeddings.json").write_text(
        json.dumps(_make_embeddings(["SPEAKER_00", "SPEAKER_01"]))
    )

    # Session without embeddings
    bare_id = "20260102-180000"
    bare_dir = tmp_path / "sessions" / bare_id
    bare_dir.mkdir()
    (bare_dir / "audio.m4a").write_bytes(b"fake")

    # Pre-seed one profile
    profiles_path = str(tmp_path / "profiles.json")
    profiles_data = {"profiles": [_make_profile("spk_aaa111", "Saurin")]}
    Path(profiles_path).write_text(json.dumps(profiles_data))

    app = create_app(
        sessions_dir=str(tmp_path / "sessions"),
        profiles_path=profiles_path,
    )
    app.config["TESTING"] = True

    class Env:
        pass

    e = Env()
    e.tmp_path = tmp_path
    e.session_id = session_id
    e.bare_id = bare_id
    e.session_dir = session_dir
    e.profiles_path = profiles_path
    e.app = app
    with app.test_client() as c:
        e.client = c
        yield e


# ---------------------------------------------------------------------------
# Individual action types
# ---------------------------------------------------------------------------

def test_confirm_action(env):
    """Confirm assigns the speaker's embedding to the given profile."""
    resp = env.client.post(
        f"/api/sessions/{env.session_id}/confirm-speakers",
        json={"decisions": [
            {"speaker_key": "SPEAKER_00", "action": "confirm", "profile_id": "spk_aaa111"},
        ]},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["applied"] >= 1

    # Profile should now have a second embedding (seed + this one)
    profiles = json.loads(Path(env.profiles_path).read_text())
    profile = profiles["profiles"][0]
    session_embeddings = [
        e for e in profile["embeddings"] if e.get("session_id") == env.session_id
    ]
    assert len(session_embeddings) == 1


def test_confirm_variant_action(env):
    """confirm_variant adds a voice variant, not an embedding."""
    resp = env.client.post(
        f"/api/sessions/{env.session_id}/confirm-speakers",
        json={"decisions": [
            {"speaker_key": "SPEAKER_01", "action": "confirm_variant", "profile_id": "spk_aaa111"},
        ]},
    )
    assert resp.status_code == 200

    profiles = json.loads(Path(env.profiles_path).read_text())
    profile = profiles["profiles"][0]
    assert len(profile["voice_variants"]) == 1
    assert profile["voice_variants"][0]["session_id"] == env.session_id


def test_create_action(env):
    """Create action makes a new profile and enrolls the embedding."""
    resp = env.client.post(
        f"/api/sessions/{env.session_id}/confirm-speakers",
        json={"decisions": [
            {"speaker_key": "SPEAKER_01", "action": "create", "new_name": "Robin", "new_role": "child"},
        ]},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["created_profiles"]) == 1
    new_id = data["created_profiles"][0]
    assert new_id.startswith("spk_")

    # Verify profile exists with embedding enrolled
    profiles = json.loads(Path(env.profiles_path).read_text())
    new_profile = next(p for p in profiles["profiles"] if p["id"] == new_id)
    assert new_profile["name"] == "Robin"
    assert new_profile["role"] == "child"
    assert len(new_profile["embeddings"]) == 1
    assert new_profile["centroid"] is not None


def test_reassign_action(env):
    """Reassign adds the embedding to a different profile than originally suggested."""
    resp = env.client.post(
        f"/api/sessions/{env.session_id}/confirm-speakers",
        json={"decisions": [
            {"speaker_key": "SPEAKER_01", "action": "reassign", "profile_id": "spk_aaa111"},
        ]},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["applied"] >= 1


def test_skip_action(env):
    """Skip is a no-op — the speaker is ignored."""
    resp = env.client.post(
        f"/api/sessions/{env.session_id}/confirm-speakers",
        json={"decisions": [
            {"speaker_key": "SPEAKER_00", "action": "skip"},
        ]},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["skipped"] == 1
    assert data["applied"] == 0


# ---------------------------------------------------------------------------
# Mixed batch
# ---------------------------------------------------------------------------

def test_mixed_batch(env):
    """All action types in one batch request."""
    resp = env.client.post(
        f"/api/sessions/{env.session_id}/confirm-speakers",
        json={"decisions": [
            {"speaker_key": "SPEAKER_00", "action": "confirm", "profile_id": "spk_aaa111"},
            {"speaker_key": "SPEAKER_01", "action": "create", "new_name": "Robin", "new_role": "child"},
        ]},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["applied"] == 2
    assert data["skipped"] == 0
    assert len(data["created_profiles"]) == 1
    # Fresh identifications should be returned
    assert "identifications" in data
    assert data["identifications"]["session_id"] == env.session_id


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_confirm_idempotent(env):
    """Confirming the same speaker twice doesn't create duplicate embeddings."""
    body = {"decisions": [
        {"speaker_key": "SPEAKER_00", "action": "confirm", "profile_id": "spk_aaa111"},
    ]}
    env.client.post(f"/api/sessions/{env.session_id}/confirm-speakers", json=body)
    env.client.post(f"/api/sessions/{env.session_id}/confirm-speakers", json=body)

    profiles = json.loads(Path(env.profiles_path).read_text())
    profile = profiles["profiles"][0]
    session_embeddings = [
        e for e in profile["embeddings"] if e.get("session_id") == env.session_id
    ]
    assert len(session_embeddings) == 1


# ---------------------------------------------------------------------------
# Re-identification
# ---------------------------------------------------------------------------

def test_response_includes_fresh_identifications(env):
    """After saving, fresh identification results should be returned."""
    resp = env.client.post(
        f"/api/sessions/{env.session_id}/confirm-speakers",
        json={"decisions": [
            {"speaker_key": "SPEAKER_00", "action": "confirm", "profile_id": "spk_aaa111"},
        ]},
    )
    data = resp.get_json()
    ident = data["identifications"]
    assert ident["session_id"] == env.session_id
    assert len(ident["identifications"]) == 2  # both speakers re-identified


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_missing_body(env):
    resp = env.client.post(
        f"/api/sessions/{env.session_id}/confirm-speakers",
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_missing_decisions_key(env):
    resp = env.client.post(
        f"/api/sessions/{env.session_id}/confirm-speakers",
        json={"not_decisions": []},
    )
    assert resp.status_code == 400


def test_unknown_action(env):
    resp = env.client.post(
        f"/api/sessions/{env.session_id}/confirm-speakers",
        json={"decisions": [
            {"speaker_key": "SPEAKER_00", "action": "explode"},
        ]},
    )
    assert resp.status_code == 400
    assert "action" in resp.get_json()["error"].lower()


def test_confirm_missing_profile_id(env):
    resp = env.client.post(
        f"/api/sessions/{env.session_id}/confirm-speakers",
        json={"decisions": [
            {"speaker_key": "SPEAKER_00", "action": "confirm"},
        ]},
    )
    assert resp.status_code == 400


def test_create_missing_name(env):
    resp = env.client.post(
        f"/api/sessions/{env.session_id}/confirm-speakers",
        json={"decisions": [
            {"speaker_key": "SPEAKER_00", "action": "create", "new_role": "child"},
        ]},
    )
    assert resp.status_code == 400


def test_profile_not_found(env):
    resp = env.client.post(
        f"/api/sessions/{env.session_id}/confirm-speakers",
        json={"decisions": [
            {"speaker_key": "SPEAKER_00", "action": "confirm", "profile_id": "spk_nonexistent"},
        ]},
    )
    assert resp.status_code == 404
    assert "profile" in resp.get_json()["error"].lower()


def test_session_not_found(env):
    resp = env.client.post(
        "/api/sessions/20260199-000000/confirm-speakers",
        json={"decisions": []},
    )
    assert resp.status_code == 404


def test_no_embeddings(env):
    """Session without embeddings.json should return 400."""
    resp = env.client.post(
        f"/api/sessions/{env.bare_id}/confirm-speakers",
        json={"decisions": [
            {"speaker_key": "SPEAKER_00", "action": "skip"},
        ]},
    )
    assert resp.status_code == 400
    assert "embeddings" in resp.get_json()["error"].lower()


def test_speaker_key_not_in_embeddings(env):
    """Decision for a speaker_key not in embeddings.json should return 400."""
    resp = env.client.post(
        f"/api/sessions/{env.session_id}/confirm-speakers",
        json={"decisions": [
            {"speaker_key": "SPEAKER_99", "action": "confirm", "profile_id": "spk_aaa111"},
        ]},
    )
    assert resp.status_code == 400
    assert "speaker" in resp.get_json()["error"].lower()


# ---------------------------------------------------------------------------
# Confirmed field persistence
# ---------------------------------------------------------------------------

def test_confirm_action_sets_confirmed_fields(env):
    """Confirm action annotates the identification with confirmed fields."""
    resp = env.client.post(
        f"/api/sessions/{env.session_id}/confirm-speakers",
        json={"decisions": [
            {"speaker_key": "SPEAKER_00", "action": "confirm", "profile_id": "spk_aaa111"},
        ]},
    )
    data = resp.get_json()
    ident = next(
        i for i in data["identifications"]["identifications"]
        if i["speaker_key"] == "SPEAKER_00"
    )
    assert ident["confirmed"] is True
    assert ident["confirmed_action"] == "confirm"
    assert ident["confirmed_profile_id"] == "spk_aaa111"


def test_confirm_variant_sets_confirmed_fields(env):
    """confirm_variant action stores variant-specific confirmed_action."""
    resp = env.client.post(
        f"/api/sessions/{env.session_id}/confirm-speakers",
        json={"decisions": [
            {"speaker_key": "SPEAKER_01", "action": "confirm_variant", "profile_id": "spk_aaa111"},
        ]},
    )
    data = resp.get_json()
    ident = next(
        i for i in data["identifications"]["identifications"]
        if i["speaker_key"] == "SPEAKER_01"
    )
    assert ident["confirmed"] is True
    assert ident["confirmed_action"] == "confirm_variant"
    assert ident["confirmed_profile_id"] == "spk_aaa111"


def test_create_sets_confirmed_profile_id(env):
    """Create action maps confirmed_profile_id to the newly-created profile."""
    resp = env.client.post(
        f"/api/sessions/{env.session_id}/confirm-speakers",
        json={"decisions": [
            {"speaker_key": "SPEAKER_01", "action": "create", "new_name": "Robin", "new_role": "child"},
        ]},
    )
    data = resp.get_json()
    new_id = data["created_profiles"][0]
    ident = next(
        i for i in data["identifications"]["identifications"]
        if i["speaker_key"] == "SPEAKER_01"
    )
    assert ident["confirmed"] is True
    assert ident["confirmed_action"] == "create"
    assert ident["confirmed_profile_id"] == new_id


def test_skip_has_no_confirmed_fields(env):
    """Skipped speakers should not have confirmed fields."""
    resp = env.client.post(
        f"/api/sessions/{env.session_id}/confirm-speakers",
        json={"decisions": [
            {"speaker_key": "SPEAKER_00", "action": "skip"},
            {"speaker_key": "SPEAKER_01", "action": "confirm", "profile_id": "spk_aaa111"},
        ]},
    )
    data = resp.get_json()
    ident_00 = next(
        i for i in data["identifications"]["identifications"]
        if i["speaker_key"] == "SPEAKER_00"
    )
    assert "confirmed" not in ident_00


def test_confirmed_fields_persisted_to_disk(env):
    """GET session after save should return the confirmed fields from disk."""
    env.client.post(
        f"/api/sessions/{env.session_id}/confirm-speakers",
        json={"decisions": [
            {"speaker_key": "SPEAKER_00", "action": "confirm", "profile_id": "spk_aaa111"},
        ]},
    )

    # Read identifications.json directly from disk
    ident_path = env.session_dir / "identifications.json"
    assert ident_path.exists()
    ident_data = json.loads(ident_path.read_text())
    ident_00 = next(
        i for i in ident_data["identifications"]
        if i["speaker_key"] == "SPEAKER_00"
    )
    assert ident_00["confirmed"] is True
    assert ident_00["confirmed_action"] == "confirm"
    assert ident_00["confirmed_profile_id"] == "spk_aaa111"
