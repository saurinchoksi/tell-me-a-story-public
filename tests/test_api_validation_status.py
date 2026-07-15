"""Tests for the session validation-status endpoint."""

import json

import pytest

from api.app import create_app


@pytest.fixture
def sessions_dir(tmp_path):
    """Create a temp sessions directory with one session fixture."""
    s1 = tmp_path / "20260101-120000"
    s1.mkdir()
    (s1 / "audio.m4a").write_bytes(b"fake audio")
    return tmp_path


@pytest.fixture
def client(sessions_dir, tmp_path):
    profiles_path = str(tmp_path / "profiles.json")
    app = create_app(sessions_dir=sessions_dir, profiles_path=profiles_path)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# --- PUT /api/sessions/:id/validation-status ---

def test_save_validation_status(client, sessions_dir):
    """Save a status and verify it persists to session-metadata.json."""
    resp = client.put(
        "/api/sessions/20260101-120000/validation-status",
        json={"status": "in_progress"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["validationStatus"] == "in_progress"
    assert "updatedAt" in data

    metadata_path = sessions_dir / "20260101-120000" / "session-metadata.json"
    assert metadata_path.exists()
    saved = json.loads(metadata_path.read_text())
    assert saved["validationStatus"] == "in_progress"
    assert "updatedAt" in saved


def test_save_validation_status_all_values(client):
    """Each of the three valid states is accepted."""
    for status in ("not_started", "in_progress", "done"):
        resp = client.put(
            "/api/sessions/20260101-120000/validation-status",
            json={"status": status},
        )
        assert resp.status_code == 200
        assert resp.get_json()["validationStatus"] == status


def test_save_validation_status_invalid_value(client):
    resp = client.put(
        "/api/sessions/20260101-120000/validation-status",
        json={"status": "bogus"},
    )
    assert resp.status_code == 400


def test_save_validation_status_missing_key(client):
    resp = client.put(
        "/api/sessions/20260101-120000/validation-status",
        json={"wrong_key": "done"},
    )
    assert resp.status_code == 400


def test_save_validation_status_missing_body(client):
    resp = client.put(
        "/api/sessions/20260101-120000/validation-status",
        data="not json",
        content_type="text/plain",
    )
    assert resp.status_code == 400


def test_save_validation_status_invalid_session_id(client):
    resp = client.put(
        "/api/sessions/bad-id/validation-status",
        json={"status": "done"},
    )
    assert resp.status_code == 400


def test_save_validation_status_session_not_found(client):
    resp = client.put(
        "/api/sessions/20260199-000000/validation-status",
        json={"status": "done"},
    )
    assert resp.status_code == 404


def test_save_validation_status_preserves_note(client, sessions_dir):
    """Saving a status keeps an existing note (read-merge-write)."""
    metadata_path = sessions_dir / "20260101-120000" / "session-metadata.json"
    metadata_path.write_text(json.dumps({"note": "keep me"}))

    client.put(
        "/api/sessions/20260101-120000/validation-status",
        json={"status": "done"},
    )

    saved = json.loads(metadata_path.read_text())
    assert saved["note"] == "keep me"
    assert saved["validationStatus"] == "done"


# --- GET /api/sessions reflects the status ---

def test_list_sessions_includes_saved_status(client):
    """After a PUT, the status appears in the sessions list."""
    client.put(
        "/api/sessions/20260101-120000/validation-status",
        json={"status": "in_progress"},
    )
    resp = client.get("/api/sessions")
    sessions = {s["id"]: s for s in resp.get_json()["sessions"]}
    assert sessions["20260101-120000"]["validation_status"] == "in_progress"


def test_list_sessions_corrupt_transcript(client, sessions_dir):
    """A corrupt transcript-rich.json surfaces as 500, not a silent skip (Fail Loud)."""
    transcript_path = sessions_dir / "20260101-120000" / "transcript-rich.json"
    transcript_path.write_text("{bad json")

    resp = client.get("/api/sessions")
    assert resp.status_code == 500
    assert "Corrupt" in resp.get_json()["error"]
