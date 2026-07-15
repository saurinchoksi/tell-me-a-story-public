"""Tests for the session-level note endpoint."""

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


# --- PUT /api/sessions/:id/note ---

def test_save_note(client, sessions_dir):
    """Save a note and verify it persists to session-metadata.json."""
    resp = client.put(
        "/api/sessions/20260101-120000/note",
        json={"note": "Mahabharata story, my daughter + me"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["note"] == "Mahabharata story, my daughter + me"
    assert "updatedAt" in data

    metadata_path = sessions_dir / "20260101-120000" / "session-metadata.json"
    assert metadata_path.exists()
    saved = json.loads(metadata_path.read_text())
    assert saved["note"] == "Mahabharata story, my daughter + me"
    assert "updatedAt" in saved


def test_save_note_replaces(client):
    """A second save overwrites the first."""
    client.put("/api/sessions/20260101-120000/note", json={"note": "first"})
    client.put("/api/sessions/20260101-120000/note", json={"note": "second"})

    resp = client.get("/api/sessions")
    sessions = {s["id"]: s for s in resp.get_json()["sessions"]}
    assert sessions["20260101-120000"]["note"] == "second"


def test_save_note_empty_string(client):
    """An empty note is valid (clears the note)."""
    resp = client.put(
        "/api/sessions/20260101-120000/note",
        json={"note": ""},
    )
    assert resp.status_code == 200
    assert resp.get_json()["note"] == ""


def test_save_note_preserves_other_metadata(client, sessions_dir):
    """Saving a note keeps unrelated metadata fields (forward-compat for tags)."""
    metadata_path = sessions_dir / "20260101-120000" / "session-metadata.json"
    metadata_path.write_text(json.dumps({"note": "old", "tags": ["mahabharata"]}))

    client.put("/api/sessions/20260101-120000/note", json={"note": "new"})

    saved = json.loads(metadata_path.read_text())
    assert saved["note"] == "new"
    assert saved["tags"] == ["mahabharata"]


def test_save_note_invalid_session_id(client):
    resp = client.put("/api/sessions/bad-id/note", json={"note": "hi"})
    assert resp.status_code == 400


def test_save_note_session_not_found(client):
    resp = client.put("/api/sessions/20260199-000000/note", json={"note": "hi"})
    assert resp.status_code == 404


def test_save_note_missing_note_key(client):
    resp = client.put(
        "/api/sessions/20260101-120000/note",
        json={"wrong_key": "hi"},
    )
    assert resp.status_code == 400


def test_save_note_not_a_string(client):
    resp = client.put(
        "/api/sessions/20260101-120000/note",
        json={"note": 123},
    )
    assert resp.status_code == 400


def test_save_note_missing_body(client):
    resp = client.put(
        "/api/sessions/20260101-120000/note",
        data="not json",
        content_type="text/plain",
    )
    assert resp.status_code == 400


# --- GET /api/sessions includes the note ---

def test_list_sessions_note_empty_by_default(client):
    """A session with no metadata file reports note as ''."""
    resp = client.get("/api/sessions")
    sessions = {s["id"]: s for s in resp.get_json()["sessions"]}
    assert sessions["20260101-120000"]["note"] == ""


def test_list_sessions_includes_saved_note(client):
    """After a PUT, the note appears in the sessions list."""
    client.put(
        "/api/sessions/20260101-120000/note",
        json={"note": "English fairy tale, short"},
    )
    resp = client.get("/api/sessions")
    sessions = {s["id"]: s for s in resp.get_json()["sessions"]}
    assert sessions["20260101-120000"]["note"] == "English fairy tale, short"


def test_list_sessions_corrupt_metadata(client, sessions_dir):
    """Corrupt session-metadata.json returns 500, not a silent skip (Fail Loud)."""
    metadata_path = sessions_dir / "20260101-120000" / "session-metadata.json"
    metadata_path.write_text("{bad json")

    resp = client.get("/api/sessions")
    assert resp.status_code == 500
    assert "Corrupt" in resp.get_json()["error"]
