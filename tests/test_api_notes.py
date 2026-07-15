"""Tests for API validation notes endpoints."""

import json

import pytest

from api.app import create_app


@pytest.fixture
def sessions_dir(tmp_path):
    """Create a temp session directory."""
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


# --- GET /api/sessions/:id/notes ---

def test_get_notes_empty(client):
    """No notes file returns empty array."""
    resp = client.get("/api/sessions/20260101-120000/notes")
    assert resp.status_code == 200
    assert resp.get_json() == {"notes": []}


def test_get_notes_with_data(client, sessions_dir):
    """Returns notes from existing file."""
    notes = [
        {"id": "abc123", "segmentId": 3, "text": "Check this word"}
    ]
    notes_path = sessions_dir / "20260101-120000" / "validation-notes.json"
    notes_path.write_text(json.dumps({"notes": notes}))

    resp = client.get("/api/sessions/20260101-120000/notes")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["notes"]) == 1
    assert data["notes"][0]["id"] == "abc123"


def test_get_notes_session_not_found(client):
    resp = client.get("/api/sessions/20260199-000000/notes")
    assert resp.status_code == 404


def test_get_notes_bad_format(client):
    resp = client.get("/api/sessions/bad-id/notes")
    assert resp.status_code == 400


# --- POST /api/sessions/:id/notes ---

def test_save_notes(client, sessions_dir):
    """Save and retrieve roundtrip."""
    notes = [
        {"id": "n1", "segmentId": 1, "text": "Speaker mismatch"},
        {"id": "n2", "segmentId": 5, "wordIndex": 3, "text": "Wrong name"},
    ]
    resp = client.post(
        "/api/sessions/20260101-120000/notes",
        json={"notes": notes},
    )
    assert resp.status_code == 200
    assert resp.get_json()["saved"] == 2

    # Verify persisted to disk
    notes_path = sessions_dir / "20260101-120000" / "validation-notes.json"
    assert notes_path.exists()
    saved = json.loads(notes_path.read_text())
    assert len(saved["notes"]) == 2

    # Verify GET returns the saved data
    resp = client.get("/api/sessions/20260101-120000/notes")
    assert len(resp.get_json()["notes"]) == 2


def test_save_notes_replaces(client, sessions_dir):
    """Second save replaces first, not appends."""
    client.post(
        "/api/sessions/20260101-120000/notes",
        json={"notes": [{"id": "n1", "text": "first"}]},
    )
    client.post(
        "/api/sessions/20260101-120000/notes",
        json={"notes": [{"id": "n2", "text": "second"}]},
    )
    resp = client.get("/api/sessions/20260101-120000/notes")
    data = resp.get_json()
    assert len(data["notes"]) == 1
    assert data["notes"][0]["id"] == "n2"


def test_save_notes_empty_array(client):
    """Saving empty array is valid (clears notes)."""
    resp = client.post(
        "/api/sessions/20260101-120000/notes",
        json={"notes": []},
    )
    assert resp.status_code == 200
    assert resp.get_json()["saved"] == 0


def test_save_notes_missing_body(client):
    resp = client.post(
        "/api/sessions/20260101-120000/notes",
        data="not json",
        content_type="text/plain",
    )
    assert resp.status_code == 400


def test_save_notes_missing_notes_key(client):
    resp = client.post(
        "/api/sessions/20260101-120000/notes",
        json={"wrong_key": []},
    )
    assert resp.status_code == 400


def test_save_notes_notes_not_array(client):
    resp = client.post(
        "/api/sessions/20260101-120000/notes",
        json={"notes": "not an array"},
    )
    assert resp.status_code == 400


def test_save_notes_session_not_found(client):
    resp = client.post(
        "/api/sessions/20260199-000000/notes",
        json={"notes": []},
    )
    assert resp.status_code == 404


def test_get_notes_corrupt_json(client, sessions_dir):
    """Corrupt JSON returns 500, not silent empty array (Fail Loud)."""
    notes_path = sessions_dir / "20260101-120000" / "validation-notes.json"
    notes_path.write_text("{bad json")

    resp = client.get("/api/sessions/20260101-120000/notes")
    assert resp.status_code == 500
    assert "Corrupt" in resp.get_json()["error"]
