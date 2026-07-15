"""Tests for API audio endpoint."""

import json

import pytest

from api.app import create_app


@pytest.fixture
def client(tmp_path):
    """Flask test client with a session that has an audio file."""
    sessions_dir = tmp_path / "sessions"
    session = sessions_dir / "20260101-120000"
    session.mkdir(parents=True)
    (session / "audio.m4a").write_bytes(b"fake audio data for testing")

    # Session without audio
    empty = sessions_dir / "20260102-180000"
    empty.mkdir()

    app = create_app(sessions_dir=sessions_dir)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_get_audio(client):
    resp = client.get("/api/sessions/20260101-120000/audio")
    assert resp.status_code == 200
    assert resp.data == b"fake audio data for testing"


def test_get_audio_no_file(client):
    resp = client.get("/api/sessions/20260102-180000/audio")
    assert resp.status_code == 404


def test_get_audio_session_not_found(client):
    resp = client.get("/api/sessions/20260199-000000/audio")
    assert resp.status_code == 404


def test_get_audio_path_traversal(client):
    # Flask normalizes ../ before routing; either 400 or 404 indicates blocked traversal
    resp = client.get("/api/sessions/..%2F..%2Fetc/audio")
    assert resp.status_code in (400, 404)
