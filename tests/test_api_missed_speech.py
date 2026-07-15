"""Tests for GET /api/sessions/:id/missed-speech (TMAS-48).

The endpoint serves the canonical missed-speech gap detection (src/gaps.py)
live from transcript-rich.json + diarization.json. Read-only: computing a
view must never touch session files.
"""

import json

import pytest

from api.app import create_app

SESSION_ID = "20260101-120000"


def make_transcript():
    """A transcript with real words, an [unintelligible] placeholder, and a
    bare stretch (2.0-4.0) that diarization covers — the one true gap."""
    return {
        "segments": [
            {"id": 1, "start": 0.0, "end": 1.0, "text": "hi"},
            {"id": "gap_1.0", "start": 1.0, "end": 2.0,
             "text": "[unintelligible]", "_source": "diarization_gap"},
            {"id": 2, "start": 4.0, "end": 5.0, "text": "bye"},
        ]
    }


def make_diarization():
    return {"segments": [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]}


@pytest.fixture
def sessions_dir(tmp_path):
    s1 = tmp_path / SESSION_ID
    s1.mkdir()
    (s1 / "transcript-rich.json").write_text(json.dumps(make_transcript()))
    (s1 / "diarization.json").write_text(json.dumps(make_diarization()))
    return tmp_path


@pytest.fixture
def client(sessions_dir, tmp_path):
    profiles_path = str(tmp_path / "profiles.json")
    app = create_app(sessions_dir=sessions_dir, profiles_path=profiles_path)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_missed_speech_default_threshold(client):
    """Default min_gap is the 1.0s high-confidence floor; the [unintelligible]
    placeholder counts as coverage, so only the bare 2.0-4.0 stretch is flagged."""
    resp = client.get(f"/api/sessions/{SESSION_ID}/missed-speech")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["min_gap"] == 1.0
    assert data["candidates"] == [
        {"start": 2.0, "end": 4.0, "dur": 2.0, "speaker": "SPEAKER_00"}
    ]


def test_missed_speech_candidates_are_bare_spans(client):
    """find_gaps attaches prev/next segment dicts for the sweep's HTML table —
    the API must strip them (they carry transcript text)."""
    resp = client.get(f"/api/sessions/{SESSION_ID}/missed-speech")
    for c in resp.get_json()["candidates"]:
        assert set(c.keys()) == {"start", "end", "dur", "speaker"}


def test_missed_speech_min_gap_param(client, sessions_dir):
    """?min_gap widens/narrows the flagged set (0.3 = the sweep's candidate tier)."""
    # Split the tail coverage so a second, short (0.4s) bare stretch exists:
    # covered (0-2, 4-4.5, 4.9-5) -> gaps 2.0-4.0 (2.0s) and 4.5-4.9 (0.4s).
    transcript = make_transcript()
    transcript["segments"][2] = {"id": 2, "start": 4.0, "end": 4.5, "text": "and"}
    transcript["segments"].append(
        {"id": 3, "start": 4.9, "end": 5.0, "text": "bye"})
    path = sessions_dir / SESSION_ID / "transcript-rich.json"
    path.write_text(json.dumps(transcript))

    resp = client.get(f"/api/sessions/{SESSION_ID}/missed-speech?min_gap=0.3")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["min_gap"] == 0.3
    assert [(c["start"], c["end"]) for c in data["candidates"]] == [
        (2.0, 4.0), (4.5, 4.9)]

    # Default 1.0s still hides the sliver.
    resp = client.get(f"/api/sessions/{SESSION_ID}/missed-speech")
    assert [(c["start"], c["end"]) for c in resp.get_json()["candidates"]] == [
        (2.0, 4.0)]


def test_missed_speech_invalid_min_gap(client):
    resp = client.get(f"/api/sessions/{SESSION_ID}/missed-speech?min_gap=abc")
    assert resp.status_code == 400
    resp = client.get(f"/api/sessions/{SESSION_ID}/missed-speech?min_gap=0")
    assert resp.status_code == 400
    resp = client.get(f"/api/sessions/{SESSION_ID}/missed-speech?min_gap=-1")
    assert resp.status_code == 400


def test_missed_speech_bad_session_id(client):
    resp = client.get("/api/sessions/bad-id/missed-speech")
    assert resp.status_code == 400


def test_missed_speech_session_not_found(client):
    resp = client.get("/api/sessions/20260199-000000/missed-speech")
    assert resp.status_code == 404


def test_missed_speech_missing_transcript(client, sessions_dir):
    """A session without a transcript can't compute — explicit 404, not
    a silent empty list."""
    (sessions_dir / SESSION_ID / "transcript-rich.json").unlink()
    resp = client.get(f"/api/sessions/{SESSION_ID}/missed-speech")
    assert resp.status_code == 404
    assert "transcript-rich.json" in resp.get_json()["error"]


def test_missed_speech_missing_diarization(client, sessions_dir):
    (sessions_dir / SESSION_ID / "diarization.json").unlink()
    resp = client.get(f"/api/sessions/{SESSION_ID}/missed-speech")
    assert resp.status_code == 404
    assert "diarization.json" in resp.get_json()["error"]


def test_missed_speech_is_read_only(client, sessions_dir):
    """Viewing never writes: the session dir contents are unchanged after a GET."""
    sdir = sessions_dir / SESSION_ID
    before = {p.name: p.read_bytes() for p in sdir.iterdir()}
    client.get(f"/api/sessions/{SESSION_ID}/missed-speech")
    after = {p.name: p.read_bytes() for p in sdir.iterdir()}
    assert before == after
