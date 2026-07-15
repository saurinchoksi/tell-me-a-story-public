"""Tests for GET/POST /api/sessions/:id/missed-speech/hearings (TMAS-75).

The hear-box sidecar: per-session ear verdicts on missed-speech candidates,
keyed by candidate span. Human-authored data — the POST must write ONLY
missed-speech-verdicts.json, never a transcript.
"""

import json

import pytest

from api.app import create_app

SESSION_ID = "20260101-120000"

HEARINGS_URL = f"/api/sessions/{SESSION_ID}/missed-speech/hearings"


def make_hearing(**overrides):
    """A valid hear-box entry; override fields to probe validation."""
    entry = {
        "start": 138.314,
        "end": 139.34,
        "min_gap": 1.0,
        "hear": "Wowwwww — tail of a partially captured word",
        "verdict": "boundary-tail",
        "createdAt": "2026-07-03T00:00:00.000Z",
        "updatedAt": "2026-07-03T00:00:00.000Z",
    }
    entry.update(overrides)
    return entry


@pytest.fixture
def sessions_dir(tmp_path):
    s1 = tmp_path / SESSION_ID
    s1.mkdir()
    (s1 / "transcript-rich.json").write_text(json.dumps({"segments": []}))
    return tmp_path


@pytest.fixture
def client(sessions_dir, tmp_path):
    profiles_path = str(tmp_path / "profiles.json")
    app = create_app(sessions_dir=sessions_dir, profiles_path=profiles_path)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_get_hearings_empty(client):
    """No sidecar yet — empty list, not an error (nothing has been heard)."""
    resp = client.get(HEARINGS_URL)
    assert resp.status_code == 200
    assert resp.get_json() == {"hearings": []}


def test_post_then_get_roundtrip(client):
    hearings = [make_hearing(),
                make_hearing(start=298.08, end=300.962,
                             hear="garbage stretch upstream",
                             verdict="hallucination-adjacent")]
    resp = client.post(HEARINGS_URL, json={"hearings": hearings})
    assert resp.status_code == 200
    assert resp.get_json() == {"saved": 2}

    resp = client.get(HEARINGS_URL)
    assert resp.get_json() == {"hearings": hearings}


def test_post_replaces_not_appends(client):
    """Full array replacement, same contract as validation notes."""
    client.post(HEARINGS_URL, json={"hearings": [make_hearing()]})
    resp = client.post(HEARINGS_URL, json={"hearings": []})
    assert resp.get_json() == {"saved": 0}
    assert client.get(HEARINGS_URL).get_json() == {"hearings": []}


def test_post_null_verdict_allowed(client):
    """Typed text with no verdict yet is a legitimate in-progress state."""
    resp = client.post(
        HEARINGS_URL, json={"hearings": [make_hearing(verdict=None)]})
    assert resp.status_code == 200


def test_post_missing_hearings_key(client):
    resp = client.post(HEARINGS_URL, json={"notes": []})
    assert resp.status_code == 400
    resp = client.post(HEARINGS_URL, json={"hearings": "not-a-list"})
    assert resp.status_code == 400


def test_post_non_object_body(client):
    """A bare JSON scalar body must be a clean 400, not a TypeError 500."""
    for raw in ["5", "true", '"hearings"', "[]"]:
        resp = client.post(HEARINGS_URL, data=raw,
                           content_type="application/json")
        assert resp.status_code == 400, raw


def test_post_rejects_non_finite_min_gap(client):
    """min_gap is the provenance tier; NaN would serialize as a bare token
    that the browser's strict JSON.parse rejects on every later GET."""
    for bad in [make_hearing(min_gap=None),
                {k: v for k, v in make_hearing().items() if k != "min_gap"},
                make_hearing(min_gap="1.0")]:
        resp = client.post(HEARINGS_URL, json={"hearings": [bad]})
        assert resp.status_code == 400, bad
    resp = client.post(
        HEARINGS_URL,
        data='{"hearings": [{"start": 1.0, "end": 2.0, "min_gap": NaN, '
             '"hear": "x", "verdict": null}]}',
        content_type="application/json")
    assert resp.status_code == 400


def test_get_rejects_malformed_sidecar_shape(client, sessions_dir):
    """Valid JSON that isn't {"hearings": [...]} is a clean 500, not a
    TypeError 500 or a shape the frontend would crash on."""
    sidecar = sessions_dir / SESSION_ID / "missed-speech-verdicts.json"
    for raw in ["[]", '"x"', '{"hearings": {"a": 1}}', '{"hearings": null}']:
        sidecar.write_text(raw)
        resp = client.get(HEARINGS_URL)
        assert resp.status_code == 500, raw
        assert "hearings" in resp.get_json()["error"]


def test_post_rejects_unusable_span_key(client):
    """A hearing that can't join back to a candidate is a wasted listen —
    reject loudly instead of storing it."""
    for bad in [make_hearing(start="2:17"),      # string, not seconds
                make_hearing(end=None),
                make_hearing(start=float("nan")),
                make_hearing(end=True),          # bool is not a number here
                {"hear": "no span at all", "verdict": "nothing"}]:
        resp = client.post(HEARINGS_URL, json={"hearings": [bad]})
        assert resp.status_code == 400, bad


def test_post_rejects_unknown_verdict(client):
    resp = client.post(
        HEARINGS_URL, json={"hearings": [make_hearing(verdict="maybe")]})
    assert resp.status_code == 400
    assert "verdict" in resp.get_json()["error"]


def test_post_rejects_non_string_hear(client):
    resp = client.post(
        HEARINGS_URL, json={"hearings": [make_hearing(hear=None)]})
    assert resp.status_code == 400


def test_post_writes_only_the_sidecar(client, sessions_dir):
    """The one file the hear-box may touch is its own sidecar — the
    transcript (and everything else in the session dir) stays byte-identical."""
    sdir = sessions_dir / SESSION_ID
    before = {p.name: p.read_bytes() for p in sdir.iterdir()}
    client.post(HEARINGS_URL, json={"hearings": [make_hearing()]})
    after = {p.name: p.read_bytes() for p in sdir.iterdir()}
    assert set(after) - set(before) == {"missed-speech-verdicts.json"}
    for name in before:
        assert after[name] == before[name]


def test_bad_session_id(client):
    resp = client.get("/api/sessions/bad-id/missed-speech/hearings")
    assert resp.status_code == 400
    resp = client.post("/api/sessions/bad-id/missed-speech/hearings",
                       json={"hearings": []})
    assert resp.status_code == 400


def test_session_not_found(client):
    resp = client.get("/api/sessions/20260199-000000/missed-speech/hearings")
    assert resp.status_code == 404
