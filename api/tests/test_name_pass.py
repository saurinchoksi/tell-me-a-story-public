"""Tests for the name-pass endpoints (#4 step 1).

The recall engine (emp/src/name_truth.py) roots every path at its own module-level
ROOT, so we monkeypatch name_truth.ROOT to tmp_path. That isolates BOTH the session
reads AND the key writes (emp/results/visuals/<id>/name-truth.json) into the temp dir,
so a test never reads a real session or clobbers a real gold key.
"""

import json

import pytest

from api.app import create_app

SID = "20990101-000000"  # synthetic + far-future, so it can't collide with a real session

# Two mid-sentence-capitalized names (Zorbon, Kwelly) the model-free sweep must surface;
# deliberately non-roster nonsense words so the result never depends on the real roster.
TRANSCRIPT = {
    "segments": [
        {"id": 0, "start": 0.0, "end": 2.0, "text": "Once Zorbon met Kwelly.",
         "words": [
             {"word": "Once", "start": 0.0, "end": 0.4, "probability": 0.9},
             {"word": "Zorbon", "start": 0.5, "end": 0.9, "probability": 0.9},
             {"word": "met", "start": 1.0, "end": 1.2, "probability": 0.9},
             {"word": "Kwelly.", "start": 1.3, "end": 1.8, "probability": 0.9},
         ]},
        {"id": 1, "start": 2.0, "end": 4.0, "text": "Zorbon liked Kwelly.",
         "words": [
             {"word": "Zorbon", "start": 2.1, "end": 2.5, "probability": 0.9},
             {"word": "liked", "start": 2.6, "end": 2.9, "probability": 0.9},
             {"word": "Kwelly.", "start": 3.0, "end": 3.5, "probability": 0.9},
         ]},
    ]
}


# A synthetic roster (example-twin names, per data/name_roster.example.json): the frozen
# M9a detector fail-louds without a roster FILE, so a hermetic run must write one — but
# its names are phonetically disjoint from the transcript's, so no result depends on it.
ROSTER = {
    "people": [
        {"id": "child", "canonical": "Marta", "role": "child", "circle": "family", "nicknames": ["Kiki"]},
        {"id": "parent", "canonical": "Kiran", "role": "parent", "circle": "family"},
    ],
    "aliases": [{"token": "bemarta", "person_id": "child"}],
}


def _make_app(tmp_path, monkeypatch, write_roster=True):
    sessions = tmp_path / "sessions"
    (sessions / SID).mkdir(parents=True)
    (sessions / SID / "transcript-rich.json").write_text(json.dumps(TRANSCRIPT))
    if write_roster:
        (tmp_path / "roster.json").write_text(json.dumps(ROSTER))

    app = create_app(
        sessions_dir=sessions,
        profiles_path=str(tmp_path / "profiles.json"),
        roster_path=tmp_path / "roster.json",
        home_canon_path=tmp_path / "home.json",
    )
    app.config["TESTING"] = True

    # emp/src is importable now (create_app imported the name_pass blueprint). Root the
    # engine at tmp so its session reads and its key writes stay in the temp dir.
    import name_truth
    monkeypatch.setattr(name_truth, "ROOT", tmp_path)
    return app


@pytest.fixture
def client(tmp_path, monkeypatch):
    with _make_app(tmp_path, monkeypatch).test_client() as c:
        yield c


def test_get_returns_cards_and_empty_seed(client):
    resp = client.get(f"/api/sessions/{SID}/name-pass")
    assert resp.status_code == 200
    data = resp.get_json()

    names = {card["cleaned"] for card in data["cards"]}
    assert {"zorbon", "kwelly"} <= names

    # flat candidate index (occ_id -> card key) drives the transcript highlighter
    assert data["candidate_index"]
    assert all(":" in occ_id for occ_id in data["candidate_index"])

    # engine sets serialized to lists — if any leaked, jsonify would have 500'd
    card = next(c for c in data["cards"] if c["cleaned"] == "zorbon")
    assert isinstance(card["surfaces"], list)
    assert isinstance(card["source"], list)
    assert card["occurrences"][0]["seg_id"] == 0  # int seg id preserved from occ_id

    # unseeded session -> empty seed, but the current transcript is fingerprinted
    assert data["seed"]["items"] == {}
    assert data["seed"]["fingerprint"] is None
    assert data["fingerprint"]


def test_post_writes_field_and_preserves_siblings(client):
    client.post(f"/api/sessions/{SID}/name-truth",
                json={"kind": "item", "id": "zorbon", "field": "category", "value": "tv-canon"})
    r2 = client.post(f"/api/sessions/{SID}/name-truth",
                     json={"kind": "item", "id": "zorbon", "field": "true_spelling", "value": "Zorbon"})
    assert r2.status_code == 200 and r2.get_json()["ok"] is True

    # both fields survive — the field-routed save never wiped the sibling
    data = client.get(f"/api/sessions/{SID}/name-pass").get_json()
    item = data["seed"]["items"]["zorbon"]
    assert item["category"] == "tv-canon"
    assert item["true_spelling"] == "Zorbon"

    # first save backfilled the stamp against the current transcript
    assert data["seed"]["fingerprint"] == data["fingerprint"]
    assert data["seed"]["marked_at"]


def test_post_occurrence_true_name(client):
    r = client.post(f"/api/sessions/{SID}/name-truth",
                    json={"kind": "occ", "id": "0:1", "field": "true_name", "value": "Zorbon"})
    assert r.status_code == 200
    data = client.get(f"/api/sessions/{SID}/name-pass").get_json()
    assert data["seed"]["occurrences"]["0:1"]["true_name"] == "Zorbon"


def test_post_rejects_bad_body(client):
    # unknown field is not in the item defaults
    assert client.post(f"/api/sessions/{SID}/name-truth",
                       json={"kind": "item", "id": "zorbon", "field": "nope", "value": "x"}).status_code == 400
    # missing id
    assert client.post(f"/api/sessions/{SID}/name-truth",
                       json={"kind": "item", "field": "category", "value": "x"}).status_code == 400
    # unknown kind
    assert client.post(f"/api/sessions/{SID}/name-truth",
                       json={"kind": "bogus", "id": "z", "field": "category", "value": "x"}).status_code == 400


def test_bad_session_id_and_missing_session(client):
    assert client.get("/api/sessions/not-an-id/name-pass").status_code == 400
    assert client.get("/api/sessions/29990101-000000/name-pass").status_code == 404


def test_missing_roster_names_the_roster_not_the_transcript(tmp_path, monkeypatch):
    """A fresh clone/export has this session's transcript but no roster, so the frozen
    M9a fail-louds. The GET must surface THAT message — it names the roster and the fix —
    never the old "No transcript" mislabel that pointed a new runner at the wrong file."""
    app = _make_app(tmp_path, monkeypatch, write_roster=False)
    with app.test_client() as c:
        resp = c.get(f"/api/sessions/{SID}/name-pass")
    assert resp.status_code == 500
    err = resp.get_json()["error"]
    assert "roster" in err.lower()
    assert "transcript" not in err.lower()


# --- Phase 2: Opus suggestions + the heard-gate commit ---

def test_get_returns_suggestions(client, tmp_path):
    sug = tmp_path / "emp" / "results" / "visuals" / SID
    sug.mkdir(parents=True, exist_ok=True)
    (sug / "name-suggestions.json").write_text(json.dumps(
        {"suggestions": {"zorbon": {"suggested": "Zorbon", "confidence": "high", "note": "x"}}}))
    data = client.get(f"/api/sessions/{SID}/name-pass").get_json()
    assert data["suggestions"]["zorbon"]["suggested"] == "Zorbon"


def test_suggestions_empty_when_absent(client):
    assert client.get(f"/api/sessions/{SID}/name-pass").get_json()["suggestions"] == {}


def test_commit_records_provenance_and_heard_gate(client):
    # inline correction — one item, ear-corrected, no gate stamp yet
    r = client.post(f"/api/sessions/{SID}/name-truth/commit",
                    json={"items": {"zorbon": {"true_spelling": "Zorbon", "provenance": "ear-corrected"}}})
    assert r.status_code == 200 and r.get_json()["committed"] == 1
    assert client.get(f"/api/sessions/{SID}/name-pass").get_json()["seed"]["heard_confirmed"] is None
    # "Mark heard" — bulk accept, suggestion-confirmed, stamps the ear-gate
    r2 = client.post(f"/api/sessions/{SID}/name-truth/commit",
                     json={"items": {"kwelly": {"true_spelling": "Kwelly", "provenance": "suggestion-confirmed"}},
                           "heard": True})
    assert r2.status_code == 200
    data = client.get(f"/api/sessions/{SID}/name-pass").get_json()
    items = data["seed"]["items"]
    assert items["zorbon"]["provenance"] == "ear-corrected"
    assert items["kwelly"]["true_spelling"] == "Kwelly"
    assert items["kwelly"]["provenance"] == "suggestion-confirmed"
    assert data["seed"]["heard_confirmed"]  # gate stamped


def test_commit_rejects_bad_body(client):
    # fail-loud: bad shape, empty id, non-dict fields, or an unknown field all 400 — and write nothing
    for body in (
        {"items": "nope"},
        {"items": {"": {"true_spelling": "Ghost"}}},   # empty id would pollute the answer key
        {"items": {"x": "not-a-dict"}},
        {"items": {"x": {"bogus": "y"}}},              # unknown field
    ):
        assert client.post(f"/api/sessions/{SID}/name-truth/commit", json=body).status_code == 400
    assert client.get(f"/api/sessions/{SID}/name-pass").get_json()["seed"]["items"] == {}


def test_lowercase_net_cleans_home_canon_name(client, tmp_path, monkeypatch):
    # A home character with an apostrophe. The lowercase-sound net must clean() the canonical
    # before phonetic-coding it — else codes("O'Malley") != codes("omalley") and a lowercase
    # garble is silently missed (the review bug).
    import home_canon
    monkeypatch.setattr(home_canon, "canonical_forms", lambda path=None: {"omalley": "O'Malley"})
    tx = {"segments": [
        {"id": 0, "start": 0.0, "end": 2.0, "text": "Once Zorbon met omalley.",
         "words": [
             {"word": "Once", "start": 0.0, "end": 0.4, "probability": 0.9},
             {"word": "Zorbon", "start": 0.5, "end": 0.9, "probability": 0.9},
             {"word": "met", "start": 1.0, "end": 1.2, "probability": 0.9},
             {"word": "omalley", "start": 1.3, "end": 1.8, "probability": 0.9},
         ]},
    ]}
    (tmp_path / "sessions" / SID / "transcript-rich.json").write_text(json.dumps(tx))
    names = {c["cleaned"] for c in client.get(f"/api/sessions/{SID}/name-pass").get_json()["cards"]}
    assert "omalley" in names  # only the lowercase-sound net can surface this lowercase garble
