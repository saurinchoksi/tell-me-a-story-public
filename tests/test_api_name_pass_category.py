"""The gold key's write path must refuse a name-kind it does not recognise.

A `category` outside the vocabulary is the quietest way to corrupt the ruler: it is not
blank, so score_transform's `unjudged` guard never fires — but it belongs to neither
KINDS nor NEGATIVE_KINDS, so the decision silently vanishes from every per-kind table.
Now the machine DRAFTS post a category, so a single typo in a hand-written
name-suggestions.json ("improvized", "canon") would reach name-truth.json unchallenged.

Fail Loud (docs/principles.md): refuse it at the write path, both routes.

NB: name_truth writes to the REAL emp/results/visuals/<sid>/ — the path is not injectable
— so these tests only exercise the reject path (which returns 400 before any write) and
stub update_notes for the accept path. Nothing here touches a real key.
"""

import json

import pytest

from api.app import create_app


@pytest.fixture
def sid():
    return "20260101-000000"


@pytest.fixture
def client(tmp_path, sid):
    sdir = tmp_path / sid
    sdir.mkdir()
    (sdir / "transcript-rich.json").write_text(json.dumps({"segments": []}))
    app = create_app(sessions_dir=tmp_path, profiles_path=str(tmp_path / "profiles.json"))
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def no_write(monkeypatch):
    """Stub the ONLY writer, so an accepted category never touches a real key file."""
    import name_truth
    calls = []
    monkeypatch.setattr(name_truth, "update_notes", lambda s, mutate: calls.append((s, mutate)))
    return calls


# --- /name-truth/commit (the bulk path: "Mark heard", and every accept) ---

@pytest.mark.parametrize("bogus", ["improvized", "canon", "tv canon", "Roster", "invented"])
def test_commit_refuses_a_category_outside_the_vocabulary(client, sid, no_write, bogus):
    resp = client.post(f"/api/sessions/{sid}/name-truth/commit",
                       json={"items": {"jama": {"true_spelling": "Jammus", "category": bogus}}})
    assert resp.status_code == 400
    assert "unknown category" in resp.get_json()["error"]
    assert no_write == []            # refused BEFORE any write reached the key


@pytest.mark.parametrize("good", ["roster", "tv-canon", "improvised", "place-other", "not-a-name", ""])
def test_commit_accepts_every_real_category(client, sid, no_write, good):
    """Including "" — the '— unset —' default is a legitimate value the tool can send."""
    resp = client.post(f"/api/sessions/{sid}/name-truth/commit",
                       json={"items": {"jama": {"true_spelling": "Jammus", "category": good}}})
    assert resp.status_code == 200
    assert len(no_write) == 1


def test_commit_still_refuses_an_unknown_field(client, sid, no_write):
    """The pre-existing field-NAME guard must survive the new value guard."""
    resp = client.post(f"/api/sessions/{sid}/name-truth/commit",
                       json={"items": {"jama": {"kind": "improvised"}}})
    assert resp.status_code == 400
    assert "unknown field" in resp.get_json()["error"]
    assert no_write == []


# --- /name-truth (the field-router: one field at a time) ---

def test_field_router_refuses_a_bogus_category(client, sid, no_write):
    resp = client.post(f"/api/sessions/{sid}/name-truth",
                       json={"kind": "item", "id": "jama", "field": "category", "value": "improvized"})
    assert resp.status_code == 400
    assert "unknown category" in resp.get_json()["error"]
    assert no_write == []


def test_field_router_accepts_a_real_category(client, sid, no_write):
    resp = client.post(f"/api/sessions/{sid}/name-truth",
                       json={"kind": "item", "id": "jama", "field": "category", "value": "improvised"})
    assert resp.status_code == 200
    assert len(no_write) == 1


def test_field_router_does_not_police_other_fields(client, sid, no_write):
    """The guard is scoped to `category` — a true_spelling is free text and must stay so."""
    resp = client.post(f"/api/sessions/{sid}/name-truth",
                       json={"kind": "item", "id": "jama", "field": "true_spelling", "value": "Jammus"})
    assert resp.status_code == 200
    assert len(no_write) == 1
