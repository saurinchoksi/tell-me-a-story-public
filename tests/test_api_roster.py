"""Tests for the /api/roster endpoints — the add-only roster page."""
import json

import pytest

from api.app import create_app


@pytest.fixture
def client(tmp_path):
    roster = tmp_path / "name_roster.json"
    roster.write_text(json.dumps({
        "people": [{"id": "child", "canonical": "Marta", "circle": "family",
                    "nicknames": ["Kiki"]}],
        "aliases": [],
    }))
    app = create_app(sessions_dir=tmp_path, profiles_path=str(tmp_path / "p.json"),
                     name_book_dir=tmp_path / "name-book", roster_path=roster)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c, roster


def test_get_roster(client):
    c, _ = client
    data = c.get("/api/roster").get_json()
    assert data["people"][0]["canonical"] == "Marta"
    assert data["people"][0]["nicknames"] == ["Kiki"]


def test_add_person(client):
    c, roster = client
    r = c.post("/api/roster/people", json={"canonical": "Priya", "circle": "friend"})
    assert r.status_code == 200
    assert r.get_json()["person"]["id"] == "priya"
    assert json.loads(roster.read_text())["people"][-1]["circle"] == "friend"


def test_add_person_validation_400(client):
    c, _ = client
    r = c.post("/api/roster/people", json={"canonical": "Kiki"})   # nickname collision
    assert r.status_code == 400 and "collides" in r.get_json()["error"]
    r = c.post("/api/roster/people", json={})
    assert r.status_code == 400


def test_add_nickname_and_alias(client):
    c, roster = client
    r = c.post("/api/roster/people/child/nicknames", json={"nickname": "Mimi"})
    assert r.status_code == 200
    assert "Mimi" in r.get_json()["person"]["nicknames"]
    r = c.post("/api/roster/people/child/aliases", json={"token": "Okmarta"})
    assert r.status_code == 200
    assert json.loads(roster.read_text())["aliases"][0]["token"] == "okmarta"
    # redundant alias (sound-alike of the canonical) refused with the reason
    r = c.post("/api/roster/people/child/aliases", json={"token": "Marda"})
    assert r.status_code == 400 and "redundant" in r.get_json()["error"]
    # unknown person
    r = c.post("/api/roster/people/ghost/nicknames", json={"nickname": "X"})
    assert r.status_code == 400
