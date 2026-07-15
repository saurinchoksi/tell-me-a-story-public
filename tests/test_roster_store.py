"""Tests for the roster store (src/roster_store.py) — add-only, validated writes.
All names are FAKE (the schema-example names); the real roster never appears in tests."""
import json

import pytest

import roster_store


@pytest.fixture
def path(tmp_path):
    p = tmp_path / "name_roster.json"
    p.write_text(json.dumps({
        "people": [
            {"id": "child", "canonical": "Marta", "role": "child",
             "circle": "family", "nicknames": ["Kiki"]},
            {"id": "parent", "canonical": "Kiran", "circle": "family"},
        ],
        "aliases": [{"token": "bemarta", "person_id": "child"}],
    }))
    return p


def test_absent_file_is_the_empty_roster(tmp_path):
    assert roster_store.load(tmp_path / "nope.json") == {"people": [], "aliases": []}


def test_add_person_with_circle(path):
    person = roster_store.add_person("Priya", circle="friend", path=path)
    assert person == {"id": "priya", "canonical": "Priya", "circle": "friend"}
    roster = roster_store.load(path)
    assert roster["people"][-1]["circle"] == "friend"


def test_add_person_rejects_collisions_and_bad_circle(path):
    with pytest.raises(ValueError, match="collides"):
        roster_store.add_person("marta", path=path)         # existing canonical
    with pytest.raises(ValueError, match="collides"):
        roster_store.add_person("Kiki", path=path)          # existing nickname
    with pytest.raises(ValueError, match="collides"):
        roster_store.add_person("Bemarta", path=path)       # existing alias
    with pytest.raises(ValueError, match="circle"):
        roster_store.add_person("Priya", circle="stranger", path=path)
    with pytest.raises(ValueError, match="non-empty"):
        roster_store.add_person("  ", path=path)


def test_add_nickname(path):
    person = roster_store.add_nickname("parent", "Kiku", path=path)
    assert person["nicknames"] == ["Kiku"]
    with pytest.raises(ValueError, match="unknown person_id"):
        roster_store.add_nickname("ghost", "X", path=path)
    with pytest.raises(ValueError, match="collides"):
        roster_store.add_nickname("child", "Kiku", path=path)   # now taken


def test_add_alias_rejects_redundant_and_collisions(path):
    # a distortion whose codes can't reach any form -> accepted, cleaned
    alias = roster_store.add_alias("parent", " Okiran. ", path=path)
    assert alias == {"token": "okiran", "person_id": "parent"}
    # sound-alike of a canonical -> the phonetic layer already has it
    with pytest.raises(ValueError, match="redundant"):
        roster_store.add_alias("child", "Marda", path=path)
    # sound-alike of a NICKNAME -> also redundant (nicknames are full name forms)
    with pytest.raises(ValueError, match="redundant"):
        roster_store.add_alias("child", "Keekee", path=path)
    with pytest.raises(ValueError, match="collides"):
        roster_store.add_alias("child", "bemarta", path=path)


def test_writes_are_valid_json_round_trips(path):
    roster_store.add_person("Priya", circle="friend", path=path)
    roster_store.add_nickname("priya", "Pri", path=path)
    roster = json.loads(path.read_text())
    assert roster["people"][-1]["nicknames"] == ["Pri"]
    assert len(roster["people"]) == 3 and len(roster["aliases"]) == 1
