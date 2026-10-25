"""Tests for home canon (src/home_canon.py) — the family's own characters.
Fake names only, as everywhere."""
import json

import pytest

import home_canon


@pytest.fixture
def paths(tmp_path):
    cast = tmp_path / "home-canon.json"
    roster = tmp_path / "name_roster.json"
    roster.write_text(json.dumps({
        "people": [{"id": "child", "canonical": "Marta", "nicknames": ["Kiki"]}],
        "aliases": [{"token": "bemarta", "person_id": "child"}],
    }))
    return cast, roster


def test_absent_cast_is_empty(tmp_path):
    assert home_canon.load(tmp_path / "nope.json") == {"characters": []}
    assert home_canon.canonical_forms(tmp_path / "nope.json") == {}
    assert home_canon.fingerprint(tmp_path / "nope.json") == "absent"


def test_add_character_round_trip(paths):
    cast, roster = paths
    entry = home_canon.add_character("Jammus", path=cast, roster_path=roster)
    assert entry == {"id": "jammus", "canonical": "Jammus"}
    assert home_canon.canonical_forms(cast) == {"jammus": "Jammus"}
    fp1 = home_canon.fingerprint(cast)
    home_canon.add_character("Marley", path=cast, roster_path=roster)
    assert home_canon.fingerprint(cast) != fp1   # registration re-runs the fixers


def test_add_character_refuses_duplicates_and_roster_collisions(paths):
    cast, roster = paths
    home_canon.add_character("Jammus", path=cast, roster_path=roster)
    with pytest.raises(ValueError, match="already registered"):
        home_canon.add_character("jammus", path=cast, roster_path=roster)
    # a character can't share a form with a person (canonical, nickname, or alias)
    for taken in ("Marta", "Kiki", "Bemarta"):
        with pytest.raises(ValueError, match="collides with a roster form"):
            home_canon.add_character(taken, path=cast, roster_path=roster)
    with pytest.raises(ValueError, match="non-empty"):
        home_canon.add_character("  ", path=cast, roster_path=roster)


# ------------------------------ the stage-1 fixer effects ------------------------------
def _w(word, start):
    return {"word": word, "start": start, "end": start + 0.3, "probability": 0.9}


def _transcript(words_by_seg):
    return {"text": "", "segments": [
        {"id": i, "text": "".join(x[0] for x in words),
         "words": [_w(word, start) for word, start in words]}
        for i, words in enumerate(words_by_seg)]}


def test_familyfix_never_questions_a_registered_character(paths, tmp_path):
    """The Marley class dies at the root: a registered character's correct spelling is
    never a family candidate, however much it sounds like a real person."""
    import familyfix
    cast, roster = paths
    # "Marda" sounds like Marta — without registration it's a candidate...
    t = _transcript([[(" Marda", 0.0), (" laughed", 0.4)]])
    r = familyfix.load_roster(roster)
    assert len(familyfix.family_candidates(t, r)) == 1
    # ...registered, it is left alone everywhere
    home_canon.add_character("Marda", path=cast, roster_path=roster)
    forms = frozenset(home_canon.canonical_forms(cast))
    assert familyfix.family_candidates(t, r, home_forms=forms) == []


def test_spellingfix_registered_spelling_outranks_the_vote(paths, tmp_path):
    """The registry beats frequency: Jammus is registered, so a cluster where 'Jameis'
    is the majority still proposes Jammus (method home-canon), and the registered form
    is never a deviant."""
    import spellingfix
    cast, roster = paths
    wordlist = tmp_path / "words"
    wordlist.write_text("story\n")
    home_canon.add_character("Jammus", path=cast, roster_path=roster)
    t = _transcript([[(" Jameis", 0.0), (" Jameis", 0.4), (" Jammus", 0.8)]])
    res = spellingfix.run_spellingfix(t, {}, wordlist_path=wordlist, home_path=cast)
    assert len(res["groups"]) == 1
    g = res["groups"][0]
    assert g["heard_cleaned"] == "jameis"      # the majority form is the deviant now
    assert g["canonical"] == "Jammus" and g["method"] == "home-canon"


def test_spellingfix_worldless_clusters_file_under_home(paths, tmp_path):
    """A cluster outside every recognized story world gets world 'Home' — its picks
    finally have a durable book file (name-book/home.json)."""
    import spellingfix
    cast, _ = paths
    wordlist = tmp_path / "words"
    wordlist.write_text("story\n")
    t = _transcript([[(" Khadijah", 0.0)], [(" Khadija", 1.0)]])
    res = spellingfix.run_spellingfix(t, {}, wordlist_path=wordlist, home_path=cast)
    assert res["groups"][0]["world"] == home_canon.WORLD


def test_spellingfix_defer_to_namefix_stays_senior(paths, tmp_path):
    """A cluster namefix already asks about is namefix's territory even when a home
    character sits in it — the canon-dedup precedence is unchanged."""
    import spellingfix
    cast, roster = paths
    wordlist = tmp_path / "words"
    wordlist.write_text("story\n")
    home_canon.add_character("Jammus", path=cast, roster_path=roster)
    t = _transcript([[(" Jameis", 0.0), (" Jammus", 0.4)]])
    res = spellingfix.run_spellingfix(t, {}, wordlist_path=wordlist, home_path=cast,
                                      defer_words={"jameis"})
    assert res["groups"] == []


def test_fingerprints_track_the_cast(paths):
    import familyfix
    import spellingfix
    cast, roster = paths
    ff1 = familyfix.config_fingerprint(roster, home_path=cast)
    sf1 = spellingfix.config_fingerprint(roster_path=roster, home_path=cast)
    home_canon.add_character("Jammus", path=cast, roster_path=roster)
    assert familyfix.config_fingerprint(roster, home_path=cast) != ff1
    assert spellingfix.config_fingerprint(roster_path=roster, home_path=cast) != sf1


def test_api_home_canon_endpoints(tmp_path):
    from api.app import create_app
    cast = tmp_path / "home-canon.json"
    roster = tmp_path / "name_roster.json"
    roster.write_text(json.dumps({
        "people": [{"id": "child", "canonical": "Marta"}], "aliases": []}))
    app = create_app(sessions_dir=tmp_path, profiles_path=str(tmp_path / "p.json"),
                     name_book_dir=tmp_path / "name-book", roster_path=roster,
                     home_canon_path=cast)
    app.config["TESTING"] = True
    with app.test_client() as c:
        assert c.get("/api/home-canon").get_json() == {"characters": []}
        r = c.post("/api/home-canon/characters", json={"canonical": "Jammus"})
        assert r.status_code == 200
        assert r.get_json()["character"]["id"] == "jammus"
        r = c.post("/api/home-canon/characters", json={"canonical": "Marta"})
        assert r.status_code == 400 and "roster form" in r.get_json()["error"]
        assert json.loads(cast.read_text())["characters"][0]["canonical"] == "Jammus"
