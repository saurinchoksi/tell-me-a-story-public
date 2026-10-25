"""Tests for spellingfix (the M9b fixer) — m9b cluster semantics in memory, the
majority proposal, and the one-verdict-rules-the-cluster group shape."""
import json

import pytest

import namefix
import spellingfix


def w(word, start):
    return {"word": word, "start": start, "end": start + 0.3, "probability": 0.9}


def make_transcript(words_by_seg, stories=None):
    t = {"text": "", "segments": [
        {"id": i, "text": "".join(x[0] for x in words),
         "words": [w(word, start) for word, start in words]}
        for i, words in enumerate(words_by_seg)]}
    if stories is not None:
        t["_stories"] = stories
    return t


@pytest.fixture
def wordlist(tmp_path):
    """A small dictionary standing in for /usr/share/dict/words."""
    path = tmp_path / "words"
    path.write_text("\n".join(["what", "yeah", "where", "there", "engine", "story"]))
    return path


# ------------------------------ clustering (m9b semantics) ------------------------------
def test_inconsistent_invented_name_clusters(wordlist):
    """Jammus/Jameis (sound-alike, both non-words) form one cluster keyed by the min
    cleaned form; a consistently-spelled name never clusters."""
    t = make_transcript([[(" Jammus", 0.0), (" went", 0.4)],
                         [(" Jameis", 1.0), (" Jammus", 1.4)],
                         [(" Pataki", 2.0), (" Pataki", 2.4)]])
    clusters = spellingfix.find_clusters(t, wordlist_path=wordlist)
    assert len(clusters) == 1
    cl = clusters[0]
    assert cl["cluster_id"] == "jameis"
    assert set(cl["forms"]) == {"jammus", "jameis"}
    assert cl["forms"]["jammus"]["count"] == 2


def test_all_common_cluster_dropped(wordlist):
    """What/Whats (all ordinary words, incl. the trailing-s rule) never become a
    question — the Bibi-class boundary, deferred to the offline judge."""
    t = make_transcript([[(" What", 0.0), (" Whats", 0.4), (" There", 0.8), (" Theres", 1.2)]])
    assert spellingfix.find_clusters(t, wordlist_path=wordlist) == []


def test_one_nonword_spelling_keeps_the_cluster(wordlist):
    """A cluster with at least one non-dictionary spelling is a clear name — the
    misspelling is the signal (Engine/Enjin: 'enjin' is not a word)."""
    t = make_transcript([[(" Engine", 0.0), (" Enjin", 0.4)]])
    clusters = spellingfix.find_clusters(t, wordlist_path=wordlist)
    assert len(clusters) == 1


def test_short_and_lowercase_tokens_never_cluster(wordlist):
    t = make_transcript([[(" Jam", 0.0), (" jammus", 0.4), (" jameis", 0.8)]])
    assert spellingfix.find_clusters(t, wordlist_path=wordlist) == []


def test_roster_names_excluded(tmp_path, wordlist):
    """Family-name spellings belong to familyfix — spellingfix must not double-ask."""
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps({
        "people": [{"id": "child", "canonical": "Marta"}], "aliases": []}))
    t = make_transcript([[(" Marta", 0.0), (" Marda", 0.4)]])
    assert spellingfix.find_clusters(t, wordlist_path=wordlist,
                                     roster_path=roster) == []
    # without the roster, the same pair clusters (proves the exclusion did the work)
    assert len(spellingfix.find_clusters(t, wordlist_path=wordlist)) == 1


# ------------------------------ the proposal ------------------------------
def test_majority_wins():
    forms = {"jammus": {"count": 2}, "jameis": {"count": 1}}
    assert spellingfix.propose_spelling(forms) == "jammus"


def test_tie_breaks_alphabetically():
    forms = {"jammus": {"count": 1}, "jameis": {"count": 1}}
    assert spellingfix.propose_spelling(forms) == "jameis"


# ------------------------------ the groups ------------------------------
STORIES = [{"start_id": 0, "end_id": 1, "world": "", "title": "T"}]


def test_groups_one_per_deviant_sharing_cluster(wordlist):
    """Three sound-alike spellings, majority Jammus -> TWO questions (Jameis, Jamus),
    same cluster_id, same proposal — one verdict on either card rules the cluster."""
    t = make_transcript([[(" Jammus", 0.0), (" Jammus", 0.4), (" Jamus", 0.8)],
                         [(" Jameis", 1.0)]], stories=STORIES)
    res = spellingfix.run_spellingfix(t, {0: "Thomas & Friends"}, wordlist_path=wordlist)
    groups = sorted(res["groups"], key=lambda g: g["heard_cleaned"])
    assert len(groups) == 2
    assert [g["heard_cleaned"] for g in groups] == ["jameis", "jamus"]
    for g in groups:
        assert g["canonical"] == "Jammus" and g["cluster_id"] == "jameis"
        assert g["label"] == "inconsistent-spelling" and g["method"] == "majority"
        assert g["world"] == "Thomas & Friends"
        assert g["majority_count"] == 2
        assert g["cluster_spellings"] == ["Jameis", "Jammus", "Jamus"]
        assert g["story_id"] is None
    assert groups[0]["gid"] == "spelling:jameis:jameis"
    assert groups[1]["gid"] == "spelling:jameis:jamus"
    # the majority spelling's own spots are not questioned
    assert all(o["token"].strip().rstrip(".") != "Jammus"
               for g in groups for o in g["occurrences"])


def test_worldless_files_under_home_mixed_gets_none(wordlist):
    """v1.2.0: a cluster unanimously OUTSIDE recognized story worlds files
    under 'Home' — the family's own canon is the ruling home for world-less clusters.
    A cluster genuinely SPANNING two recognized worlds still gets no home."""
    t = make_transcript([[(" Jammus", 0.0)], [(" Jameis", 1.0)]])   # no _stories
    res = spellingfix.run_spellingfix(t, {}, wordlist_path=wordlist)
    assert res["groups"][0]["world"] == "Home"

    stories = [{"start_id": 0, "end_id": 0, "world": "", "title": "A"},
               {"start_id": 1, "end_id": 1, "world": "", "title": "B"}]
    t2 = make_transcript([[(" Jammus", 0.0)], [(" Jameis", 1.0)]], stories=stories)
    res2 = spellingfix.run_spellingfix(t2, {0: "Thomas & Friends", 1: "Mahabharata"},
                                       wordlist_path=wordlist)
    assert res2["groups"][0]["world"] == ""    # spans two worlds -> genuinely ambiguous


def test_ui_chips_stay_quiet(wordlist):
    """Spelling questions carry neutral namefix signals: no echo chip, and >=1 sibling
    by construction so the 'spelled the same every time' chip can never fire."""
    t = make_transcript([[(" Jammus", 0.0), (" Jammus", 0.4)], [(" Jameis", 1.0)]])
    g = spellingfix.run_spellingfix(t, {}, wordlist_path=wordlist)["groups"][0]
    assert g["echo_in_sentence"] is False
    assert g["n_sibling_spellings"] >= 1


def test_cluster_defers_to_namefix(wordlist):
    """v1.1.0: a cluster touching any word namefix asks about (heard OR suggested
    canonical) is dropped — a majority vote on a canon-garble family proposes the
    wrong thing (Bhishma→Bushma; Jammus→James). Unrelated clusters survive."""
    t = make_transcript([[(" Bushma", 0.0), (" Bishma", 0.4), (" Bhishma", 0.8)],
                         [(" Khadijah", 1.0), (" Khadija", 1.4)]])
    nf_groups = [{"heard_cleaned": "bushma", "canonical": "Bhishma"},
                 {"heard_cleaned": "bishma", "canonical": "Bhishma"}]
    defer = spellingfix.defer_words_from(nf_groups)
    assert defer == {"bushma", "bishma", "bhishma"}
    res = spellingfix.run_spellingfix(t, {}, wordlist_path=wordlist,
                                      defer_words=defer)
    heards = {g["heard_cleaned"] for g in res["groups"]}
    assert "bhishma" not in heards and "bishma" not in heards
    assert heards == {"khadijah"}    # the genuine invented cluster still asks
    # without the defer set, the canon-garble cluster would have asked (the baseline)
    res_base = spellingfix.run_spellingfix(t, {}, wordlist_path=wordlist)
    assert any(g["cluster_id"] == "bhishma" or "Bushma" in g["cluster_spellings"]
               for g in res_base["groups"])


def test_groups_flow_into_shared_queue(tmp_path, wordlist):
    t = make_transcript([[(" Jammus", 0.0)], [(" Jameis", 1.0)]])
    res = spellingfix.run_spellingfix(t, {}, wordlist_path=wordlist)
    namefix.write_queue(tmp_path, spellingfix.FIXER, res["groups"], t,
                        spellingfix.VERSION,
                        spellingfix.config_fingerprint(wordlist_path=wordlist))
    data = json.loads((tmp_path / "pending-name-corrections.json").read_text())
    assert data["groups"][0]["fixer"] == "spelling"
    assert "spelling" in data["fixers"]


def test_config_fingerprint_tracks_wordlist(tmp_path, wordlist):
    fp1 = spellingfix.config_fingerprint(wordlist_path=wordlist)
    assert fp1 == spellingfix.config_fingerprint(wordlist_path=wordlist)
    other = tmp_path / "words2"
    other.write_text("different\n")
    assert spellingfix.config_fingerprint(wordlist_path=other) != fp1
