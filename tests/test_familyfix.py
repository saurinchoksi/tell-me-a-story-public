"""Tests for familyfix (the M9a fixer) — detector-mirror candidates, the re-listen gate,
and the coverage-first fallback. All names here are FAKE (the schema-example names);
the real roster is gitignored and never appears in tests."""
import json

import pytest

import familyfix
import namefix


@pytest.fixture
def roster_path(tmp_path):
    """A fake-family roster in the real schema: two people + one front-of-word alias."""
    path = tmp_path / "name_roster.json"
    path.write_text(json.dumps({
        "people": [
            {"id": "child", "canonical": "Marta", "role": "child"},
            {"id": "parent", "canonical": "Kiran", "role": "parent"},
        ],
        "aliases": [{"token": "bemarta", "person_id": "child"}],
    }))
    return path


def w(word, start):
    return {"word": word, "start": start, "end": start + 0.3, "probability": 0.9}


def make_transcript(words_by_seg):
    """words_by_seg: list of lists of (word, start) tuples."""
    return {"text": "", "segments": [
        {"id": i, "text": "".join(t[0] for t in words),
         "words": [w(word, start) for word, start in words]}
        for i, words in enumerate(words_by_seg)]}


def fake_redecode_factory(mapping):
    """A redecode injectable: for each job whose blind (cleaned) is in `mapping`, gate a
    decision as if Whisper re-heard that word at the right offset. Jobs not in the
    mapping produce nothing (the re-listen 'heard nothing usable')."""
    from detectors.phonetics import clean

    def fake_redecode(audio_path, jobs, singles):
        decisions = []
        for j in jobs:
            heard = mapping.get(clean(j["cand"]["blind"]))
            if heard is None:
                continue
            d = namefix.gate_decision([(heard, 0.0)], j["cand"]["blind"],
                                      j["cast_clean"], set(singles))
            if d:
                decisions.append({**j["cand"], **d, "world": j["world"]})
        return decisions
    return fake_redecode


# ------------------------------ candidates (detector mirror) ------------------------------
def test_candidates_mirror_detector_semantics(roster_path):
    """The frozen detector's exact matching: phonetic garble matches, exact alias
    matches, correctly-spelled canonical skipped, lowercase homophone dropped by the
    capitalization gate (checked AFTER matching), unrelated words untouched."""
    t = make_transcript([[
        (" Marta", 0.0),        # correctly spelled canonical — never a candidate
        (" Marda", 0.4),        # phonetic garble of Marta -> candidate
        (" marda", 0.8),        # same garble, lowercase -> capitalization gate drops it
        (" Bemarta", 1.2),      # the alias, capitalized -> candidate (exact layer)
        (" dinner", 1.6),       # unrelated -> nothing
    ]])
    roster = familyfix.load_roster(roster_path)
    cands = familyfix.family_candidates(t, roster)
    assert [(c["blind"], c["matched"]) for c in cands] == [
        ("Marda", ["Marta"]), ("Bemarta", ["Marta"])]


def test_candidates_have_no_dictionary_gate(roster_path):
    """The un-gated-M9a rule: a capitalized common word that sounds like a family
    member IS a candidate — the disguise is exactly what must become a question."""
    t = make_transcript([[(" Mart", 0.0)]])   # a real word ("mart") sounding like Marta
    roster = familyfix.load_roster(roster_path)
    cands = familyfix.family_candidates(t, roster)
    assert len(cands) == 1 and cands[0]["matched"] == ["Marta"]


def test_load_roster_rejects_redundant_alias(tmp_path):
    path = tmp_path / "roster.json"
    path.write_text(json.dumps({
        "people": [{"id": "child", "canonical": "Marta"}],
        "aliases": [{"token": "marda", "person_id": "child"}],  # DM-reachable already
    }))
    with pytest.raises(ValueError, match="Redundant alias"):
        familyfix.load_roster(path)


def test_load_roster_rejects_unknown_person(tmp_path):
    path = tmp_path / "roster.json"
    path.write_text(json.dumps({
        "people": [{"id": "child", "canonical": "Marta"}],
        "aliases": [{"token": "bemarta", "person_id": "ghost"}],
    }))
    with pytest.raises(ValueError, match="unknown person_id"):
        familyfix.load_roster(path)


# ------------------------------ the run: gate + coverage-first ------------------------------
def test_redecode_heard_name_makes_a_sounds_like_question(roster_path):
    t = make_transcript([[(" saw", 0.0), (" Marda", 0.4), (" today", 0.8)]])
    res = familyfix.run_familyfix(t, "unused.m4a", roster_path=roster_path,
                                  redecode=fake_redecode_factory({"marda": "Marta"}))
    assert len(res["groups"]) == 1
    g = res["groups"][0]
    assert g["gid"] == "family:marda:marta"
    assert g["world"] == "Family" and g["canonical"] == "Marta"
    assert g["label"] == "sounds-like" and g["redecode_heard"] is True
    assert g["method"] == "exact"     # the re-decode landed exactly on the cast spelling


def test_coverage_first_silent_redecode_still_asks(roster_path):
    """The coverage-first dial: the re-listen produced nothing usable, but the detector
    knows this spot — the question still goes up, suggested from the roster match,
    carrying redecode_heard=False so verdicts can judge the machine's second silence."""
    t = make_transcript([[(" saw", 0.0), (" Marda", 0.4)]])
    res = familyfix.run_familyfix(t, "unused.m4a", roster_path=roster_path,
                                  redecode=fake_redecode_factory({}))  # hears nothing
    assert len(res["groups"]) == 1
    g = res["groups"][0]
    assert g["canonical"] == "Marta" and g["method"] == "roster"
    assert g["redecode_heard"] is False
    assert g["label"] == "sounds-like"    # marda ~ marta: computed the normal way


def test_ordinary_word_disguise_is_always_a_human_call(roster_path):
    """A common-word disguise ('Mart') gets the ordinary-word label whether or not the
    re-listen helps — the machine never trusts itself on real English words."""
    t = make_transcript([[(" the", 0.0), (" Mart", 0.4), (" opened", 0.8)]])
    res = familyfix.run_familyfix(t, "unused.m4a", roster_path=roster_path,
                                  redecode=fake_redecode_factory({}))
    g = res["groups"][0]
    assert g["label"] == "ordinary-word" and g["redecode_heard"] is False


def test_alias_maps_back_to_its_person(roster_path):
    """A re-listen that lands on the alias form still identifies the person (the alias
    is in the re-decode vocabulary), and an unheard alias falls back to its person."""
    t = make_transcript([[(" Bemarta", 0.0), (" laughed", 0.4)]])
    res = familyfix.run_familyfix(t, "unused.m4a", roster_path=roster_path,
                                  redecode=fake_redecode_factory({"bemarta": "Marta"}))
    g = res["groups"][0]
    assert g["canonical"] == "Marta" and g["redecode_heard"] is True


def test_no_candidates_no_model(roster_path):
    """A transcript with nothing roster-shaped returns empty WITHOUT invoking the
    redecode at all (no model spin-up for nothing)."""
    t = make_transcript([[(" hello", 0.0), (" there", 0.4)]])

    def exploding_redecode(*a):
        raise AssertionError("redecode must not run with no candidates")
    res = familyfix.run_familyfix(t, "unused.m4a", roster_path=roster_path,
                                  redecode=exploding_redecode)
    assert res["groups"] == []


def test_missing_roster_fails_loud(tmp_path):
    with pytest.raises(FileNotFoundError):
        familyfix.run_familyfix({"segments": []}, "unused.m4a",
                                roster_path=tmp_path / "absent.json")


def test_groups_flow_into_shared_queue(roster_path, tmp_path):
    """End of the chain: familyfix groups land in the shared queue tagged
    fixer='family', merging beside namefix's groups."""
    t = make_transcript([[(" saw", 0.0), (" Marda", 0.4)]])
    namefix.write_queue(tmp_path, "namefix", [], t, "2.1.0", "nf-fp", worlds=[])
    res = familyfix.run_familyfix(t, "unused.m4a", roster_path=roster_path,
                                  redecode=fake_redecode_factory({"marda": "Marta"}))
    namefix.write_queue(tmp_path, familyfix.FIXER, res["groups"], t,
                        familyfix.VERSION, familyfix.config_fingerprint(roster_path))
    data = json.loads((tmp_path / "pending-name-corrections.json").read_text())
    assert data["groups"][0]["fixer"] == "family"
    assert data["groups"][0]["gid"] == "family:marda:marta"
    assert set(data["fixers"]) == {"namefix", "family"}


# ------------------------------ nicknames (TMAS-56) ------------------------------
@pytest.fixture
def nickname_roster(tmp_path):
    """Marta has two nicknames: 'Marto' (shares her sound codes) and 'Kiki' (its own
    sound). Plus a friend — same roster, circle tag."""
    path = tmp_path / "name_roster.json"
    path.write_text(json.dumps({
        "people": [
            {"id": "child", "canonical": "Marta", "circle": "family",
             "nicknames": ["Marto", "Kiki"]},
            {"id": "friend", "canonical": "Priya", "circle": "friend"},
        ],
        "aliases": [],
    }))
    return path


def test_nickname_is_correct_as_heard_never_a_candidate(nickname_roster):
    """A nickname on tape is right, not a garble — same skip as a canonical."""
    t = make_transcript([[(" Marto", 0.0), (" Kiki", 0.4), (" Marta", 0.8)]])
    roster = familyfix.load_roster(nickname_roster)
    assert familyfix.family_candidates(t, roster) == []


def test_distinct_sound_nickname_garble_suggests_the_nickname(nickname_roster):
    """A garble of 'Kiki' is caught by the nickname's own sound codes and corrected
    toward the nickname as spoken — never silently promoted to 'Marta'."""
    t = make_transcript([[(" saw", 0.0), (" Keekee", 0.4)]])
    res = familyfix.run_familyfix(t, "unused.m4a", roster_path=nickname_roster,
                                  redecode=fake_redecode_factory({}))
    assert len(res["groups"]) == 1
    assert res["groups"][0]["canonical"] == "Kiki"


def test_shared_sound_nickname_does_not_double_the_cards(nickname_roster):
    """The canonical-preference rule: 'Marto' shares Marta's codes, so a garble like
    'Marda' suggests Marta alone — one card, not one per form. The re-listen can still
    land on the nickname (it's in the cast vocabulary mapping to itself)."""
    t = make_transcript([[(" saw", 0.0), (" Marda", 0.4)]])
    res = familyfix.run_familyfix(t, "unused.m4a", roster_path=nickname_roster,
                                  redecode=fake_redecode_factory({}))
    assert [g["canonical"] for g in res["groups"]] == ["Marta"]
    # audio wins when it speaks: a re-listen that hears the nickname suggests it
    res2 = familyfix.run_familyfix(t, "unused.m4a", roster_path=nickname_roster,
                                   redecode=fake_redecode_factory({"marda": "Marto"}))
    assert [g["canonical"] for g in res2["groups"]] == ["Marto"]


def test_friend_is_a_person_like_any_other(nickname_roster):
    """circle='friend' changes nothing mechanically — a friend's garble is caught."""
    t = make_transcript([[(" saw", 0.0), (" Preeya", 0.4)]])
    res = familyfix.run_familyfix(t, "unused.m4a", roster_path=nickname_roster,
                                  redecode=fake_redecode_factory({}))
    assert [g["canonical"] for g in res["groups"]] == ["Priya"]


def test_prompt_carries_nicknames(nickname_roster):
    """Whisper can only hear a form it was primed with — nicknames ride in the ear."""
    name_forms, _, _, _ = familyfix.load_roster(nickname_roster)
    assert set(name_forms) == {"Marta", "Marto", "Kiki", "Priya"}
    prompt = familyfix.PROMPT_TEMPLATE.format(names=", ".join(name_forms))
    assert "Kiki" in prompt and "Marto" in prompt


def test_alias_redundant_against_nickname_codes(tmp_path):
    """The loader's alias rule extends to nicknames: an alias whose codes reach a
    nickname is redundant (the phonetic layer already has it)."""
    path = tmp_path / "roster.json"
    path.write_text(json.dumps({
        "people": [{"id": "child", "canonical": "Marta", "nicknames": ["Kiki"]}],
        "aliases": [{"token": "keekee", "person_id": "child"}],
    }))
    with pytest.raises(ValueError, match="Redundant alias"):
        familyfix.load_roster(path)


def test_config_fingerprint_tracks_roster_content(roster_path, tmp_path):
    fp1 = familyfix.config_fingerprint(roster_path)
    assert fp1 == familyfix.config_fingerprint(roster_path)
    changed = tmp_path / "roster2.json"
    changed.write_text(json.dumps({
        "people": [{"id": "child", "canonical": "Marta"}], "aliases": []}))
    assert familyfix.config_fingerprint(changed) != fp1
