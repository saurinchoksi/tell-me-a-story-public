"""Tests for the Name Book (src/namebook.py) — the per-world human rulings store."""
import json

import pytest

import namebook


def test_slug():
    assert namebook.slug("Thomas & Friends") == "thomas-friends"
    assert namebook.slug("  Mahabharata ") == "mahabharata"
    assert namebook.slug("") == ""


def test_empty_world_is_first_contact(tmp_path):
    assert namebook.load_book("Mahabharata", base=tmp_path) == {}
    assert namebook.load_yes_map("Mahabharata", base=tmp_path) == {}
    assert namebook.load_no_words("Mahabharata", base=tmp_path) == set()
    assert namebook.fingerprint("Mahabharata", base=tmp_path) == ""


def test_corrupt_book_fails_loud(tmp_path):
    (tmp_path / "mahabharata.json").write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        namebook.load_book("Mahabharata", base=tmp_path)


def test_bless_writes_a_yes(tmp_path):
    namebook.bless("Mahabharata", "Bushma", "Bhishma", provenance="bless:s1",
                   base=tmp_path, now="t0")
    book = json.loads((tmp_path / "mahabharata.json").read_text())
    assert book["_world"] == "Mahabharata" and book["_version"] == 2
    assert book["yes"] == [{"heard": "Bushma", "canonical": "Bhishma",
                            "provenance": "bless:s1", "at": "t0"}]
    assert book["no"] == [] and book["spellings"] == []
    assert namebook.load_yes_map("Mahabharata", base=tmp_path) == {"bushma": "Bhishma"}


def test_bless_same_heard_is_idempotent_and_rebless_replaces(tmp_path):
    namebook.bless("Mahabharata", "Bushma", "Bhishma", base=tmp_path, now="t0")
    namebook.bless("Mahabharata", "bushma", "Bhishma", base=tmp_path, now="t1")
    book = namebook.load_book("Mahabharata", base=tmp_path)
    assert len(book["yes"]) == 1 and book["yes"][0]["at"] == "t0"
    # the human changed their mind: same heard form, different canonical -> replaced
    namebook.bless("Mahabharata", "Bushma", "Bhima", base=tmp_path, now="t2")
    book = namebook.load_book("Mahabharata", base=tmp_path)
    assert len(book["yes"]) == 1
    assert book["yes"][0]["canonical"] == "Bhima" and book["yes"][0]["at"] == "t2"


def test_bless_requires_world_and_names():
    with pytest.raises(ValueError):
        namebook.bless("", "Bushma", "Bhishma")
    with pytest.raises(ValueError):
        namebook.bless("Mahabharata", " ", "Bhishma", base="unused")


def test_reject_is_word_level_and_idempotent(tmp_path):
    namebook.reject("Thomas & Friends", "jammus", provenance="reject:s1",
                    base=tmp_path, now="t0")
    namebook.reject("Thomas & Friends", "jammus", base=tmp_path)  # idempotent on the word
    book = namebook.load_book("Thomas & Friends", base=tmp_path)
    assert len(book["no"]) == 1
    assert "canonical_cleaned" not in book["no"][0]      # word-level: no suggestion stored
    assert namebook.load_no_words("Thomas & Friends", base=tmp_path) == {"jammus"}


def test_reject_requires_world_and_word():
    with pytest.raises(ValueError):
        namebook.reject("", "jammus")
    with pytest.raises(ValueError):
        namebook.reject("Thomas & Friends", " ", base="unused")


def test_unbless_lifts_only_the_matching_yes(tmp_path):
    namebook.bless("Mahabharata", "Bushma", "Bhishma", base=tmp_path)
    namebook.bless("Mahabharata", "Bondos", "Pandavas", base=tmp_path)
    assert namebook.unbless("Mahabharata", "bushma", base=tmp_path) is True  # case-insensitive
    assert namebook.load_yes_map("Mahabharata", base=tmp_path) == {"bondos": "Pandavas"}
    assert namebook.unbless("Mahabharata", "Bushma", base=tmp_path) is False
    assert namebook.unbless("Narnia", "x", base=tmp_path) is False


def test_unreject_lifts_only_the_matching_word(tmp_path):
    namebook.reject("Thomas & Friends", "jammus", base=tmp_path)
    namebook.reject("Thomas & Friends", "pataki", base=tmp_path)
    assert namebook.unreject("Thomas & Friends", "jammus", base=tmp_path) is True
    assert namebook.load_no_words("Thomas & Friends", base=tmp_path) == {"pataki"}
    # lifting an absent word (or an absent world) is a no-op, reported as such
    assert namebook.unreject("Thomas & Friends", "jammus", base=tmp_path) is False
    assert namebook.unreject("Narnia", "x", base=tmp_path) is False


def test_pick_spelling_round_trip(tmp_path):
    """A pick rules one invented-name cluster: unify to a spelling, or None = don't
    unify. One ruling per cluster; a re-pick replaces; unpick lifts it."""
    namebook.pick_spelling("Thomas & Friends", "jameis", "Jammus",
                           provenance="bless:s1", base=tmp_path, now="t0")
    picks = namebook.load_spellings("Thomas & Friends", base=tmp_path)
    assert picks["jameis"]["spelling"] == "Jammus"
    # identical re-pick keeps the original stamp; a changed pick replaces
    namebook.pick_spelling("Thomas & Friends", "jameis", "Jammus", base=tmp_path, now="t1")
    assert namebook.load_spellings("Thomas & Friends", base=tmp_path)["jameis"]["at"] == "t0"
    namebook.pick_spelling("Thomas & Friends", "jameis", "Jameis", base=tmp_path, now="t2")
    picks = namebook.load_spellings("Thomas & Friends", base=tmp_path)
    assert picks["jameis"]["spelling"] == "Jameis" and picks["jameis"]["at"] == "t2"
    assert namebook.unpick_spelling("Thomas & Friends", "jameis", base=tmp_path) is True
    assert namebook.load_spellings("Thomas & Friends", base=tmp_path) == {}
    assert namebook.unpick_spelling("Thomas & Friends", "jameis", base=tmp_path) is False
    assert namebook.unpick_spelling("Narnia", "x", base=tmp_path) is False


def test_pick_spelling_null_means_dont_unify(tmp_path):
    """spelling=None is the narrow spelling-No: don't unify this cluster, never ask
    again — and it writes NO word-level no (the word rulings are untouched)."""
    namebook.pick_spelling("Thomas & Friends", "jameis", None, base=tmp_path, now="t0")
    picks = namebook.load_spellings("Thomas & Friends", base=tmp_path)
    assert picks["jameis"]["spelling"] is None
    assert namebook.load_no_words("Thomas & Friends", base=tmp_path) == set()


def test_pick_spelling_validation():
    with pytest.raises(ValueError):
        namebook.pick_spelling("", "jameis", "Jammus")
    with pytest.raises(ValueError):
        namebook.pick_spelling("Thomas & Friends", " ", "Jammus", base="unused")
    with pytest.raises(ValueError):
        # empty string is ambiguous — None is the explicit "don't unify"
        namebook.pick_spelling("Thomas & Friends", "jameis", "  ", base="unused")


def test_yes_and_no_coexist_and_fingerprint_tracks_changes(tmp_path):
    namebook.bless("Mahabharata", "Bushma", "Bhishma", base=tmp_path)
    fp1 = namebook.fingerprint("Mahabharata", base=tmp_path)
    assert fp1
    namebook.reject("Mahabharata", "arrows", base=tmp_path)
    fp2 = namebook.fingerprint("Mahabharata", base=tmp_path)
    assert fp2 and fp2 != fp1
    book = namebook.load_book("Mahabharata", base=tmp_path)
    assert len(book["yes"]) == 1 and len(book["no"]) == 1
