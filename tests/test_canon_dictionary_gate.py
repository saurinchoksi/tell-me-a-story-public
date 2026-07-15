"""The canon dictionary gate drops M9c flags that are really ordinary words dressed up
as a misspelled canon name (the held-out 'arrows -> Arrow' failure), WITHOUT removing a
real canon name that merely happens to be a dictionary entry.

The dictionary is stubbed with a small fixed set so the test never reads the system
wordlist — what matters is the gate's logic, not which words macOS happens to list.
No model is loaded (the gate is pure post-processing of verdict flags)."""
import pytest

from detectors.story_names import _audit


@pytest.fixture(autouse=True)
def stub_dictionary(monkeypatch):
    """Treat a small fixed set as 'common', with the same trailing-'s' tolerance the real
    _is_common has — so 'arrows'->'arrow' and 'wars'->'war' read as common. Crucially the
    canon names Drona/Dhritarashtra are ALSO in the dict here, to prove the proper-name
    guard (not the bare dictionary) is what protects them."""
    common = {"arrow", "war", "king", "drona", "dhritarashtra"}

    def fake_is_common(word):
        base = word[:-1] if word.endswith("s") else word
        return base in common

    monkeypatch.setattr(_audit._NCD, "_is_common", fake_is_common)


def _m9c(canonical, wrong):
    """A minimal canon flag as _flag_from_verdict would build it."""
    return {"case": "M9c", "canonical": canonical,
            "wrong_cleaned": sorted(wrong), "wrong_surface": sorted(wrong)}


def test_ordinary_canonical_with_common_cluster_drops_whole_flag():
    """The 'arrows/wars/Urzi -> Arrow' card: canonical 'Arrow' is an ordinary word AND its
    cluster contains ordinary words ('arrows'/'wars'), so the entire flag is dropped — even
    the capitalized non-word 'urzi' member goes, because the canon claim is incoherent."""
    flags = [_m9c("Arrow", ["arrows", "wars", "urzi"])]
    kept = _audit.gate_canon_flags(flags, singles={"urzi"})
    assert kept == []


def test_real_catch_with_nonword_canonical_survives():
    """Urjun -> Arjuna: neither the canonical nor the token is a dictionary word, so the
    real catch is untouched."""
    flags = [_m9c("Arjuna", ["urjun"])]
    kept = _audit.gate_canon_flags(flags, singles={"urjun"})
    assert len(kept) == 1
    assert kept[0]["wrong_cleaned"] == ["urjun"]
    assert "dict_gated" not in kept[0]  # nothing was removed


def test_dictionary_word_canon_name_kept_when_used_as_a_name():
    """The recall guard: 'Dhritarashtra' is a dictionary entry, but in its own story it is
    a capitalized name (in `singles`), so a Dhrashtra -> Dhritarashtra catch must survive.
    This is the case a bare dictionary gate would have wrongly deleted."""
    flags = [_m9c("Dhritarashtra", ["dhrashtra"])]
    kept = _audit.gate_canon_flags(flags, singles={"dhritarashtra", "dhrashtra"})
    assert len(kept) == 1
    assert kept[0]["canonical"] == "Dhritarashtra"


def test_dictionary_word_canon_name_kept_when_misspelling_is_a_nonword():
    """The recall fix for the real Pandavas regression: 'Dhritarashtra' is a dictionary
    entry and its correct form never appears capitalized in this story (NOT in `singles`),
    so the canonical alone reads as ordinary — BUT its only misspelling 'dhrashtra' is a
    non-word, which is exactly what a real canon name spelled wrong looks like. The cluster
    contains no ordinary word, so the canonical-cut must NOT fire and the catch survives."""
    flags = [_m9c("Dhritarashtra", ["dhrashtra"])]
    kept = _audit.gate_canon_flags(flags, singles={"dhrashtra"})
    assert len(kept) == 1
    assert kept[0]["canonical"] == "Dhritarashtra"
    assert kept[0]["wrong_cleaned"] == ["dhrashtra"]


def test_partial_gate_trims_common_wrong_spellings_and_records_them():
    """A flag with a legit canonical but a common word clustered into its wrong spellings:
    the common word is removed, the real misspelling stays, and `dict_gated` records the cut."""
    flags = [_m9c("Arjuna", ["urjun", "war"])]
    kept = _audit.gate_canon_flags(flags, singles={"urjun"})
    assert len(kept) == 1
    assert kept[0]["wrong_cleaned"] == ["urjun"]
    assert kept[0]["wrong_surface"] == ["urjun"]
    assert kept[0]["dict_gated"] == ["war"]


def test_m9b_and_m9d_flags_pass_through_untouched():
    """The gate is canon-only — an inconsistency (M9b) or substitution (M9d) flag whose
    canonical is an ordinary word is NOT its concern and must pass straight through."""
    m9b = {"case": "M9b", "canonical": "King", "wrong_cleaned": ["kingg"], "wrong_surface": ["kingg"]}
    m9d = {"case": "M9d-suspect", "canonical": "War", "wrong_cleaned": ["wahr"], "wrong_surface": ["wahr"]}
    kept = _audit.gate_canon_flags([m9b, m9d], singles=set())
    assert kept == [m9b, m9d]
