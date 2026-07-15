"""Unit tests for the family-name detector (fake names only — never real ones)."""

import json

import pytest

from detectors import DETECTORS, get_detector
from detectors.family_names import FamilyNameDetector, clean, codes, is_capitalized


# --- Fixtures -----------------------------------------------------------------

@pytest.fixture
def roster_path(tmp_path):
    """Fake-name roster: 'Marta' (child) + alias 'bemarta' (front-distortion)."""
    path = tmp_path / "name_roster.json"
    path.write_text(json.dumps({
        "people": [
            {"id": "child", "canonical": "Marta", "role": "child"},
            {"id": "parent", "canonical": "Kiran", "role": "parent"},
        ],
        "aliases": [
            {"token": "bemarta", "person_id": "child"},
        ],
    }))
    return path


def make_session(tmp_path, words):
    """Session dir with a one-segment transcript built from word strings."""
    session_dir = tmp_path / "20260101-120000"
    session_dir.mkdir(exist_ok=True)
    (session_dir / "transcript-rich.json").write_text(json.dumps({
        "segments": [{
            "id": 0,
            "start": 0.0,
            "end": 10.0,
            "text": " ".join(w.strip() for w in words),
            "words": [
                {"word": f" {w}", "start": float(i), "end": float(i) + 0.4}
                for i, w in enumerate(words)
            ],
        }],
    }))
    return session_dir


def run_on(tmp_path, roster_path, words):
    det = FamilyNameDetector(roster_path=roster_path)
    return det.run(make_session(tmp_path, words))


# --- Helpers ------------------------------------------------------------------

def test_clean_strips_punctuation_and_possessive():
    assert clean(" Marta's,") == "marta"
    assert clean("Marta’s") == "marta"
    assert clean("...") == ""


def test_is_capitalized():
    assert is_capitalized("Marta")
    assert not is_capitalized("marta")
    assert is_capitalized('"Marta')  # leading punctuation skipped


# --- Phonetic layer -----------------------------------------------------------

def test_phonetic_primary_match(tmp_path, roster_path):
    result = run_on(tmp_path, roster_path, ["Marda", "went", "home"])
    assert len(result["flags"]) == 1
    flag = result["flags"][0]
    assert flag["match_type"] == "phonetic"
    assert flag["matched_canonicals"] == ["Marta"]
    assert flag["token"] == "Marda"


def test_secondary_code_th_distortion_match(tmp_path, roster_path):
    # "Martha" encodes to primary MR0 (theta) + secondary MRT — only the
    # SECONDARY code reaches canonical "Marta" (MRT). The load-bearing
    # both-codes behavior: a single-code matcher would miss this.
    assert codes("martha") == {"MR0", "MRT"}
    assert codes("marta") == {"MRT"}
    result = run_on(tmp_path, roster_path, ["Martha", "smiled"])
    assert len(result["flags"]) == 1
    assert result["flags"][0]["matched_canonicals"] == ["Marta"]


def test_canonical_spelling_not_flagged(tmp_path, roster_path):
    result = run_on(tmp_path, roster_path, ["Marta", "and", "Kiran"])
    assert result["flags"] == []
    assert result["n_word_tokens"] == 3


def test_possessive_variant_flagged(tmp_path, roster_path):
    result = run_on(tmp_path, roster_path, ["Marda's,", "toy"])
    assert len(result["flags"]) == 1
    assert result["flags"][0]["cleaned"] == "marda"


# --- Capitalization gate ------------------------------------------------------

def test_cap_gate_drops_lowercase_homophone(tmp_path, roster_path):
    lower = run_on(tmp_path, roster_path, ["marda", "fell"])
    upper = run_on(tmp_path, roster_path, ["Marda", "fell"])
    assert lower["flags"] == []
    assert len(upper["flags"]) == 1


# --- Alias layer --------------------------------------------------------------

def test_alias_exact_match(tmp_path, roster_path):
    result = run_on(tmp_path, roster_path, ["Bemarta's", "story"])
    assert len(result["flags"]) == 1
    flag = result["flags"][0]
    assert flag["match_type"] == "alias"
    assert flag["matched_person_ids"] == ["child"]


def test_redundant_alias_rejected(tmp_path):
    # "marda" reaches canonical "Marta" phonetically — alias is redundant
    path = tmp_path / "roster.json"
    path.write_text(json.dumps({
        "people": [{"id": "child", "canonical": "Marta", "role": "child"}],
        "aliases": [{"token": "marda", "person_id": "child"}],
    }))
    det = FamilyNameDetector(roster_path=path)
    with pytest.raises(ValueError, match="Redundant alias"):
        det.run(make_session(tmp_path, ["hello"]))


def test_alias_unknown_person_rejected(tmp_path):
    path = tmp_path / "roster.json"
    path.write_text(json.dumps({
        "people": [{"id": "child", "canonical": "Marta", "role": "child"}],
        "aliases": [{"token": "bemarta", "person_id": "ghost"}],
    }))
    det = FamilyNameDetector(roster_path=path)
    with pytest.raises(ValueError, match="unknown person_id"):
        det.run(make_session(tmp_path, ["hello"]))


# --- Fail-loud cases ----------------------------------------------------------

def test_missing_roster_fails_loud(tmp_path):
    det = FamilyNameDetector(roster_path=tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError, match="name_roster.example.json"):
        det.run(make_session(tmp_path, ["hello"]))


def test_config_fingerprint_missing_roster_fails_loud(tmp_path):
    det = FamilyNameDetector(roster_path=tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError, match="name_roster.example.json"):
        det.config_fingerprint()


def test_roster_edit_picked_up_between_runs(tmp_path, roster_path):
    """The roster cache is keyed by content hash — an edit while the detector
    instance lives (long-running API process) takes effect on the next run."""
    det = FamilyNameDetector(roster_path=roster_path)
    # "Soran" — a phonetic variant of "Soren", who isn't in the roster yet
    session = make_session(tmp_path, ["Soran", "arrived"])
    assert det.run(session)["flags"] == []

    roster = json.loads(roster_path.read_text())
    roster["people"].append({"id": "uncle", "canonical": "Soren", "role": "family"})
    roster_path.write_text(json.dumps(roster))

    flags = det.run(session)["flags"]
    assert len(flags) == 1                   # same instance, new roster
    assert flags[0]["matched_canonicals"] == ["Soren"]


def test_missing_transcript_fails_loud(tmp_path, roster_path):
    session_dir = tmp_path / "20260101-130000"
    session_dir.mkdir()
    det = FamilyNameDetector(roster_path=roster_path)
    with pytest.raises(FileNotFoundError, match="transcript-rich.json"):
        det.run(session_dir)


def test_gap_segments_without_words_skipped(tmp_path, roster_path):
    session_dir = tmp_path / "20260101-140000"
    session_dir.mkdir()
    (session_dir / "transcript-rich.json").write_text(json.dumps({
        "segments": [{"id": "gap_5.000", "start": 5.0, "end": 6.0,
                      "text": "[unintelligible]"}],
    }))
    det = FamilyNameDetector(roster_path=roster_path)
    result = det.run(session_dir)
    assert result == {"n_word_tokens": 0, "flags": []}


# --- Registry -----------------------------------------------------------------

def test_registry_contains_m9a():
    assert "m9a-family-names" in [d.id for d in DETECTORS]
    assert get_detector("m9a-family-names").failure_mode == "M9a"


def test_unknown_detector_rejected():
    with pytest.raises(ValueError, match="Unknown detector"):
        get_detector("nope")
