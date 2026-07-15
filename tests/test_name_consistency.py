"""Unit tests for the M9b name-consistency detector (fake names only)."""

import json

import pytest

from detectors import phonetics
from detectors.name_consistency import NameConsistencyDetector


def make_session(tmp_path, words, extra_segments=None):
    """One-segment transcript built from word strings, plus optional extra segments."""
    session_dir = tmp_path / "20260101-120000"
    session_dir.mkdir(exist_ok=True)
    segments = [{
        "id": 0,
        "start": 0.0,
        "end": float(len(words)),
        "text": " ".join(w.strip() for w in words),
        "words": [
            {"word": f" {w}", "start": float(i), "end": float(i) + 0.4}
            for i, w in enumerate(words)
        ],
    }]
    if extra_segments:
        segments.extend(extra_segments)
    (session_dir / "transcript-rich.json").write_text(json.dumps({"segments": segments}))
    return session_dir


def run_on(tmp_path, words, extra_segments=None, wordlist_path=None, roster_path=None):
    # wordlist_path=None → stoplist-only, so the fake names below aren't filtered as
    # "common words". roster_path defaults to a nonexistent file → no roster exclusion,
    # so tests never depend on the real gitignored data/name_roster.json.
    det = NameConsistencyDetector(wordlist_path=wordlist_path,
                                  roster_path=roster_path or (tmp_path / "no-roster.json"))
    return det.run(make_session(tmp_path, words, extra_segments))


# --- Core behavior -------------------------------------------------------------

def test_inconsistent_cluster_all_occurrences_flagged(tmp_path):
    # "Zerk" and "Zerg" are one phonetic cluster spelled two ways → both flagged.
    result = run_on(tmp_path, ["Zerk", "met", "Zerg", "today"])
    assert len(result["flags"]) == 2
    tokens = sorted(f["token"] for f in result["flags"])
    assert tokens == ["Zerg", "Zerk"]
    flag = result["flags"][0]
    assert flag["cluster_spellings"] == ["Zerg", "Zerk"]
    assert flag["n_cluster_occurrences"] == 2


def test_consistent_name_not_flagged(tmp_path):
    # Same name spelled the same way every time → inconsistency signal absent.
    result = run_on(tmp_path, ["Bibi", "and", "Bibi", "played"])
    assert result["flags"] == []


def test_three_spellings_cluster(tmp_path):
    result = run_on(tmp_path, ["Jarko", "Jarco", "Jarkko", "raced"])
    assert len(result["flags"]) == 3
    # cluster_spellings is sorted alphabetically; all three share a phonetic code
    assert result["flags"][0]["cluster_spellings"] == ["Jarco", "Jarkko", "Jarko"]


def test_majority_spelling_not_flagged_only_deviation(tmp_path):
    # Zerk ×3 with one stray Zerg: only the stray is flagged. The 3 Zerks are this
    # recording's majority (intended) spelling and are left alone — so a name spelled
    # right dozens of times with one typo no longer floods the board.
    result = run_on(tmp_path, ["Zerk", "Zerk", "Zerk", "Zerg"])
    assert len(result["flags"]) == 1
    f = result["flags"][0]
    assert f["token"] == "Zerg"
    assert f["majority_spelling"] == "Zerk"   # the in-recording reference (not a global canonical)
    assert f["majority_count"] == 3


def test_no_clear_majority_flags_all(tmp_path):
    # Evenly split (Zerk ×2 / Zerg ×2): no winner, the name is unsettled, so every
    # occurrence is flagged for review, with no majority reference.
    result = run_on(tmp_path, ["Zerk", "Zerk", "Zerg", "Zerg"])
    assert len(result["flags"]) == 4
    assert all(f["majority_spelling"] is None for f in result["flags"])


def test_cap_gate_drops_lowercase(tmp_path):
    # A lowercase token sharing the cluster's sound is not a name candidate, so it
    # neither gets flagged nor counts toward the cluster.
    result = run_on(tmp_path, ["Zerk", "zerg", "ran"])
    assert result["flags"] == []  # only one capitalized spelling → consistent


def test_single_capitalized_name_not_flagged(tmp_path):
    result = run_on(tmp_path, ["Marco", "went", "home"])
    assert result["flags"] == []


def test_gap_segment_without_words_skipped(tmp_path):
    gap = {"id": "gap_99.0", "start": 99.0, "end": 100.0, "text": "[unintelligible]"}
    result = run_on(tmp_path, ["Zerk", "Zerg"], extra_segments=[gap])
    assert len(result["flags"]) == 2  # gap contributed nothing, no crash


def test_two_distinct_names_not_merged(tmp_path):
    # Phonetically distinct names must not cluster, even when each is consistent.
    result = run_on(tmp_path, ["Marco", "and", "Tilly", "and", "Marco"])
    assert result["flags"] == []


def test_n_word_tokens_counts_all(tmp_path):
    result = run_on(tmp_path, ["Zerk", "Zerg", "went"])
    assert result["n_word_tokens"] == 3


def test_flag_anchor_fields_present(tmp_path):
    # Shares the M9a anchor fields so the API segment-join + audio playback work.
    flag = run_on(tmp_path, ["Zerk", "Zerg"])["flags"][0]
    for k in ("segment_id", "word_index", "start", "end", "token", "cleaned",
              "dm_codes", "cluster_id", "cluster_spellings", "n_cluster_occurrences"):
        assert k in flag


# --- Precision layer -----------------------------------------------------------

def test_short_names_dropped_by_min_length(tmp_path):
    # A 3-char inconsistent pair would cluster, but min-length removes it before
    # clustering — short tokens are function words / interjections, not names.
    result = run_on(tmp_path, ["Cat", "Kat", "ran"])
    assert result["flags"] == []


def test_all_common_cluster_dropped(tmp_path):
    # Both spellings are ordinary words → not an improvised name → dropped.
    wordlist = tmp_path / "words.txt"
    wordlist.write_text("zerk\nzerg\n")
    result = run_on(tmp_path, ["Zerk", "Zerg"], wordlist_path=wordlist)
    assert result["flags"] == []


def test_cluster_kept_when_one_spelling_is_not_a_word(tmp_path):
    # One variant is a dictionary word, the other (the misspelling) is not →
    # the cluster survives. This is the Pataki/Bacchus case.
    wordlist = tmp_path / "words.txt"
    wordlist.write_text("zerg\n")          # only one of the two is "common"
    result = run_on(tmp_path, ["Zerk", "Zerg"], wordlist_path=wordlist)
    assert len(result["flags"]) == 2


# --- Contract ------------------------------------------------------------------

def test_config_fingerprint_tracks_roster(tmp_path):
    # m9b now excludes roster names, so the roster is an input: absent → None
    # (roster-agnostic), present → its content hash (a roster edit re-runs m9b).
    assert NameConsistencyDetector(roster_path=tmp_path / "absent.json").config_fingerprint() is None
    roster = tmp_path / "roster.json"
    roster.write_text('{"people": [{"id": "c", "canonical": "Marta"}], "aliases": []}')
    fp = NameConsistencyDetector(roster_path=roster).config_fingerprint()
    assert fp is not None and fp.startswith("sha256:")


def test_missing_transcript_fails_loud(tmp_path):
    session_dir = tmp_path / "20260101-130000"
    session_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="transcript-rich.json"):
        NameConsistencyDetector().run(session_dir)


def test_registered_in_registry():
    from detectors import DETECTORS, get_detector
    assert "m9b-name-consistency" in [d.id for d in DETECTORS]
    assert get_detector("m9b-name-consistency").failure_mode == "M9b"


# --- Offline judge layer -------------------------------------------------------

def test_judge_recovers_all_common_cluster(tmp_path):
    # Both spellings are "common", so code-only drops the cluster — but a judge
    # that recognizes it as a name recovers it (the Bibi/Bacchus case).
    wordlist = tmp_path / "words.txt"
    wordlist.write_text("zerk\nzerg\n")
    det = NameConsistencyDetector(wordlist_path=wordlist)
    session = make_session(tmp_path, ["Zerk", "Zerg", "ran"])
    assert det.run(session)["flags"] == []          # no judge → dropped

    seen = {}
    def judge(candidates):
        seen["c"] = candidates
        return {c["cluster_id"] for c in candidates}  # keep everything shown
    flags = det.run(session, judge=judge)["flags"]
    assert len(flags) == 2
    assert len(seen["c"]) == 1                        # the one all-common cluster
    assert set(seen["c"][0]["spellings"]) == {"Zerk", "Zerg"}


def test_judge_drop_keeps_cluster_out(tmp_path):
    wordlist = tmp_path / "words.txt"
    wordlist.write_text("zerk\nzerg\n")
    det = NameConsistencyDetector(wordlist_path=wordlist)
    session = make_session(tmp_path, ["Zerk", "Zerg"])
    assert det.run(session, judge=lambda c: set())["flags"] == []  # judge drops → empty


def test_judge_not_called_for_clear_names(tmp_path):
    # A cluster with a non-dictionary spelling is auto-kept; the judge never sees it.
    det = NameConsistencyDetector(wordlist_path=None)  # stoplist-only
    session = make_session(tmp_path, ["Zerk", "Zerg"])
    calls = []
    det.run(session, judge=lambda c: (calls.append(c), set())[1])
    assert calls == [[]] or calls == []  # judge given no candidates (or not called)


# --- Roster exclusion (defer family names to m9a) ------------------------------

def test_roster_names_excluded_deferred_to_m9a(tmp_path):
    # A family-roster name spelled inconsistently is m9a's job — m9b must not
    # double-count it. With the roster loaded, Marta/Martah (canonical + a "th"
    # misspelling that shares its metaphone code) is excluded, while a non-roster
    # improvised cluster (Zerk/Zerg) still flags.
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps({
        "people": [{"id": "child", "canonical": "Marta", "role": "child"}],
        "aliases": [],
    }))
    det = NameConsistencyDetector(wordlist_path=None, roster_path=roster)
    session = make_session(tmp_path, ["Marta", "Martah", "Zerk", "Zerg"])
    tokens = sorted(f["token"] for f in det.run(session)["flags"])
    assert tokens == ["Zerg", "Zerk"]  # roster names dropped, improvised kept


def test_no_roster_file_is_roster_agnostic(tmp_path):
    # Graceful degradation: with no roster file, m9b keeps its original behavior
    # and flags the inconsistent cluster (Marta/Martah) like any other.
    det = NameConsistencyDetector(wordlist_path=None, roster_path=tmp_path / "absent.json")
    tokens = sorted(f["token"] for f in det.run(make_session(tmp_path, ["Marta", "Martah"]))["flags"])
    assert tokens == ["Marta", "Martah"]


# --- Contraction handling (precision) ------------------------------------------

def test_contraction_forms_treated_as_common(tmp_path):
    # clean() strips the apostrophe, so "You're"->"youre", "We've"->"weve" — forms
    # the dictionary lacks. Treating them as common lets a contraction-only cluster
    # get dropped instead of flagged as a name (the Cruel Baby We're/Where/You're case).
    det = NameConsistencyDetector(wordlist_path=None)  # stoplist + contractions only
    assert det._is_common("youre") and det._is_common("weve") and det._is_common("dont")


def test_name_possessive_is_not_treated_as_common(tmp_path):
    # The possessive of a real name must stay a candidate: "Zerk's" -> "zerks" is not
    # common (its stem "zerk" is not a word), unlike a contraction. This keeps real
    # name possessives (e.g. the child's "Martie's") in the clustering.
    det = NameConsistencyDetector(wordlist_path=None)
    assert not det._is_common("zerks")


# --- Phonetics module move did not break M9a's helpers -------------------------

def test_phonetics_helpers_intact():
    # The shared module must keep the validated behavior the M9a probe relied on:
    # "th" names carry both a theta primary and a plain-t secondary code.
    assert phonetics.codes("martha") == {"MR0", "MRT"}
    assert phonetics.codes("marta") == {"MRT"}
    assert phonetics.clean(" Marta's,") == "marta"
    assert phonetics.is_capitalized("Marta") and not phonetics.is_capitalized("marta")
