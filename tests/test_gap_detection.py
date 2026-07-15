"""Tests for detect_unintelligible_gaps and its helpers in speaker.py."""

import copy

import pytest

from speaker import detect_unintelligible_gaps, enrich_with_diarization, _dominant_speaker, _word_coverage


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

def _word(text, start, end, speaker=None):
    """Build a word dict, optionally with a _speaker label."""
    w = {"word": text, "start": start, "end": end, "probability": 0.9}
    if speaker is not None:
        w["_speaker"] = {"label": speaker, "coverage": 1.0}
    return w


def _segment(words, start=None, end=None):
    """Build a transcript segment from a list of word dicts.

    Stamps segment-level _speaker to simulate what enrich_with_diarization
    produces before detect_unintelligible_gaps runs.
    """
    s = words[0]["start"] if words else 0.0
    e = words[-1]["end"] if words else 0.0
    seg = {
        "start": start if start is not None else s,
        "end": end if end is not None else e,
        "text": " ".join(w["word"].strip() for w in words),
        "words": words,
    }
    dominant = _dominant_speaker(seg)
    if dominant is not None:
        seg["_speaker"] = {"label": dominant, "source": "dominant"}
    return seg


def _transcript(*segments):
    """Wrap segments in a minimal transcript dict."""
    return {"text": "", "segments": list(segments)}


def _diarization(*segs):
    """Build a diarization dict from (start, end, speaker) tuples."""
    return {
        "_generator_version": "test",
        "segments": [{"start": s, "end": e, "speaker": sp} for s, e, sp in segs],
    }


# ---------------------------------------------------------------------------
# _dominant_speaker
# ---------------------------------------------------------------------------

def test_dominant_speaker_returns_majority_label():
    seg = _segment([
        _word("a", 0.0, 0.5, speaker="SPEAKER_00"),
        _word("b", 0.5, 1.0, speaker="SPEAKER_01"),
        _word("c", 1.0, 1.5, speaker="SPEAKER_00"),
    ])
    assert _dominant_speaker(seg) == "SPEAKER_00"


def test_dominant_speaker_no_words_returns_none():
    seg = {"start": 0.0, "end": 1.0, "text": "", "words": []}
    assert _dominant_speaker(seg) is None


def test_dominant_speaker_no_speaker_labels_returns_none():
    seg = _segment([
        _word("a", 0.0, 0.5),
        _word("b", 0.5, 1.0),
    ])
    assert _dominant_speaker(seg) is None


# ---------------------------------------------------------------------------
# _word_coverage
# ---------------------------------------------------------------------------

def test_word_coverage_full():
    words = [_word("x", 5.0, 7.0)]
    assert _word_coverage(5.0, 7.0, words) == pytest.approx(1.0)


def test_word_coverage_half():
    words = [_word("x", 5.0, 6.0)]
    assert _word_coverage(5.0, 7.0, words) == pytest.approx(0.5)


def test_word_coverage_none():
    words = [_word("x", 10.0, 12.0)]
    assert _word_coverage(5.0, 7.0, words) == pytest.approx(0.0)


def test_word_coverage_zero_duration_segment():
    words = [_word("x", 5.0, 5.5)]
    assert _word_coverage(5.0, 5.0, words) == 0.0


# ---------------------------------------------------------------------------
# detect_unintelligible_gaps — core filter logic
# ---------------------------------------------------------------------------

def test_gap_detected_when_speaker_differs_from_both_neighbors():
    """A low-coverage diarization segment whose speaker differs from both
    neighboring transcript segments should produce an [unintelligible] segment."""
    seg_a = _segment([_word("hello", 0.0, 1.0, "SPEAKER_00"),
                      _word("there", 1.0, 2.0, "SPEAKER_00")])
    seg_b = _segment([_word("yes", 8.0, 9.0, "SPEAKER_00"),
                      _word("indeed", 9.0, 10.0, "SPEAKER_00")])

    transcript = _transcript(seg_a, seg_b)
    # Gap at [3.0, 5.0] with SPEAKER_01 — no transcript words cover it
    diarization = _diarization(
        (0.0, 2.0, "SPEAKER_00"),
        (3.0, 5.0, "SPEAKER_01"),
        (8.0, 10.0, "SPEAKER_00"),
    )

    result, entry = detect_unintelligible_gaps(transcript, diarization)

    unintelligible = [s for s in result["segments"] if s.get("_source") == "diarization_gap"]
    assert len(unintelligible) == 1
    gap = unintelligible[0]
    assert gap["start"] == 3.0
    assert gap["end"] == 5.0
    assert gap["text"] == "[unintelligible]"
    assert gap["words"] == []
    assert gap["_speaker"] == {"label": "SPEAKER_01", "coverage": 1.0}
    assert entry["gaps_found"] == 1


def test_gap_not_detected_when_speaker_matches_preceding_neighbor():
    """Gap speaker matching the preceding neighbor = monologue pause, not a turn."""
    seg_a = _segment([_word("hello", 0.0, 2.0, "SPEAKER_01")])
    seg_b = _segment([_word("yes", 8.0, 10.0, "SPEAKER_00")])

    transcript = _transcript(seg_a, seg_b)
    diarization = _diarization(
        (0.0, 2.0, "SPEAKER_01"),
        (3.0, 5.0, "SPEAKER_01"),   # matches preceding
        (8.0, 10.0, "SPEAKER_00"),
    )

    result, entry = detect_unintelligible_gaps(transcript, diarization)

    unintelligible = [s for s in result["segments"] if s.get("_source") == "diarization_gap"]
    assert len(unintelligible) == 0
    assert entry["gaps_found"] == 0


def test_gap_not_detected_when_speaker_matches_following_neighbor():
    """Gap speaker matching the following neighbor = lead-in, not a turn."""
    seg_a = _segment([_word("hello", 0.0, 2.0, "SPEAKER_00")])
    seg_b = _segment([_word("yes", 8.0, 10.0, "SPEAKER_01")])

    transcript = _transcript(seg_a, seg_b)
    diarization = _diarization(
        (0.0, 2.0, "SPEAKER_00"),
        (3.0, 5.0, "SPEAKER_01"),   # matches following
        (8.0, 10.0, "SPEAKER_01"),
    )

    result, entry = detect_unintelligible_gaps(transcript, diarization)

    unintelligible = [s for s in result["segments"] if s.get("_source") == "diarization_gap"]
    assert len(unintelligible) == 0
    assert entry["gaps_found"] == 0


def test_gap_not_detected_when_word_coverage_at_threshold():
    """Word coverage >= 30% should NOT trigger gap injection.

    Uses a 50% coverage value (1.0s word / 2.0s gap) to avoid floating-point
    edge cases near the 0.3 boundary.
    """
    seg_a = _segment([_word("hello", 0.0, 2.0, "SPEAKER_00")])
    seg_b = _segment([_word("yes", 8.0, 10.0, "SPEAKER_00")])
    # Word covers [4.0, 5.0] = 1.0s of a 2.0s gap = 50%
    seg_gap = _segment([_word("maybe", 4.0, 5.0, "SPEAKER_01")],
                       start=4.0, end=5.0)

    transcript = _transcript(seg_a, seg_gap, seg_b)
    diarization = _diarization(
        (0.0, 2.0, "SPEAKER_00"),
        (4.0, 6.0, "SPEAKER_01"),
        (8.0, 10.0, "SPEAKER_00"),
    )

    result, entry = detect_unintelligible_gaps(transcript, diarization)

    unintelligible = [s for s in result["segments"] if s.get("_source") == "diarization_gap"]
    assert len(unintelligible) == 0
    assert entry["gaps_found"] == 0


def test_gap_detected_when_word_coverage_below_threshold():
    """Coverage clearly below 30% (25%) should trigger gap injection."""
    seg_a = _segment([_word("hello", 0.0, 2.0, "SPEAKER_00")])
    seg_b = _segment([_word("yes", 8.0, 10.0, "SPEAKER_00")])
    # Word covers [4.0, 5.0] = 1.0s of a 4.0s gap = 25%
    seg_gap = _segment([_word("maybe", 4.0, 5.0, "SPEAKER_01")],
                       start=4.0, end=5.0)

    transcript = _transcript(seg_a, seg_gap, seg_b)
    diarization = _diarization(
        (0.0, 2.0, "SPEAKER_00"),
        (4.0, 8.0, "SPEAKER_01"),   # 4.0s gap; word covers 1.0s = 25%
        (10.0, 12.0, "SPEAKER_00"),
    )

    # Need seg_b to start >= diar_end (8.0) for following neighbor to be found
    seg_b2 = _segment([_word("yes", 10.0, 12.0, "SPEAKER_00")])
    transcript2 = _transcript(seg_a, seg_gap, seg_b2)

    result, entry = detect_unintelligible_gaps(transcript2, diarization)

    unintelligible = [s for s in result["segments"] if s.get("_source") == "diarization_gap"]
    assert len(unintelligible) == 1
    assert entry["gaps_found"] == 1


# ---------------------------------------------------------------------------
# Sorting and timestamp preservation
# ---------------------------------------------------------------------------

def test_injected_segments_sorted_by_start_time():
    """Multiple injected gaps should be interleaved with existing segments in
    chronological order."""
    seg_a = _segment([_word("a", 0.0, 1.0, "SPEAKER_00")])
    seg_b = _segment([_word("b", 6.0, 7.0, "SPEAKER_00")])
    seg_c = _segment([_word("c", 14.0, 15.0, "SPEAKER_00")])

    transcript = _transcript(seg_a, seg_b, seg_c)
    diarization = _diarization(
        (0.0, 1.0, "SPEAKER_00"),
        (2.0, 4.0, "SPEAKER_01"),    # gap between seg_a and seg_b
        (6.0, 7.0, "SPEAKER_00"),
        (9.0, 11.0, "SPEAKER_01"),   # gap between seg_b and seg_c
        (14.0, 15.0, "SPEAKER_00"),
    )

    result, entry = detect_unintelligible_gaps(transcript, diarization)

    starts = [s["start"] for s in result["segments"]]
    assert starts == sorted(starts), f"Segments not sorted: {starts}"
    assert entry["gaps_found"] == 2


def test_injected_segment_timestamps_not_clipped():
    """The injected segment's start/end must exactly match the diarization segment."""
    seg_a = _segment([_word("a", 0.0, 1.0, "SPEAKER_00")])
    seg_b = _segment([_word("b", 8.0, 9.0, "SPEAKER_00")])

    transcript = _transcript(seg_a, seg_b)
    diarization = _diarization(
        (0.0, 1.0, "SPEAKER_00"),
        (3.14, 5.92, "SPEAKER_01"),
        (8.0, 9.0, "SPEAKER_00"),
    )

    result, _ = detect_unintelligible_gaps(transcript, diarization)

    gaps = [s for s in result["segments"] if s.get("_source") == "diarization_gap"]
    assert len(gaps) == 1
    assert gaps[0]["start"] == 3.14
    assert gaps[0]["end"] == 5.92


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_diarization_returns_unchanged_transcript():
    seg = _segment([_word("hello", 0.0, 1.0, "SPEAKER_00")])
    transcript = _transcript(seg)
    diarization = _diarization()

    result, entry = detect_unintelligible_gaps(transcript, diarization)

    assert len(result["segments"]) == 1
    assert entry["gaps_found"] == 0
    assert entry["status"] == "success"


def test_empty_transcript_returns_unchanged():
    transcript = _transcript()
    diarization = _diarization((0.0, 5.0, "SPEAKER_01"))

    result, entry = detect_unintelligible_gaps(transcript, diarization)

    assert result["segments"] == []
    assert entry["gaps_found"] == 0


def test_empty_both_returns_gracefully():
    result, entry = detect_unintelligible_gaps({"segments": []}, {"segments": []})
    assert result["segments"] == []
    assert entry["gaps_found"] == 0


def test_missing_segments_key_in_both():
    """Totally bare dicts (no 'segments' key) should not crash."""
    result, entry = detect_unintelligible_gaps({}, {})
    assert entry["status"] == "success"
    assert entry["gaps_found"] == 0


def test_gap_at_start_of_recording_not_injected():
    """A gap with no preceding transcript segment should be skipped."""
    seg = _segment([_word("b", 8.0, 9.0, "SPEAKER_00")])
    transcript = _transcript(seg)
    diarization = _diarization(
        (0.0, 2.0, "SPEAKER_01"),   # no preceding segment → skip
        (8.0, 9.0, "SPEAKER_00"),
    )

    result, entry = detect_unintelligible_gaps(transcript, diarization)

    unintelligible = [s for s in result["segments"] if s.get("_source") == "diarization_gap"]
    assert len(unintelligible) == 0


def test_gap_at_end_of_recording_not_injected():
    """A gap with no following transcript segment should be skipped."""
    seg = _segment([_word("a", 0.0, 1.0, "SPEAKER_00")])
    transcript = _transcript(seg)
    diarization = _diarization(
        (0.0, 1.0, "SPEAKER_00"),
        (3.0, 5.0, "SPEAKER_01"),   # no following segment → skip
    )

    result, entry = detect_unintelligible_gaps(transcript, diarization)

    unintelligible = [s for s in result["segments"] if s.get("_source") == "diarization_gap"]
    assert len(unintelligible) == 0


# ---------------------------------------------------------------------------
# Processing entry
# ---------------------------------------------------------------------------

def test_processing_entry_structure():
    result, entry = detect_unintelligible_gaps({"segments": []}, {"segments": []})

    assert entry["stage"] == "gap_detection"
    assert entry["status"] == "success"
    assert "gaps_found" in entry
    assert "timestamp" in entry
    assert isinstance(entry["gaps_found"], int)


# ---------------------------------------------------------------------------
# Deep copy — input not mutated
# ---------------------------------------------------------------------------

def test_input_transcript_not_mutated():
    seg = _segment([_word("hello", 0.0, 1.0, "SPEAKER_00")])
    seg_b = _segment([_word("yes", 8.0, 9.0, "SPEAKER_00")])
    transcript = _transcript(seg, seg_b)
    original_segment_count = len(transcript["segments"])

    diarization = _diarization(
        (0.0, 1.0, "SPEAKER_00"),
        (3.0, 5.0, "SPEAKER_01"),
        (8.0, 9.0, "SPEAKER_00"),
    )

    detect_unintelligible_gaps(transcript, diarization)

    # Original transcript must be unchanged
    assert len(transcript["segments"]) == original_segment_count
    assert not any(s.get("_source") == "diarization_gap" for s in transcript["segments"])


def test_injected_gap_has_stable_id():
    """Each injected gap must have an id derived from its start time."""
    seg_a = _segment([_word("hello", 0.0, 1.0, "SPEAKER_00")])
    seg_b = _segment([_word("yes", 8.0, 9.0, "SPEAKER_00")])
    transcript = _transcript(seg_a, seg_b)
    diarization = _diarization(
        (0.0, 1.0, "SPEAKER_00"),
        (3.14, 5.92, "SPEAKER_01"),
        (8.0, 9.0, "SPEAKER_00"),
    )

    result, _ = detect_unintelligible_gaps(transcript, diarization)

    gaps = [s for s in result["segments"] if s.get("_source") == "diarization_gap"]
    assert len(gaps) == 1
    assert gaps[0]["id"] == "gap_3.140"


# ---------------------------------------------------------------------------
# enrich_with_diarization — segment-level _speaker stamping
# ---------------------------------------------------------------------------

def test_enrich_stamps_segment_level_speaker():
    """enrich_with_diarization should stamp _speaker on the segment itself."""
    seg = {
        "start": 0.0, "end": 2.0, "text": "hello there",
        "words": [
            {"word": "hello", "start": 0.0, "end": 1.0},
            {"word": "there", "start": 1.0, "end": 2.0},
        ]
    }
    transcript = {"segments": [seg]}
    diarization = _diarization((0.0, 2.0, "SPEAKER_00"))

    result, _ = enrich_with_diarization(transcript, diarization)

    segment = result["segments"][0]
    assert "_speaker" in segment
    assert segment["_speaker"]["label"] == "SPEAKER_00"
    assert segment["_speaker"]["source"] == "dominant"


def test_enrich_segment_speaker_majority_vote():
    """Segment _speaker should reflect the majority speaker across words."""
    seg = {
        "start": 0.0, "end": 3.0, "text": "a b c",
        "words": [
            {"word": "a", "start": 0.0, "end": 1.0},
            {"word": "b", "start": 1.0, "end": 2.0},
            {"word": "c", "start": 2.0, "end": 3.0},
        ]
    }
    transcript = {"segments": [seg]}
    # SPEAKER_00 covers words a and c; SPEAKER_01 covers only b
    diarization = _diarization(
        (0.0, 1.0, "SPEAKER_00"),
        (1.0, 2.0, "SPEAKER_01"),
        (2.0, 3.0, "SPEAKER_00"),
    )

    result, _ = enrich_with_diarization(transcript, diarization)

    assert result["segments"][0]["_speaker"]["label"] == "SPEAKER_00"


def test_enrich_segment_no_speaker_when_no_coverage():
    """Segments with no diarization coverage should not get _speaker."""
    seg = {
        "start": 10.0, "end": 12.0, "text": "something",
        "words": [{"word": "something", "start": 10.0, "end": 12.0}]
    }
    transcript = {"segments": [seg]}
    diarization = _diarization((0.0, 2.0, "SPEAKER_00"))

    result, _ = enrich_with_diarization(transcript, diarization)

    assert "_speaker" not in result["segments"][0]


def test_enrich_does_not_mutate_input():
    """enrich_with_diarization should not mutate the input transcript."""
    seg = {
        "start": 0.0, "end": 1.0, "text": "hi",
        "words": [{"word": "hi", "start": 0.0, "end": 1.0}]
    }
    transcript = {"segments": [seg]}
    diarization = _diarization((0.0, 1.0, "SPEAKER_00"))

    enrich_with_diarization(transcript, diarization)

    assert "_speaker" not in transcript["segments"][0]
    assert "_speaker" not in transcript["segments"][0]["words"][0]
