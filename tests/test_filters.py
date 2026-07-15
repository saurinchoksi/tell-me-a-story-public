"""Tests for src/filters.py — hallucination detection predicates."""

from filters import silence_gap, near_zero_probability, find_duplicate_segments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word(speaker_label="SPEAKER_00", coverage=1.0, probability=0.9,
          include_speaker=True):
    """Build a minimal word dict with optional _speaker block."""
    w = {"word": "hello", "start": 0.0, "end": 0.5, "probability": probability}
    if include_speaker:
        w["_speaker"] = {"label": speaker_label, "coverage": coverage}
    return w


def _segment(seg_id, text):
    """Build a minimal segment dict."""
    return {"id": seg_id, "text": text}


# ---------------------------------------------------------------------------
# silence_gap
# ---------------------------------------------------------------------------

def test_silence_gap_true_when_no_speaker_and_zero_coverage():
    word = _word(speaker_label=None, coverage=0.0)
    assert silence_gap(word) is True


def test_silence_gap_false_when_label_present():
    word = _word(speaker_label="SPEAKER_00", coverage=0.0)
    assert silence_gap(word) is False


def test_silence_gap_false_when_coverage_nonzero():
    word = _word(speaker_label=None, coverage=0.5)
    assert silence_gap(word) is False


def test_silence_gap_false_when_speaker_key_missing():
    word = _word(include_speaker=False)
    # Missing _speaker → defaults to {} → label None, but coverage defaults 1.0
    assert silence_gap(word) is False


def test_silence_gap_false_when_valid_speaker():
    word = _word(speaker_label="SPEAKER_01", coverage=0.95)
    assert silence_gap(word) is False


# ---------------------------------------------------------------------------
# near_zero_probability
# ---------------------------------------------------------------------------

def test_near_zero_true_below_threshold():
    word = _word(probability=0.001)
    assert near_zero_probability(word) is True


def test_near_zero_false_above_threshold():
    word = _word(probability=0.5)
    assert near_zero_probability(word) is False


def test_near_zero_false_at_exact_threshold():
    # Default threshold is 0.01; exactly at threshold should be False (strict <)
    word = _word(probability=0.01)
    assert near_zero_probability(word) is False


def test_near_zero_false_when_probability_missing():
    word = {"word": "hello", "start": 0.0, "end": 0.5}
    assert near_zero_probability(word) is False


def test_near_zero_custom_threshold():
    word = _word(probability=0.05)
    assert near_zero_probability(word, threshold=0.1) is True


# ---------------------------------------------------------------------------
# find_duplicate_segments
# ---------------------------------------------------------------------------

def test_duplicates_empty_list():
    assert find_duplicate_segments([]) == set()


def test_duplicates_none_found():
    segments = [_segment(0, "hello"), _segment(1, "world"), _segment(2, "goodbye")]
    assert find_duplicate_segments(segments) == set()


def test_duplicates_pair_returns_later_id():
    segments = [_segment(0, "hello"), _segment(1, "hello")]
    assert find_duplicate_segments(segments) == {1}


def test_duplicates_triple_returns_second_and_third():
    segments = [_segment(0, "hello"), _segment(1, "hello"), _segment(2, "hello")]
    assert find_duplicate_segments(segments) == {1, 2}


def test_duplicates_mixed_with_unique():
    segments = [_segment(0, "hello"), _segment(1, "world"), _segment(2, "hello"), _segment(3, "goodbye")]
    assert find_duplicate_segments(segments) == {2}


def test_duplicates_empty_text_skipped():
    # Two segments with empty text should not be treated as duplicates
    segments = [_segment(0, ""), _segment(1, "")]
    assert find_duplicate_segments(segments) == set()


def test_duplicates_whitespace_only_skipped():
    # Whitespace-only text strips to empty, should be skipped
    segments = [_segment(0, "   "), _segment(1, "   ")]
    assert find_duplicate_segments(segments) == set()
