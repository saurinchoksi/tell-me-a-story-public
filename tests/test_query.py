"""Tests for query module."""

from pipeline import (
    to_utterances,
    format_transcript,
)


# --- to_utterances tests ---

def test_to_utterances_consolidates_same_speaker():
    """Consecutive same-speaker words become one utterance."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello", "speaker": "SPEAKER_00"},
        {"start": 0.5, "end": 1.0, "word": " there", "speaker": "SPEAKER_00"},
    ]
    result = to_utterances(words)
    assert len(result) == 1
    assert result[0]["text"] == "Hello there"
    assert result[0]["speaker"] == "SPEAKER_00"
    assert len(result[0]["words"]) == 2


def test_to_utterances_splits_different_speakers():
    """Different speakers produce separate utterances."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello", "speaker": "SPEAKER_00"},
        {"start": 0.5, "end": 1.0, "word": " Hi", "speaker": "SPEAKER_01"},
    ]
    result = to_utterances(words)
    assert len(result) == 2
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[1]["speaker"] == "SPEAKER_01"


def test_to_utterances_does_not_consolidate_none():
    """None speaker utterances stay separate."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Uh", "speaker": None},
        {"start": 0.5, "end": 1.0, "word": " huh", "speaker": None},
    ]
    result = to_utterances(words)
    assert len(result) == 2
    assert result[0]["text"] == "Uh"
    assert result[1]["text"] == "huh"


def test_to_utterances_timestamps():
    """Utterances have correct start/end timestamps."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello", "speaker": "SPEAKER_00"},
        {"start": 0.5, "end": 1.0, "word": " there", "speaker": "SPEAKER_00"},
    ]
    result = to_utterances(words)
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 1.0


def test_to_utterances_strips_whitespace():
    """Word text is stripped of whitespace."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello ", "speaker": "SPEAKER_00"},
    ]
    result = to_utterances(words)
    assert result[0]["text"] == "Hello"


def test_to_utterances_empty():
    """Empty input returns empty result."""
    assert to_utterances([]) == []


def test_to_utterances_enriched_format():
    """Words with _speaker.label (enriched format) work directly."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello",
         "_speaker": {"label": "SPEAKER_00", "coverage": 0.9}},
        {"start": 0.5, "end": 1.0, "word": " there",
         "_speaker": {"label": "SPEAKER_00", "coverage": 0.8}},
        {"start": 1.0, "end": 1.5, "word": " Hi",
         "_speaker": {"label": "SPEAKER_01", "coverage": 0.95}},
    ]
    result = to_utterances(words)
    assert len(result) == 2
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[0]["text"] == "Hello there"
    assert result[1]["speaker"] == "SPEAKER_01"


def test_to_utterances_enriched_none_speaker():
    """Enriched words with label=None stay separate (same as flat None)."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Uh",
         "_speaker": {"label": None, "coverage": 0.0}},
        {"start": 0.5, "end": 1.0, "word": " huh",
         "_speaker": {"label": None, "coverage": 0.0}},
    ]
    result = to_utterances(words)
    assert len(result) == 2


# --- format_transcript tests ---

def test_format_transcript_basic():
    """Formats utterances as SPEAKER: text lines."""
    utterances = [
        {"speaker": "SPEAKER_00", "text": "Hello"},
        {"speaker": "SPEAKER_01", "text": "Hi there"},
    ]
    result = format_transcript(utterances)
    assert "SPEAKER_00: Hello" in result
    assert "SPEAKER_01: Hi there" in result


def test_format_transcript_none_speaker():
    """None speaker shows as UNKNOWN."""
    utterances = [
        {"speaker": None, "text": "mystery"},
    ]
    result = format_transcript(utterances)
    assert "UNKNOWN: mystery" in result


def test_format_transcript_empty():
    """Empty utterances return empty string."""
    result = format_transcript([])
    assert result == ""
