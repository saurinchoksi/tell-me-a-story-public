"""Tests for transcribe module."""

from transcribe import transcribe, clean_transcript, make_processing_entry, MODEL


# --- Unit tests (fast, no model needed) ---


# --- make_processing_entry tests ---

def test_make_processing_entry():
    """make_processing_entry returns dict with all expected keys."""
    entry = make_processing_entry("sha256:abc123", "2026-01-01T00:00:00+00:00")
    assert entry["stage"] == "transcription"
    assert entry["model"] == MODEL
    assert entry["status"] == "success"
    assert entry["audio_hash"] == "sha256:abc123"
    assert entry["timestamp"] == "2026-01-01T00:00:00+00:00"


# --- clean_transcript tests ---

def test_clean_removes_zero_duration_words():
    """Zero-duration words (fabrications) are removed."""
    transcript = {
        "text": "Hello world",
        "segments": [
            {
                "start": 0.0, "end": 2.0, "text": " Hello world",
                "words": [
                    {"word": " Hello", "start": 0.0, "end": 0.5, "probability": 0.95},
                    {"word": " fake", "start": 1.0, "end": 1.0, "probability": 0.9},  # zero duration
                    {"word": " world", "start": 1.5, "end": 2.0, "probability": 0.92}
                ]
            }
        ]
    }

    result = clean_transcript(transcript)

    assert len(result["segments"]) == 1
    assert len(result["segments"][0]["words"]) == 2
    words = [w["word"] for w in result["segments"][0]["words"]]
    assert " Hello" in words
    assert " world" in words
    assert " fake" not in words


def test_clean_keeps_normal_words():
    """Words with positive duration pass through unchanged."""
    transcript = {
        "text": "Once upon a time",
        "segments": [
            {
                "start": 0.0, "end": 3.0, "text": " Once upon a time",
                "words": [
                    {"word": " Once", "start": 0.0, "end": 0.5, "probability": 0.98},
                    {"word": " upon", "start": 0.5, "end": 1.0, "probability": 0.97},
                    {"word": " a", "start": 1.0, "end": 1.2, "probability": 0.99},
                    {"word": " time", "start": 1.2, "end": 1.8, "probability": 0.96}
                ]
            }
        ]
    }

    result = clean_transcript(transcript)

    assert len(result["segments"][0]["words"]) == 4
    # Verify original transcript not mutated
    assert len(transcript["segments"][0]["words"]) == 4


def test_clean_adds_generator_version():
    """clean_transcript adds _generator_version to result."""
    transcript = {
        "text": "Hello",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": " Hello", "words": []}
        ]
    }

    result = clean_transcript(transcript)

    assert "_generator_version" in result


def test_clean_empty_segments():
    """Empty segments list returns empty result."""
    transcript = {
        "text": "",
        "segments": []
    }

    result = clean_transcript(transcript)

    assert result["segments"] == []


def test_clean_removes_garbage_segments():
    """Segments with no text or zero duration are removed."""
    transcript = {
        "text": "Hello",
        "segments": [
            # Good segment
            {"start": 0.0, "end": 1.0, "text": " Hello", "words": []},
            # Garbage: empty text, zero duration
            {"start": 115.56, "end": 115.56, "text": "", "words": []},
            # Garbage: whitespace-only text
            {"start": 2.0, "end": 3.0, "text": "   ", "words": []},
            # Garbage: zero duration even with text
            {"start": 5.0, "end": 5.0, "text": " Something", "words": []},
        ]
    }

    result = clean_transcript(transcript)

    assert len(result["segments"]) == 1
    assert result["segments"][0]["text"] == " Hello"



# --- Inegration test (slow, runs actual model) ---
# Mark with pytest.mark.slow so we can skip in quick runs

import pytest

@pytest.mark.slow
def test_transcribe_returns_expected_structure():
    """transcribe() should return dict with text, language, segments."""
    result = transcribe("tests/fixtures/audio.m4a")

    assert "text" in result
    assert "language" in result
    assert "segments" in result
    assert isinstance(result["text"], str)
    assert isinstance(result["segments"], list)
    assert len(result["text"]) > 0 # Should have some context

@pytest.mark.slow
def test_transcribe_segments_have_timestamps():
    """Each segment should have start, end, and text."""
    result = transcribe("tests/fixtures/audio.m4a")

    assert len(result["segments"]) > 0

    for seg in result["segments"]:
        assert "start" in seg
        assert "end" in seg
        assert "text" in seg
        assert seg["end"] > seg["start"]


@pytest.mark.slow
def test_transcribe_with_word_timestamps():
    """Transcribe should include words array in segments."""
    result = transcribe("tests/fixtures/audio.m4a")

    # Find a segment with content (some may be empty)
    seg_with_words = None
    for seg in result["segments"]:
        if seg.get("words"):
            seg_with_words = seg
            break

    assert seg_with_words is not None, "Expected at least one segment with words"

    # Check word structure
    word = seg_with_words["words"][0]
    assert "start" in word
    assert "end" in word
    assert "word" in word
