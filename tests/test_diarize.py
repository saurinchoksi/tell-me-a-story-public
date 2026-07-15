"""Tests for diarize module."""

import os
import tempfile

import pytest

from diarize import prepare_audio_for_diarization, diarize


# --- Integration tests (need real audio + ffmpeg) ---


@pytest.mark.slow
def test_prepare_audio_for_diarization_creates_file():
    """prepare_audio_for_diarization should create a WAV file."""
    source = "tests/fixtures/audio.m4a"
    
    wav_path = prepare_audio_for_diarization(source)
    
    try:
        assert os.path.exists(wav_path)
        assert wav_path.endswith(".wav")
    finally:
        os.unlink(wav_path)


@pytest.mark.slow
def test_prepare_audio_for_diarization_correct_sample_rate():
    """Converted WAV should be 16kHz mono."""
    source = "tests/fixtures/audio.m4a"
    
    wav_path = prepare_audio_for_diarization(source)
    
    try:
        # Use wave module to check format
        import wave
        with wave.open(wav_path, 'rb') as w:
            assert w.getframerate() == 16000  # 16kHz
            assert w.getnchannels() == 1      # mono
    finally:
        os.unlink(wav_path)


# --- Integration tests (slow, loads model) ---


@pytest.mark.slow
def test_diarize_returns_expected_structure():
    """diarize() should return dict with schema version and segments."""
    result = diarize("tests/fixtures/audio.m4a")

    assert isinstance(result, dict)
    assert "_generator_version" in result
    assert "segments" in result

    segments = result["segments"]
    assert isinstance(segments, list)
    assert len(segments) > 0

    # Check first segment has expected keys
    first = segments[0]
    assert "start" in first
    assert "end" in first
    assert "speaker" in first


@pytest.mark.slow
def test_diarize_segments_have_valid_timestamps():
    """Each segment should have valid timestamps (end > start)."""
    result = diarize("tests/fixtures/audio.m4a")
    segments = result["segments"]

    for seg in segments:
        assert isinstance(seg["start"], float)
        assert isinstance(seg["end"], float)
        assert seg["end"] > seg["start"]


@pytest.mark.slow
def test_diarize_segments_are_chronological():
    """Segments should be in chronological order."""
    result = diarize("tests/fixtures/audio.m4a")
    segments = result["segments"]

    for i in range(1, len(segments)):
        # Each segment should start at or after the previous one started
        assert segments[i]["start"] >= segments[i-1]["start"]
