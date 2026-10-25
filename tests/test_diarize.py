"""Tests for diarize module."""

import os
import tempfile
import wave
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from diarize import prepare_audio_for_diarization, diarize, load_wav


def _write_wav(path, samples, sample_rate=16000, channels=1):
    """Write int16 samples (interleaved if multi-channel) as a PCM WAV."""
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(np.asarray(samples, dtype="<i2").tobytes())


# --- In-memory audio (torchcodec can't load against Homebrew's FFmpeg 9) ---


def test_load_wav_returns_pyannote_waveform_dict(tmp_path):
    """load_wav gives pyannote's in-memory form: (channel, time) float32 in [-1, 1]."""
    path = str(tmp_path / "a.wav")
    _write_wav(path, [0, 16384, -32768, 32767])

    audio = load_wav(path)

    assert audio["sample_rate"] == 16000
    assert audio["waveform"].dtype == torch.float32
    assert audio["waveform"].shape == (1, 4)
    assert audio["waveform"][0].tolist() == pytest.approx([0.0, 0.5, -1.0, 32767 / 32768])


def test_load_wav_stereo_is_channel_first(tmp_path):
    """Interleaved stereo frames split into (2, time)."""
    path = str(tmp_path / "s.wav")
    _write_wav(path, [100, -100, 200, -200, 300, -300], channels=2)

    waveform = load_wav(path)["waveform"]

    assert waveform.shape == (2, 3)
    assert (waveform[0] > 0).all() and (waveform[1] < 0).all()


def test_diarize_hands_model_a_waveform_not_a_path(tmp_path):
    """The model gets audio in memory: a path makes pyannote decode via torchcodec."""
    path = str(tmp_path / "prepared.wav")
    _write_wav(path, np.zeros(16000, dtype=np.int16))
    seen = {}

    def stub_model(audio, hook=None, num_speakers=None):
        seen["audio"] = audio
        turn = SimpleNamespace(start=0.0, end=1.0)
        return SimpleNamespace(exclusive_speaker_diarization=[(turn, "SPEAKER_00")])

    with patch("diarize.prepare_audio_for_diarization", return_value=path):
        result = diarize("/fake/audio.m4a", model=stub_model)

    audio = seen["audio"]
    assert isinstance(audio, dict)
    assert audio["sample_rate"] == 16000
    assert audio["waveform"].shape == (1, 16000)
    assert result["segments"] == [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
    assert not os.path.exists(path)  # temp WAV still cleaned up


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
