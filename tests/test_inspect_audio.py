"""Tests for inspect_audio module."""

import pytest

from pipeline import get_audio_info


@pytest.mark.slow
def test_get_audio_info_returns_dict_for_valid_file():
    """Valid audio file should return a dict with expected keys."""
    info = get_audio_info("tests/fixtures/audio.m4a")

    assert info is not None
    assert "filename" in info
    assert "format" in info
    assert "duration_seconds" in info
    assert "sample_rate" in info
    assert "channels" in info


@pytest.mark.slow
def test_get_audio_info_returns_none_for_missing_file():
    """Missing file should return None."""
    info = get_audio_info("sessions/audio/does_not_exist.m4a")

    assert info is None


@pytest.mark.slow
def test_audio_file_has_expected_properties():
    """Our test file should have the properties we observed."""
    info = get_audio_info("tests/fixtures/audio.m4a")

    assert info["sample_rate"] == 48000
    assert info["channels"] == 2
    assert info["duration_seconds"] > 300 # roughly 5+ minutes
