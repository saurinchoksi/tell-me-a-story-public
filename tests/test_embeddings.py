"""Tests for embeddings module."""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from embeddings import extract_speaker_embeddings, save_embeddings, load_embedding_model


# --- Test helpers ---

def _diarization(segments=None):
    """Build a diarization dict matching diarize() output."""
    return {
        "_generator_version": "pyannote-speaker-diarization-community-1",
        "segments": segments or [],
    }


def _segment(speaker, start, end):
    """Build a single diarization segment."""
    return {"speaker": speaker, "start": start, "end": end}


def _mock_model(dim=256):
    """Return a mock embedding model that produces numpy arrays.

    Each call returns a slightly different vector (based on input shape)
    so averaged embeddings aren't trivially identical across speakers.
    Uses spec=[] to prevent MagicMock from auto-creating attributes
    like min_num_samples on hasattr checks.
    """
    call_count = [0]

    def side_effect(batch):
        call_count[0] += 1
        # Use call count to produce different vectors per call
        rng = np.random.RandomState(call_count[0])
        return rng.randn(1, dim).astype(np.float32)

    model = MagicMock(side_effect=side_effect, spec=[])
    model.dimension = dim
    return model


# --- extract_speaker_embeddings tests ---

@patch("embeddings.prepare_audio_for_diarization")
@patch("embeddings.torchaudio.load")
def test_two_speakers_both_get_vectors(mock_load, mock_prepare):
    """Two speakers with usable segments → both present in result."""
    import torch

    mock_prepare.return_value = "/tmp/fake.wav"
    # 10 seconds of audio at 16kHz
    mock_load.return_value = (torch.randn(1, 160000), 16000)

    diarization = _diarization([
        _segment("SPEAKER_00", 0.0, 3.0),
        _segment("SPEAKER_01", 3.0, 6.0),
        _segment("SPEAKER_00", 6.0, 9.0),
    ])

    model = _mock_model()

    with patch("embeddings.os.unlink"):
        result = extract_speaker_embeddings(model, "/fake/audio.m4a", diarization)

    assert "SPEAKER_00" in result["speakers"]
    assert "SPEAKER_01" in result["speakers"]
    assert result["speakers"]["SPEAKER_00"]["num_segments"] == 2
    assert result["speakers"]["SPEAKER_01"]["num_segments"] == 1


@patch("embeddings.prepare_audio_for_diarization")
@patch("embeddings.torchaudio.load")
def test_short_segments_excluded(mock_load, mock_prepare):
    """Speaker with all segments < 0.5s → excluded from result."""
    import torch

    mock_prepare.return_value = "/tmp/fake.wav"
    mock_load.return_value = (torch.randn(1, 160000), 16000)

    diarization = _diarization([
        _segment("SPEAKER_00", 0.0, 3.0),   # usable
        _segment("SPEAKER_01", 3.0, 3.3),   # too short
        _segment("SPEAKER_01", 4.0, 4.2),   # too short
    ])

    model = _mock_model()

    with patch("embeddings.os.unlink"):
        result = extract_speaker_embeddings(model, "/fake/audio.m4a", diarization)

    assert "SPEAKER_00" in result["speakers"]
    assert "SPEAKER_01" not in result["speakers"]


@patch("embeddings.prepare_audio_for_diarization")
@patch("embeddings.torchaudio.load")
def test_empty_diarization(mock_load, mock_prepare):
    """Empty diarization → empty speakers dict."""
    import torch

    mock_prepare.return_value = "/tmp/fake.wav"
    mock_load.return_value = (torch.randn(1, 160000), 16000)

    diarization = _diarization([])
    model = _mock_model()

    with patch("embeddings.os.unlink"):
        result = extract_speaker_embeddings(model, "/fake/audio.m4a", diarization)

    assert result["speakers"] == {}


@patch("embeddings.prepare_audio_for_diarization")
@patch("embeddings.torchaudio.load")
def test_single_segment_per_speaker(mock_load, mock_prepare):
    """Single segment per speaker → valid embedding."""
    import torch

    mock_prepare.return_value = "/tmp/fake.wav"
    mock_load.return_value = (torch.randn(1, 160000), 16000)

    diarization = _diarization([
        _segment("SPEAKER_00", 1.0, 4.0),
    ])

    model = _mock_model()

    with patch("embeddings.os.unlink"):
        result = extract_speaker_embeddings(model, "/fake/audio.m4a", diarization)

    assert "SPEAKER_00" in result["speakers"]
    assert result["speakers"]["SPEAKER_00"]["num_segments"] == 1
    vec = result["speakers"]["SPEAKER_00"]["vector"]
    assert len(vec) == 256


@patch("embeddings.prepare_audio_for_diarization")
@patch("embeddings.torchaudio.load")
def test_nan_in_model_output_skipped(mock_load, mock_prepare):
    """NaN in model output → that segment is skipped gracefully."""
    import torch

    mock_prepare.return_value = "/tmp/fake.wav"
    mock_load.return_value = (torch.randn(1, 160000), 16000)

    call_count = [0]

    def nan_on_first(batch):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call returns NaN
            result = np.full((1, 256), np.nan, dtype=np.float32)
            return result
        # Subsequent calls return valid data
        rng = np.random.RandomState(42)
        return rng.randn(1, 256).astype(np.float32)

    model = MagicMock(side_effect=nan_on_first, spec=[])
    model.dimension = 256

    diarization = _diarization([
        _segment("SPEAKER_00", 0.0, 2.0),  # will produce NaN
        _segment("SPEAKER_00", 3.0, 5.0),  # will produce valid
    ])

    with patch("embeddings.os.unlink"):
        result = extract_speaker_embeddings(model, "/fake/audio.m4a", diarization)

    # Speaker should still be present (second segment succeeded)
    assert "SPEAKER_00" in result["speakers"]
    assert result["speakers"]["SPEAKER_00"]["num_segments"] == 1


@patch("embeddings.prepare_audio_for_diarization")
@patch("embeddings.torchaudio.load")
def test_vectors_are_plain_lists(mock_load, mock_prepare):
    """Output vectors must be plain Python lists, not numpy arrays."""
    import torch

    mock_prepare.return_value = "/tmp/fake.wav"
    mock_load.return_value = (torch.randn(1, 160000), 16000)

    diarization = _diarization([
        _segment("SPEAKER_00", 0.0, 3.0),
    ])

    model = _mock_model()

    with patch("embeddings.os.unlink"):
        result = extract_speaker_embeddings(model, "/fake/audio.m4a", diarization)

    vec = result["speakers"]["SPEAKER_00"]["vector"]
    assert isinstance(vec, list)
    assert all(isinstance(v, float) for v in vec)


@patch("embeddings.prepare_audio_for_diarization")
@patch("embeddings.torchaudio.load")
def test_vectors_are_256_dim(mock_load, mock_prepare):
    """Each speaker vector must be exactly 256-dimensional."""
    import torch

    mock_prepare.return_value = "/tmp/fake.wav"
    mock_load.return_value = (torch.randn(1, 160000), 16000)

    diarization = _diarization([
        _segment("SPEAKER_00", 0.0, 3.0),
        _segment("SPEAKER_01", 3.0, 6.0),
    ])

    model = _mock_model(dim=256)

    with patch("embeddings.os.unlink"):
        result = extract_speaker_embeddings(model, "/fake/audio.m4a", diarization)

    for speaker_data in result["speakers"].values():
        assert len(speaker_data["vector"]) == 256


@patch("embeddings.prepare_audio_for_diarization")
@patch("embeddings.torchaudio.load")
def test_vectors_are_finite(mock_load, mock_prepare):
    """All values in output vectors must be finite floats."""
    import torch

    mock_prepare.return_value = "/tmp/fake.wav"
    mock_load.return_value = (torch.randn(1, 160000), 16000)

    diarization = _diarization([
        _segment("SPEAKER_00", 0.0, 3.0),
    ])

    model = _mock_model()

    with patch("embeddings.os.unlink"):
        result = extract_speaker_embeddings(model, "/fake/audio.m4a", diarization)

    vec = result["speakers"]["SPEAKER_00"]["vector"]
    assert all(np.isfinite(v) for v in vec)


@patch("embeddings.prepare_audio_for_diarization")
@patch("embeddings.torchaudio.load")
def test_output_schema_keys(mock_load, mock_prepare):
    """Output has _generator_version, _dimension, speakers keys."""
    import torch

    mock_prepare.return_value = "/tmp/fake.wav"
    mock_load.return_value = (torch.randn(1, 160000), 16000)

    diarization = _diarization([
        _segment("SPEAKER_00", 0.0, 3.0),
    ])

    model = _mock_model()

    with patch("embeddings.os.unlink"):
        result = extract_speaker_embeddings(model, "/fake/audio.m4a", diarization)

    assert result["_generator_version"] == "wespeaker-voxceleb-resnet34-LM"
    assert result["_dimension"] == 256
    assert "speakers" in result


@patch("embeddings.prepare_audio_for_diarization")
@patch("embeddings.torchaudio.load")
def test_total_duration_includes_short_segments(mock_load, mock_prepare):
    """total_duration_s includes ALL segments, even ones too short to embed."""
    import torch

    mock_prepare.return_value = "/tmp/fake.wav"
    mock_load.return_value = (torch.randn(1, 160000), 16000)

    diarization = _diarization([
        _segment("SPEAKER_00", 0.0, 3.0),   # 3.0s, usable
        _segment("SPEAKER_00", 5.0, 5.3),   # 0.3s, too short to embed
    ])

    model = _mock_model()

    with patch("embeddings.os.unlink"):
        result = extract_speaker_embeddings(model, "/fake/audio.m4a", diarization)

    # Total duration should include both segments
    assert result["speakers"]["SPEAKER_00"]["total_duration_s"] == 3.3
    # But only one segment was actually embedded
    assert result["speakers"]["SPEAKER_00"]["num_segments"] == 1


@patch("embeddings.prepare_audio_for_diarization")
@patch("embeddings.torchaudio.load")
def test_temp_wav_cleaned_up_on_success(mock_load, mock_prepare):
    """Temp WAV is cleaned up after successful extraction."""
    import torch

    mock_prepare.return_value = "/tmp/fake.wav"
    mock_load.return_value = (torch.randn(1, 160000), 16000)

    diarization = _diarization([])
    model = _mock_model()

    with patch("embeddings.os.unlink") as mock_unlink:
        extract_speaker_embeddings(model, "/fake/audio.m4a", diarization)

    mock_unlink.assert_called_once_with("/tmp/fake.wav")


@patch("embeddings.prepare_audio_for_diarization")
@patch("embeddings.torchaudio.load")
def test_temp_wav_cleaned_up_on_error(mock_load, mock_prepare):
    """Temp WAV is cleaned up even if extraction fails."""
    mock_prepare.return_value = "/tmp/fake.wav"
    mock_load.side_effect = RuntimeError("audio load failed")

    diarization = _diarization([])
    model = _mock_model()

    with patch("embeddings.os.unlink") as mock_unlink:
        with pytest.raises(RuntimeError, match="audio load failed"):
            extract_speaker_embeddings(model, "/fake/audio.m4a", diarization)

    mock_unlink.assert_called_once_with("/tmp/fake.wav")


# --- save_embeddings tests ---

def test_save_embeddings_writes_valid_json():
    """save_embeddings writes JSON that round-trips correctly."""
    data = {
        "_generator_version": "wespeaker-voxceleb-resnet34-LM",
        "_dimension": 256,
        "speakers": {
            "SPEAKER_00": {
                "vector": [0.1] * 256,
                "num_segments": 3,
                "total_duration_s": 15.5,
            }
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "embeddings.json")
        save_embeddings(data, path)

        with open(path) as f:
            loaded = json.load(f)

        assert loaded == data


def test_save_embeddings_creates_parent_dirs():
    """save_embeddings creates parent directories if needed."""
    data = {"_generator_version": "test", "_dimension": 256, "speakers": {}}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "nested", "dir", "embeddings.json")
        save_embeddings(data, path)

        assert os.path.isfile(path)


# --- load_embedding_model tests ---

@patch("embeddings.PretrainedSpeakerEmbedding")
def test_load_model_calls_correct_model(mock_pse):
    """load_embedding_model passes the correct model name."""
    mock_pse.return_value = MagicMock()

    load_embedding_model()

    mock_pse.assert_called_once_with("pyannote/wespeaker-voxceleb-resnet34-LM")


# --- Integration test (slow, loads real model + real audio) ---

@pytest.mark.slow
def test_real_embedding_extraction():
    """Load real model, extract embeddings from test session."""
    from embeddings import load_embedding_model, extract_speaker_embeddings

    # any local session with audio + diarization works; skip-guarded for machines without
    session_dir = "sessions/20260117-202237"
    audio_path = os.path.join(session_dir, "audio.m4a")
    diar_path = os.path.join(session_dir, "diarization.json")

    if not os.path.exists(audio_path) or not os.path.exists(diar_path):
        pytest.skip("Test session audio/diarization not available")

    with open(diar_path) as f:
        diarization = json.load(f)

    model = load_embedding_model()
    result = extract_speaker_embeddings(model, audio_path, diarization)

    # Should have at least one speaker
    assert len(result["speakers"]) > 0
    assert result["_generator_version"] == "wespeaker-voxceleb-resnet34-LM"
    assert result["_dimension"] == 256

    for speaker, data in result["speakers"].items():
        vec = data["vector"]
        assert len(vec) == 256
        assert all(isinstance(v, float) for v in vec)
        assert all(np.isfinite(v) for v in vec)
        assert data["num_segments"] > 0
        assert data["total_duration_s"] > 0
