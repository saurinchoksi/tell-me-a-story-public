"""Tests for pipeline module."""

import contextlib
import copy
import json
import os
import subprocess
import sys
import tempfile
from unittest.mock import patch, MagicMock

from pipeline import compute_file_hash, save_computed, run_pipeline, guard_re_enrich


# --- compute_file_hash tests ---

def test_compute_file_hash_returns_sha256():
    """Hash has correct prefix and length."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        temp_path = f.name

    try:
        result = compute_file_hash(temp_path)
        assert result.startswith("sha256:")
        # SHA256 hex is 64 chars + 7 for "sha256:"
        assert len(result) == 71
    finally:
        os.unlink(temp_path)


def test_compute_file_hash_deterministic():
    """Same content produces same hash."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("deterministic")
        temp_path = f.name

    try:
        hash1 = compute_file_hash(temp_path)
        hash2 = compute_file_hash(temp_path)
        assert hash1 == hash2
    finally:
        os.unlink(temp_path)


# --- save_computed tests ---

def test_save_computed_creates_directory():
    """save_computed creates session directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        session_dir = os.path.join(temp_dir, "test-session")

        save_computed(
            session_dir=session_dir,
            transcript_raw={"text": "hello", "segments": []},
            transcript={"text": "hello", "segments": []},
            diarization={"segments": []},
        )

        assert os.path.isdir(session_dir)


def test_save_computed_creates_files():
    """save_computed creates all expected files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        session_dir = os.path.join(temp_dir, "test-session")

        save_computed(
            session_dir=session_dir,
            transcript_raw={"text": "hello", "segments": []},
            transcript={"text": "hello", "segments": []},
            diarization={"segments": []},
        )

        assert os.path.isfile(os.path.join(session_dir, "transcript-raw.json"))
        assert os.path.isfile(os.path.join(session_dir, "transcript-rich.json"))
        assert os.path.isfile(os.path.join(session_dir, "diarization.json"))


def test_save_computed_content():
    """save_computed writes correct JSON content."""
    with tempfile.TemporaryDirectory() as temp_dir:
        session_dir = os.path.join(temp_dir, "test-session")

        transcript_raw = {"text": "hello world", "segments": []}
        transcript = {"text": "hello world", "segments": [], "audio": {"duration": 10.5}}
        diarization = {"segments": [{"start": 0, "end": 1, "speaker": "A"}]}

        save_computed(
            session_dir=session_dir,
            transcript_raw=transcript_raw,
            transcript=transcript,
            diarization=diarization,
        )

        with open(os.path.join(session_dir, "transcript-raw.json")) as f:
            assert json.load(f) == transcript_raw

        with open(os.path.join(session_dir, "transcript-rich.json")) as f:
            assert json.load(f) == transcript

        with open(os.path.join(session_dir, "diarization.json")) as f:
            assert json.load(f) == diarization


# --- run_pipeline normalization tests ---

_FAKE_TRANSCRIPT = {
    "text": "hello world",
    "segments": [{
        "text": "hello world",
        "words": [
            {"word": " hello", "start": 0.0, "end": 0.5, "probability": 0.9},
            {"word": " world", "start": 0.5, "end": 1.0, "probability": 0.8},
        ]
    }]
}


def _find_stage(processing, stage_name):
    """Find a processing entry by stage name."""
    return next(e for e in processing if e["stage"] == stage_name)


def _apply_pipeline_mocks(stack, **overrides):
    """Enter common mock patches onto an ExitStack and return a dict of active mocks.

    Any key in overrides replaces the default mock for that target name
    (use the short name after 'pipeline.', e.g. enrich_transcript=MagicMock(...)).
    """
    defaults = {
        "transcribe": MagicMock(return_value=copy.deepcopy(_FAKE_TRANSCRIPT)),
        "clean_transcript": MagicMock(side_effect=lambda x: x),
        "realign_words": MagicMock(side_effect=lambda t, p, verbose=True: (
            t, {"stage": "word_realignment", "model": "torchaudio-MMS_FA", "status": "success"})),
        "diarize": MagicMock(return_value={"segments": []}),
        "get_audio_info": MagicMock(return_value={"duration": 10.0}),
        "compute_file_hash": MagicMock(return_value="sha256:fake"),
        "os.path.exists": MagicMock(return_value=True),
        "load_embedding_model": MagicMock(return_value=MagicMock()),
        "extract_speaker_embeddings": MagicMock(return_value=None),
    }
    defaults.update(overrides)

    active = {}
    for short_name, mock_obj in defaults.items():
        target = f"pipeline.{short_name}"
        active[short_name] = stack.enter_context(patch(target, mock_obj))
    return active


def test_pipeline_dictionary_normalization_succeeds():
    """Dictionary normalization succeeds and counts are recorded."""
    fake_transcript = copy.deepcopy(_FAKE_TRANSCRIPT)

    enrichment_processing = [
        {"stage": "diarization_enrichment", "model": "pyannote/speaker-diarization-community-1", "status": "success"},
        {"stage": "gap_detection", "gaps_found": 0, "status": "success"},
        {"stage": "dictionary_normalization", "library": "data/mahabharata.json", "status": "success", "corrections_applied": 3},
    ]

    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(
            stack,
            enrich_transcript=MagicMock(
                return_value=(fake_transcript, enrichment_processing, {"dict_count": 3})
            ),
        )
        result = run_pipeline("/fake/20260101-120000/audio.m4a", verbose=False)

    processing = result["transcript"]["_processing"]
    assert len(processing) == 7

    transcription = _find_stage(processing, "transcription")
    assert transcription["status"] == "success"
    assert transcription["audio_hash"] == "sha256:fake"
    assert "timestamp" in transcription
    assert "started_at" in transcription
    assert "duration_seconds" in transcription

    assert _find_stage(processing, "diarization")["status"] == "success"
    assert "duration_seconds" in _find_stage(processing, "diarization")

    assert _find_stage(processing, "diarization_enrichment")["status"] == "success"
    assert _find_stage(processing, "gap_detection")["status"] == "success"

    dict_entry = _find_stage(processing, "dictionary_normalization")
    assert dict_entry["status"] == "success"
    assert dict_entry["corrections_applied"] == 3

    emb = _find_stage(processing, "embedding_extraction")
    assert emb["status"] == "success"
    assert "duration_seconds" in emb

    assert "llm_count" not in result
    assert "dict_count" not in result

    # Verify _stats
    assert "_stats" in result["transcript"]
    stats = result["transcript"]["_stats"]
    assert "pipeline_started_at" in stats
    assert "pipeline_duration_seconds" in stats
    assert stats["segments"] == 1
    assert stats["words"] == 2


def test_pipeline_empty_corrections():
    """Empty correction lists result in zero counts but success status."""
    fake_transcript = copy.deepcopy(_FAKE_TRANSCRIPT)

    enrichment_processing = [
        {"stage": "diarization_enrichment", "model": "pyannote/speaker-diarization-community-1", "status": "success"},
        {"stage": "gap_detection", "gaps_found": 0, "status": "success"},
        {"stage": "dictionary_normalization", "library": "data/mahabharata.json", "status": "success", "corrections_applied": 0},
    ]

    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(
            stack,
            enrich_transcript=MagicMock(
                return_value=(copy.deepcopy(fake_transcript), enrichment_processing, {"dict_count": 0})
            ),
        )
        result = run_pipeline("/fake/20260101-120000/audio.m4a", verbose=False)

    processing = result["transcript"]["_processing"]
    assert len(processing) == 7

    assert _find_stage(processing, "dictionary_normalization")["status"] == "success"
    assert _find_stage(processing, "dictionary_normalization")["corrections_applied"] == 0

    # Words should be unchanged
    words = result["transcript"]["segments"][0]["words"]
    assert words[0]["word"] == " hello"
    assert words[1]["word"] == " world"


def test_pipeline_diarization_enrichment_succeeds():
    """Diarization enrichment stage is recorded in processing."""
    fake_transcript = copy.deepcopy(_FAKE_TRANSCRIPT)

    enrichment_processing = [
        {"stage": "diarization_enrichment", "model": "pyannote/speaker-diarization-community-1", "status": "success"},
        {"stage": "gap_detection", "gaps_found": 0, "status": "success"},
        {"stage": "dictionary_normalization", "library": "data/mahabharata.json", "status": "success", "corrections_applied": 0},
    ]

    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(
            stack,
            enrich_transcript=MagicMock(
                return_value=(fake_transcript, enrichment_processing, {"dict_count": 0})
            ),
        )
        result = run_pipeline("/fake/20260101-120000/audio.m4a", verbose=False)

    processing = result["transcript"]["_processing"]
    assert len(processing) == 7

    enrichment_entry = _find_stage(processing, "diarization_enrichment")
    assert enrichment_entry["status"] == "success"
    assert enrichment_entry["model"] == "pyannote/speaker-diarization-community-1"


def test_pipeline_diarization_enrichment_fails_gracefully():
    """Diarization enrichment failure is recorded but pipeline continues."""
    fake_transcript = copy.deepcopy(_FAKE_TRANSCRIPT)

    enrichment_processing = [
        {"stage": "diarization_enrichment", "model": "pyannote/speaker-diarization-community-1", "status": "error", "error": "Enrichment crashed"},
        {"stage": "gap_detection", "gaps_found": 0, "status": "success"},
        {"stage": "dictionary_normalization", "library": "data/mahabharata.json", "status": "success", "corrections_applied": 0},
    ]

    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(
            stack,
            enrich_transcript=MagicMock(
                return_value=(fake_transcript, enrichment_processing, {"dict_count": 0})
            ),
        )
        result = run_pipeline("/fake/20260101-120000/audio.m4a", verbose=False)

    processing = result["transcript"]["_processing"]
    assert len(processing) == 7

    enrichment_entry = _find_stage(processing, "diarization_enrichment")
    assert enrichment_entry["status"] == "error"
    assert "Enrichment crashed" in enrichment_entry["error"]

    # Pipeline still returns valid result
    assert "transcript" in result
    assert "diarization" in result


# --- session_id validation tests ---

import pytest


def test_pipeline_rejects_invalid_session_id():
    """Non-YYYYMMDD-HHMMSS parent directory raises ValueError."""
    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(stack)
        with pytest.raises(ValueError, match="Invalid session ID 'random-folder'"):
            run_pipeline("/fake/random-folder/audio.m4a", verbose=False)


def test_pipeline_rejects_uuid_session_id():
    """UUID-style directory name is not a valid session ID."""
    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(stack)
        with pytest.raises(ValueError, match="Invalid session ID"):
            run_pipeline("/fake/a1b2c3d4-e5f6-7890-abcd-ef1234567890/audio.m4a", verbose=False)


def test_pipeline_accepts_valid_session_id():
    """Standard YYYYMMDD-HHMMSS session ID passes validation."""
    fake_transcript = copy.deepcopy(_FAKE_TRANSCRIPT)
    enrichment_processing = [
        {"stage": "diarization_enrichment", "model": "pyannote/speaker-diarization-community-1", "status": "success"},
        {"stage": "gap_detection", "gaps_found": 0, "status": "success"},
        {"stage": "dictionary_normalization", "library": "data/mahabharata.json", "status": "success", "corrections_applied": 0},
    ]
    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(
            stack,
            enrich_transcript=MagicMock(
                return_value=(fake_transcript, enrichment_processing, {"dict_count": 0})
            ),
        )
        result = run_pipeline("/sessions/20260217-143000/audio.m4a", verbose=False)
    assert result["session_id"] == "20260217-143000"


def test_pipeline_embedding_failure_returns_none():
    """Embedding extraction failure → pipeline continues, embeddings is None."""
    fake_transcript = copy.deepcopy(_FAKE_TRANSCRIPT)
    enrichment_processing = [
        {"stage": "diarization_enrichment", "model": "pyannote/speaker-diarization-community-1", "status": "success"},
        {"stage": "gap_detection", "gaps_found": 0, "status": "success"},
        {"stage": "dictionary_normalization", "library": "data/mahabharata.json", "status": "success", "corrections_applied": 0},
    ]
    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(
            stack,
            enrich_transcript=MagicMock(
                return_value=(fake_transcript, enrichment_processing, {"dict_count": 0})
            ),
            load_embedding_model=MagicMock(side_effect=RuntimeError("Model download failed")),
        )
        result = run_pipeline("/sessions/20260301-120000/audio.m4a", verbose=False)

    assert result["embeddings"] is None
    # Pipeline still produces valid transcript and diarization
    assert "transcript" in result
    assert "diarization" in result
    assert result["session_id"] == "20260301-120000"

    # Embedding failure is still recorded in processing
    emb_entry = _find_stage(result["transcript"]["_processing"], "embedding_extraction")
    assert emb_entry["status"] == "error"
    assert "Model download failed" in emb_entry["error"]


def test_pipeline_stats_structure():
    """_stats contains all expected keys with correct types."""
    fake_transcript = copy.deepcopy(_FAKE_TRANSCRIPT)
    # Add _speaker to words so speaker count is nonzero
    for word in fake_transcript["segments"][0]["words"]:
        word["_speaker"] = {"label": "SPEAKER_00"}
    # A gap/unassigned word carries a null label; _stats.speakers must count
    # only real speakers, not the null (regression test for TMAS-41).
    fake_transcript["segments"][0]["words"][0]["_speaker"] = {"label": None}

    enrichment_processing = [
        {"stage": "diarization_enrichment", "model": "pyannote/speaker-diarization-community-1", "status": "success"},
        {"stage": "gap_detection", "gaps_found": 0, "status": "success"},
        {"stage": "dictionary_normalization", "library": "data/mahabharata.json", "status": "success", "corrections_applied": 0},
    ]
    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(
            stack,
            enrich_transcript=MagicMock(
                return_value=(fake_transcript, enrichment_processing, {"dict_count": 0})
            ),
        )
        result = run_pipeline("/sessions/20260310-120000/audio.m4a", verbose=False)

    stats = result["transcript"]["_stats"]
    assert isinstance(stats["pipeline_started_at"], str)
    assert isinstance(stats["pipeline_duration_seconds"], float)
    assert stats["segments"] == 1
    assert stats["words"] == 2
    assert stats["speakers"] == 1


def test_pipeline_processing_stage_order():
    """Processing entries appear in pipeline execution order."""
    fake_transcript = copy.deepcopy(_FAKE_TRANSCRIPT)
    enrichment_processing = [
        {"stage": "diarization_enrichment", "model": "pyannote/speaker-diarization-community-1", "status": "success"},
        {"stage": "gap_detection", "gaps_found": 0, "status": "success"},
        {"stage": "dictionary_normalization", "library": "data/mahabharata.json", "status": "success", "corrections_applied": 0},
    ]
    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(
            stack,
            enrich_transcript=MagicMock(
                return_value=(fake_transcript, enrichment_processing, {"dict_count": 0})
            ),
        )
        result = run_pipeline("/sessions/20260310-120000/audio.m4a", verbose=False)

    stages = [e["stage"] for e in result["transcript"]["_processing"]]
    assert stages == [
        "transcription",
        "word_realignment",
        "diarization",
        "diarization_enrichment",
        "gap_detection",
        "dictionary_normalization",
        "embedding_extraction",
    ]


# --- --re-enrich guard tests (TMAS-105) ---

def test_guard_re_enrich_refuses_coded_session(tmp_path):
    """A session with axial-labels.json refuses --re-enrich, naming the session and the risk."""
    (tmp_path / "axial-labels.json").write_text("{}")
    with pytest.raises(SystemExit) as exc:
        guard_re_enrich(str(tmp_path))
    message = str(exc.value)
    assert str(tmp_path) in message
    assert "axial-labels.json" in message
    assert "--force" in message


def test_guard_re_enrich_force_proceeds_on_coded_session(tmp_path):
    """--force overrides the guard on a coded session."""
    (tmp_path / "axial-labels.json").write_text("{}")
    guard_re_enrich(str(tmp_path), force=True)


def test_guard_re_enrich_allows_uncoded_session(tmp_path):
    """A session without axial-labels.json proceeds without --force."""
    guard_re_enrich(str(tmp_path))


@pytest.mark.slow
def test_cli_re_enrich_wires_guard(tmp_path):
    """The --re-enrich CLI branch actually calls the guard (regression net for the
    __main__ wiring, which the unit tests above can't see): a coded session refuses
    with exit 1, and --force gets past the guard to the next pre-existing check."""
    (tmp_path / "axial-labels.json").write_text("{}")
    script = os.path.join(os.path.dirname(__file__), "..", "src", "pipeline.py")

    refused = subprocess.run([sys.executable, script, "--re-enrich", str(tmp_path)],
                             capture_output=True, text=True)
    assert refused.returncode == 1
    assert "REFUSED" in refused.stderr

    forced = subprocess.run([sys.executable, script, "--re-enrich", "--force", str(tmp_path)],
                            capture_output=True, text=True)
    assert "REFUSED" not in forced.stderr
    assert "No transcript-raw.json" in forced.stderr


def test_pipeline_realign_runs_before_raw_snapshot():
    """Realignment runs before the raw snapshot, so transcript-raw.json carries the
    corrected word timings (raw and rich agree) — and before diarization."""
    def fake_realign(transcript, audio_path, verbose=True):
        # Mark every word as realigned so we can detect the snapshot ordering.
        for seg in transcript["segments"]:
            for wd in seg["words"]:
                wd["_align_conf"] = 0.9
        return transcript, {"stage": "word_realignment",
                            "model": "torchaudio-MMS_FA", "status": "success"}

    enrichment_processing = [
        {"stage": "diarization_enrichment", "status": "success"},
        {"stage": "gap_detection", "gaps_found": 0, "status": "success"},
        {"stage": "dictionary_normalization", "status": "success", "corrections_applied": 0},
    ]
    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(
            stack,
            realign_words=MagicMock(side_effect=fake_realign),
            enrich_transcript=MagicMock(
                return_value=(copy.deepcopy(_FAKE_TRANSCRIPT), enrichment_processing,
                              {"dict_count": 0})
            ),
        )
        result = run_pipeline("/sessions/20260401-120000/audio.m4a", verbose=False)

    # The raw snapshot was taken AFTER realign, so it carries the _align_conf marks.
    raw_words = result["transcript_raw"]["segments"][0]["words"]
    assert raw_words and all("_align_conf" in w for w in raw_words)

    stages = [e["stage"] for e in result["transcript"]["_processing"]]
    assert stages.index("word_realignment") < stages.index("diarization")
