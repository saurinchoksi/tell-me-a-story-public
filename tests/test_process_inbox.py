"""Tests for process_inbox module."""

import contextlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from process_inbox import process_inbox, _print_summary


def _make_inbox(tmp_dir, filenames):
    """Create dummy files in a temp inbox directory and return the inbox Path."""
    inbox = Path(tmp_dir) / "inbox"
    inbox.mkdir()
    for name in filenames:
        (inbox / name).write_bytes(b"fake audio")
    return inbox


def _patch_dirs(stack, inbox_dir, sessions_dir):
    """Patch module-level directory constants onto an ExitStack.

    Also empties the detector registry so tests stay hermetic — the real
    family-name detector needs the gitignored data/name_roster.json.
    Detector-specific tests re-patch the registry with a fake.
    """
    stack.enter_context(patch("process_inbox.INBOX_DIR", inbox_dir))
    stack.enter_context(patch("process_inbox.SESSIONS_DIR", sessions_dir))
    stack.enter_context(patch("detectors.DETECTORS", []))


def _make_pipeline_result(embeddings=None, processing=()):
    """Build a mock pipeline result with _stats and _processing (matching real
    run_pipeline output — the fixer summary reads _processing)."""
    return {
        "transcript_raw": {"segments": []},
        "transcript": {"segments": [], "_stats": {}, "_processing": list(processing)},
        "diarization": {"segments": []},
        "embeddings": embeddings,
    }


def _make_saving_mock(session_dir):
    """Create a save_computed mock that writes actual files (needed for file-size measurement)."""
    def _save(sd, raw, transcript, diarization):
        for name, data in [("transcript-raw.json", raw), ("transcript-rich.json", transcript), ("diarization.json", diarization)]:
            with open(Path(sd) / name, "w") as f:
                json.dump(data, f)
    return MagicMock(side_effect=_save)


# --- process_inbox tests ---


def test_process_inbox_empty_inbox(capsys):
    """Empty inbox prints message and never calls pipeline."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = _make_inbox(tmp, [])
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()

        mock_pipeline = MagicMock()
        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            stack.enter_context(patch("process_inbox.run_pipeline", mock_pipeline))
            process_inbox()

        output = capsys.readouterr().out
        assert "Inbox is empty" in output
        mock_pipeline.assert_not_called()


def test_process_inbox_init_and_pipeline_success(capsys):
    """Successful init and pipeline run records session as created."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = _make_inbox(tmp, ["story.m4a"])
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()
        # Create the session dir that init would create
        session_dir = sessions / "20260301-120000"
        session_dir.mkdir()
        (session_dir / "audio.m4a").write_bytes(b"fake audio")

        mock_init = MagicMock(return_value=("20260301-120000", "audio.m4a"))
        mock_pipeline = MagicMock(return_value=_make_pipeline_result())
        mock_save = _make_saving_mock(session_dir)

        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            stack.enter_context(patch("process_inbox.init_session", mock_init))
            stack.enter_context(patch("process_inbox.run_pipeline", mock_pipeline))
            stack.enter_context(patch("process_inbox.save_computed", mock_save))
            process_inbox()

        mock_pipeline.assert_called_once()
        mock_save.assert_called_once()
        output = capsys.readouterr().out
        assert "20260301-120000" in output


def test_process_inbox_says_what_fixers_did_and_didnt(capsys):
    """ each session prints the fixer-stage ending (counts per fixer, NOT RUN
    with the reason, shadow reminder), and the run summary says loudly that family
    names were never checked — 'never ran' must not look like 'nothing found'."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = _make_inbox(tmp, ["story.m4a"])
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()
        session_dir = sessions / "20260301-120000"
        session_dir.mkdir()
        (session_dir / "audio.m4a").write_bytes(b"fake audio")

        result = _make_pipeline_result(processing=[
            {"stage": "namefix", "status": "success", "groups": 2,
             "from_cache": False, "shadow": True},
            {"stage": "family_namefix", "status": "skipped", "reason": "no_roster"},
            {"stage": "spellingfix", "status": "success", "groups": 1, "shadow": True},
        ])
        mock_init = MagicMock(return_value=("20260301-120000", "audio.m4a"))
        mock_pipeline = MagicMock(return_value=result)
        mock_save = _make_saving_mock(session_dir)

        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            stack.enter_context(patch("process_inbox.init_session", mock_init))
            stack.enter_context(patch("process_inbox.run_pipeline", mock_pipeline))
            stack.enter_context(patch("process_inbox.save_computed", mock_save))
            process_inbox()

        output = capsys.readouterr().out
        assert "Name fixers (shadow — no transcript touched):" in output
        assert "universe-name-fix: 2 labeled questions" in output
        assert "family-name-fix: NOT RUN — no roster yet" in output
        assert "invented-spelling-fix: 1 labeled question" in output
        assert "Family names NOT checked (1):" in output
        assert "fix_session.py" in output


def test_process_inbox_saves_embeddings(capsys):
    """Embeddings are saved to session directory when present in pipeline result."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = _make_inbox(tmp, ["story.m4a"])
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()
        session_dir = sessions / "20260301-120000"
        session_dir.mkdir()
        (session_dir / "audio.m4a").write_bytes(b"fake audio")

        fake_embeddings = {
            "_generator_version": "wespeaker-voxceleb-resnet34-LM",
            "_dimension": 256,
            "speakers": {"SPEAKER_00": {"vector": [0.1] * 256, "num_segments": 2, "total_duration_s": 5.0}},
        }

        mock_init = MagicMock(return_value=("20260301-120000", "audio.m4a"))
        mock_pipeline = MagicMock(return_value=_make_pipeline_result(embeddings=fake_embeddings))
        mock_save = _make_saving_mock(session_dir)

        def _save_emb(data, path):
            with open(path, "w") as f:
                json.dump(data, f)

        mock_save_embeddings = MagicMock(side_effect=_save_emb)

        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            stack.enter_context(patch("process_inbox.init_session", mock_init))
            stack.enter_context(patch("process_inbox.run_pipeline", mock_pipeline))
            stack.enter_context(patch("process_inbox.save_computed", mock_save))
            stack.enter_context(patch("process_inbox.save_embeddings", mock_save_embeddings))
            process_inbox()

        mock_save_embeddings.assert_called_once_with(
            fake_embeddings,
            str(sessions / "20260301-120000" / "embeddings.json"),
        )
        output = capsys.readouterr().out
        assert "Done" in output


def test_process_inbox_skips_embeddings_when_none(capsys):
    """Embeddings are not saved when pipeline returns None for embeddings."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = _make_inbox(tmp, ["story.m4a"])
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()
        session_dir = sessions / "20260301-120000"
        session_dir.mkdir()
        (session_dir / "audio.m4a").write_bytes(b"fake audio")

        mock_init = MagicMock(return_value=("20260301-120000", "audio.m4a"))
        mock_pipeline = MagicMock(return_value=_make_pipeline_result())
        mock_save = _make_saving_mock(session_dir)
        mock_save_embeddings = MagicMock()

        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            stack.enter_context(patch("process_inbox.init_session", mock_init))
            stack.enter_context(patch("process_inbox.run_pipeline", mock_pipeline))
            stack.enter_context(patch("process_inbox.save_computed", mock_save))
            stack.enter_context(patch("process_inbox.save_embeddings", mock_save_embeddings))
            process_inbox()

        mock_save_embeddings.assert_not_called()


def test_process_inbox_init_returns_none(capsys):
    """Init returning None records file as skipped; pipeline not called."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = _make_inbox(tmp, ["dup.m4a"])
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()

        mock_init = MagicMock(return_value=None)
        mock_pipeline = MagicMock()

        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            stack.enter_context(patch("process_inbox.init_session", mock_init))
            stack.enter_context(patch("process_inbox.run_pipeline", mock_pipeline))
            stack.enter_context(patch("process_inbox.save_computed", MagicMock()))
            process_inbox()

        mock_pipeline.assert_not_called()
        output = capsys.readouterr().out
        assert "No new sessions to process" in output


def test_process_inbox_init_raises(capsys):
    """Init raising an exception records failure; processing continues."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = _make_inbox(tmp, ["bad.m4a"])
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()

        mock_init = MagicMock(side_effect=Exception("corrupt file"))
        mock_pipeline = MagicMock()

        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            stack.enter_context(patch("process_inbox.init_session", mock_init))
            stack.enter_context(patch("process_inbox.run_pipeline", mock_pipeline))
            stack.enter_context(patch("process_inbox.save_computed", MagicMock()))
            process_inbox()

        mock_pipeline.assert_not_called()
        output = capsys.readouterr().out
        assert "Init failed" in output
        assert "corrupt file" in output


def test_process_inbox_pipeline_raises(capsys):
    """Init succeeds but pipeline raises records session as failed."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = _make_inbox(tmp, ["story.m4a"])
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()
        session_dir = sessions / "20260401-100000"
        session_dir.mkdir()
        (session_dir / "audio.m4a").write_bytes(b"fake audio")

        mock_init = MagicMock(return_value=("20260401-100000", "audio.m4a"))
        mock_pipeline = MagicMock(side_effect=RuntimeError("GPU OOM"))

        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            stack.enter_context(patch("process_inbox.init_session", mock_init))
            stack.enter_context(patch("process_inbox.run_pipeline", mock_pipeline))
            stack.enter_context(patch("process_inbox.save_computed", MagicMock()))
            process_inbox()

        output = capsys.readouterr().out
        assert "Pipeline failed" in output
        assert "GPU OOM" in output


def test_process_inbox_mixed_results(capsys):
    """Multiple files: one succeeds, one skipped, one fails — all recorded."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = _make_inbox(tmp, ["a.m4a", "b.m4a", "c.m4a"])
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()
        session_dir = sessions / "20260501-100000"
        session_dir.mkdir()
        (session_dir / "audio.m4a").write_bytes(b"fake audio")

        def _init_side_effect(path):
            if path.name == "a.m4a":
                return ("20260501-100000", "audio.m4a")  # success
            elif path.name == "b.m4a":
                return None  # skipped (duplicate)
            else:
                raise Exception("bad file")  # failure

        mock_pipeline = MagicMock(return_value=_make_pipeline_result())
        mock_save = _make_saving_mock(session_dir)

        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            stack.enter_context(patch("process_inbox.init_session", side_effect=_init_side_effect))
            stack.enter_context(patch("process_inbox.run_pipeline", mock_pipeline))
            stack.enter_context(patch("process_inbox.save_computed", mock_save))
            process_inbox()

        output = capsys.readouterr().out
        # All three categories appear in summary with correct counts
        assert "Created (1):" in output
        assert "Skipped" in output
        assert "Failed (1):" in output


from detectors.base import Detector


class _FakeDetector(Detector):
    id = "fake-detector"
    label = "Fake detector"
    failure_mode = "M0"
    version = "0.0.1"

    def run(self, session_dir):
        return {"n_word_tokens": 1, "flags": []}


class _FailingDetector(_FakeDetector):
    def run(self, session_dir):
        raise RuntimeError("roster missing")


def test_process_inbox_runs_detectors(capsys):
    """Registered detectors run on each created session and write detections.json."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = _make_inbox(tmp, ["story.m4a"])
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()
        session_dir = sessions / "20260301-120000"
        session_dir.mkdir()
        (session_dir / "audio.m4a").write_bytes(b"fake audio")

        mock_init = MagicMock(return_value=("20260301-120000", "audio.m4a"))
        mock_pipeline = MagicMock(return_value=_make_pipeline_result())
        mock_save = _make_saving_mock(session_dir)

        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            stack.enter_context(patch("process_inbox.init_session", mock_init))
            stack.enter_context(patch("process_inbox.run_pipeline", mock_pipeline))
            stack.enter_context(patch("process_inbox.save_computed", mock_save))
            stack.enter_context(patch("detectors.DETECTORS", [_FakeDetector()]))
            process_inbox()

        data = json.loads((session_dir / "detections.json").read_text())
        assert data["detectors"]["fake-detector"]["n_flags"] == 0
        output = capsys.readouterr().out
        assert "Done" in output


def test_process_inbox_detector_failure_recorded(capsys):
    """A detector exception is reported separately — the session still counts
    as created, because its transcription artifacts are saved and usable."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = _make_inbox(tmp, ["story.m4a"])
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()
        session_dir = sessions / "20260301-120000"
        session_dir.mkdir()
        (session_dir / "audio.m4a").write_bytes(b"fake audio")

        mock_init = MagicMock(return_value=("20260301-120000", "audio.m4a"))
        mock_pipeline = MagicMock(return_value=_make_pipeline_result())
        mock_save = _make_saving_mock(session_dir)

        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            stack.enter_context(patch("process_inbox.init_session", mock_init))
            stack.enter_context(patch("process_inbox.run_pipeline", mock_pipeline))
            stack.enter_context(patch("process_inbox.save_computed", mock_save))
            stack.enter_context(patch("detectors.DETECTORS", [_FailingDetector()]))
            process_inbox()

        output = capsys.readouterr().out
        assert "Detectors failed" in output
        assert "roster missing" in output
        assert "Pipeline failed" not in output      # not conflated with pipeline errors
        assert "Created (1):" in output             # session still counts as created
        # Transcription artifacts were already saved before the detector ran
        assert (session_dir / "transcript-rich.json").exists()


# --- _print_summary tests ---


def test_print_summary_all_populated(capsys):
    """All three lists populated prints all sections."""
    _print_summary(["sess-1"], ["dup.m4a"], [("bad.m4a", "error")])
    output = capsys.readouterr().out
    assert "Created" in output
    assert "Skipped" in output
    assert "Failed" in output


def test_print_summary_all_empty(capsys):
    """Empty lists prints nothing-processed message."""
    _print_summary([], [], [])
    output = capsys.readouterr().out
    assert "(nothing processed)" in output


def test_print_summary_only_created(capsys):
    """Only created list populated prints just that section."""
    _print_summary(["sess-1"], [], [])
    output = capsys.readouterr().out
    assert "Created" in output
    assert "Skipped" not in output
    assert "Failed" not in output


def test_print_summary_only_skipped(capsys):
    """Only skipped list populated prints just that section."""
    _print_summary([], ["dup.m4a"], [])
    output = capsys.readouterr().out
    assert "Skipped" in output
    assert "Created" not in output
    assert "Failed" not in output


def test_print_summary_only_failed(capsys):
    """Only failed list populated prints just that section."""
    _print_summary([], [], [("bad.m4a", "oops")])
    output = capsys.readouterr().out
    assert "Failed" in output
    assert "Created" not in output
    assert "Skipped" not in output


def test_process_inbox_target_file_valid(capsys):
    """target_file processes only the specified file from inbox."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = _make_inbox(tmp, ["target.m4a", "other.m4a"])
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()
        session_dir = sessions / "20260701-100000"
        session_dir.mkdir()
        (session_dir / "audio.m4a").write_bytes(b"fake audio")

        mock_init = MagicMock(return_value=("20260701-100000", "audio.m4a"))
        mock_pipeline = MagicMock(return_value=_make_pipeline_result())
        mock_save = _make_saving_mock(session_dir)

        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            stack.enter_context(patch("process_inbox.init_session", mock_init))
            stack.enter_context(patch("process_inbox.run_pipeline", mock_pipeline))
            stack.enter_context(patch("process_inbox.save_computed", mock_save))
            process_inbox(target_file="target.m4a")

        # init_session called with only the target file, not both
        mock_init.assert_called_once()
        called_path = mock_init.call_args[0][0]
        assert called_path.name == "target.m4a"


def test_process_inbox_target_file_not_found():
    """target_file pointing to nonexistent file raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = _make_inbox(tmp, [])
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()

        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            with pytest.raises(FileNotFoundError, match="ghost.m4a"):
                process_inbox(target_file="ghost.m4a")


def test_process_inbox_target_file_unsupported_format():
    """target_file with unsupported extension raises ValueError."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = _make_inbox(tmp, ["notes.txt"])
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()

        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            with pytest.raises(ValueError, match="Unsupported format"):
                process_inbox(target_file="notes.txt")


def test_process_inbox_missing_inbox_dir():
    """Missing inbox directory raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = Path(tmp) / "nonexistent"
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()

        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            with pytest.raises(FileNotFoundError):
                process_inbox()


def test_process_inbox_save_computed_raises(capsys):
    """save_computed failure is recorded as a pipeline failure."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = _make_inbox(tmp, ["story.m4a"])
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()

        mock_init = MagicMock(return_value=("20260601-100000", "audio.m4a"))
        mock_pipeline = MagicMock(return_value=_make_pipeline_result())
        mock_save = MagicMock(side_effect=OSError("disk full"))

        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            stack.enter_context(patch("process_inbox.init_session", mock_init))
            stack.enter_context(patch("process_inbox.run_pipeline", mock_pipeline))
            stack.enter_context(patch("process_inbox.save_computed", mock_save))
            process_inbox()

        output = capsys.readouterr().out
        assert "Pipeline failed" in output
        assert "disk full" in output
