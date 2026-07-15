"""Tests for detections.json persistence and the freshness gate."""

import json

import pytest

from detectors.base import (
    Detector,
    build_section,
    scan_session,
    section_is_stale,
    transcript_fingerprint,
    write_detections,
)


class FakeDetector(Detector):
    id = "fake-detector"
    label = "Fake detector"
    failure_mode = "M0"
    version = "0.1.0"

    def __init__(self):
        self.run_count = 0

    def run(self, session_dir):
        self.run_count += 1
        return {"n_word_tokens": 100, "flags": [{"segment_id": 0, "token": "Xyz"}]}


RESULT = {"n_word_tokens": 100, "flags": [{"segment_id": 0, "token": "Xyz"}]}


@pytest.fixture
def session_dir(tmp_path):
    """Session dir with a minimal transcript (fingerprint source)."""
    (tmp_path / "transcript-rich.json").write_text(json.dumps({"segments": []}))
    return tmp_path


def test_write_creates_file_with_section_shape(session_dir):
    section = write_detections(session_dir, FakeDetector(), RESULT)
    data = json.loads((session_dir / "detections.json").read_text())
    assert set(data) == {"_about", "detectors"}
    assert data["detectors"]["fake-detector"] == section
    assert section["label"] == "Fake detector"
    assert section["failure_mode"] == "M0"
    assert section["detector_version"] == "0.1.0"
    assert section["n_word_tokens"] == 100
    assert section["n_flags"] == 1
    assert section["flags"] == RESULT["flags"]
    assert "run_at" in section
    assert section["transcript_fingerprint"] == transcript_fingerprint(session_dir)
    assert section["config_fingerprint"] is None  # FakeDetector has no config


def test_write_requires_transcript(tmp_path):
    with pytest.raises(FileNotFoundError, match="transcript-rich.json"):
        write_detections(tmp_path, FakeDetector(), RESULT)


def test_rerun_overwrites_only_own_section(session_dir):
    # Pre-seed a section from another detector
    (session_dir / "detections.json").write_text(json.dumps({
        "_about": "x",
        "detectors": {"other-detector": {"n_flags": 7, "flags": []}},
    }))
    write_detections(session_dir, FakeDetector(), RESULT)
    data = json.loads((session_dir / "detections.json").read_text())
    assert data["detectors"]["other-detector"]["n_flags"] == 7
    assert data["detectors"]["fake-detector"]["n_flags"] == 1

    # Re-run with new results replaces only the fake-detector section
    write_detections(session_dir, FakeDetector(), {"n_word_tokens": 100, "flags": []})
    data = json.loads((session_dir / "detections.json").read_text())
    assert data["detectors"]["fake-detector"]["n_flags"] == 0
    assert data["detectors"]["other-detector"]["n_flags"] == 7


def test_corrupt_detections_file_fails_loud(session_dir):
    (session_dir / "detections.json").write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        write_detections(session_dir, FakeDetector(), RESULT)


def test_build_section_counts_flags():
    section = build_section(FakeDetector(), {"n_word_tokens": 5, "flags": [{}, {}]},
                            "sha256:x", None)
    assert section["n_flags"] == 2
    assert section["transcript_fingerprint"] == "sha256:x"


def test_save_leaves_no_temp_file(session_dir):
    write_detections(session_dir, FakeDetector(), RESULT)
    assert (session_dir / "detections.json").exists()
    assert not (session_dir / "detections.json.tmp").exists()


# --- scan_session (the scan entry point) --------------------------------------

def test_scan_runs_when_never_scanned(session_dir):
    det = FakeDetector()
    data = scan_session(session_dir, [det])
    assert det.run_count == 1
    assert data["detectors"]["fake-detector"]["n_flags"] == 1
    on_disk = json.loads((session_dir / "detections.json").read_text())
    assert on_disk["detectors"]["fake-detector"]["n_flags"] == 1


def test_scan_skips_when_fingerprint_matches(session_dir):
    det = FakeDetector()
    first = scan_session(session_dir, [det])["detectors"]["fake-detector"]
    second = scan_session(session_dir, [det])["detectors"]["fake-detector"]
    assert det.run_count == 1          # fresh → not re-run
    assert second["run_at"] == first["run_at"]


def test_scan_force_reruns_even_when_fresh(session_dir):
    det = FakeDetector()
    scan_session(session_dir, [det])
    scan_session(session_dir, [det], force=True)
    assert det.run_count == 2           # force re-runs regardless of fingerprint


def test_scan_reruns_when_transcript_changes(session_dir):
    det = FakeDetector()
    scan_session(session_dir, [det])
    old_fp = transcript_fingerprint(session_dir)
    (session_dir / "transcript-rich.json").write_text(
        json.dumps({"segments": [], "changed": True}))
    data = scan_session(session_dir, [det])
    assert det.run_count == 2
    new_fp = data["detectors"]["fake-detector"]["transcript_fingerprint"]
    assert new_fp == transcript_fingerprint(session_dir) and new_fp != old_fp


def test_scan_preserves_unregistered_sections(session_dir):
    (session_dir / "detections.json").write_text(json.dumps({
        "_about": "x",
        "detectors": {"other-detector": {"n_flags": 7, "flags": []}},
    }))
    data = scan_session(session_dir, [FakeDetector()])
    assert data["detectors"]["other-detector"]["n_flags"] == 7
    assert data["detectors"]["fake-detector"]["n_flags"] == 1


def test_scan_requires_transcript(tmp_path):
    with pytest.raises(FileNotFoundError, match="transcript-rich.json"):
        scan_session(tmp_path, [FakeDetector()])


def test_scan_corrupt_detections_fails_loud(session_dir):
    (session_dir / "detections.json").write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        scan_session(session_dir, [FakeDetector()])


def test_scan_reruns_when_config_changes(session_dir):
    class ConfigDetector(FakeDetector):
        def __init__(self):
            super().__init__()
            self.config = "v1"

        def config_fingerprint(self):
            return f"sha256:{self.config}"

    det = ConfigDetector()
    scan_session(session_dir, [det])
    scan_session(session_dir, [det])
    assert det.run_count == 1
    det.config = "v2"
    scan_session(session_dir, [det])
    assert det.run_count == 2


def test_scan_passes_judge_to_accepting_detector(session_dir):
    seen = {}

    class JudgeDetector(FakeDetector):
        accepts_judge = True

        def run(self, session_dir, judge=None):
            self.run_count += 1
            seen["judge"] = judge
            return {"n_word_tokens": 1, "flags": []}

    det = JudgeDetector()
    marker = lambda c: set()
    data = scan_session(session_dir, [det], judge=marker)
    assert seen["judge"] is marker
    assert data["detectors"]["fake-detector"]["judge_applied"] is True


def test_scan_withholds_judge_from_plain_detector(session_dir):
    # FakeDetector.run takes no judge param; scan must not pass one (accepts_judge=False).
    det = FakeDetector()
    data = scan_session(session_dir, [det], judge=lambda c: set())
    assert data["detectors"]["fake-detector"]["judge_applied"] is False


# --- offline_only gate (keeps the slow per-story name auditor out of web requests) ---

class OfflineDetector(FakeDetector):
    id = "offline-detector"
    offline_only = True


def test_scan_skips_offline_only_without_run_offline(session_dir):
    det = OfflineDetector()
    data = scan_session(session_dir, [det])        # default run_offline=False
    assert det.run_count == 0                       # never invoked
    assert "offline-detector" not in data["detectors"]  # no section written


def test_scan_runs_offline_only_with_run_offline(session_dir):
    det = OfflineDetector()
    data = scan_session(session_dir, [det], run_offline=True)
    assert det.run_count == 1
    assert data["detectors"]["offline-detector"]["n_flags"] == 1


def test_scan_mixed_runs_normal_skips_offline(session_dir):
    # Mirrors the production registry: code detectors run, the offline auditor is
    # skipped unless explicitly opted into (the API scan routes never opt in).
    normal, offline = FakeDetector(), OfflineDetector()
    data = scan_session(session_dir, [normal, offline])
    assert normal.run_count == 1 and offline.run_count == 0
    assert "fake-detector" in data["detectors"]
    assert "offline-detector" not in data["detectors"]


def test_section_is_stale(session_dir):
    det = FakeDetector()
    scan_session(session_dir, [det])
    section = json.loads((session_dir / "detections.json").read_text())["detectors"]["fake-detector"]
    assert section_is_stale(section, session_dir) is False
    (session_dir / "transcript-rich.json").write_text(json.dumps({"segments": [], "x": 1}))
    assert section_is_stale(section, session_dir) is True
