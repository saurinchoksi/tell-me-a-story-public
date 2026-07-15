"""Tests for the detections (monitor) API routes.

Viewing is read-only — GET routes serve whatever the last scan wrote and
never run a detector; scans happen only via the POST scan routes. Tests
inject a fake detector via create_app(detectors=...) so they never need
the gitignored name roster.
"""

import json

import pytest

from api.app import create_app
from detectors.base import Detector, transcript_fingerprint

TRANSCRIPT = {
    "audio": {"duration_seconds": 60.0},
    "_stories": [
        {"index": 0, "start_id": 0, "end_id": 0, "title": "The Marda Mix-up", "world": ""},
    ],
    "segments": [
        {
            "id": 0,
            "start": 1.0,
            "end": 4.0,
            "text": " And then Martha smiled.",
            "_speaker": {"label": "SPEAKER_00", "source": "dominant"},
            "words": [{"word": " And"}, {"word": " then"},
                      {"word": " Martha"}, {"word": " smiled."}],
        },
    ],
}

# Canned section pre-written for the "scanned" session. Same detector id as
# the fake so the freshness gate applies; the token differs from the fake's
# output ("Martha" canned vs "Marda" fresh) so tests can tell which served.
CANNED_FLAG = {
    "segment_id": 0,
    "word_index": 2,
    "start": 2.0,
    "end": 2.4,
    "token": "Martha",
    "cleaned": "martha",
    "dm_codes": ["MR0", "MRT"],
    "match_type": "phonetic",
    "matched_person_ids": ["child"],
    "matched_canonicals": ["Marta"],
}


class _FakeDetector(Detector):
    id = "fake-detector"
    label = "Fake detector"
    failure_mode = "M0"
    version = "0.1.0"

    def __init__(self):
        self.run_count = 0

    def run(self, session_dir):
        self.run_count += 1
        return {"n_word_tokens": 4,
                "flags": [{**CANNED_FLAG, "token": "Marda", "cleaned": "marda"}]}


def _canned_detections(session_dir):
    return {
        "_about": "test",
        "detectors": {
            "fake-detector": {
                "label": "Fake detector",
                "failure_mode": "M0",
                "detector_version": "0.1.0",
                "run_at": "2026-06-11T12:00:00+00:00",
                "transcript_fingerprint": transcript_fingerprint(session_dir),
                "config_fingerprint": None,
                "n_word_tokens": 4,
                "n_flags": 1,
                "flags": [CANNED_FLAG],
            },
        },
    }


@pytest.fixture
def fake_detector():
    return _FakeDetector()


@pytest.fixture
def sessions_dir(tmp_path):
    # Scanned session: transcript + detections with a MATCHING fingerprint
    scanned = tmp_path / "sessions" / "20260101-120000"
    scanned.mkdir(parents=True)
    (scanned / "transcript-rich.json").write_text(json.dumps(TRANSCRIPT))
    (scanned / "detections.json").write_text(json.dumps(_canned_detections(scanned)))
    (scanned / "audio.m4a").write_bytes(b"fake audio")  # drives has_audio=True

    # Unscanned session: transcript only, no audio — auto-scanned on first view
    unscanned = tmp_path / "sessions" / "20260102-120000"
    unscanned.mkdir()
    (unscanned / "transcript-rich.json").write_text(json.dumps(TRANSCRIPT))

    # Untranscribed session: excluded from the rollup entirely
    raw = tmp_path / "sessions" / "20260103-120000"
    raw.mkdir()

    return tmp_path / "sessions"


@pytest.fixture
def client(sessions_dir, tmp_path, fake_detector):
    profiles_path = str(tmp_path / "profiles.json")
    app = create_app(sessions_dir=sessions_dir, profiles_path=profiles_path,
                     detectors=[fake_detector])
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# --- Rollup -------------------------------------------------------------------

def test_rollup_lists_injected_detectors(client):
    resp = client.get("/api/detections")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["detectors"] == [{"id": "fake-detector", "label": "Fake detector",
                                  "failure_mode": "M0", "version": "0.1.0"}]


def test_rollup_is_read_only(client, fake_detector):
    data = client.get("/api/detections").get_json()
    by_id = {s["session_id"]: s for s in data["sessions"]}
    # Untranscribed session excluded; newest first
    assert list(by_id) == ["20260102-120000", "20260101-120000"]
    assert fake_detector.run_count == 0                  # viewing never runs a detector
    assert by_id["20260101-120000"]["results"]["fake-detector"]["n_flags"] == 1  # canned
    assert by_id["20260102-120000"]["results"] == {}     # unscanned: empty, not auto-run
    assert by_id["20260101-120000"]["duration_seconds"] == 60.0
    assert by_id["20260101-120000"]["stale"] is False
    assert data["totals"]["fake-detector"] == 1


def test_rollup_includes_story_summary(client):
    """Each rollup session carries the derived story summary for the Monitor."""
    data = client.get("/api/detections").get_json()
    by_id = {s["session_id"]: s for s in data["sessions"]}
    st = by_id["20260101-120000"]["stories"]
    assert st is not None
    assert st["label"] == "The Marda Mix-up"
    assert st["n_stories"] == 1


def test_rollup_second_view_runs_nothing(client, fake_detector):
    client.get("/api/detections")
    runs_after_first = fake_detector.run_count
    client.get("/api/detections")
    assert fake_detector.run_count == runs_after_first


def test_rollup_marks_stale_on_transcript_change(client, fake_detector, sessions_dir):
    (sessions_dir / "20260101-120000" / "transcript-rich.json").write_text(
        json.dumps({**TRANSCRIPT, "_changed": True}))
    data = client.get("/api/detections").get_json()
    assert fake_detector.run_count == 0     # surfaced, not recomputed on view
    by_id = {s["session_id"]: s for s in data["sessions"]}
    assert by_id["20260101-120000"]["stale"] is True


def test_rollup_empty_sessions_dir(tmp_path, fake_detector):
    app = create_app(sessions_dir=tmp_path / "empty",
                     profiles_path=str(tmp_path / "profiles.json"),
                     detectors=[fake_detector])
    app.config["TESTING"] = True
    with app.test_client() as c:
        data = c.get("/api/detections").get_json()
    assert data["sessions"] == []
    assert [d["id"] for d in data["detectors"]] == ["fake-detector"]


def test_rollup_corrupt_detections_is_500(client, sessions_dir):
    (sessions_dir / "20260102-120000" / "detections.json").write_text("{nope")
    resp = client.get("/api/detections")
    assert resp.status_code == 500
    assert "Corrupt detections.json" in resp.get_json()["error"]


class _BrokenDetector(_FakeDetector):
    """Simulates detector setup failure (e.g. missing roster)."""

    def run(self, session_dir):
        raise FileNotFoundError("Family-name roster not found at data/name_roster.json.")


def _broken_client(sessions_dir, tmp_path):
    app = create_app(sessions_dir=sessions_dir,
                     profiles_path=str(tmp_path / "profiles.json"),
                     detectors=[_BrokenDetector()])
    app.config["TESTING"] = True
    return app.test_client()


# Detector failures now surface during a SCAN (POST), never a view (GET).

def test_scan_all_detector_failure_is_legible_500(sessions_dir, tmp_path):
    resp = _broken_client(sessions_dir, tmp_path).post("/api/detections/scan")
    assert resp.status_code == 500
    error = resp.get_json()["error"]
    assert "Scan failed on 20260102-120000" in error and "roster not found" in error


def test_scan_one_detector_failure_is_legible_500(sessions_dir, tmp_path):
    resp = _broken_client(sessions_dir, tmp_path).post(
        "/api/sessions/20260102-120000/detections/scan")
    assert resp.status_code == 500
    assert "roster not found" in resp.get_json()["error"]


# --- Scan (POST) --------------------------------------------------------------

def test_scan_one_runs_and_returns_detail(client, fake_detector):
    resp = client.post("/api/sessions/20260102-120000/detections/scan")
    assert resp.status_code == 200
    assert fake_detector.run_count == 1
    flag = resp.get_json()["detectors"]["fake-detector"]["flags"][0]
    assert flag["token"] == "Marda"                            # fresh run
    assert flag["segment_text"] == "And then Martha smiled."   # joined


def test_scan_one_force_reruns_fresh_session(client, fake_detector):
    client.post("/api/sessions/20260101-120000/detections/scan")
    assert fake_detector.run_count == 1   # force=True re-runs even the fresh canned session


def test_scan_all_runs_missing_skips_fresh(client, fake_detector):
    client.post("/api/detections/scan")
    # unscanned session ran; the canned (fresh) one was skipped
    assert fake_detector.run_count == 1


def test_scan_one_no_transcript_400(client):
    resp = client.post("/api/sessions/20260103-120000/detections/scan")
    assert resp.status_code == 400


# --- Per-session detail ---------------------------------------------------------

def test_session_detections_bad_id(client):
    assert client.get("/api/sessions/not-a-session/detections").status_code == 400


def test_session_detections_unknown_session(client):
    assert client.get("/api/sessions/20269999-000000/detections").status_code == 404


def test_session_detections_unscanned_reads_empty(client, fake_detector):
    resp = client.get("/api/sessions/20260102-120000/detections")
    assert resp.status_code == 200
    assert fake_detector.run_count == 0                       # viewing never runs
    assert resp.get_json()["detectors"] == {}                # never scanned → empty


def test_session_detections_reads_canned_with_stale_flag(client, fake_detector):
    data = client.get("/api/sessions/20260101-120000/detections").get_json()
    assert fake_detector.run_count == 0
    section = data["detectors"]["fake-detector"]
    assert section["run_at"] == "2026-06-11T12:00:00+00:00"   # canned, read as-is
    assert section["flags"][0]["token"] == "Martha"
    assert section["stale"] is False


def test_session_detections_joins_segment_context(client):
    data = client.get("/api/sessions/20260101-120000/detections").get_json()
    section = data["detectors"]["fake-detector"]
    assert section["n_flags"] == 1
    flag = section["flags"][0]
    assert flag["segment_text"] == "And then Martha smiled."
    assert flag["segment_start"] == 1.0
    assert flag["segment_end"] == 4.0
    assert flag["segment_speaker"] == "SPEAKER_00"


def test_session_detections_no_transcript_no_detections(client):
    resp = client.get("/api/sessions/20260103-120000/detections")
    assert resp.status_code == 200
    assert resp.get_json() == {"session_id": "20260103-120000", "has_audio": False,
                               "stories": None, "name_verdicts": [], "detectors": {}}


def test_session_detections_includes_story_summary(client):
    """The detail payload carries the story summary for the page header — even on
    a transcribed session that was never scanned (no detections.json)."""
    scanned = client.get("/api/sessions/20260101-120000/detections").get_json()
    assert scanned["stories"]["label"] == "The Marda Mix-up"
    assert scanned["stories"]["n_stories"] == 1

    unscanned = client.get("/api/sessions/20260102-120000/detections").get_json()
    assert unscanned["detectors"] == {}                      # never scanned
    assert unscanned["stories"]["label"] == "The Marda Mix-up"  # but still identifiable


def test_session_detections_reports_has_audio(client):
    # 20260101-120000 has an audio.m4a fixture; 20260102-120000 does not
    with_audio = client.get("/api/sessions/20260101-120000/detections").get_json()
    without_audio = client.get("/api/sessions/20260102-120000/detections").get_json()
    assert with_audio["has_audio"] is True
    assert without_audio["has_audio"] is False


def test_session_detections_missing_transcript_joins_null(client, sessions_dir,
                                                          fake_detector):
    (sessions_dir / "20260101-120000" / "transcript-rich.json").unlink()
    data = client.get("/api/sessions/20260101-120000/detections").get_json()
    assert fake_detector.run_count == 0        # nothing to scan — passive read
    flag = data["detectors"]["fake-detector"]["flags"][0]
    assert flag["segment_text"] is None
    assert flag["segment_speaker"] is None
    assert flag["token"] == "Martha"           # the flag itself is still real data


# --- canon de-dup (m9b defers to m9c) -----------------------------------------

def test_canon_dedup_suppresses_m9b_flag_owned_by_m9c():
    # A canon name the reader (m9c) flagged is dropped from m9b, so it shows only
    # under M9c; an improvised name m9c did not claim stays in m9b.
    from api.routes.detections import _apply_canon_dedup
    sections = {
        "m9b-name-consistency": {"n_flags": 2, "flags": [
            {"cleaned": "pondavas", "token": "Pondavas", "cluster_id": "pondavas"},  # canon — m9c owns it
            {"cleaned": "jammus", "token": "Jammus", "cluster_id": "jammus"},         # improvised — m9b keeps
        ]},
        "m9c-canon": {"n_flags": 1, "flags": [
            {"cleaned": "pondavas", "wrong_cleaned": ["pondavas"], "canonical": "Pandavas"},
        ]},
    }
    _apply_canon_dedup(sections)
    m9b = sections["m9b-name-consistency"]
    assert m9b["n_flags"] == 1
    assert [f["cleaned"] for f in m9b["flags"]] == ["jammus"]


def test_canon_dedup_drops_whole_cluster_when_m9c_owns_one_token():
    # Regression for the split-name bug: M9b's unit is the CLUSTER. If M9c owns even ONE token of a
    # cluster, the whole cluster defers to M9c — so a canon name never appears half in M9b (the
    # stranded "james"/"jamis" remnant after M9c took the other James spellings).
    from api.routes.detections import _apply_canon_dedup
    sections = {
        "m9b-name-consistency": {"n_flags": 4, "flags": [
            {"cleaned": "jameis", "token": "Jameis", "cluster_id": "jameis"},  # m9c owns this token
            {"cleaned": "james", "token": "James", "cluster_id": "jameis"},     # same cluster -> also defers
            {"cleaned": "jamis", "token": "Jamis", "cluster_id": "jameis"},     # same cluster -> also defers
            {"cleaned": "jiraki", "token": "Jiraki", "cluster_id": "jiraki"},   # separate cluster -> stays
        ]},
        "m9c-canon": {"n_flags": 1, "flags": [
            {"cleaned": "jameis", "wrong_cleaned": ["jameis"], "canonical": "James"},
        ]},
    }
    _apply_canon_dedup(sections)
    m9b = sections["m9b-name-consistency"]
    assert m9b["n_flags"] == 1
    assert [f["cleaned"] for f in m9b["flags"]] == ["jiraki"]  # the whole James cluster deferred to M9c


def test_canon_tier_derives_three_tiers():
    # confident = sounds alike; best_guess = not sound-alike but judge vote >= BEST_GUESS_VOTE_MIN
    # (3, = the judge's catch floor); low = below that (structurally empty now, no catch votes < 3).
    from api.helpers import canon_tier
    assert canon_tier({"suggestion_confident": True, "vote_count": 7}) == "confident"
    assert canon_tier({"suggestion_confident": True}) == "confident"                      # confident wins
    assert canon_tier({"suggestion_confident": False, "vote_count": 5}) == "best_guess"
    assert canon_tier({"suggestion_confident": False, "vote_count": 3}) == "best_guess"   # at the floor
    assert canon_tier({"suggestion_confident": False, "vote_count": 2}) == "low"
    assert canon_tier({"suggestion_confident": False}) == "low"                           # no votes -> low


def test_annotate_canon_tiers_tags_flags_and_counts_default_visible():
    # Each M9c flag gets a tier; n_flags becomes the DEFAULT-VISIBLE count (confident + best_guess),
    # while the 'low' flag stays in the list (the UI reveals it under "Show all").
    from api.helpers import annotate_canon_tiers
    sections = {"m9c-canon": {"n_flags": 4, "flags": [
        {"cleaned": "pondavas", "canonical": "Pandavas", "suggestion_confident": True},
        {"cleaned": "dhrashtra", "canonical": "Dhritarashtra", "suggestion_confident": False, "vote_count": 5},
        {"cleaned": "yudhisthir", "canonical": "Yudhishthira", "suggestion_confident": False, "vote_count": 2},
        {"cleaned": "duryodhan", "canonical": "Duryodhana", "suggestion_confident": True},
    ]}}
    annotate_canon_tiers(sections)
    flags = sections["m9c-canon"]["flags"]
    assert [f["tier"] for f in flags] == ["confident", "best_guess", "low", "confident"]
    assert sections["m9c-canon"]["n_flags"] == 3   # the 'low' yudhisthir is not counted in the badge
    assert len(flags) == 4                          # but it stays in the payload for "Show all"


def test_canon_dedup_skips_low_tier_m9c_claim():
    # A LOW-confidence M9c claim must NOT hide a real M9b inconsistency (a confident/best-guess one
    # still does — covered above). Tiers are annotated before the dedup runs.
    from api.helpers import annotate_canon_tiers
    from api.routes.detections import _apply_canon_dedup
    sections = {
        "m9b-name-consistency": {"n_flags": 2, "flags": [
            {"cleaned": "pondavas", "token": "Pondavas", "cluster_id": "pondavas"},   # owned by a CONFIDENT m9c
            {"cleaned": "yudister", "token": "Yudister", "cluster_id": "yudister"},   # only a LOW m9c claim -> stays
        ]},
        "m9c-canon": {"n_flags": 2, "flags": [
            {"cleaned": "pondavas", "wrong_cleaned": ["pondavas"], "canonical": "Pandavas",
             "suggestion_confident": True},
            {"cleaned": "yudister", "wrong_cleaned": ["yudister"], "canonical": "Yudhishthira",
             "suggestion_confident": False, "vote_count": 2},   # low tier
        ]},
    }
    annotate_canon_tiers(sections)
    _apply_canon_dedup(sections)
    m9b = sections["m9b-name-consistency"]
    assert [f["cleaned"] for f in m9b["flags"]] == ["yudister"]   # the low-tier claim did NOT defer it
    assert m9b["n_flags"] == 1


def test_apply_not_canon_drops_m9c_and_releases_m9b():
    # The core of the feature: a 'not canon' verdict drops the M9c claim, and because it
    # runs BEFORE the dedup, M9b's (correct) inconsistency is no longer deferred away.
    from api.helpers import apply_name_verdicts
    from api.routes.detections import _apply_canon_dedup
    sections = {
        "m9b-name-consistency": {"n_flags": 3, "flags": [
            {"cleaned": "james", "token": "James", "cluster_id": "jameis"},
            {"cleaned": "jameis", "token": "Jameis", "cluster_id": "jameis"},
            {"cleaned": "jammus", "token": "Jammus", "cluster_id": "jameis"},
        ]},
        "m9c-canon": {"n_flags": 2, "flags": [
            {"cleaned": "jameis", "canonical": "James", "wrong_cleaned": ["jameis"]},
            {"cleaned": "jammus", "canonical": "James", "wrong_cleaned": ["jammus"]},
        ]},
    }
    apply_name_verdicts(sections, [{"type": "not_canon", "name": "James"}])
    assert sections["m9c-canon"]["n_flags"] == 0          # dropped from M9c
    _apply_canon_dedup(sections)                          # M9c now owns nothing -> no deferral
    assert sections["m9b-name-consistency"]["n_flags"] == 3  # the inconsistency surfaces under M9b


def test_apply_correct_suppresses_spelling_in_m9b_m9c_but_not_m9a():
    from api.helpers import apply_name_verdicts
    sections = {
        "m9a-family-names": {"n_flags": 1, "flags": [
            {"cleaned": "james", "token": "James", "matched_canonicals": ["James"]}]},
        "m9b-name-consistency": {"n_flags": 3, "flags": [
            {"cleaned": "james", "token": "James", "cluster_id": "jameis",
             "cluster_spellings": ["James", "Jameis", "Jammus"]},
            {"cleaned": "jameis", "token": "Jameis", "cluster_id": "jameis",
             "cluster_spellings": ["James", "Jameis", "Jammus"]},
            {"cleaned": "jammus", "token": "Jammus", "cluster_id": "jameis",
             "cluster_spellings": ["James", "Jameis", "Jammus"]},
        ]},
        "m9c-canon": {"n_flags": 1, "flags": [{"cleaned": "james", "canonical": "James"}]},
    }
    apply_name_verdicts(sections, [{"type": "correct", "cleaned": "james"}])
    # 'james' rows gone from m9b and m9c...
    assert [f["cleaned"] for f in sections["m9b-name-consistency"]["flags"]] == ["jameis", "jammus"]
    assert sections["m9c-canon"]["n_flags"] == 0
    # ...the un-gated family detector (m9a) is left alone...
    assert sections["m9a-family-names"]["n_flags"] == 1
    # ...and the corrected spelling stops being listed in the cluster header.
    assert all("James" not in f["cluster_spellings"] for f in sections["m9b-name-consistency"]["flags"])


def test_name_verdict_status_marks_stale_on_spelling_drift():
    from api.helpers import name_verdict_status
    sections = {"m9c-canon": {"flags": [
        {"cleaned": "jameis", "canonical": "James"},
        {"cleaned": "jammus", "canonical": "James"},
    ]}}
    fresh = name_verdict_status(
        sections, [{"type": "not_canon", "name": "James", "cleaned_at_review": ["jameis", "jammus"]}])
    assert fresh[0]["stale"] is False
    drifted = name_verdict_status(
        sections, [{"type": "not_canon", "name": "James", "cleaned_at_review": ["jameis"]}])
    assert drifted[0]["stale"] is True


def test_name_verdict_post_toggles_and_persists(client, sessions_dir):
    sd = sessions_dir / "20260101-120000"
    r = client.post("/api/sessions/20260101-120000/name-verdicts",
                    json={"type": "not_canon", "name": "James", "cleaned_at_review": ["jammus"]})
    assert r.status_code == 200
    saved = json.loads((sd / "name-verdicts.json").read_text())["verdicts"]
    assert len(saved) == 1 and saved[0]["name"] == "James" and "at" in saved[0]
    assert any(v["name"] == "James" for v in r.get_json()["name_verdicts"])  # echoed in detail

    # Re-sending the same verdict toggles it off (undo).
    client.post("/api/sessions/20260101-120000/name-verdicts",
                json={"type": "not_canon", "name": "James"})
    assert json.loads((sd / "name-verdicts.json").read_text())["verdicts"] == []


def test_name_verdict_post_validates(client):
    assert client.post("/api/sessions/20260101-120000/name-verdicts",
                       json={"type": "correct"}).status_code == 400        # missing 'cleaned'
    assert client.post("/api/sessions/20260101-120000/name-verdicts",
                       json={"type": "bogus", "name": "x"}).status_code == 400


def test_canon_dedup_noop_when_m9c_found_nothing():
    # When the canon reader found no canon (e.g. it didn't recognize the world),
    # m9b keeps its catches — they are the only thing flagging those names.
    from api.routes.detections import _apply_canon_dedup
    sections = {
        "m9b-name-consistency": {"n_flags": 2, "flags": [
            {"cleaned": "bishma", "token": "Bishma"},
            {"cleaned": "garn", "token": "Garn"},
        ]},
        "m9c-canon": {"n_flags": 0, "flags": []},
    }
    _apply_canon_dedup(sections)
    assert sections["m9b-name-consistency"]["n_flags"] == 2
