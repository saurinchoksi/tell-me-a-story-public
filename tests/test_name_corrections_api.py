"""Tests for the labeled shadow queue API (calibration: record-only verdicts, gid identity)."""
import json

import pytest

import namefix
from api.app import create_app


@pytest.fixture
def client(tmp_path):
    profiles_path = str(tmp_path / "profiles.json")
    app = create_app(sessions_dir=tmp_path, profiles_path=profiles_path,
                     name_book_dir=tmp_path / "name-book")
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


SID = "20260101-120000"

TRANSCRIPT = {"text": "", "segments": [
    {"id": 0, "text": " the Bandos were good",
     "words": [
         {"word": " the", "start": 0.0, "end": 0.2, "probability": 0.9},
         {"word": " Bandos", "start": 0.2, "end": 0.6, "probability": 0.9},
         {"word": " were", "start": 0.6, "end": 0.8, "probability": 0.9},
         {"word": " good", "start": 0.8, "end": 1.0, "probability": 0.9},
     ]},
    {"id": 2, "text": " Bandos.",
     "words": [{"word": " Bandos.", "start": 2.0, "end": 2.4, "probability": 0.9}]},
]}


def read_book(tmp_path, world_slug):
    path = tmp_path / "name-book" / f"{world_slug}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def make_session(tmp_path, pending=None):
    sdir = tmp_path / SID
    sdir.mkdir()
    (sdir / "transcript-rich.json").write_text(json.dumps(TRANSCRIPT))
    if pending is not None:
        (sdir / "pending-name-corrections.json").write_text(json.dumps(pending))
    return sdir


GID = "0:bandos:pandavas"


def group_fixture(**over):
    g = {
        "gid": GID,
        "world": "Mahabharata", "story_id": 0, "heard": "Bandos",
        "heard_cleaned": "bandos", "canonical": "Pandavas",
        "method": "exact", "label": "audio-only", "dm_share": False,
        "echo_in_sentence": True, "n_sibling_spellings": 0,
        "occurrences": [
            {"segment_id": 0, "word_index": 1, "start": 0.2, "token": "Bandos"},
            {"segment_id": 2, "word_index": 0, "start": 2.0, "token": "Bandos."},
        ],
    }
    g.update(over)
    return g


def pending_fixture(groups=None, stale=False):
    """A queue file stamped against TRANSCRIPT (fresh) — or deliberately stale."""
    return {
        "_about": "test", "namefix_version": "2.1.0",
        "config_fingerprint": "cfg",
        "transcript_fingerprint": ("drifted" if stale
                                   else namefix.transcript_fingerprint(TRANSCRIPT)),
        "run_at": "2026-07-02T00:00:00Z",
        "worlds": [{"story_id": 0, "recognized_world": "Mahabharata"}],
        "groups": groups if groups is not None else [group_fixture()],
        "_rejected": [],
    }


def test_rollup_grouped_by_label_world_name(client, tmp_path):
    make_session(tmp_path, pending_fixture())
    r = client.get("/api/name-corrections")
    assert r.status_code == 200
    data = r.get_json()
    assert data["n_pending_groups"] == 1
    assert data["n_suppressed_groups"] == 0 and data["n_covered_by_yes"] == 0
    assert data["stale_sessions"] == []
    section = data["labels"][0]
    assert section["label"] == "audio-only"
    assert section["worlds"][0]["world"] == "Mahabharata"
    name = section["worlds"][0]["names"][0]
    assert name["heard_cleaned"] == "bandos" and name["canonical"] == "Pandavas"
    assert name["label"] == "audio-only" and name["echo_in_sentence"] is True
    sess = name["sessions"][0]
    assert sess["session_id"] == SID and sess["gid"] == GID and sess["stale"] is False
    assert len(sess["occurrences"]) == 2
    # occurrences enriched with sentence context
    assert "segment_text" in sess["occurrences"][0]


def test_rollup_label_sections_ordered(client, tmp_path):
    groups = [
        group_fixture(),
        group_fixture(gid="0:bushma:bhishma", heard="Bushma", heard_cleaned="bushma",
                      canonical="Bhishma", label="sounds-like", dm_share=True,
                      echo_in_sentence=False,
                      occurrences=[{"segment_id": 0, "word_index": 2, "start": 0.6,
                                    "token": "were"}]),
    ]
    make_session(tmp_path, pending_fixture(groups))
    data = client.get("/api/name-corrections").get_json()
    assert [s["label"] for s in data["labels"]] == ["sounds-like", "audio-only"]


def test_rollup_two_suggestions_one_word_are_distinct_items(client, tmp_path):
    """The Jammus case: one heard word, two suggestions -> two independent items, each
    actionable by its own gid."""
    groups = [
        group_fixture(gid="0:bandos:pandavas", canonical="Pandavas"),
        group_fixture(gid="0:bandos:bhishma", canonical="Bhishma", label="sounds-like",
                      occurrences=[{"segment_id": 0, "word_index": 1, "start": 0.2,
                                    "token": "Bandos"}]),
    ]
    make_session(tmp_path, pending_fixture(groups))
    data = client.get("/api/name-corrections").get_json()
    assert data["n_pending_groups"] == 2
    gids = {n["sessions"][0]["gid"]
            for sec in data["labels"] for w in sec["worlds"] for n in w["names"]}
    assert gids == {"0:bandos:pandavas", "0:bandos:bhishma"}


def test_rollup_fails_loud_on_legacy_shape(client, tmp_path):
    """A queue file without 'groups' (pre-2.0 'pending' schema, or corrupt) must scream,
    not read as an empty queue (CLAUDE.md: Fail Loud). Under TESTING Flask re-raises;
    in production this surfaces as a loud 500."""
    make_session(tmp_path, {"pending": [group_fixture()], "_rejected": []})
    with pytest.raises(KeyError):
        client.get("/api/name-corrections")


def test_rollup_empty_when_no_questions(client, tmp_path):
    make_session(tmp_path, pending=None)
    r = client.get("/api/name-corrections")
    assert r.status_code == 200
    assert r.get_json()["n_pending_groups"] == 0


def test_rollup_reports_family_fixer_status(client, tmp_path):
    """ the rollup says whether the family side RAN per session — checked (a
    metadata entry with no 'status'), skipped (with the reason), or unknown (queue
    files from before skip-recording AND sessions with no queue file at all) — plus
    the LIVE family question count."""
    # session 1: familyfix recorded a skip (no roster)
    pending1 = pending_fixture()
    pending1["fixers"] = {"family": {"status": "skipped", "reason": "no_roster",
                                     "version": "1.0.0",
                                     "run_at": "2026-07-03T00:00:00Z"}}
    make_session(tmp_path, pending1)
    # session 2: familyfix ran and asked one (live) question
    sdir2 = tmp_path / "20260102-120000"
    sdir2.mkdir()
    (sdir2 / "transcript-rich.json").write_text(json.dumps(TRANSCRIPT))
    pending2 = pending_fixture(groups=[group_fixture(
        gid="family:bandos:marta", fixer="family", world="Family")])
    pending2["fixers"] = {"family": {"version": "1.0.0", "config_fingerprint": "fp",
                                     "run_at": "2026-07-03T00:00:00Z"}}
    (sdir2 / "pending-name-corrections.json").write_text(json.dumps(pending2))
    # session 3: a queue from before skip-recording (no fixers dict at all)
    sdir3 = tmp_path / "20260103-120000"
    sdir3.mkdir()
    (sdir3 / "transcript-rich.json").write_text(json.dumps(TRANSCRIPT))
    (sdir3 / "pending-name-corrections.json").write_text(
        json.dumps(pending_fixture(groups=[])))
    # session 4: processed but the fixers never wrote a queue file — must NOT be
    # invisible (that invisibility was the original hole)
    sdir4 = tmp_path / "20260104-120000"
    sdir4.mkdir()
    (sdir4 / "transcript-rich.json").write_text(json.dumps(TRANSCRIPT))

    ff = client.get("/api/name-corrections").get_json()["family_fixer"]
    assert ff == {"checked": 1, "skipped": 1, "stale": 0, "unknown": 2,
                  "questions": 1, "skip_reasons": ["no_roster"]}


def test_rollup_family_fixer_stale_is_not_checked(client, tmp_path):
    """A queue whose transcript changed since the family run is 'stale', not
    'checked' — the current text was never checked — and its drifted family
    questions don't inflate the open count."""
    pending = pending_fixture(groups=[group_fixture(
        gid="family:bandos:marta", fixer="family", world="Family")], stale=True)
    pending["fixers"] = {"family": {"version": "1.0.0", "config_fingerprint": "fp",
                                    "run_at": "2026-07-03T00:00:00Z"}}
    make_session(tmp_path, pending)
    ff = client.get("/api/name-corrections").get_json()["family_fixer"]
    assert ff["stale"] == 1 and ff["checked"] == 0
    assert ff["questions"] == 0


def test_session_get_404_without_file(client, tmp_path):
    make_session(tmp_path, pending=None)
    assert client.get(f"/api/sessions/{SID}/name-corrections").status_code == 404


def test_bless_records_only_transcript_untouched(client, tmp_path):
    """Calibration: a yes is RECORDED (book + audit with label/signals) — the transcript
    stays byte-identical. Nothing applies until the apply pass."""
    sdir = make_session(tmp_path, pending_fixture())
    before = (sdir / "transcript-rich.json").read_text()
    r = client.post(f"/api/sessions/{SID}/name-corrections/bless", json={"gid": GID})
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "recorded" and body["verdict"] == "yes"
    assert body["applied_occurrences"] == 0 and body["n_pending"] == 0

    assert (sdir / "transcript-rich.json").read_text() == before   # SHADOW

    book = read_book(tmp_path, "mahabharata")
    assert book["yes"][0]["heard"] == "Bandos"
    assert book["yes"][0]["canonical"] == "Pandavas"

    p = json.loads((sdir / "pending-name-corrections.json").read_text())
    assert p["groups"] == []
    blessed = p["_blessed"][0]
    assert blessed["gid"] == GID and blessed["heard_cleaned"] == "bandos"
    # the audit entry is a calibration datapoint: label + signals ride with the verdict
    assert blessed["label"] == "audio-only" and blessed["echo_in_sentence"] is True
    assert blessed["partial"] is False and blessed["n_occurrences"] == 2


def test_bless_can_override_spelling(client, tmp_path):
    make_session(tmp_path, pending_fixture())
    r = client.post(f"/api/sessions/{SID}/name-corrections/bless",
                    json={"gid": GID, "canonical": "The Pandavas"})
    assert r.status_code == 200 and r.get_json()["canonical"] == "The Pandavas"
    assert read_book(tmp_path, "mahabharata")["yes"][0]["canonical"] == "The Pandavas"


def test_verdicts_refused_on_stale_queue(client, tmp_path):
    """A queue whose transcript changed since the namefix run may carry drifted positions:
    verdicts are refused (409) and the rollup flags the session."""
    make_session(tmp_path, pending_fixture(stale=True))
    data = client.get("/api/name-corrections").get_json()
    assert data["stale_sessions"] == [SID]
    assert data["labels"][0]["worlds"][0]["names"][0]["sessions"][0]["stale"] is True
    for verb, body in (("bless", {"gid": GID}), ("reject", {"gid": GID})):
        r = client.post(f"/api/sessions/{SID}/name-corrections/{verb}", json=body)
        assert r.status_code == 409, verb
    # nothing recorded anywhere
    p = json.loads((tmp_path / SID / "pending-name-corrections.json").read_text())
    assert len(p["groups"]) == 1 and "_blessed" not in p
    assert read_book(tmp_path, "mahabharata") is None


def test_reject_records_word_no_transcript_untouched(client, tmp_path):
    sdir = make_session(tmp_path, pending_fixture())
    before = (sdir / "transcript-rich.json").read_text()
    r = client.post(f"/api/sessions/{SID}/name-corrections/reject", json={"gid": GID})
    assert r.status_code == 200 and r.get_json()["n_pending"] == 0
    assert (sdir / "transcript-rich.json").read_text() == before
    p = json.loads((sdir / "pending-name-corrections.json").read_text())
    assert p["_rejected"][0]["gid"] == GID
    assert p["_rejected"][0]["label"] == "audio-only"   # the datapoint keeps its label
    book = read_book(tmp_path, "mahabharata")
    assert book["no"][0]["heard_cleaned"] == "bandos"
    assert "canonical_cleaned" not in book["no"][0]     # word-level


def test_standing_no_folds_question(client, tmp_path):
    """The quiet-but-viewable rule: a question matching a Name Book no leaves the live
    queue and rides in `suppressed` — whatever the current suggestion is (word-level)."""
    import namebook
    make_session(tmp_path, pending_fixture([group_fixture(canonical="Arjuna",
                                                          gid="0:bandos:arjuna")]))
    namebook.reject("Mahabharata", "bandos", base=tmp_path / "name-book")

    data = client.get("/api/name-corrections").get_json()
    assert data["n_pending_groups"] == 0
    assert data["n_suppressed_groups"] == 1
    assert data["labels"] == []
    sup = data["suppressed"][0]
    assert sup["world"] == "Mahabharata"
    assert sup["names"][0]["heard_cleaned"] == "bandos"

    s = client.get(f"/api/sessions/{SID}/name-corrections").get_json()
    assert s["groups"] == [] and s["suppressed"][0]["heard_cleaned"] == "bandos"

    # the file on disk is untouched — the fold is view-time only
    p = json.loads((tmp_path / SID / "pending-name-corrections.json").read_text())
    assert len(p["groups"]) == 1


def test_standing_yes_folds_question_even_with_punctuated_heard(client, tmp_path):
    """A standing yes folds the question into covered_by_yes — including when the blessed
    heard form carries punctuation clean() strips (the lowered-vs-cleaned mismatch)."""
    import namebook
    make_session(tmp_path, pending_fixture())
    namebook.bless("Mahabharata", "Bandos'", "Pandavas", base=tmp_path / "name-book")

    data = client.get("/api/name-corrections").get_json()
    assert data["n_pending_groups"] == 0
    assert data["n_covered_by_yes"] == 1
    assert data["covered_by_yes"][0]["names"][0]["heard_cleaned"] == "bandos"

    s = client.get(f"/api/sessions/{SID}/name-corrections").get_json()
    assert s["groups"] == [] and s["covered_by_yes"][0]["heard_cleaned"] == "bandos"


def test_lifting_a_yes_restores_the_question(client, tmp_path):
    """The view-time promise: folds are computed live, so removing a yes brings its
    questions straight back — no stale label may pin them folded."""
    import namebook
    make_session(tmp_path, pending_fixture())
    namebook.bless("Mahabharata", "Bandos", "Pandavas", base=tmp_path / "name-book")
    assert client.get("/api/name-corrections").get_json()["n_covered_by_yes"] == 1
    namebook.unbless("Mahabharata", "Bandos", base=tmp_path / "name-book")
    data = client.get("/api/name-corrections").get_json()
    assert data["n_covered_by_yes"] == 0 and data["n_pending_groups"] == 1


def test_unrule_yes_restores_question_and_book(client, tmp_path):
    """Undo a recorded yes: the audit entry pops, the book yes is lifted, the question
    returns to the queue — the reviewer's exact pre-tap state."""
    sdir = make_session(tmp_path, pending_fixture())
    client.post(f"/api/sessions/{SID}/name-corrections/bless", json={"gid": GID})
    assert read_book(tmp_path, "mahabharata")["yes"]     # ruled
    data = client.get("/api/name-corrections").get_json()
    assert data["n_pending_groups"] == 0 and data["n_ruled"] == 1
    ruled = data["ruled"][0]
    assert ruled["verdict"] == "yes" and ruled["gid"] == GID and ruled["undoable"] is True

    r = client.post(f"/api/sessions/{SID}/name-corrections/unrule",
                    json={"verdict": "yes", "gid": GID})
    assert r.status_code == 200 and r.get_json()["status"] == "unruled"

    data = client.get("/api/name-corrections").get_json()
    assert data["n_pending_groups"] == 1 and data["n_ruled"] == 0
    assert read_book(tmp_path, "mahabharata")["yes"] == []
    p = json.loads((sdir / "pending-name-corrections.json").read_text())
    assert p["_blessed"] == [] and len(p["groups"][0]["occurrences"]) == 2


def test_unrule_no_restores_question_and_lifts_the_no(client, tmp_path):
    make_session(tmp_path, pending_fixture())
    client.post(f"/api/sessions/{SID}/name-corrections/reject", json={"gid": GID})
    import namebook
    assert namebook.load_no_words("Mahabharata", base=tmp_path / "name-book") == {"bandos"}

    r = client.post(f"/api/sessions/{SID}/name-corrections/unrule",
                    json={"verdict": "no", "gid": GID})
    assert r.status_code == 200
    assert namebook.load_no_words("Mahabharata", base=tmp_path / "name-book") == set()
    data = client.get("/api/name-corrections").get_json()
    assert data["n_pending_groups"] == 1 and data["n_suppressed_groups"] == 0


def test_unrule_partial_yes_merges_back(client, tmp_path):
    """Undoing a partial yes merges the ruled spots back into the still-queued remainder
    group (same gid) — one whole question again, no book to reverse."""
    make_session(tmp_path, pending_fixture())
    client.post(f"/api/sessions/{SID}/name-corrections/bless",
                json={"gid": GID, "canonical": "Arjuna",
                      "occurrences": [{"segment_id": 0, "word_index": 1}]})
    p = json.loads((tmp_path / SID / "pending-name-corrections.json").read_text())
    assert len(p["groups"][0]["occurrences"]) == 1       # remainder, same gid
    assert p["groups"][0]["gid"] == GID
    assert read_book(tmp_path, "mahabharata") is None    # partial never writes the book

    r = client.post(f"/api/sessions/{SID}/name-corrections/unrule",
                    json={"verdict": "yes", "gid": GID})
    assert r.status_code == 200
    p = json.loads((tmp_path / SID / "pending-name-corrections.json").read_text())
    assert len(p["groups"]) == 1                         # merged into the same-gid group
    assert len(p["groups"][0]["occurrences"]) == 2
    assert p["_blessed"] == []


def test_unrule_unknown_404_and_missing_fields_400(client, tmp_path):
    make_session(tmp_path, pending_fixture())
    r = client.post(f"/api/sessions/{SID}/name-corrections/unrule",
                    json={"verdict": "yes", "gid": "0:nope:nope"})
    assert r.status_code == 404
    r = client.post(f"/api/sessions/{SID}/name-corrections/unrule", json={"gid": GID})
    assert r.status_code == 400


def test_unreject_lifts_the_no(client, tmp_path):
    import namebook
    make_session(tmp_path, pending_fixture())
    namebook.reject("Mahabharata", "bandos", base=tmp_path / "name-book")
    assert client.get("/api/name-corrections").get_json()["n_pending_groups"] == 0
    r = client.post("/api/name-book/unreject",
                    json={"world": "Mahabharata", "heard_cleaned": "bandos"})
    assert r.status_code == 200
    assert client.get("/api/name-corrections").get_json()["n_pending_groups"] == 1


def test_unreject_unknown_no_404(client, tmp_path):
    r = client.post("/api/name-book/unreject",
                    json={"world": "Mahabharata", "heard_cleaned": "nope"})
    assert r.status_code == 404


def test_bless_unknown_gid_404(client, tmp_path):
    make_session(tmp_path, pending_fixture())
    r = client.post(f"/api/sessions/{SID}/name-corrections/bless",
                    json={"gid": "0:nope:nope"})
    assert r.status_code == 404


def test_bless_requires_gid(client, tmp_path):
    make_session(tmp_path, pending_fixture())
    r = client.post(f"/api/sessions/{SID}/name-corrections/bless",
                    json={"heard_cleaned": "bandos"})
    assert r.status_code == 400


def test_bless_per_occurrence_split(client, tmp_path):
    """The same-sound-two-referents case: rule on ONE spot with an override spelling; the
    other stays queued; a partial yes never writes the Name Book (the Bheem guard) — and
    still, nothing touches the transcript."""
    sdir = make_session(tmp_path, pending_fixture())
    before = (sdir / "transcript-rich.json").read_text()
    r = client.post(f"/api/sessions/{SID}/name-corrections/bless",
                    json={"gid": GID, "canonical": "Arjuna",
                          "occurrences": [{"segment_id": 0, "word_index": 1}]})
    assert r.status_code == 200
    assert r.get_json()["applied_occurrences"] == 0            # SHADOW

    assert (sdir / "transcript-rich.json").read_text() == before
    p = json.loads((sdir / "pending-name-corrections.json").read_text())
    assert len(p["groups"]) == 1                               # remainder re-queued
    assert p["groups"][0]["occurrences"][0]["segment_id"] == 2
    blessed = p["_blessed"][0]
    assert blessed["partial"] is True and blessed["canonical"] == "Arjuna"
    assert blessed["ruled_occurrences"] == [{"segment_id": 0, "word_index": 1}]
    assert read_book(tmp_path, "mahabharata") is None          # no book write on partial


def test_ambiguous_word_yes_teaches_nothing(client, tmp_path):
    """The word-level Bheem guard (found live on `arrows`): a yes on one suggestion while
    a SIBLING question for the same word is still queued records the verdict but writes
    no forever-rule — and the sibling stays live instead of folding."""
    groups = [
        group_fixture(gid="0:bandos:kauravas", canonical="Kauravas",
                      occurrences=[{"segment_id": 0, "word_index": 1, "start": 0.2,
                                    "token": "Bandos"}]),
        group_fixture(gid="0:bandos:arjuna", canonical="Arjuna",
                      occurrences=[{"segment_id": 2, "word_index": 0, "start": 2.0,
                                    "token": "Bandos."}]),
    ]
    make_session(tmp_path, pending_fixture(groups))
    r = client.post(f"/api/sessions/{SID}/name-corrections/bless",
                    json={"gid": "0:bandos:kauravas"})
    assert r.status_code == 200 and r.get_json()["book_written"] is False
    assert read_book(tmp_path, "mahabharata") is None          # no forever-rule

    data = client.get("/api/name-corrections").get_json()
    assert data["n_pending_groups"] == 1                       # sibling still LIVE
    assert data["n_covered_by_yes"] == 0                       # not folded away
    p = json.loads((tmp_path / SID / "pending-name-corrections.json").read_text())
    assert p["_blessed"][0]["book_written"] is False


def test_ambiguous_word_mixed_verdicts_stay_per_spot(client, tmp_path):
    """arrows end-to-end: no on one suggestion, then yes on the other — both record as
    per-spot data, the book learns nothing (the word is proven ambiguous), and the queue
    empties without folding anything."""
    groups = [
        group_fixture(gid="0:bandos:kauravas", canonical="Kauravas",
                      occurrences=[{"segment_id": 0, "word_index": 1, "start": 0.2,
                                    "token": "Bandos"}]),
        group_fixture(gid="0:bandos:arjuna", canonical="Arjuna",
                      occurrences=[{"segment_id": 2, "word_index": 0, "start": 2.0,
                                    "token": "Bandos."}]),
    ]
    make_session(tmp_path, pending_fixture(groups))
    r1 = client.post(f"/api/sessions/{SID}/name-corrections/reject",
                     json={"gid": "0:bandos:kauravas"})
    assert r1.get_json()["book_written"] is False              # sibling still queued
    r2 = client.post(f"/api/sessions/{SID}/name-corrections/bless",
                     json={"gid": "0:bandos:arjuna"})
    assert r2.get_json()["book_written"] is False              # prior no on the word
    book = read_book(tmp_path, "mahabharata")
    assert book is None                                        # nothing taught
    data = client.get("/api/name-corrections").get_json()
    assert data["n_pending_groups"] == 0 and data["n_ruled"] == 2


def test_reject_per_occurrence_split(client, tmp_path):
    """A no can be per-spot too: the checked spot records, the remainder stays queued,
    and no word-level no is written."""
    make_session(tmp_path, pending_fixture())
    r = client.post(f"/api/sessions/{SID}/name-corrections/reject",
                    json={"gid": GID,
                          "occurrences": [{"segment_id": 0, "word_index": 1}]})
    assert r.status_code == 200 and r.get_json()["book_written"] is False
    p = json.loads((tmp_path / SID / "pending-name-corrections.json").read_text())
    assert len(p["groups"]) == 1                               # remainder re-queued
    assert p["groups"][0]["occurrences"][0]["segment_id"] == 2
    rej = p["_rejected"][0]
    assert rej["partial"] is True and rej["book_written"] is False
    assert read_book(tmp_path, "mahabharata") is None


def test_unrule_ambiguous_yes_leaves_book_alone(client, tmp_path):
    """Undoing a verdict that never wrote the book must not lift someone else's rule."""
    import namebook
    groups = [
        group_fixture(gid="0:bandos:kauravas", canonical="Kauravas",
                      occurrences=[{"segment_id": 0, "word_index": 1, "start": 0.2,
                                    "token": "Bandos"}]),
        group_fixture(gid="0:bandos:arjuna", canonical="Arjuna",
                      occurrences=[{"segment_id": 2, "word_index": 0, "start": 2.0,
                                    "token": "Bandos."}]),
    ]
    make_session(tmp_path, pending_fixture(groups))
    # an unrelated standing yes for the same heard word, recorded directly in the book
    namebook.bless("Mahabharata", "Bandos", "Pandavas", base=tmp_path / "name-book")
    client.post(f"/api/sessions/{SID}/name-corrections/bless",
                json={"gid": "0:bandos:kauravas"})            # ambiguous -> no book write
    r = client.post(f"/api/sessions/{SID}/name-corrections/unrule",
                    json={"verdict": "yes", "gid": "0:bandos:kauravas"})
    assert r.status_code == 200
    # the pre-existing book yes survives the undo of a verdict that never wrote it
    assert read_book(tmp_path, "mahabharata")["yes"][0]["canonical"] == "Pandavas"


def test_bless_per_occurrence_no_match_400(client, tmp_path):
    make_session(tmp_path, pending_fixture())
    r = client.post(f"/api/sessions/{SID}/name-corrections/bless",
                    json={"gid": GID,
                          "occurrences": [{"segment_id": 99, "word_index": 0}]})
    assert r.status_code == 400
    # question not lost — still queued
    r2 = client.get(f"/api/sessions/{SID}/name-corrections")
    assert len(r2.get_json()["groups"]) == 1


# ------------------------- spelling questions (the M9b fixer) -------------------------
SPELL_GID = "spelling:jameis:jameis"


def spelling_group(**over):
    """An inconsistent-spelling question: the cluster Jameis/Jammus, majority Jammus,
    so the deviant spelling Jameis carries the question."""
    g = {
        "gid": SPELL_GID, "fixer": "spelling",
        "world": "Thomas & Friends", "story_id": None,
        "heard": "Jameis", "heard_cleaned": "jameis", "canonical": "Jammus",
        "method": "majority", "label": "inconsistent-spelling", "dm_share": True,
        "cluster_id": "jameis", "cluster_spellings": ["Jameis", "Jammus"],
        "echo_in_sentence": False, "n_sibling_spellings": 1,
        "occurrences": [
            {"segment_id": 0, "word_index": 1, "start": 0.2, "token": "Jameis"},
        ],
    }
    g.update(over)
    return g


def test_spelling_yes_records_cluster_pick_not_word_yes(client, tmp_path):
    """Yes on a spelling card -> the spellings slot (unify the cluster), never a word
    yes — spelling rulings live apart from word rulings by design."""
    make_session(tmp_path, pending_fixture([spelling_group()]))
    r = client.post(f"/api/sessions/{SID}/name-corrections/bless",
                    json={"gid": SPELL_GID})
    assert r.status_code == 200
    assert r.get_json()["book_written"] is True
    book = read_book(tmp_path, "thomas-friends")
    assert book["spellings"] == [{"cluster": "jameis", "spelling": "Jammus",
                                  "provenance": f"bless:{SID}",
                                  "at": book["spellings"][0]["at"]}]
    assert book["yes"] == [] and book["no"] == []


def test_spelling_yes_override_uses_edited_spelling(client, tmp_path):
    """Override-then-yes: he edits the proposed spelling before blessing — the pick
    records HIS spelling (there is no 'no + different spelling' gesture)."""
    make_session(tmp_path, pending_fixture([spelling_group()]))
    client.post(f"/api/sessions/{SID}/name-corrections/bless",
                json={"gid": SPELL_GID, "canonical": "Jameis"})
    book = read_book(tmp_path, "thomas-friends")
    assert book["spellings"][0]["spelling"] == "Jameis"


def test_spelling_no_is_dont_unify_never_a_word_no(client, tmp_path):
    """No on a spelling card = "don't unify these" — a null pick for the cluster. It
    must write NO word-level no (a spelling opinion never mutes other fixers' questions
    about the word — the arrows lesson)."""
    make_session(tmp_path, pending_fixture([spelling_group()]))
    r = client.post(f"/api/sessions/{SID}/name-corrections/reject",
                    json={"gid": SPELL_GID})
    assert r.status_code == 200
    assert r.get_json()["book_written"] is True
    book = read_book(tmp_path, "thomas-friends")
    assert book["spellings"][0] == {"cluster": "jameis", "spelling": None,
                                    "provenance": f"reject:{SID}",
                                    "at": book["spellings"][0]["at"]}
    assert book["no"] == []          # the load-bearing assertion
    audit = json.loads((tmp_path / SID / "pending-name-corrections.json").read_text())
    assert audit["_rejected"][0]["label"] == "inconsistent-spelling"
    assert audit["_rejected"][0]["fixer"] == "spelling"


def test_standing_pick_folds_cluster_questions_at_view_time(client, tmp_path):
    """A pick folds the cluster's questions live: unify -> covered_by_yes (showing the
    picked spelling), don't-unify -> suppressed. folded_by tells the UI which lift to
    offer. A word question about the same word is NOT folded by the pick."""
    import namebook
    word_q = group_fixture(gid="0:jameis:james", world="Thomas & Friends",
                           heard="Jameis", heard_cleaned="jameis", canonical="James")
    make_session(tmp_path, pending_fixture([spelling_group(), word_q]))
    namebook.pick_spelling("Thomas & Friends", "jameis", "Jammus",
                           base=tmp_path / "name-book")
    data = client.get("/api/name-corrections").get_json()
    assert data["n_pending_groups"] == 1          # the word question stays live
    assert data["n_covered_by_yes"] == 1
    folded = data["covered_by_yes"][0]["names"][0]
    assert folded["folded_by"] == "pick-unify" and folded["cluster_id"] == "jameis"
    assert folded["canonical"] == "Jammus"        # the picked spelling, not the proposal

    namebook.pick_spelling("Thomas & Friends", "jameis", None,
                           base=tmp_path / "name-book")
    data = client.get("/api/name-corrections").get_json()
    assert data["n_suppressed_groups"] == 1 and data["n_covered_by_yes"] == 0
    folded = data["suppressed"][0]["names"][0]
    assert folded["folded_by"] == "pick-leave"


def test_word_no_still_folds_spelling_question(client, tmp_path):
    """Standing word-level nos compose: a spelling question whose heard word is no'd
    folds under the no (his jammus/jameis nos keep working)."""
    import namebook
    make_session(tmp_path, pending_fixture([spelling_group()]))
    namebook.reject("Thomas & Friends", "jameis", base=tmp_path / "name-book")
    data = client.get("/api/name-corrections").get_json()
    assert data["n_suppressed_groups"] == 1
    assert data["suppressed"][0]["names"][0]["folded_by"] == "no"


def test_unrule_spelling_verdict_lifts_the_pick(client, tmp_path):
    """Undo on a spelling verdict pops the audit entry and removes the cluster pick
    (both yes- and no-kind picks reverse through unpick, never unbless/unreject)."""
    make_session(tmp_path, pending_fixture([spelling_group()]))
    client.post(f"/api/sessions/{SID}/name-corrections/reject", json={"gid": SPELL_GID})
    assert read_book(tmp_path, "thomas-friends")["spellings"]
    r = client.post(f"/api/sessions/{SID}/name-corrections/unrule",
                    json={"verdict": "no", "gid": SPELL_GID})
    assert r.status_code == 200
    assert read_book(tmp_path, "thomas-friends")["spellings"] == []
    data = client.get("/api/name-corrections").get_json()
    assert data["n_pending_groups"] == 1          # the question is back


def test_partial_spelling_verdict_is_audit_only(client, tmp_path):
    """Un-checking spots makes a spelling verdict partial -> recorded as calibration
    data, but no pick is written (the cluster ruling needs the whole card)."""
    g = spelling_group(occurrences=[
        {"segment_id": 0, "word_index": 1, "start": 0.2, "token": "Jameis"},
        {"segment_id": 2, "word_index": 0, "start": 2.0, "token": "Jameis."},
    ])
    make_session(tmp_path, pending_fixture([g]))
    r = client.post(f"/api/sessions/{SID}/name-corrections/bless",
                    json={"gid": SPELL_GID,
                          "occurrences": [{"segment_id": 0, "word_index": 1}]})
    assert r.status_code == 200
    assert r.get_json()["book_written"] is False
    book = read_book(tmp_path, "thomas-friends")
    assert book is None or book.get("spellings", []) == []


def test_unpick_spelling_endpoint(client, tmp_path):
    import namebook
    namebook.pick_spelling("Thomas & Friends", "jameis", "Jammus",
                           base=tmp_path / "name-book")
    r = client.post("/api/name-book/unpick-spelling",
                    json={"world": "Thomas & Friends", "cluster": "jameis"})
    assert r.status_code == 200
    assert read_book(tmp_path, "thomas-friends")["spellings"] == []
    r = client.post("/api/name-book/unpick-spelling",
                    json={"world": "Thomas & Friends", "cluster": "jameis"})
    assert r.status_code == 404
    r = client.post("/api/name-book/unpick-spelling", json={"world": "X"})
    assert r.status_code == 400


# ------------------------- the spot-answer fold (the third fold) -------------------------
def _regenerate(tmp_path, groups):
    """Simulate a fixer re-run: rewrite the queue's groups from scratch (same transcript),
    keeping the audits — exactly what write_queue does on a config-change re-run."""
    path = tmp_path / SID / "pending-name-corrections.json"
    data = json.loads(path.read_text())
    data["groups"] = groups
    path.write_text(json.dumps(data))


def test_arrows_case_mixed_per_spot_verdicts_fold_both_cards(client, tmp_path):
    """The case that started this: same word, two suggestions, ruled spot-by-spot
    (yes→Kauravas on one spot, no on the other). A ruled spot settles the WORD there,
    so after regeneration BOTH cards fold, each listing its covering verdicts."""
    spot_a = {"segment_id": 0, "word_index": 1, "start": 0.2, "token": "arrows"}
    spot_b = {"segment_id": 2, "word_index": 0, "start": 2.0, "token": "arrows."}
    g1 = group_fixture(gid="0:arrows:kauravas", heard="arrows", heard_cleaned="arrows",
                       canonical="Kauravas", label="ordinary-word",
                       occurrences=[dict(spot_a), dict(spot_b)])
    g2 = group_fixture(gid="0:arrows:arjuna", heard="arrows", heard_cleaned="arrows",
                       canonical="Arjuna", label="ordinary-word",
                       occurrences=[dict(spot_a), dict(spot_b)])
    make_session(tmp_path, pending_fixture([g1, g2]))
    r = client.post(f"/api/sessions/{SID}/name-corrections/bless",
                    json={"gid": "0:arrows:kauravas",
                          "occurrences": [{"segment_id": 0, "word_index": 1}]})
    assert r.status_code == 200 and r.get_json()["book_written"] is False
    r = client.post(f"/api/sessions/{SID}/name-corrections/reject",
                    json={"gid": "0:arrows:arjuna",
                          "occurrences": [{"segment_id": 2, "word_index": 0}]})
    assert r.status_code == 200 and r.get_json()["book_written"] is False

    # the re-run brings both full cards back...
    _regenerate(tmp_path, [g1, g2])
    data = client.get("/api/name-corrections").get_json()
    # ...and both fold: every spot is settled by a same-word ruled spot
    assert data["n_pending_groups"] == 0
    assert data["n_spot_answered"] == 2
    names = {n["canonical"]: n for w in data["spot_answered"] for n in w["names"]}
    assert set(names) == {"Kauravas", "Arjuna"}
    covering = names["Kauravas"]["sessions"][0]["covering"]
    assert {(c["verdict"], c["n_spots"]) for c in covering} == {("yes", 1), ("no", 1)}


def test_partial_coverage_stays_live_with_unruled_spots_only(client, tmp_path):
    """Strict rule: a re-run that finds a NEW spot brings the card back live, showing
    just the un-ruled spot, with the folded count riding along."""
    g = group_fixture()   # spots (0,1) and (2,0)
    make_session(tmp_path, pending_fixture([g]))
    # a PARTIAL yes on one spot only (writes no standing rule — the Bheem guard)
    r = client.post(f"/api/sessions/{SID}/name-corrections/bless",
                    json={"gid": GID,
                          "occurrences": [{"segment_id": 0, "word_index": 1}]})
    assert r.get_json()["book_written"] is False
    # re-run regenerates the card AND finds a third spot the fixer previously missed
    g3 = group_fixture(occurrences=g["occurrences"] + [
        {"segment_id": 0, "word_index": 2, "start": 0.6, "token": "Bandos"}])
    _regenerate(tmp_path, [g3])
    data = client.get("/api/name-corrections").get_json()
    assert data["n_pending_groups"] == 1 and data["n_spot_answered"] == 0
    sess = data["labels"][0]["worlds"][0]["names"][0]["sessions"][0]
    assert {(o["segment_id"], o["word_index"]) for o in sess["occurrences"]} \
        == {(2, 0), (0, 2)}   # the ruled spot (0,1) is folded out
    assert sess["n_spot_answered_occ"] == 1


def test_no_fold_across_transcript_change(client, tmp_path):
    """A verdict stamped under a different transcript version never folds anything —
    drifted positions don't get to act (the stale guard's instinct, applied to history)."""
    pending = pending_fixture([group_fixture()])
    pending["_blessed"] = [{
        "gid": GID, "heard_cleaned": "bandos", "canonical": "Pandavas",
        "transcript_fingerprint": "an-older-transcript",
        "ruled_occurrences": [{"segment_id": 0, "word_index": 1},
                              {"segment_id": 2, "word_index": 0}],
        "group": group_fixture(), "at": "t0"}]
    make_session(tmp_path, pending)
    data = client.get("/api/name-corrections").get_json()
    assert data["n_pending_groups"] == 1 and data["n_spot_answered"] == 0


def test_no_fold_for_a_different_word_at_the_same_spots(client, tmp_path):
    """Coverage matches the WORD at the spot — an answer about another word that happens
    to share positions covers nothing."""
    g = group_fixture()
    pending = pending_fixture([g])
    pending["_rejected"] = [{
        "gid": "0:other:pandavas", "heard_cleaned": "other",
        "transcript_fingerprint": namefix.transcript_fingerprint(TRANSCRIPT),
        "ruled_occurrences": [{"segment_id": 0, "word_index": 1},
                              {"segment_id": 2, "word_index": 0}],
        "group": g, "at": "t0"}]
    make_session(tmp_path, pending)
    data = client.get("/api/name-corrections").get_json()
    assert data["n_pending_groups"] == 1 and data["n_spot_answered"] == 0


def test_worldless_pick_folds_on_regeneration(client, tmp_path):
    """A world-less spelling yes writes no book entry (audit-only), so before this fold
    a re-run re-asked it forever. Now it folds under the recorded answer."""
    g = spelling_group(world="")
    make_session(tmp_path, pending_fixture([g]))
    r = client.post(f"/api/sessions/{SID}/name-corrections/bless",
                    json={"gid": SPELL_GID})
    assert r.get_json()["book_written"] is False
    _regenerate(tmp_path, [g])
    data = client.get("/api/name-corrections").get_json()
    assert data["n_pending_groups"] == 0 and data["n_spot_answered"] == 1
    folded = data["spot_answered"][0]["names"][0]
    assert folded["sessions"][0]["covering"][0]["verdict"] == "yes"


def test_verdicts_are_stamped_with_the_transcript_fingerprint(client, tmp_path):
    make_session(tmp_path, pending_fixture([group_fixture()]))
    client.post(f"/api/sessions/{SID}/name-corrections/bless", json={"gid": GID})
    data = json.loads((tmp_path / SID / "pending-name-corrections.json").read_text())
    assert (data["_blessed"][0]["transcript_fingerprint"]
            == namefix.transcript_fingerprint(TRANSCRIPT))


def test_undo_covering_verdict_frees_the_folded_card(client, tmp_path):
    """The inline undo: unruling a covering verdict pops the audit, so the fold
    (computed live) releases the card back to the queue on the next GET."""
    g = spelling_group(world="")
    make_session(tmp_path, pending_fixture([g]))
    client.post(f"/api/sessions/{SID}/name-corrections/bless", json={"gid": SPELL_GID})
    _regenerate(tmp_path, [g])
    assert client.get("/api/name-corrections").get_json()["n_spot_answered"] == 1
    r = client.post(f"/api/sessions/{SID}/name-corrections/unrule",
                    json={"verdict": "yes", "gid": SPELL_GID})
    assert r.status_code == 200
    data = client.get("/api/name-corrections").get_json()
    assert data["n_spot_answered"] == 0 and data["n_pending_groups"] == 1


# ------------------------- the frequency-asymmetry signal (the Marley lesson) -------------------------
def test_frequency_signal_rides_on_views_and_verdicts(client, tmp_path):
    """`Bandos` appears twice in TRANSCRIPT, "Pandavas" zero times -> the card carries
    2 vs 0, computed at view time (no re-run needed) and stamped into the audit when
    the verdict is recorded. Logged only — nothing gates on it."""
    make_session(tmp_path, pending_fixture([group_fixture()]))
    item = (client.get("/api/name-corrections").get_json()
            ["labels"][0]["worlds"][0]["names"][0])
    assert item["n_heard_in_transcript"] == 2
    assert item["n_suggested_in_transcript"] == 0
    client.post(f"/api/sessions/{SID}/name-corrections/reject", json={"gid": GID})
    audit = json.loads((tmp_path / SID / "pending-name-corrections.json").read_text())
    snap = audit["_rejected"][0]["group"]
    assert snap["n_heard_in_transcript"] == 2
    assert snap["n_suggested_in_transcript"] == 0


# ------------------------- the dictionary-corrected spot -------------------------
def _make_session_with(tmp_path, transcript, pending):
    """A session with a caller-supplied transcript (these tests need `_original` trails)."""
    sdir = tmp_path / SID
    sdir.mkdir()
    (sdir / "transcript-rich.json").write_text(json.dumps(transcript))
    (sdir / "pending-name-corrections.json").write_text(json.dumps(pending))
    return sdir


# A transcript where the dictionary rewrote "Arjun" -> "Arjuna" in place at seg 0 wi 0:
# the live word says "Arjuna" but keeps its pre-rewrite `_original` = "Arjun".
DICT_TRANSCRIPT = {"text": "", "segments": [
    {"id": 0, "text": " Arjuna drew his bow",
     "words": [
         {"word": " Arjuna", "start": 0.0, "end": 0.5, "probability": 0.9,
          "_original": "Arjun",
          "_corrections": [{"stage": "dictionary", "from": "Arjun", "to": "Arjuna"}]},
         {"word": " drew", "start": 0.5, "end": 0.8, "probability": 0.9},
         {"word": " his", "start": 0.8, "end": 1.0, "probability": 0.9},
         {"word": " bow", "start": 1.0, "end": 1.3, "probability": 0.9},
     ]},
]}


def _arjun_group():
    return {
        "gid": "0:arjun:arjuna", "world": "Mahabharata", "story_id": 0,
        "heard": "Arjun", "heard_cleaned": "arjun", "canonical": "Arjuna",
        "method": "exact", "label": "sounds-like", "dm_share": True,
        "echo_in_sentence": False, "n_sibling_spellings": 0,
        "occurrences": [{"segment_id": 0, "word_index": 0, "start": 0.0, "token": "Arjun"}],
    }


def test_dictionary_corrected_spot_is_annotated(client, tmp_path):
    """ a question filed against "Arjun" whose spot the dictionary then rewrote to
    "Arjuna" is still LIVE (the position-preserving rewrite doesn't stale it) — so the card
    must say plainly what happened to the word. The occurrence carries the honest marker
    (`corrected_by_dictionary` + `now_reads`), and the heard count of 0 rides with an
    explanatory `n_heard_dictionary_corrected` so it isn't a bare contradiction."""
    pending = {
        "_about": "test", "namefix_version": "2.1.0", "config_fingerprint": "cfg",
        # the fixer stamped the PRE-dictionary words — restorable via `_original`, so fresh
        "transcript_fingerprint": namefix.pre_correction_fingerprint(DICT_TRANSCRIPT),
        "run_at": "2026-07-02T00:00:00Z",
        "worlds": [{"story_id": 0, "recognized_world": "Mahabharata"}],
        "groups": [_arjun_group()], "_rejected": [],
    }
    _make_session_with(tmp_path, DICT_TRANSCRIPT, pending)
    data = client.get("/api/name-corrections").get_json()
    assert data["stale_sessions"] == []          # not stale — the rewrite was 1:1 in place
    assert data["n_pending_groups"] == 1         # still a live card, not folded away
    item = next(n for sec in data["labels"] for w in sec["worlds"]
                for n in w["names"] if n["heard_cleaned"] == "arjun")
    occ = item["sessions"][0]["occurrences"][0]
    assert occ["corrected_by_dictionary"] is True
    assert occ["now_reads"] == "Arjuna"
    # honest counts: the heard form is gone from the live text (0), but the explanation
    # rides alongside so the 0 is never bare
    assert item["n_heard_in_transcript"] == 0
    assert item["n_suggested_in_transcript"] == 1
    assert item["n_heard_dictionary_corrected"] == 1
    # honest audits going forward: BLESSING stamps the same count into the durable
    # _blessed snapshot — the verdict path never enriches, so the count must come from
    # the transcript join in _annotate_frequency, never from display markers
    r = client.post(f"/api/sessions/{SID}/name-corrections/bless",
                    json={"gid": "0:arjun:arjuna"})
    assert r.status_code == 200
    audit = json.loads((tmp_path / SID / "pending-name-corrections.json").read_text())
    snap = audit["_blessed"][0]["group"]
    assert snap["n_heard_in_transcript"] == 0
    assert snap["n_heard_dictionary_corrected"] == 1


def test_namefix_stage_correction_gets_no_marker(client, tmp_path):
    """Provenance (review fix): `_original`'s mere presence doesn't mean the DICTIONARY
    rewrote the spot — apply_corrections stamps it for ANY stage, and the future apply
    pass will write namefix-class stages. Only a recorded "dictionary" stage earns the
    marker; a namefix-stage rewrite gets none and counts nothing."""
    transcript = {"text": "", "segments": [
        {"id": 0, "text": " Arjuna drew his bow",
         "words": [
             {"word": " Arjuna", "start": 0.0, "end": 0.5, "probability": 0.9,
              "_original": "Arjun",
              "_corrections": [{"stage": "namefix", "from": "Arjun", "to": "Arjuna"}]},
             {"word": " drew", "start": 0.5, "end": 0.8, "probability": 0.9},
         ]},
    ]}
    pending = {
        "_about": "test", "namefix_version": "2.1.0", "config_fingerprint": "cfg",
        # pre_correction_fingerprint restores through `_original` whatever the stage,
        # so the queue reads fresh and the card is live — exactly the dangerous surface
        "transcript_fingerprint": namefix.pre_correction_fingerprint(transcript),
        "run_at": "2026-07-02T00:00:00Z",
        "worlds": [{"story_id": 0, "recognized_world": "Mahabharata"}],
        "groups": [_arjun_group()], "_rejected": [],
    }
    _make_session_with(tmp_path, transcript, pending)
    data = client.get("/api/name-corrections").get_json()
    assert data["n_pending_groups"] == 1
    item = next(n for sec in data["labels"] for w in sec["worlds"]
                for n in w["names"] if n["heard_cleaned"] == "arjun")
    occ = item["sessions"][0]["occurrences"][0]
    assert "corrected_by_dictionary" not in occ
    assert "now_reads" not in occ
    assert item["n_heard_dictionary_corrected"] == 0


def test_rollup_survives_pre_tmas79_snapshot_fields(client, tmp_path):
    """Robustness (review fix): an older group (e.g. restored via unrule from an
    old verdict snapshot) carries the two older frequency fields baked in but NOT
    `n_heard_dictionary_corrected`. With the transcript unavailable nothing recomputes
    them — the rollup must pass the old fields through without a KeyError; 0 is the
    honest default for a field that didn't exist when the snapshot was stamped."""
    g = _arjun_group()
    g["n_heard_in_transcript"] = 3
    g["n_suggested_in_transcript"] = 1
    pending = {
        "_about": "test", "namefix_version": "2.1.0", "config_fingerprint": "cfg",
        "transcript_fingerprint": "whatever", "run_at": "2026-07-02T00:00:00Z",
        "worlds": [{"story_id": 0, "recognized_world": "Mahabharata"}],
        "groups": [g], "_rejected": [],
    }
    sdir = tmp_path / SID
    sdir.mkdir()   # deliberately NO transcript-rich.json
    (sdir / "pending-name-corrections.json").write_text(json.dumps(pending))
    r = client.get("/api/name-corrections")
    assert r.status_code == 200
    item = next(n for sec in r.get_json()["labels"] for w in sec["worlds"]
                for n in w["names"] if n["heard_cleaned"] == "arjun")
    assert item["n_heard_in_transcript"] == 3
    assert item["n_suggested_in_transcript"] == 1
    assert item["n_heard_dictionary_corrected"] == 0


def test_legacy_occurrence_without_token_never_crashes(client, tmp_path):
    """Legacy tolerance (review fix): an occurrence with no `token` (hand-edited or
    pre-schema file) gets no marker and no count — never a 500 taking down the whole
    page (the _read_pending backfill precedent). The transcript spot here IS
    dictionary-corrected, so only the missing token stands between it and a marker."""
    g = _arjun_group()
    g["occurrences"] = [{"segment_id": 0, "word_index": 0, "start": 0.0}]   # no token
    pending = {
        "_about": "test", "namefix_version": "2.1.0", "config_fingerprint": "cfg",
        "transcript_fingerprint": namefix.pre_correction_fingerprint(DICT_TRANSCRIPT),
        "run_at": "2026-07-02T00:00:00Z",
        "worlds": [{"story_id": 0, "recognized_world": "Mahabharata"}],
        "groups": [g], "_rejected": [],
    }
    _make_session_with(tmp_path, DICT_TRANSCRIPT, pending)
    r = client.get("/api/name-corrections")
    assert r.status_code == 200
    item = next(n for sec in r.get_json()["labels"] for w in sec["worlds"]
                for n in w["names"] if n["heard_cleaned"] == "arjun")
    occ = item["sessions"][0]["occurrences"][0]
    assert "segment_text" in occ                 # the sentence join still works
    assert "corrected_by_dictionary" not in occ
    assert "now_reads" not in occ
    assert item["n_heard_dictionary_corrected"] == 0


def test_real_drift_without_original_gets_no_marker(client, tmp_path):
    """The negative: a spot whose live word changed WITHOUT an `_original` trail is real
    drift — that is staleness's job (the queue reads stale and verdicts are refused), NOT
    this annotation's. No `corrected_by_dictionary` marker, and the dictionary count is 0."""
    drifted = {"text": "", "segments": [
        {"id": 0, "text": " Krishna drew his bow",
         "words": [
             {"word": " Krishna", "start": 0.0, "end": 0.5, "probability": 0.9},
             {"word": " drew", "start": 0.5, "end": 0.8, "probability": 0.9},
         ]},
    ]}
    pending = {
        "_about": "test", "namefix_version": "2.1.0", "config_fingerprint": "cfg",
        "transcript_fingerprint": "stamped-when-the-spot-said-arjun",  # no longer matches
        "run_at": "2026-07-02T00:00:00Z",
        "worlds": [{"story_id": 0, "recognized_world": "Mahabharata"}],
        "groups": [_arjun_group()], "_rejected": [],
    }
    _make_session_with(tmp_path, drifted, pending)
    data = client.get("/api/name-corrections").get_json()
    assert data["stale_sessions"] == [SID]       # real drift IS stale
    item = next(n for sec in data["labels"] for w in sec["worlds"]
                for n in w["names"] if n["heard_cleaned"] == "arjun")
    occ = item["sessions"][0]["occurrences"][0]
    assert "corrected_by_dictionary" not in occ
    assert "now_reads" not in occ
    assert item["n_heard_dictionary_corrected"] == 0
