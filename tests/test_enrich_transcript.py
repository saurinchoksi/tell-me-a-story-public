"""Tests for enrich_transcript function (in pipeline module).

The world-blind LLM name-normalizer (old Pass 3) was removed 2026-07-01 — it
confidently substituted wrong names. Enrichment now runs: diarization (Pass 1),
gap detection (Pass 2), story segmentation (Pass 3, only with a cache_dir),
namefix (Pass 4 — the audio-grounded replacement; needs cache_dir + audio_path
+ _stories, else records a "skipped" entry), and dictionary normalization
(Pass 5, off by default).
"""

import copy
from unittest.mock import patch, MagicMock

from pipeline import enrich_transcript


# --- Test fixtures ---

_CLEAN_TRANSCRIPT = {
    "text": "hello world",
    "segments": [{
        "text": "hello world",
        "words": [
            {"word": " hello", "start": 0.0, "end": 0.5, "probability": 0.9},
            {"word": " world", "start": 0.5, "end": 1.0, "probability": 0.8},
        ],
    }],
}

_FAKE_DIARIZATION = {
    "segments": [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
    ],
}

_DIAR_ENTRY = {"stage": "diarization_enrichment",
               "model": "pyannote/speaker-diarization-community-1",
               "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}
_GAP_ENTRY = {"stage": "gap_detection", "gaps_found": 0,
              "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}


# --- enrich_transcript tests ---

def test_enrich_all_stages_succeed():
    """With a library (and no cache_dir/audio), the stages run: diar, gap,
    namefix(skipped), dictionary."""
    transcript = copy.deepcopy(_CLEAN_TRANSCRIPT)
    diarization = copy.deepcopy(_FAKE_DIARIZATION)

    with patch("pipeline.extract_text", return_value="hello world"), \
         patch("pipeline.apply_corrections", return_value=(copy.deepcopy(transcript), 1)), \
         patch("pipeline.load_library", return_value={"entries": []}), \
         patch("pipeline.build_variant_map", return_value={}), \
         patch("pipeline.normalize_variants", return_value=[]), \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, _DIAR_ENTRY)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, _GAP_ENTRY)):

        result, processing, counts = enrich_transcript(
            transcript, diarization, library_path="fake/lib.json", verbose=False
        )

    assert len(processing) == 6
    assert processing[0]["stage"] == "diarization_enrichment"
    assert processing[0]["status"] == "success"
    assert processing[1]["stage"] == "gap_detection"
    assert processing[1]["status"] == "success"
    assert processing[2]["stage"] == "namefix"
    assert processing[2]["status"] == "skipped"
    assert processing[3]["stage"] == "family_namefix"
    assert processing[3]["status"] == "skipped"
    assert processing[3]["reason"] == "no_cache_dir"
    assert processing[4]["stage"] == "spellingfix"
    assert processing[4]["status"] == "skipped"
    assert processing[4]["reason"] == "no_cache_dir"
    assert processing[5]["stage"] == "dictionary_normalization"
    assert processing[5]["status"] == "success"
    assert counts["dict_count"] == 1
    assert counts["namefix_groups"] == 0
    assert counts["family_groups"] == 0
    assert counts["spelling_groups"] == 0
    assert "llm_count" not in counts


def test_enrich_does_not_set_processing_on_transcript():
    """enrich_transcript does not set _processing on the returned transcript."""
    transcript = copy.deepcopy(_CLEAN_TRANSCRIPT)
    diarization = copy.deepcopy(_FAKE_DIARIZATION)

    with patch("pipeline.extract_text", return_value="hello world"), \
         patch("pipeline.apply_corrections", return_value=(copy.deepcopy(transcript), 0)), \
         patch("pipeline.load_library", return_value={"entries": []}), \
         patch("pipeline.build_variant_map", return_value={}), \
         patch("pipeline.normalize_variants", return_value=[]), \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, _DIAR_ENTRY)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, _GAP_ENTRY)):

        result, processing, counts = enrich_transcript(
            transcript, diarization, library_path="fake/lib.json", verbose=False
        )

    assert "_processing" not in result


def test_enrich_skips_dictionary_when_no_library():
    """Dictionary normalization is skipped when no library_path is provided."""
    transcript = copy.deepcopy(_CLEAN_TRANSCRIPT)
    diarization = copy.deepcopy(_FAKE_DIARIZATION)

    with patch("pipeline.load_library") as mock_load_lib, \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, _DIAR_ENTRY)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, _GAP_ENTRY)):

        result, processing, counts = enrich_transcript(
            transcript, diarization, verbose=False
        )

    # diar, gap, namefix / family / spelling (all skipped), dict(skipped) = 6
    assert len(processing) == 6
    assert processing[2]["stage"] == "namefix"
    assert processing[2]["status"] == "skipped"
    assert processing[3]["stage"] == "family_namefix"
    assert processing[3]["status"] == "skipped"
    assert processing[4]["stage"] == "spellingfix"
    assert processing[4]["status"] == "skipped"
    assert processing[5]["stage"] == "dictionary_normalization"
    assert processing[5]["status"] == "skipped"
    assert processing[5]["reason"] == "no_library_path"
    assert counts["dict_count"] == 0
    mock_load_lib.assert_not_called()


def test_enrich_namefix_runs_in_shadow(tmp_path):
    """With cache_dir + audio_path + stories, Pass 4 runs namefix in SHADOW: the labeled
    queue is written to the session dir, counts reported — and the transcript's words are
    NEVER touched (calibration: nothing applies until the apply pass)."""
    transcript = {"segments": [{
        "id": 0, "start": 0.0, "end": 2.0, "text": " about Bushma today",
        "words": [
            {"word": " about", "start": 0.0, "end": 0.5, "probability": 0.9},
            {"word": " Bushma", "start": 0.5, "end": 1.0, "probability": 0.9},
            {"word": " today", "start": 1.0, "end": 1.5, "probability": 0.9},
        ],
    }]}
    diarization = {"segments": []}
    stories = [{"start_id": 0, "end_id": 0, "title": "T", "world": ""}]
    fake_result = {
        "groups": [
            {"world": "Mahabharata", "story_id": 0, "heard": "Bushma",
             "heard_cleaned": "bushma", "suggestion": "Bhishma", "canonical": "Bhishma",
             "method": "exact", "label": "sounds-like", "dm_share": True,
             "echo_in_sentence": False, "n_sibling_spellings": 0,
             "occurrences": [{"segment_id": 0, "word_index": 1, "start": 0.5,
                              "token": "Bushma"}]},
            {"world": "Mahabharata", "story_id": 0, "heard": "arrows",
             "heard_cleaned": "arrows", "suggestion": "Kauravas",
             "canonical": "Kauravas", "method": "exact", "label": "ordinary-word",
             "dm_share": False, "echo_in_sentence": False, "n_sibling_spellings": 0,
             "occurrences": []}],
        "worlds": [{"story_id": 0, "recognized_world": "Mahabharata"}],
    }
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")

    with patch("pipeline.segment_transcript", return_value=stories), \
         patch("namefix.run_namefix", return_value=fake_result) as mock_nf, \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, _DIAR_ENTRY)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, _GAP_ENTRY)):
        result, processing, counts = enrich_transcript(
            transcript, diarization, cache_dir=tmp_path, audio_path=str(audio),
            family_roster_path=tmp_path / "absent-roster.json",  # isolate from the real one
            verbose=False)

    mock_nf.assert_called_once()
    nf = [p for p in processing if p["stage"] == "namefix"][0]
    assert nf["status"] == "success" and nf["shadow"] is True and nf["groups"] == 2
    assert nf["by_label"] == {"sounds-like": 1, "ordinary-word": 1}
    ff = [p for p in processing if p["stage"] == "family_namefix"][0]
    assert ff["status"] == "skipped" and ff["reason"] == "no_roster"
    # SHADOW: the word is untouched — no correction, no _original trail
    assert result["segments"][0]["words"][1]["word"].strip() == "Bushma"
    assert "_original" not in result["segments"][0]["words"][1]
    assert counts["namefix_groups"] == 2
    assert (tmp_path / "pending-name-corrections.json").exists()
    # TMAS-72: the familyfix skip is recorded WHERE THE REVIEW PAGE LOOKS, not just
    # in _processing — "never ran" must not read as "checked, nothing found"
    import json as _json
    queue = _json.loads((tmp_path / "pending-name-corrections.json").read_text())
    fam = queue["fixers"]["family"]
    assert fam["status"] == "skipped" and fam["reason"] == "no_roster"
    # namefix RAN — its metadata entry carries no skip status
    assert "status" not in queue["fixers"]["namefix"]


def test_enrich_familyfix_runs_in_shadow(tmp_path):
    """With cache_dir + audio + a roster, Pass 4b runs familyfix in SHADOW: labeled
    family questions merge into the shared queue (fixer='family'), counts reported,
    the transcript untouched. No _stories needed — family names apply everywhere."""
    import json as _json
    transcript = {"segments": [{
        "id": 0, "start": 0.0, "end": 2.0, "text": " saw Marda today",
        "words": [
            {"word": " saw", "start": 0.0, "end": 0.5, "probability": 0.9},
            {"word": " Marda", "start": 0.5, "end": 1.0, "probability": 0.9},
            {"word": " today", "start": 1.0, "end": 1.5, "probability": 0.9},
        ],
    }]}
    diarization = {"segments": []}
    roster = tmp_path / "roster.json"
    roster.write_text(_json.dumps({
        "people": [{"id": "child", "canonical": "Marta"}], "aliases": []}))
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")
    fake_result = {"groups": [
        {"gid": "family:marda:marta", "world": "Family", "story_id": "family",
         "heard": "Marda", "heard_cleaned": "marda", "canonical": "Marta",
         "method": "exact", "label": "sounds-like", "dm_share": True,
         "echo_in_sentence": False, "n_sibling_spellings": 0, "redecode_heard": True,
         "occurrences": [{"segment_id": 0, "word_index": 1, "start": 0.5,
                          "token": "Marda"}]}]}

    with patch("familyfix.run_familyfix", return_value=fake_result) as mock_ff, \
         patch("pipeline.segment_transcript", return_value=[]), \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, _DIAR_ENTRY)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, _GAP_ENTRY)):
        result, processing, counts = enrich_transcript(
            transcript, diarization, cache_dir=tmp_path, audio_path=str(audio),
            family_roster_path=roster, verbose=False)

    mock_ff.assert_called_once()
    ff = [p for p in processing if p["stage"] == "family_namefix"][0]
    assert ff["status"] == "success" and ff["shadow"] is True and ff["groups"] == 1
    assert counts["family_groups"] == 1
    # SHADOW: the transcript word is untouched
    assert result["segments"][0]["words"][1]["word"].strip() == "Marda"
    queue = _json.loads((tmp_path / "pending-name-corrections.json").read_text())
    assert queue["groups"][0]["fixer"] == "family"
    assert "family" in queue["fixers"]


def test_enrich_familyfix_error_leaves_queue_trace(tmp_path):
    """A CRASHED familyfix run also leaves its trace in the queue metadata (reason
    'error') — a crash must not read as 'never stamped' at /name-review (TMAS-72)."""
    import json as _json
    transcript = {"segments": [{
        "id": 0, "start": 0.0, "end": 2.0, "text": " saw Marda today",
        "words": [
            {"word": " saw", "start": 0.0, "end": 0.5, "probability": 0.9},
            {"word": " Marda", "start": 0.5, "end": 1.0, "probability": 0.9},
            {"word": " today", "start": 1.0, "end": 1.5, "probability": 0.9},
        ],
    }]}
    roster = tmp_path / "roster.json"
    roster.write_text(_json.dumps({
        "people": [{"id": "child", "canonical": "Marta"}], "aliases": []}))
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")

    with patch("familyfix.run_familyfix", side_effect=RuntimeError("whisper died")), \
         patch("pipeline.segment_transcript", return_value=[]), \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, _DIAR_ENTRY)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, _GAP_ENTRY)):
        _, processing, _ = enrich_transcript(
            transcript, {"segments": []}, cache_dir=tmp_path, audio_path=str(audio),
            family_roster_path=roster, verbose=False)

    ff = [p for p in processing if p["stage"] == "family_namefix"][0]
    assert ff["status"] == "error" and "whisper died" in ff["error"]
    assert "queue_record_error" not in ff   # the queue record itself succeeded
    queue = _json.loads((tmp_path / "pending-name-corrections.json").read_text())
    fam = queue["fixers"]["family"]
    assert fam["status"] == "skipped" and fam["reason"] == "error"


def test_enrich_spellingfix_runs_in_shadow(tmp_path):
    """With a cache_dir, Pass 4c clusters inconsistent invented-name spellings and
    writes labeled questions (fixer='spelling') into the shared queue — pure code, no
    model, transcript untouched."""
    import json as _json
    transcript = {"segments": [{
        "id": 0, "start": 0.0, "end": 2.0, "text": " Jammus and Jameis",
        "words": [
            {"word": " Jammus", "start": 0.0, "end": 0.5, "probability": 0.9},
            {"word": " and", "start": 0.5, "end": 0.8, "probability": 0.9},
            {"word": " Jameis", "start": 0.8, "end": 1.2, "probability": 0.9},
            {"word": " Jammus", "start": 1.2, "end": 1.6, "probability": 0.9},
        ],
    }]}
    diarization = {"segments": []}

    with patch("pipeline.segment_transcript", return_value=[]), \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, _DIAR_ENTRY)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, _GAP_ENTRY)):
        result, processing, counts = enrich_transcript(
            transcript, diarization, cache_dir=tmp_path,
            family_roster_path=tmp_path / "absent-roster.json",
            verbose=False)

    sf = [p for p in processing if p["stage"] == "spellingfix"][0]
    assert sf["status"] == "success" and sf["shadow"] is True and sf["groups"] == 1
    assert counts["spelling_groups"] == 1
    # SHADOW: no transcript word changed
    assert result["segments"][0]["words"][2]["word"].strip() == "Jameis"
    queue = _json.loads((tmp_path / "pending-name-corrections.json").read_text())
    g = queue["groups"][0]
    assert g["fixer"] == "spelling" and g["label"] == "inconsistent-spelling"
    assert g["heard_cleaned"] == "jameis" and g["canonical"] == "Jammus"
    assert g["world"] == "Home"   # world-less clusters file under the home canon (TMAS-57)
    # TMAS-72: the fixers that DIDN'T run (no audio here) left their trace too
    assert queue["fixers"]["namefix"]["status"] == "skipped"
    assert queue["fixers"]["namefix"]["reason"] == "no_audio_path"
    assert queue["fixers"]["family"]["status"] == "skipped"
    assert queue["fixers"]["family"]["reason"] == "no_audio_path"
    assert "status" not in queue["fixers"]["spelling"]   # spellingfix ran


def test_enrich_does_not_realign(tmp_path):
    """Realignment left the enrichment loop: enrich_transcript emits no word_realignment
    stage and never re-times words — so --re-enrich reuses transcript-raw's corrected
    timings, even when a session audio file sits in cache_dir."""
    transcript = {"segments": [{
        "id": 0, "text": " hello world",
        "words": [
            {"word": " hello", "start": 0.11, "end": 0.52, "probability": 0.9, "_align_conf": 0.88},
            {"word": " world", "start": 0.52, "end": 1.03, "probability": 0.8, "_align_conf": 0.91},
        ],
    }]}
    before = copy.deepcopy(transcript["segments"][0]["words"])
    diarization = {"segments": []}
    (tmp_path / "audio.m4a").write_bytes(b"not real audio")  # present but must be ignored

    with patch("pipeline.segment_transcript", return_value=[]), \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, _DIAR_ENTRY)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, _GAP_ENTRY)):
        result, processing, _ = enrich_transcript(
            transcript, diarization, cache_dir=tmp_path, verbose=False)

    assert not any(p["stage"] == "word_realignment" for p in processing)
    assert result["segments"][0]["words"] == before  # timings untouched by enrichment


def test_enrich_pass5_segmentation_success(tmp_path):
    """With a cache_dir, Pass 5 runs and records a story_segmentation processing entry."""
    transcript = {"segments": [{"id": 0, "text": " a", "words": [{"word": " a"}]}]}
    diarization = {"segments": []}
    stories = [{"start_id": 0, "end_id": 0, "title": "A", "world": "W"}]

    with patch("pipeline.segment_transcript", return_value=stories) as mock_seg, \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, _DIAR_ENTRY)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, _GAP_ENTRY)):
        result, processing, _ = enrich_transcript(
            transcript, diarization, cache_dir=tmp_path, verbose=False)

    seg_entries = [p for p in processing if p["stage"] == "story_segmentation"]
    assert len(seg_entries) == 1
    assert seg_entries[0]["status"] == "success"
    assert seg_entries[0]["stories_found"] == 1
    assert result["_stories"][0]["title"] == "A"  # enrich_with_stories ran for real
    mock_seg.assert_called_once()


def test_enrich_pass5_degrades_safely(tmp_path):
    """A segmentation failure records an error entry but does not sink the pass."""
    transcript = {"segments": [{"id": 0, "text": " a", "words": [{"word": " a"}]}]}
    diarization = {"segments": []}

    with patch("pipeline.segment_transcript", side_effect=RuntimeError("model down")), \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, _DIAR_ENTRY)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, _GAP_ENTRY)):
        result, processing, _ = enrich_transcript(
            transcript, diarization, cache_dir=tmp_path, verbose=False)

    seg = [p for p in processing if p["stage"] == "story_segmentation"][0]
    assert seg["status"] == "error" and "model down" in seg["error"]
    # the other passes still ran
    assert any(p["stage"] == "gap_detection" for p in processing)
    assert "_stories" not in result  # segmentation failed, nothing tagged


def test_dictionary_rewrite_does_not_stale_saved_queue(tmp_path):
    """TMAS-77 regression (reworked after review): Pass 5 (dictionary normalization)
    rewrites word tokens AFTER the fixers stamp the shared queue, so the stamp truthfully
    names the pre-dictionary words — and stays that way (nothing but the fixers writes
    the queue). The fix is at READ time: _is_stale recognizes the position-preserving
    rewrite through each corrected word's `_original` and reads fresh; any other word
    change still reads stale.

    Unit level, guarding the full enrich→save→reload boundary: a tiny inline library
    where one variant applies, a cache_dir (so the queue file is written) but no audio
    (namefix/familyfix skip; spellingfix runs). Dictionary runs FOR REAL — only the ML
    stages are mocked. The enriched transcript is persisted with save_computed and
    RELOADED from disk; _is_stale is judged against the SAVED file, not the in-memory
    return."""
    import copy
    import json as _json
    import namefix
    from api.routes.name_corrections import _is_stale
    from pipeline import save_computed

    transcript = {"segments": [{
        "id": 0, "start": 0.0, "end": 2.0, "text": " Arjun drew his bow",
        "words": [
            {"word": " Arjun", "start": 0.0, "end": 0.5, "probability": 0.9},
            {"word": " drew", "start": 0.5, "end": 0.8, "probability": 0.9},
            {"word": " his", "start": 0.8, "end": 1.0, "probability": 0.9},
            {"word": " bow", "start": 1.0, "end": 1.3, "probability": 0.9},
        ],
    }]}
    transcript_raw = copy.deepcopy(transcript)
    diarization = {"segments": []}
    library = tmp_path / "lib.json"
    library.write_text(_json.dumps(
        {"entries": [{"canonical": "Arjuna", "variants": ["Arjun"], "aliases": []}]}))

    with patch("pipeline.segment_transcript", return_value=[]), \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, _DIAR_ENTRY)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, _GAP_ENTRY)):
        result, processing, counts = enrich_transcript(
            transcript, diarization, library_path=str(library), cache_dir=tmp_path,
            family_roster_path=tmp_path / "absent-roster.json", verbose=False)

    # precondition: the dictionary really rewrote a word
    assert counts["dict_count"] == 1

    # persist the way the pipeline does, then judge against the RELOADED file
    save_computed(str(tmp_path), transcript_raw, result, diarization)
    loaded = _json.loads((tmp_path / "transcript-rich.json").read_text())
    queue = _json.loads((tmp_path / "pending-name-corrections.json").read_text())
    w = loaded["segments"][0]["words"][0]
    assert w["word"].strip() == "Arjuna" and w["_original"] == "Arjun"

    # the scenario genuinely exercises the exception path: the stamp (pre-dictionary)
    # no longer matches the saved words verbatim...
    assert queue["transcript_fingerprint"] != namefix.transcript_fingerprint(loaded)
    # ...yet the queue is NOT stale — the rewrite was 1:1 in place, restorable via _original
    assert _is_stale(queue, loaded) is False

    # and the guard stays armed: a real edit (no _original trail) still reads stale
    drifted = copy.deepcopy(loaded)
    drifted["segments"][0]["words"][1]["word"] = " flew"
    assert _is_stale(queue, drifted) is True


def test_enrich_reuses_segmentation_cache_across_runs(tmp_path):
    """Two enrich runs with the same cache_dir hit the segmentation cache: the expensive
    segmenter computes once, the second run is from_cache — the 're-enrich doesn't reload
    the model' behavior, wired through the in-memory transcript fingerprints (mocks only
    the model COMPUTE, so the real cache/fingerprints run)."""
    base = {"segments": [
        {"id": 0, "start": 0.0, "end": 1.0, "text": " hello",
         "words": [{"word": " hello", "start": 0.0, "end": 1.0}]},
        {"id": 1, "start": 1.0, "end": 2.0, "text": " world",
         "words": [{"word": " world", "start": 1.0, "end": 2.0}]},
    ]}
    diarization = {"segments": []}
    stories = [{"start_id": 0, "end_id": 1, "title": "T", "world": "W"}]

    with patch("pipeline.segment_transcript", return_value=stories) as mock_seg, \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, {"stage": "diarization_enrichment"})), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, {"stage": "gap_detection", "gaps_found": 0})):
        _, p1, _ = enrich_transcript(copy.deepcopy(base), diarization, cache_dir=tmp_path, verbose=False)
        r2, p2, _ = enrich_transcript(copy.deepcopy(base), diarization, cache_dir=tmp_path, verbose=False)

    assert mock_seg.call_count == 1   # segmentation computed once, reused on the 2nd run
    assert [e for e in p1 if e["stage"] == "story_segmentation"][0]["from_cache"] is False
    assert [e for e in p2 if e["stage"] == "story_segmentation"][0]["from_cache"] is True
    assert r2["_stories"][0]["title"] == "T"  # stories still enriched in on the cached run
