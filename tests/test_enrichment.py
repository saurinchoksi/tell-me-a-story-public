"""Tests for enrichment module — speaker diarization alignment."""

from speaker import _compute_speaker_coverage, enrich_with_diarization, DIARIZATION_MODEL as MODEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transcript(words):
    """Build minimal transcript from list of (word, start, end) tuples."""
    return {
        "segments": [{
            "words": [
                {"word": w, "start": s, "end": e, "probability": 0.9}
                for w, s, e in words
            ]
        }]
    }


def _make_diarization(segments):
    """Build minimal diarization from list of (start, end, speaker) tuples."""
    return {
        "segments": [
            {"start": s, "end": e, "speaker": sp}
            for s, e, sp in segments
        ]
    }


# ---------------------------------------------------------------------------
# _compute_speaker_coverage tests
# ---------------------------------------------------------------------------

def test_word_fully_covered():
    """Word [2,4] inside diar [1,5] — full coverage."""
    diar = [{"start": 1, "end": 5, "speaker": "SPEAKER_00"}]
    result = _compute_speaker_coverage(2, 4, diar)
    assert result["label"] == "SPEAKER_00"
    assert result["coverage"] == 1.0


def test_word_partial_coverage():
    """Word [2,4] with diar [1,3] — half covered."""
    diar = [{"start": 1, "end": 3, "speaker": "SPEAKER_00"}]
    result = _compute_speaker_coverage(2, 4, diar)
    assert result["label"] == "SPEAKER_00"
    assert result["coverage"] == 0.5


def test_word_in_gap_no_speaker():
    """Word [5,7] falls in gap between diar [1,3] and [9,11]."""
    diar = [
        {"start": 1, "end": 3, "speaker": "SPEAKER_00"},
        {"start": 9, "end": 11, "speaker": "SPEAKER_01"},
    ]
    result = _compute_speaker_coverage(5, 7, diar)
    assert result["label"] is None
    assert result["coverage"] == 0.0


def test_zero_duration_word():
    """Word with start == end should return no speaker."""
    diar = [{"start": 1, "end": 10, "speaker": "SPEAKER_00"}]
    result = _compute_speaker_coverage(5, 5, diar)
    assert result["label"] is None
    assert result["coverage"] == 0.0


def test_overlapping_speakers_most_overlap_wins():
    """Word [3,5] covered by two speakers — speaker with more overlap wins."""
    diar = [
        {"start": 1, "end": 4, "speaker": "SPEAKER_00"},    # overlap = 1s
        {"start": 3.5, "end": 7, "speaker": "SPEAKER_01"},  # overlap = 1.5s
    ]
    result = _compute_speaker_coverage(3, 5, diar)
    assert result["label"] == "SPEAKER_01"
    assert result["coverage"] == 0.75  # 1.5 / 2.0


def test_coverage_clamped_at_one():
    """Overlapping segments from same speaker don't produce coverage > 1.0."""
    diar = [
        {"start": 1, "end": 3.5, "speaker": "SPEAKER_00"},  # overlap = 1.5s
        {"start": 2.5, "end": 5, "speaker": "SPEAKER_00"},  # overlap = 1.5s → total 3.0s
    ]
    # Word duration = 2s, accumulated overlap = 3.0s → raw coverage 1.5
    result = _compute_speaker_coverage(2, 4, diar)
    assert result["label"] == "SPEAKER_00"
    assert result["coverage"] == 1.0  # Clamped, not 1.5


# ---------------------------------------------------------------------------
# enrich_with_diarization tests
# ---------------------------------------------------------------------------

def test_two_words_different_speakers():
    """Two words landing in different speaker ranges."""
    transcript = _make_transcript([
        (" hello", 1.0, 2.0),
        (" world", 5.0, 6.0),
    ])
    diarization = _make_diarization([
        (0.0, 3.0, "SPEAKER_00"),
        (4.0, 7.0, "SPEAKER_01"),
    ])
    result, entry = enrich_with_diarization(transcript, diarization)
    words = result["segments"][0]["words"]

    assert words[0]["_speaker"]["label"] == "SPEAKER_00"
    assert words[0]["_speaker"]["coverage"] == 1.0
    assert words[1]["_speaker"]["label"] == "SPEAKER_01"
    assert words[1]["_speaker"]["coverage"] == 1.0


def test_empty_diarization():
    """Words exist but diarization has no segments — all get None."""
    transcript = _make_transcript([
        (" hello", 1.0, 2.0),
        (" world", 3.0, 4.0),
    ])
    diarization = _make_diarization([])
    result, entry = enrich_with_diarization(transcript, diarization)
    for word in result["segments"][0]["words"]:
        assert word["_speaker"]["label"] is None
        assert word["_speaker"]["coverage"] == 0.0


def test_empty_transcript():
    """Empty transcript with diarization data — no crash, empty segments."""
    transcript = {"segments": []}
    diarization = _make_diarization([
        (0.0, 5.0, "SPEAKER_00"),
    ])
    result, entry = enrich_with_diarization(transcript, diarization)
    assert result["segments"] == []


def test_deep_copy_no_mutation():
    """Input transcript must not be modified by enrichment."""
    transcript = _make_transcript([
        (" hello", 1.0, 2.0),
    ])
    diarization = _make_diarization([
        (0.0, 3.0, "SPEAKER_00"),
    ])

    # Snapshot original state
    original_word = transcript["segments"][0]["words"][0].copy()

    _, _ = enrich_with_diarization(transcript, diarization)

    # Original must be untouched — no _speaker key added
    assert "_speaker" not in transcript["segments"][0]["words"][0]
    assert transcript["segments"][0]["words"][0] == original_word


def test_existing_word_metadata_preserved():
    """Pre-existing underscore metadata (_original, _corrections) survives enrichment."""
    transcript = {
        "segments": [{
            "words": [{
                "word": " Gita",
                "start": 1.0,
                "end": 2.0,
                "probability": 0.9,
                "_original": "geeta",
                "_corrections": [{"stage": "dictionary", "from": "geeta", "to": "Gita"}],
            }]
        }]
    }
    diarization = _make_diarization([
        (0.0, 3.0, "SPEAKER_00"),
    ])
    result, entry = enrich_with_diarization(transcript, diarization)
    word = result["segments"][0]["words"][0]

    # Existing metadata preserved
    assert word["_original"] == "geeta"
    assert word["_corrections"] == [{"stage": "dictionary", "from": "geeta", "to": "Gita"}]

    # New metadata added
    assert word["_speaker"]["label"] == "SPEAKER_00"
    assert word["_speaker"]["coverage"] == 1.0


def test_enrich_with_diarization_returns_processing_entry():
    """Processing entry has expected stage, model, status, and timestamp."""
    transcript = _make_transcript([(" hello", 1.0, 2.0)])
    diarization = _make_diarization([(0.0, 3.0, "SPEAKER_00")])
    _, entry = enrich_with_diarization(transcript, diarization)
    assert entry["stage"] == "diarization_enrichment"
    assert entry["model"] == MODEL
    assert entry["status"] == "success"
    assert "timestamp" in entry
