"""Speaker enrichment — apply diarization results to transcripts.

Pure data transforms with no ML dependencies. These functions take
pre-computed diarization segments and annotate transcript words with
speaker labels, or inject synthetic [unintelligible] segments where
a speaker was detected but Whisper produced no transcript.
"""

import bisect
import copy
from datetime import datetime, timezone

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


def _compute_speaker_coverage(word_start, word_end, diar_segments, _seg_ends=None):
    """Compute which speaker best covers a word's time range.

    For each diarization segment, calculates the temporal overlap with the
    word. The speaker with the greatest total overlap wins.

    Uses a bisect-based scan when _seg_ends is provided: skips directly to
    the first segment whose end exceeds word_start, then stops as soon as a
    segment starts at or after word_end. This reduces the inner loop from
    O(all segments) to O(segments in the word's neighborhood).

    Args:
        word_start: Word start time in seconds.
        word_end: Word end time in seconds.
        diar_segments: List of diarization segment dicts with start, end,
            and speaker keys. Must be sorted by start time.
        _seg_ends: Optional pre-built sorted list of segment end times,
            parallel to diar_segments. Built once by enrich_with_diarization
            and passed here to avoid repeated construction.

    Returns:
        Dict with 'label' (speaker string or None) and 'coverage' (float
        0.0-1.0 indicating what fraction of the word duration is covered
        by the best-matching speaker).
    """
    word_duration = word_end - word_start
    if word_duration <= 0:
        return {"label": None, "coverage": 0.0}

    overlap_by_speaker = {}

    if _seg_ends is not None:
        # Binary search: first segment whose end > word_start can overlap.
        first = bisect.bisect_left(_seg_ends, word_start)
        for seg in diar_segments[first:]:
            if seg["start"] >= word_end:
                break
            overlap = min(word_end, seg["end"]) - max(word_start, seg["start"])
            if overlap > 0:
                speaker = seg["speaker"]
                overlap_by_speaker[speaker] = overlap_by_speaker.get(speaker, 0) + overlap
    else:
        for seg in diar_segments:
            overlap = max(0, min(word_end, seg["end"]) - max(word_start, seg["start"]))
            if overlap > 0:
                speaker = seg["speaker"]
                overlap_by_speaker[speaker] = overlap_by_speaker.get(speaker, 0) + overlap

    if not overlap_by_speaker:
        return {"label": None, "coverage": 0.0}

    best_speaker = max(overlap_by_speaker, key=overlap_by_speaker.get)
    coverage = min(overlap_by_speaker[best_speaker] / word_duration, 1.0)

    return {"label": best_speaker, "coverage": coverage}


def _dominant_speaker(segment):
    """Return the speaker label with the most words in a transcript segment.

    Only considers words that have a non-None _speaker.label. Returns None if
    the segment has no words or no words with known speaker labels.
    """
    counts = {}
    for word in segment.get("words", []):
        label = word.get("_speaker", {}).get("label")
        if label is not None:
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def enrich_with_diarization(transcript, diarization):
    """Add speaker labels to each word in a transcript using diarization data.

    Deep-copies the transcript, then assigns a _speaker dict to every word
    based on temporal overlap with diarization segments. Does not touch
    _processing — the pipeline handles that.

    Args:
        transcript: Whisper transcript dict with segments containing words.
        diarization: Diarization result dict with a 'segments' list of
            {start, end, speaker} dicts.

    Returns:
        Tuple of (enriched_transcript, processing_entry).
        enriched_transcript: Deep-copied transcript with _speaker metadata.
        processing_entry: Dict with stage metadata.
    """
    result = copy.deepcopy(transcript)
    diar_segments = diarization.get("segments", [])

    # Pre-build a sorted list of segment end times for bisect-based lookup.
    # Segments are already sorted by start time (guaranteed by pyannote output).
    # We bisect on end times to find the first segment that could overlap a word.
    seg_ends = [seg["end"] for seg in diar_segments]

    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            speaker_info = _compute_speaker_coverage(
                word["start"], word["end"], diar_segments, _seg_ends=seg_ends
            )
            word["_speaker"] = speaker_info
        dominant = _dominant_speaker(segment)
        if dominant is not None:
            segment["_speaker"] = {"label": dominant, "source": "dominant"}

    entry = {
        "stage": "diarization_enrichment",
        "model": DIARIZATION_MODEL,
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return result, entry


def _word_coverage(diar_start, diar_end, all_words):
    """Compute the fraction of a diarization segment covered by transcript words.

    Args:
        diar_start: Diarization segment start time in seconds.
        diar_end: Diarization segment end time in seconds.
        all_words: Flat list of word dicts with 'start' and 'end' keys.

    Returns:
        Float 0.0-1.0. Returns 0.0 for zero-duration segments.
    """
    diar_duration = diar_end - diar_start
    if diar_duration <= 0:
        return 0.0

    total_overlap = 0.0
    for word in all_words:
        overlap = min(diar_end, word["end"]) - max(diar_start, word["start"])
        if overlap > 0:
            total_overlap += overlap

    return min(total_overlap / diar_duration, 1.0)


def detect_unintelligible_gaps(transcript, diarization):
    """Inject [unintelligible] segments where a speaker has no transcript coverage.

    After diarization enrichment, some diarization segments may have very low
    transcript word coverage — the speaker was detected but Whisper produced no
    transcript. This function identifies those gaps and injects synthetic
    [unintelligible] segments when the gap's speaker differs from both neighboring
    transcript segments' dominant speakers (indicating a dialogue turn, not a
    monologue pause).

    Deep-copies the transcript. Does not touch _processing.

    Args:
        transcript: Enriched transcript dict with segments containing words
            that have _speaker labels (i.e., run after enrich_with_diarization).
        diarization: Diarization result dict with a 'segments' list of
            {start, end, speaker} dicts.

    Returns:
        Tuple of (enriched_transcript, processing_entry).
        enriched_transcript: Deep-copied transcript with synthetic segments injected
            and all segments re-sorted by start time.
        processing_entry: Dict with stage metadata including gaps_found count.
    """
    result = copy.deepcopy(transcript)
    diar_segments = diarization.get("segments", [])
    transcript_segments = result.get("segments", [])

    # Flatten all words once for coverage computation across all segments.
    all_words = []
    for seg in transcript_segments:
        all_words.extend(seg.get("words", []))

    injected = []

    for diar_seg in diar_segments:
        diar_start = diar_seg["start"]
        diar_end = diar_seg["end"]
        speaker = diar_seg["speaker"]

        if diar_end - diar_start <= 0:
            continue

        # Skip if transcript words already cover >= 30% of this diarization segment.
        coverage = _word_coverage(diar_start, diar_end, all_words)
        if coverage >= 0.3:
            continue

        # Find nearest preceding transcript segment (highest start < diar_start)
        # and nearest following segment (lowest start >= diar_end).
        preceding = None
        following = None
        for seg in transcript_segments:
            if seg["start"] < diar_start:
                if preceding is None or seg["start"] > preceding["start"]:
                    preceding = seg
            elif seg["start"] >= diar_end:
                if following is None or seg["start"] < following["start"]:
                    following = seg

        # Both neighbors required — skip gaps at the edges of the recording.
        if preceding is None or following is None:
            continue

        # Both neighbors must have a determinable dominant speaker.
        preceding_speaker = preceding.get("_speaker", {}).get("label")
        following_speaker = following.get("_speaker", {}).get("label")
        if preceding_speaker is None or following_speaker is None:
            continue

        # Gap is meaningful only when its speaker differs from BOTH neighbors.
        # If it matches either neighbor, this is likely a monologue pause, not a turn.
        if speaker == preceding_speaker or speaker == following_speaker:
            continue

        injected.append({
            "id": f"gap_{diar_start:.3f}",
            "start": diar_start,
            "end": diar_end,
            "text": "[unintelligible]",
            "words": [],
            "_speaker": {"label": speaker, "coverage": 1.0},
            "_source": "diarization_gap",
        })

    result.setdefault("segments", []).extend(injected)
    result["segments"].sort(key=lambda s: s["start"])

    entry = {
        "stage": "gap_detection",
        "gaps_found": len(injected),
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return result, entry
