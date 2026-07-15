"""Predicate functions for filtering transcription words.

Filters are pure functions that return True/False for inclusion.
"""

def silence_gap(word: dict) -> bool:
    """Is this word a silence-gap hallucination candidate?

    True when diarization found no speaker and zero coverage.
    Caller should also check single-word segment — a lone word
    with no speaker is suspect; the same word inside a multi-word
    segment is just a shaky moment in real speech.
    """
    speaker = word.get("_speaker", {})
    return speaker.get("label") is None and speaker.get("coverage", 1.0) == 0.0


def near_zero_probability(word: dict, threshold: float = 0.01) -> bool:
    """Is Whisper's confidence essentially zero?

    True when probability is below threshold. At 0.01, this
    catches words Whisper itself doesn't believe (e.g. prob 0.00007)
    without touching low-confidence-but-real speech (e.g. prob 0.010).
    Caller should also check single-word segment.
    """
    prob = word.get("probability")
    if prob is None:
        return False
    return prob < threshold


def find_duplicate_segments(segments: list[dict]) -> set[int]:
    """Find segment IDs whose text duplicates an earlier segment.

    Returns the IDs of the *later* occurrence. The first is kept.
    Duplicates often appear at Whisper's 30-second seek boundaries,
    but real speech repetitions also match — this is a review aid,
    not an automated suppression rule.
    """
    seen = {}
    duplicates = set()
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        if text in seen:
            duplicates.add(seg["id"])
        else:
            seen[text] = seg["id"]
    return duplicates
