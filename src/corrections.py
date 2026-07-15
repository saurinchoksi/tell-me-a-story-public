"""Pure data helpers for applying normalization corrections to transcripts.

Multi-word corrections (e.g. "Bhagwad Gita" -> "Bhagavad Gita") are silently
skipped -- apply_corrections matches per-word only. Single-word matches like
"Gita" still work.
"""

import copy
import re

# Punctuation that Whisper attaches to words (trailing only)
_TRAILING_PUNCT = re.compile(r'^(.*?)([.,;:!?"\)\]\}\']+\.{0,3})$')
_POSSESSIVE_S = re.compile(r"^(.+)(['\u2019]s)$")


def extract_text(transcript: dict) -> str:
    """Join all words from transcript segments into a single string.

    Whisper words carry natural leading spaces (e.g. " hello"), so we
    concatenate them directly and strip the leading whitespace from the result.

    Args:
        transcript: Whisper transcript dict with segments containing words.

    Returns:
        Clean text string with words joined by their natural spacing.
    """
    parts = []
    for segment in transcript.get("segments", []):
        for word in segment.get("words", []):
            parts.append(word["word"])
    return "".join(parts).strip()


def apply_corrections(
    transcript: dict, corrections: list[dict], stage: str
) -> tuple[dict, int]:
    """Apply single-word corrections to a transcript.

    Deep-copies the transcript, then for each word in each segment, checks
    if the word matches any correction (case-insensitive, stripped). Multi-word
    corrections are silently skipped. Trailing punctuation is stripped before
    lookup and reattached after correction.

    On match, the word dict gains:
        _original: The stripped original word (set only on first correction)
        _corrections: List of {stage, from, to} dicts tracking each change
        word: Updated to the corrected form, preserving any leading space

    Args:
        transcript: Whisper transcript dict with segments containing words.
        corrections: List of dicts with 'transcribed' and 'correct' keys.
        stage: Label for this correction pass (e.g. "llm", "dictionary").

    Returns:
        Tuple of (modified transcript, count of words corrected).
    """
    result = copy.deepcopy(transcript)

    # Build lookup, skipping multi-word corrections
    lookup = {}
    for correction in corrections:
        transcribed = correction["transcribed"].strip()
        if " " in transcribed:
            continue
        lookup[transcribed.lower()] = correction["correct"]

    count = 0

    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            raw = word["word"]
            stripped = raw.strip()

            # Separate trailing punctuation before lookup
            punct_match = _TRAILING_PUNCT.match(stripped)
            if punct_match:
                bare = punct_match.group(1)
                trailing_punct = punct_match.group(2)
            else:
                bare = stripped
                trailing_punct = ""

            # Separate possessive suffix before lookup
            # (also matches it's/he's — safe because corrections target proper nouns)
            possessive_suffix = ""
            poss_match = _POSSESSIVE_S.match(bare)
            if poss_match:
                bare = poss_match.group(1)
                possessive_suffix = poss_match.group(2)

            key = bare.lower()

            if key not in lookup:
                continue

            corrected_with_punct = lookup[key] + possessive_suffix + trailing_punct

            # Preserve _original from the very first correction
            if "_original" not in word:
                word["_original"] = stripped

            # Append to corrections history
            entry = {"stage": stage, "from": bare, "to": lookup[key]}
            if "_corrections" not in word:
                word["_corrections"] = []
            word["_corrections"].append(entry)

            # Update the word, preserving leading space
            leading_space = " " if raw.startswith(" ") else ""
            word["word"] = leading_space + corrected_with_punct

            count += 1

    # Keep each segment's text line in sync with its (now corrected) words — the words
    # are the source of truth. Normalization rewrites words but historically left `text`
    # stale, so every name tool had to rebuild the line itself; heal it here at the
    # source. Segments with no words (injected [unintelligible] gaps) keep their text.
    for segment in result.get("segments", []):
        words = segment.get("words")
        if words:
            segment["text"] = "".join(w["word"] for w in words)

    return result, count
