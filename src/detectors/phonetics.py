"""Shared phonetic helpers for name detectors.

Double Metaphone token cleaning, code extraction, and the capitalization gate —
used by the M9a roster detector and the M9b consistency detector alike. These
were validated as part of the M9a probe (1.00/1.00); keep them verbatim.
"""

import re

from metaphone import doublemetaphone


def clean(tok: str) -> str:
    """Lowercase; strip surrounding punctuation and the possessive; keep letters."""
    t = tok.strip().lower()
    t = re.sub(r"[^a-z'’]+$", "", t)   # trailing punctuation (keep apostrophe)
    t = re.sub(r"^[^a-z'’]+", "", t)   # leading punctuation
    t = re.sub(r"['’]s?$", "", t)      # possessive 's or trailing apostrophe
    t = re.sub(r"[^a-z]", "", t)       # letters only
    return t


def codes(word: str) -> set:
    """The non-empty Double Metaphone codes (primary + secondary) of a word."""
    return {c for c in doublemetaphone(word) if c}


def is_capitalized(raw: str) -> bool:
    """First alphabetic character is uppercase — a cheap 'name-shaped' precision
    gate that drops lowercase homophones (e.g. a contraction whose code collides
    with a real name) with no model. Caveat: it would also drop a name Whisper
    rendered lowercase, and let through a capitalized common word at a sentence
    start — neither occurred in the validated sessions."""
    first = next((ch for ch in raw if ch.isalpha()), "")
    return first.isupper()
