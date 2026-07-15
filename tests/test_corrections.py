"""Tests for corrections module."""

from corrections import extract_text, apply_corrections


# --- extract_text tests ---


def test_extract_text_joins_words():
    """Words with leading spaces are joined and leading whitespace stripped."""
    transcript = {
        "segments": [
            {
                "words": [
                    {"word": " hello", "start": 0.0, "end": 0.5, "probability": 0.9},
                    {"word": " world", "start": 0.5, "end": 1.0, "probability": 0.8},
                ]
            }
        ]
    }
    assert extract_text(transcript) == "hello world"


# --- apply_corrections tests ---


def _make_transcript(*words):
    """Helper: build a minimal transcript from word strings."""
    return {
        "segments": [
            {
                "words": [
                    {"word": w, "start": 0.0, "end": 0.5, "probability": 0.9}
                    for w in words
                ]
            }
        ]
    }


def test_apply_single_correction():
    """Single correction updates word, sets _original, records history."""
    transcript = _make_transcript(" fondos")
    corrections = [{"transcribed": "fondos", "correct": "Pandavas"}]

    result, count = apply_corrections(transcript, corrections, "llm")

    word = result["segments"][0]["words"][0]
    assert word["word"] == " Pandavas"
    assert word["_original"] == "fondos"
    assert word["_corrections"] == [
        {"stage": "llm", "from": "fondos", "to": "Pandavas"}
    ]
    assert count == 1


def test_apply_chained_corrections():
    """Second correction preserves original _original from first pass."""
    transcript = _make_transcript(" fondos")

    # Pass 1: LLM correction
    result, _ = apply_corrections(
        transcript, [{"transcribed": "fondos", "correct": "Pandvas"}], "llm"
    )

    # Pass 2: dictionary correction on the result
    result, _ = apply_corrections(
        result, [{"transcribed": "Pandvas", "correct": "Pandavas"}], "dictionary"
    )

    word = result["segments"][0]["words"][0]
    assert word["_original"] == "fondos"
    assert len(word["_corrections"]) == 2
    assert word["_corrections"][0] == {
        "stage": "llm",
        "from": "fondos",
        "to": "Pandvas",
    }
    assert word["_corrections"][1] == {
        "stage": "dictionary",
        "from": "Pandvas",
        "to": "Pandavas",
    }
    assert word["word"] == " Pandavas"


def test_apply_case_insensitive():
    """Correction matching is case-insensitive."""
    transcript = _make_transcript(" FONDOS")
    corrections = [{"transcribed": "fondos", "correct": "Pandavas"}]

    result, count = apply_corrections(transcript, corrections, "llm")

    word = result["segments"][0]["words"][0]
    assert word["word"] == " Pandavas"
    assert count == 1


def test_apply_no_match():
    """Unmatched words are left untouched with no extra keys."""
    transcript = _make_transcript(" hello")
    corrections = [{"transcribed": "fondos", "correct": "Pandavas"}]

    result, count = apply_corrections(transcript, corrections, "llm")

    word = result["segments"][0]["words"][0]
    assert count == 0
    assert "_original" not in word
    assert "_corrections" not in word


def test_apply_empty_corrections():
    """Empty corrections list leaves transcript unchanged."""
    transcript = _make_transcript(" hello", " world")

    result, count = apply_corrections(transcript, [], "llm")

    assert count == 0
    assert result["segments"][0]["words"][0]["word"] == " hello"
    assert result["segments"][0]["words"][1]["word"] == " world"


def test_apply_possessive_straight_apostrophe():
    """Possessive 's with straight apostrophe is stripped before lookup."""
    transcript = _make_transcript(" arjuna's")
    corrections = [{"transcribed": "arjuna", "correct": "Arjuna"}]

    result, count = apply_corrections(transcript, corrections, "dictionary")

    word = result["segments"][0]["words"][0]
    assert word["word"] == " Arjuna's"
    assert word["_original"] == "arjuna's"
    assert word["_corrections"] == [
        {"stage": "dictionary", "from": "arjuna", "to": "Arjuna"}
    ]
    assert count == 1


def test_apply_possessive_curly_apostrophe():
    """Possessive 's with curly apostrophe (U+2019) is stripped before lookup."""
    transcript = _make_transcript(" arjuna\u2019s")
    corrections = [{"transcribed": "arjuna", "correct": "Arjuna"}]

    result, count = apply_corrections(transcript, corrections, "dictionary")

    word = result["segments"][0]["words"][0]
    assert word["word"] == " Arjuna\u2019s"
    assert count == 1


def test_apply_possessive_with_trailing_punct():
    """Possessive with trailing punctuation: both are stripped and reattached."""
    transcript = _make_transcript(" arjuna's,")
    corrections = [{"transcribed": "arjuna", "correct": "Arjuna"}]

    result, count = apply_corrections(transcript, corrections, "dictionary")

    word = result["segments"][0]["words"][0]
    assert word["word"] == " Arjuna's,"
    assert count == 1


def test_apply_contraction_not_stripped():
    """Contractions like don't are not affected — apostrophe is mid-word."""
    transcript = _make_transcript(" don't")
    corrections = [{"transcribed": "don", "correct": "Dan"}]

    result, count = apply_corrections(transcript, corrections, "dictionary")

    assert count == 0


def test_apply_possessive_chained():
    """Two-pass correction on possessive word preserves _original."""
    transcript = _make_transcript(" yudhisthir's")

    # Pass 1: LLM
    result, _ = apply_corrections(
        transcript, [{"transcribed": "yudhisthir", "correct": "Yudhisthir"}], "llm"
    )

    # Pass 2: dictionary
    result, _ = apply_corrections(
        result,
        [{"transcribed": "Yudhisthir", "correct": "Yudhishthira"}],
        "dictionary",
    )

    word = result["segments"][0]["words"][0]
    assert word["word"] == " Yudhishthira's"
    assert word["_original"] == "yudhisthir's"
    assert len(word["_corrections"]) == 2


def test_apply_plural_possessive():
    """Plural possessive (pandavas') already works via trailing punct — regression guard."""
    transcript = _make_transcript(" pandavas'")
    corrections = [{"transcribed": "pandavas", "correct": "Pandavas"}]

    result, count = apply_corrections(transcript, corrections, "dictionary")

    word = result["segments"][0]["words"][0]
    assert word["word"] == " Pandavas'"
    assert count == 1


def test_apply_contraction_its_treated_as_possessive():
    """it's is indistinguishable from possessive — documents known behavior."""
    transcript = _make_transcript(" it's")
    corrections = [{"transcribed": "it", "correct": "IT"}]
    result, count = apply_corrections(transcript, corrections, "dictionary")
    word = result["segments"][0]["words"][0]
    assert word["word"] == " IT's"
    assert count == 1


# --- segment text is rebuilt from corrected words (the text-vs-words fix) ---


def test_apply_rebuilds_segment_text_from_corrected_words():
    """The segment's text line reflects the corrected words, not the stale original."""
    transcript = _make_transcript(" fondos", " are", " here")
    transcript["segments"][0]["text"] = " fondos are here"  # the pre-correction (stale) line
    result, _ = apply_corrections(
        transcript, [{"transcribed": "fondos", "correct": "Pandavas"}], "llm"
    )
    assert result["segments"][0]["text"] == " Pandavas are here"


def test_apply_text_synced_even_with_no_match():
    """Text is rebuilt from words whether or not a correction fired (idempotent)."""
    transcript = _make_transcript(" hello", " world")
    result, count = apply_corrections(transcript, [], "llm")
    assert count == 0
    assert result["segments"][0]["text"] == " hello world"


def test_apply_leaves_gap_segment_text_alone():
    """A segment with no words (an injected [unintelligible] gap) keeps its text."""
    transcript = {"segments": [{"id": "gap_1.5", "text": "[unintelligible]"}]}
    result, count = apply_corrections(transcript, [], "llm")
    assert result["segments"][0]["text"] == "[unintelligible]"
    assert count == 0
