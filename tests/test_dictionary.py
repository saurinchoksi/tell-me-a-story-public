"""Tests for dictionary module."""

import json
from unittest.mock import mock_open, patch

import pytest

from dictionary import build_variant_map, load_library, normalize_variants


# --- load_library tests ---


def test_load_library_valid_json():
    """Valid JSON file is loaded and returned as dict."""
    data = {"entries": [{"canonical": "Arjuna", "variants": ["Arjun"]}]}
    m = mock_open(read_data=json.dumps(data))
    with patch("builtins.open", m):
        result = load_library("fake/path.json")
    assert result == data


def test_load_library_file_not_found():
    """FileNotFoundError propagates when file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_library("/nonexistent/path.json")


def test_load_library_invalid_json():
    """JSONDecodeError propagates for malformed JSON."""
    m = mock_open(read_data="not valid json {{{")
    with patch("builtins.open", m):
        with pytest.raises(json.JSONDecodeError):
            load_library("fake/path.json")


# --- build_variant_map tests ---


def test_build_variant_map_basic():
    """Variants are mapped to their canonical names."""
    library = {
        "entries": [
            {"canonical": "Arjuna", "variants": ["Arjun"], "aliases": ["Partha"]}
        ]
    }
    result = build_variant_map(library)
    assert result == {"arjun": "Arjuna"}


def test_build_variant_map_keys_lowercase():
    """Variant keys are lowercased in the map."""
    library = {
        "entries": [
            {"canonical": "Bhima", "variants": ["Bheema", "Bheem"], "aliases": []}
        ]
    }
    result = build_variant_map(library)
    assert "bheema" in result
    assert "bheem" in result
    assert "Bheema" not in result


def test_build_variant_map_aliases_excluded():
    """Aliases are not included in the variant map."""
    library = {
        "entries": [
            {
                "canonical": "Arjuna",
                "variants": ["Arjun"],
                "aliases": ["Partha", "Dhananjaya"],
            }
        ]
    }
    result = build_variant_map(library)
    assert "partha" not in result
    assert "dhananjaya" not in result


def test_build_variant_map_canonical_not_as_key():
    """Variants matching the canonical name (case-insensitive) are skipped."""
    library = {
        "entries": [
            {"canonical": "Bhima", "variants": ["bhima", "Bheema"], "aliases": []}
        ]
    }
    result = build_variant_map(library)
    assert "bhima" not in result
    assert "bheema" in result


def test_build_variant_map_dharma_collision():
    """Two entries with same canonical handle shared variant gracefully."""
    library = {
        "entries": [
            {"canonical": "Dharma", "category": "concept", "variants": ["Dharm"], "aliases": []},
            {"canonical": "Dharma", "category": "divine", "variants": ["Dharm"], "aliases": ["Yama"]},
        ]
    }
    result = build_variant_map(library)
    assert result["dharm"] == "Dharma"
    assert "yama" not in result


def test_build_variant_map_empty_entries():
    """Empty entries list returns empty map."""
    assert build_variant_map({"entries": []}) == {}
    assert build_variant_map({}) == {}


# --- normalize_variants tests ---


SAMPLE_MAP = {
    "arjun": "Arjuna",
    "bheeshma": "Bhishma",
    "duryodhan": "Duryodhana",
    "bhagwad gita": "Bhagavad Gita",
    "geeta": "Bhagavad Gita",
    "pandav": "Pandavas",
}


def test_normalize_variants_basic():
    """Detects a known variant and returns correction."""
    result = normalize_variants("Then Arjun picked up his bow", SAMPLE_MAP)
    assert len(result) == 1
    assert result[0]["transcribed"] == "Arjun"
    assert result[0]["correct"] == "Arjuna"


def test_normalize_variants_case_insensitive():
    """Matches variants regardless of case, preserves original."""
    result = normalize_variants("BHEESHMA stood firm", SAMPLE_MAP)
    assert len(result) == 1
    assert result[0]["transcribed"] == "BHEESHMA"
    assert result[0]["correct"] == "Bhishma"


def test_normalize_variants_word_boundary():
    """Word boundaries prevent partial matches (Pandav != Pandavas)."""
    result = normalize_variants("The Pandavas won the war", SAMPLE_MAP)
    assert result == []


def test_normalize_variants_multi_word():
    """Multi-word variants like 'Bhagwad Gita' are detected."""
    result = normalize_variants("He read the Bhagwad Gita", SAMPLE_MAP)
    corrections = {c["transcribed"]: c["correct"] for c in result}
    assert "Bhagwad Gita" in corrections
    assert corrections["Bhagwad Gita"] == "Bhagavad Gita"


def test_normalize_variants_canonical_no_correction():
    """Canonical names do not appear as corrections."""
    # "Arjuna" is canonical, not in variant_map, so no match
    result = normalize_variants("Arjuna spoke to Krishna", SAMPLE_MAP)
    assert result == []


def test_normalize_variants_empty_text():
    """Empty text returns empty list."""
    assert normalize_variants("", SAMPLE_MAP) == []


def test_normalize_variants_deduplication():
    """Same variant appearing twice produces only one correction."""
    result = normalize_variants("Arjun and then Arjun again", SAMPLE_MAP)
    transcribed = [c["transcribed"].lower() for c in result]
    assert transcribed.count("arjun") == 1
