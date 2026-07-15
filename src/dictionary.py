"""Dictionary-based normalization of transliteration variants."""

import json
import re
from datetime import datetime, timezone


def make_processing_entry(library_path: str, corrections_count: int) -> dict:
    """Build a processing entry for the dictionary normalization stage."""
    return {
        "stage": "dictionary_normalization",
        "library": library_path,
        "status": "success",
        "corrections_applied": corrections_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def load_library(path: str) -> dict:
    """Load a reference library from a JSON file.

    Args:
        path: Path to the JSON library file

    Returns:
        Parsed library dict

    Raises:
        FileNotFoundError: If path does not exist
        json.JSONDecodeError: If file is not valid JSON
    """
    with open(path) as f:
        return json.load(f)


def build_variant_map(library: dict) -> dict[str, str]:
    """Build a mapping from variant spellings to canonical names.

    Iterates entries[].variants only (not aliases). Skips variants
    whose lowered form equals the canonical's lowered form.

    Args:
        library: Parsed library dict with an 'entries' list

    Returns:
        Dict mapping lowered variant strings to canonical names
    """
    variant_map = {}
    for entry in library.get("entries", []):
        canonical = entry["canonical"]
        canonical_lower = canonical.lower()
        for variant in entry.get("variants", []):
            variant_lower = variant.lower()
            if variant_lower != canonical_lower:
                variant_map[variant_lower] = canonical
    return variant_map


def normalize_variants(text: str, variant_map: dict) -> list[dict]:
    """Scan text for known variant spellings and return corrections.

    Uses word-boundary regex to find variants in text. Matches are
    case-insensitive. Results are deduplicated by lowered transcribed
    form and sorted longest-first so multi-word matches take priority.

    Args:
        text: Text to scan for variant spellings
        variant_map: Mapping from lowered variants to canonical names

    Returns:
        List of correction dicts with 'transcribed' and 'correct' keys.
        Returns [] if no variants found.
    """
    if not text:
        return []

    # Sort variants longest-first so multi-word matches take priority
    sorted_variants = sorted(variant_map.keys(), key=len, reverse=True)

    seen = {}  # lowered transcribed -> correction dict
    for variant in sorted_variants:
        canonical = variant_map[variant]
        pattern = r"\b" + re.escape(variant) + r"\b"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            transcribed = match.group()
            transcribed_lower = transcribed.lower()
            if transcribed_lower not in seen:
                seen[transcribed_lower] = {
                    "transcribed": transcribed,
                    "correct": canonical,
                }

    return list(seen.values())
