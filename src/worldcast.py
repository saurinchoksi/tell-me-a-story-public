#!/usr/bin/env python3
"""Split-prompt cast generation — the world's characters AND its groups, asked separately.

The E8 sweep (emp/results/visuals/whisper-context/notes.md) found that asking a small model
for a world's "characters" reliably drops the GROUP names — Pandavas, Kauravas, Crystal Gems,
the Foot Clan — and on the held-out Mahabharata those groups are 14 of the 36 by-ear key
corrections. No single prompt got both clean; TWO prompts did, on both Qwen and Gemma, across
all 7 swept worlds. So the cast is generated in two calls and unioned, groups first (they
matter most for correction).

Shared by the namefix correction stage and the M9c canon detector. Pure functions with an
injectable `gen` (the qwen35 reader contract); an optional on-disk cache under
`data/worlds-cache/` keyed by world slug + a prompt fingerprint, so repeat sessions in the
same world don't re-generate.
"""
import hashlib
import json
import os
import re
from pathlib import Path

from detectors.phonetics import clean
from namebook import slug

# Both prompts validated in the E8 sweep (emp/src/cast_split.py) — do not "lightly adapt"
# the wording without re-scoring; small changes move a 4B model a lot.
CHARACTERS_PROMPT = (
    'List the main characters of the fictional world "{world}" - real individual people only, '
    'never places, groups, items, species, or titles. Give the 12 to 18 most central characters, '
    'most important first, and be sure to include the main protagonist(s), the main antagonist or '
    'villain, and the most important elder or mentor. Use the correct canonical English spelling of '
    'each name. Output the names only, one name per line - no numbers, no descriptions, no '
    'commentary, and no duplicate or alternate spellings. List each character exactly once and stop '
    'once the central cast is covered.')

GROUPS_PROMPT = (
    'List the important named GROUPS of the fictional world "{world}" - the families, clans, teams, '
    'houses, factions, peoples, and collective names the story uses (for example, the name for the '
    'heroes as a group, and the name for their enemies as a group). Only real collective names from '
    'that world, not individual people and not places. Give up to 8, most important first. Use the '
    'correct canonical English spelling. Output the names only, one per line - no numbers, no '
    'descriptions, no commentary. Stop when the important groups are covered.')

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "worlds-cache"


def _fingerprint() -> str:
    """Cache key component: changing either prompt invalidates every cached cast."""
    return hashlib.sha256((CHARACTERS_PROMPT + GROUPS_PROMPT).encode()).hexdigest()[:12]


def parse_cast_lines(raw: str) -> list[str]:
    """Plain-line cast parser (E8): strip bullets/numbers and a leading "The", drop non-ASCII
    and JSON-ish lines, dedupe case-insensitively, preserve order."""
    out, seen = [], set()
    for line in (raw or "").splitlines():
        s = re.sub(r"^[\s\-\*.\d)]+", "", line).strip().strip('",')
        s = re.sub(r"^[Tt]he\s+", "", s)
        if not s or len(s) > 40 or s.lower() in seen:
            continue
        if re.search(r"[^\x00-\x7f]", s) or re.search(r"[:{}\[\]]", s):
            continue
        seen.add(s.lower())
        out.append(s)
    return out


def generate_cast_split(gen, world: str) -> dict:
    """Two model calls: {"characters": [...], "groups": [...]} for `world` ("" -> empty)."""
    if not world:
        return {"characters": [], "groups": []}
    chars = parse_cast_lines(gen(CHARACTERS_PROMPT.format(world=world), max_tokens=500))
    groups = parse_cast_lines(gen(GROUPS_PROMPT.format(world=world), max_tokens=200))
    return {"characters": chars, "groups": groups}


def correction_cast(split: dict) -> list[str]:
    """The union used for correction: groups FIRST (Pandavas/Kauravas are the highest-value
    entries), then characters, deduped by cleaned form, order preserved."""
    names, seen = [], set()
    for kind in ("groups", "characters"):
        for n in split.get(kind, []):
            c = clean(n)
            if c and c not in seen:
                seen.add(c)
                names.append(n)
    return names


def cached_cast_split(gen, world: str, cache_dir: Path = None) -> dict:
    """generate_cast_split with an on-disk cache (atomic write). The cache entry carries the
    prompt fingerprint — a prompt change re-generates rather than serving a stale cast."""
    if not world:
        return {"characters": [], "groups": []}
    cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
    path = cache_dir / f"{slug(world)}.json"
    fp = _fingerprint()
    if path.exists():
        data = json.loads(path.read_text())
        if data.get("_fingerprint") == fp:
            return {"characters": data["characters"], "groups": data["groups"]}
    split = generate_cast_split(gen, world)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"_world": world, "_fingerprint": fp, **split},
                              ensure_ascii=False, indent=2))
    os.replace(tmp, path)
    return split
