#!/usr/bin/env python3
"""The roster store — deliberate, validated, ADD-ONLY writes to data/name_roster.json.

The roster answers "who exists" (the Name Book answers "what did Saurin decide").
Design settled in the TMAS-56 grill (2026-07-03, all calls Choksi's):

- A **person** has a canonical spelling and a `circle` ("family" | "friend") — real
  people beyond the household (the child's real-life friends) live in the same roster with
  the tag; every matcher treats them identically, because the system's real question is
  "does this name belong to a real person we know."
- A **nickname** is a FULL name form, not a skip-word: correct as heard (never flagged,
  never rewritten), its sound codes join the matching layer (garbles of it get caught),
  and it is a correction target of its own — a garbled nickname is fixed toward the
  nickname as spoken, never silently promoted to the person's canonical.
- An **alias** stays what it always was: a known DISTORTION to catch exactly (a
  front-of-word mangle whose sound codes can't reach the person). The loader rejects an
  alias whose codes already reach any canonical or nickname (redundant).
- **Add-only**: the interface adds people/nicknames/aliases and never removes or renames
  (removal has ripples — recorded verdicts and book rulings reference forms — so it
  stays a rare, deliberate hand edit).

Only explicit human gestures on the /roster page write this file (the Name Book's rule,
applied to the roster). Writes are atomic and validated BEFORE saving, so a bad entry is
refused with a reason, never persisted. The file is gitignored (real names) and synced
to the private data repo; a content change re-runs the family fixer and detector scans
automatically via their config fingerprints.
"""
import json
import re
from pathlib import Path

from detectors.phonetics import clean, codes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROSTER_PATH = PROJECT_ROOT / "data" / "name_roster.json"

CIRCLES = ("family", "friend")


def load(path=None) -> dict:
    """The raw roster. Fail-loud on a corrupt file; an ABSENT file is the empty roster
    (first contact — same stance as the Name Book)."""
    p = Path(path or DEFAULT_ROSTER_PATH)
    if not p.exists():
        return {"people": [], "aliases": []}
    return json.loads(p.read_text())


def _person_by_id(roster: dict, pid: str) -> dict | None:
    return next((p for p in roster["people"] if p["id"] == pid), None)


def _all_name_codes(roster: dict) -> set:
    """DM codes of every canonical AND nickname — the phonetic reach of the roster."""
    out: set = set()
    for p in roster["people"]:
        for form in [p["canonical"]] + p.get("nicknames", []):
            out |= codes(clean(form))
    return out


def _all_forms_cleaned(roster: dict) -> set:
    out = set()
    for p in roster["people"]:
        out.add(clean(p["canonical"]))
        out |= {clean(n) for n in p.get("nicknames", [])}
    out |= {clean(a["token"]) for a in roster["aliases"]}
    return out


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")


def _save(path, roster: dict) -> None:
    from jsonio import atomic_write_json
    atomic_write_json(Path(path or DEFAULT_ROSTER_PATH), roster)


def add_person(canonical: str, circle: str = "family", path=None) -> dict:
    """Add a real person. Returns the new person entry. Refuses duplicates (same
    cleaned form as ANY existing form) and unknown circles."""
    canonical = canonical.strip()
    if not canonical:
        raise ValueError("add_person: canonical must be non-empty")
    if circle not in CIRCLES:
        raise ValueError(f"add_person: circle must be one of {CIRCLES}")
    roster = load(path)
    if clean(canonical) in _all_forms_cleaned(roster):
        raise ValueError(f"add_person: {canonical!r} collides with an existing "
                         "roster form (person, nickname, or alias)")
    pid = _slug(canonical)
    if _person_by_id(roster, pid):
        raise ValueError(f"add_person: id {pid!r} already exists")
    person = {"id": pid, "canonical": canonical, "circle": circle}
    roster["people"].append(person)
    _save(path, roster)
    return person


def add_nickname(person_id: str, nickname: str, path=None) -> dict:
    """Add a legitimate name form to a person — correct as heard, matchable, a
    correction target of its own. Refuses collisions with any existing form."""
    nickname = nickname.strip()
    if not nickname:
        raise ValueError("add_nickname: nickname must be non-empty")
    roster = load(path)
    person = _person_by_id(roster, person_id)
    if person is None:
        raise ValueError(f"add_nickname: unknown person_id {person_id!r}")
    if clean(nickname) in _all_forms_cleaned(roster):
        raise ValueError(f"add_nickname: {nickname!r} collides with an existing "
                         "roster form (person, nickname, or alias)")
    person.setdefault("nicknames", []).append(nickname)
    _save(path, roster)
    return person


def add_alias(person_id: str, token: str, path=None) -> dict:
    """Add a known distortion to catch exactly. Refuses the redundant case (its sound
    codes already reach a canonical or nickname — the phonetic layer has it) and
    collisions, mirroring the detector loader's standing rule."""
    token = clean(token)
    if not token:
        raise ValueError("add_alias: token must be non-empty")
    roster = load(path)
    person = _person_by_id(roster, person_id)
    if person is None:
        raise ValueError(f"add_alias: unknown person_id {person_id!r}")
    if token in _all_forms_cleaned(roster):
        raise ValueError(f"add_alias: {token!r} collides with an existing roster form")
    if codes(token) & _all_name_codes(roster):
        raise ValueError(f"add_alias: {token!r} is redundant — its sound codes already "
                         "reach a canonical or nickname via the phonetic layer")
    roster["aliases"].append({"token": token, "person_id": person_id})
    _save(path, roster)
    return roster["aliases"][-1]
