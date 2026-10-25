#!/usr/bin/env python3
"""Home canon — the family's OWN characters, a first-class world called "Home".

The third registry (grilled 2026-07-03, all calls Choksi's). The roster answers
"which real people exist"; the story worlds' casts answer "which canon characters
exist"; this answers "which of OUR characters exist" — Marley, the engines, the
characters the family invents and keeps. Every cross-fire failure of the fixer era
traced to this registry not existing: the spelling picker tried to fold an invented
engine into the canon "James" it sounds like, and the family fixer tried to fold the
story character "Marley" into the real friend he sounds like.

Two pieces, mirroring the people side:
- **The cast** (this file's store, `data/home-canon.json`): who exists, one canonical
  spelling each. Written ONLY by Choksi's gestures on the Roster page's "Our
  characters" section (add-only, validated). Gitignored (the family's characters are
  the family's), synced to the private data repo.
- **The rulings** (`data/name-book/home.json`): an ordinary Name Book world file —
  spelling picks for world-less clusters land there (run_spellingfix assigns world
  "Home" to clusters outside every recognized story world), foldable and liftable like
  any ruling.

Stage-1 effects on the fixers (config-fingerprinted, so a cast change re-runs them):
familyfix never questions a correctly-spelled home character (the Marley class dies at
the root); spellingfix proposes the REGISTERED spelling for a cluster containing one
(the registry outranks the vote) and never treats the registered form as a deviant.
Stage 2 (shipped 2026-07-03, namefix 2.2.0): the home cast rides in Whisper's ear on
unrecognized nights (the home ear), making garbles of home characters correctable,
and a registered spelling is untouchable to namefix in any world (the home guard).
"""
import hashlib
import json
import re
from pathlib import Path

from detectors.phonetics import clean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = PROJECT_ROOT / "data" / "home-canon.json"

WORLD = "Home"   # the world name home rulings live under (name-book/home.json)


def load(path=None) -> dict:
    """The raw cast. Absent file = no characters yet (first contact)."""
    p = Path(path or DEFAULT_PATH)
    if not p.exists():
        return {"characters": []}
    return json.loads(p.read_text())


def canonical_forms(path=None) -> dict[str, str]:
    """cleaned form -> registered canonical spelling, for the fixers."""
    return {clean(c["canonical"]): c["canonical"]
            for c in load(path)["characters"] if clean(c["canonical"])}


def fingerprint(path=None) -> str:
    """Content stamp for the fixers' config fingerprints — registering a character
    re-runs them (cheap: Whisper-only for familyfix, pure code for spellingfix)."""
    p = Path(path or DEFAULT_PATH)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "absent"


def add_character(canonical: str, path=None, roster_path=None) -> dict:
    """Register a character — Choksi's gesture only. Refuses collisions with existing
    characters AND with any roster form (a name can't be both a character and a real
    person's form; that ambiguity is exactly what the registries exist to settle)."""
    canonical = canonical.strip()
    if not canonical:
        raise ValueError("add_character: canonical must be non-empty")
    c = clean(canonical)
    cast = load(path)
    if c in {clean(ch["canonical"]) for ch in cast["characters"]}:
        raise ValueError(f"add_character: {canonical!r} is already registered")
    import roster_store
    roster = roster_store.load(roster_path)
    roster_forms = set()
    for p_ in roster["people"]:
        roster_forms.add(clean(p_["canonical"]))
        roster_forms |= {clean(n) for n in p_.get("nicknames", [])}
    roster_forms |= {clean(a["token"]) for a in roster["aliases"]}
    if c in roster_forms:
        raise ValueError(f"add_character: {canonical!r} collides with a roster form — "
                         "a name can't be both a character and a real person's")
    entry = {"id": re.sub(r"[^a-z0-9]+", "-", canonical.lower()).strip("-"),
             "canonical": canonical}
    cast["characters"].append(entry)
    from jsonio import atomic_write_json
    atomic_write_json(Path(path or DEFAULT_PATH), cast)
    return entry
