"""Shared family-roster matcher.

m9a (family_names.py) owns family-name detection. m9b (name_consistency.py) uses this
to EXCLUDE family-roster names from its improvised-name clustering, so the two detectors
partition cleanly — m9a = family, m9b = improvised — instead of both flagging the same
name (the child's name showed up under both before).

The match mirrors family_names.py's phonetic + alias test: a cleaned token matches the
roster iff it is a canonical form, OR its Double Metaphone codes intersect the roster's
codes (catches misspellings like Martie/Marthie), OR it is a known alias.

Graceful by design: if the roster file is absent, the matcher matches nothing and m9b
falls back to its original roster-agnostic behavior. No real names are hardcoded — the
roster is the gitignored data/name_roster.json (schema in data/name_roster.example.json).
"""
import json
from pathlib import Path

from detectors.phonetics import clean, codes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROSTER_PATH = PROJECT_ROOT / "data" / "name_roster.json"


class RosterMatcher:
    """is_roster_name(cleaned) -> True iff the cleaned token belongs to a family-roster
    name, by the same test m9a uses. A correctly-spelled canonical AND its misspellings
    both match, so every variant of a family name is excluded from m9b."""

    def __init__(self, roster_path=None):
        path = Path(roster_path) if roster_path else DEFAULT_ROSTER_PATH
        self._canon_forms: set[str] = set()
        self._roster_codes: set[str] = set()
        self._aliases: set[str] = set()
        if path.exists():
            roster = json.loads(path.read_text())
            self._canon_forms = {clean(p["canonical"]) for p in roster["people"]}
            for form in self._canon_forms:
                self._roster_codes |= set(codes(form))
            self._aliases = {clean(a["token"]) for a in roster.get("aliases", [])}

    def is_roster_name(self, cleaned: str) -> bool:
        if not cleaned:
            return False
        return (
            cleaned in self._canon_forms
            or cleaned in self._aliases
            or bool(set(codes(cleaned)) & self._roster_codes)
        )
