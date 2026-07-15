"""M9a — family-name mistranscription detector.

Ported from the validated EMP probe (emp/src/detect_m9a.py, which stays
untouched as the sealed eval artifact). Validated 1.00/1.00 precision/recall
on the EMP sessions; the detection logic here must stay verbatim or that
validation no longer applies.

A transcript token is flagged iff:
  (1) any Double Metaphone code (PRIMARY *or* SECONDARY) of the token matches
      any Double Metaphone code of a roster canonical name    [phonetic layer]
  OR
  (2) the token exactly equals a roster alias — a front-of-word distortion
      the phonetic layer cannot reach,                        [alias layer]
subject to the capitalization gate (lowercase tokens are never flagged).

Matching on BOTH metaphone codes is load-bearing. A name heard with a "th"
encodes to a PRIMARY theta code but a SECONDARY "t" code; it is the secondary
code that links it back to the plain-"t" canonical. A single-code matcher
would miss these.

PRIVACY: no real names are hardcoded here. The roster lives in the gitignored
data/name_roster.json; its schema is documented in data/name_roster.example.json.
"""

import hashlib
import json
from pathlib import Path

from detectors.base import Detector, load_transcript
from detectors.phonetics import clean, codes, is_capitalized

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROSTER_PATH = PROJECT_ROOT / "data" / "name_roster.json"


class FamilyNameDetector(Detector):
    id = "m9a-family-names"
    label = "Family-name mistranscription"
    failure_mode = "M9a"
    version = "1.0.0"

    def __init__(self, roster_path=None):
        self.roster_path = Path(roster_path) if roster_path else DEFAULT_ROSTER_PATH
        self._roster = None              # parsed roster, cached...
        self._roster_fingerprint = None  # ...keyed by the file's content hash

    def config_fingerprint(self) -> str:
        """Hash of the roster file. Feeding this into the staleness check
        means a roster edit re-runs every session's scan, and keying the
        in-memory cache on it means a long-lived process picks the edit up."""
        if not self.roster_path.exists():
            raise FileNotFoundError(
                f"Family-name roster not found at {self.roster_path}. "
                "Create it from the schema in data/name_roster.example.json."
            )
        return "sha256:" + hashlib.sha256(self.roster_path.read_bytes()).hexdigest()

    def _load_roster(self):
        roster = json.loads(self.roster_path.read_text())

        canon = {p["id"]: p["canonical"] for p in roster["people"]}
        canon_forms = {clean(n) for n in canon.values()}

        code_to_pid = {}  # dm code -> {person id}
        for pid, name in canon.items():
            for c in codes(clean(name)):
                code_to_pid.setdefault(c, set()).add(pid)
        roster_codes = set(code_to_pid)

        alias = {}
        for entry in roster["aliases"]:
            ct = clean(entry["token"])
            pid = entry["person_id"]
            if pid not in canon:
                raise ValueError(
                    f"Alias {entry['token']!r} references unknown person_id {pid!r}."
                )
            if codes(ct) & roster_codes:
                raise ValueError(
                    f"Redundant alias {entry['token']!r}: its Double Metaphone codes "
                    "already reach a canonical name via the phonetic layer."
                )
            alias[ct] = pid

        self._roster = (canon, canon_forms, code_to_pid, roster_codes, alias)

    def run(self, session_dir: Path) -> dict:
        fingerprint = self.config_fingerprint()  # raises if roster missing
        if self._roster is None or self._roster_fingerprint != fingerprint:
            self._load_roster()
            self._roster_fingerprint = fingerprint
        canon, canon_forms, code_to_pid, roster_codes, alias = self._roster

        data = load_transcript(session_dir)
        flags, n_tokens = [], 0
        for seg in data["segments"]:
            # injected gap segments legitimately carry no words
            for wi, w in enumerate(seg.get("words", [])):
                c = clean(w["word"])
                if not c:
                    continue
                n_tokens += 1
                if c in canon_forms:
                    continue   # correctly-spelled canonical — not an error
                tok_codes = codes(c)
                shared = tok_codes & roster_codes
                if shared:
                    matched = sorted({pid for code in shared for pid in code_to_pid[code]})
                    mtype = "phonetic"
                elif c in alias:
                    matched = [alias[c]]
                    mtype = "alias"
                else:
                    continue
                if not is_capitalized(w["word"].strip()):
                    continue   # capitalization gate — drop lowercase homophones
                flags.append({
                    "segment_id": seg["id"],
                    "word_index": wi,
                    "start": w.get("start"),
                    "end": w.get("end"),
                    "token": w["word"].strip(),
                    "cleaned": c,
                    "dm_codes": sorted(tok_codes),
                    "match_type": mtype,
                    "matched_person_ids": matched,
                    "matched_canonicals": [canon[pid] for pid in matched],
                })
        return {"n_word_tokens": n_tokens, "flags": flags}
