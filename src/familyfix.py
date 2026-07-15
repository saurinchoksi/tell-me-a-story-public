#!/usr/bin/env python3
"""familyfix — the family-name fixer (the M9a fixer), SHADOW like every fixer.

The frozen m9a detector (src/detectors/family_names.py, validated 1.00/1.00) FINDS the
spots where a family name was mis-transcribed; this stage LISTENS AGAIN at each spot and
turns its opinion into a labeled question in the shared shadow queue. Same calibration
rule as namefix: only add labeled questions — never write a transcript, never write the
Name Book; only Choksi's taps write anything durable.

The chain (namefix's validated listen-again, with the roster as the cast):
  1. CANDIDATES: transcript tokens matching the roster exactly the way the detector
     matches them — Double Metaphone against the people's canonicals (both codes) +
     the exact-alias layer + the capitalization gate, correctly-spelled canonicals
     skipped. Deliberately NOT dictionary-gated: a capitalized common-word disguise
     ("Martyr" for a family member named Marta) must become a question (the un-gated-M9a rule; it
     just carries the ordinary-word label so it is always a human call).
     The detector's matching lives inline in its run() (reads from disk) and the
     detector file is frozen, so this module carries a compact mirror of the loader +
     matcher — same schema, same reject-redundant-alias rule, injectable roster path.
  2. RE-LISTEN: ONE Whisper subprocess (no Qwen — the cast comes from the roster file);
     namefix's clip re-decode around each candidate with the family cast in Whisper's
     ear, and namefix's validated gate (same LEAD/WINDOW/TEMPORAL_TOL/MODEL).
  3. COVERAGE-FIRST (Choksi's dial, 2026-07-03): a detector-known spot where the
     re-listen produces no usable name STILL becomes a question — the suggestion falls
     back to the detector's phonetic/alias match, the label is computed the normal way
     from heard + suggestion, and the group carries `redecode_heard: false` so his
     verdicts can judge whether the machine's second silence is itself evidence.
     Nothing the detector knows about is ever silently missing.

World = "Family" — rulings land in data/name-book/family.json like any world.
gid = family:{heard_cleaned}:{canonical_cleaned} (the "family" story id keeps it
disjoint from namefix's story-int gids). Groups ride in the shared queue with
fixer="family" (namefix.write_queue merges per fixer).

PRIVACY: no real names in code or tests; the roster is the gitignored
data/name_roster.json (schema: data/name_roster.example.json); the queue files that
carry real-name canonicals live under gitignored sessions/, synced to the private repo.
"""
import hashlib
import json
from pathlib import Path

from commonwords import is_common, wordlist_fingerprint
from detectors.phonetics import clean, codes, is_capitalized
from detectors.story_names._names import proper_name_candidates
import namefix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROSTER_PATH = PROJECT_ROOT / "data" / "name_roster.json"

FIXER = "family"
WORLD = "Family"    # the Name Book world family rulings live under (family.json)
VERSION = "1.0.0"

PROMPT_TEMPLATE = ("This is a bedtime story our family recorded. "
                   "The people in the room include {names}.")


def config_fingerprint(roster_path=None, home_path=None) -> str:
    """Ties a run to the roster content + the home cast + the re-decode chain config —
    a roster edit, a character registration, or a gate change invalidates cached
    decisions, exactly like the detector's staleness."""
    import home_canon
    path = Path(roster_path or DEFAULT_ROSTER_PATH)
    roster_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "absent"
    blob = (f"{roster_hash}|{home_canon.fingerprint(home_path)}|{PROMPT_TEMPLATE}|"
            f"{namefix.MODEL}|{namefix.LEAD}|{namefix.WINDOW}|{namefix.TEMPORAL_TOL}|"
            f"{VERSION}|{wordlist_fingerprint()}")
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_roster(roster_path=None):
    """Compact mirror of the frozen detector's loader, extended for the TMAS-56 schema
    (nicknames as FULL name forms; the frozen detector itself stays untouched until
    TMAS-58). Returns (name_forms, canon_forms, code_to_names, alias_to_name):
      name_forms     — every legitimate spelling: canonicals + nicknames (the correction
                       cast and the re-listen prompt)
      canon_forms    — {cleaned canonical or nickname} (correct-as-heard -> never a
                       candidate; a nickname on tape is right, not a garble)
      code_to_names  — DM code -> {suggested form} (the phonetic layer). A nickname's
                       codes join ONLY where they exceed its person's canonical codes —
                       the canonical-preference rule: when a garble sounds like both a
                       person's canonical and their nickname (the Kiki/Kiran class:
                       the two forms share sound codes),
                       the fallback suggests the canonical alone (one card, not two);
                       the re-listen still lands on the nickname whenever the audio
                       actually says it (it's in cast_clean and the prompt).
      alias_to_name  — cleaned alias token -> the person's canonical (the exact layer;
                       aliases are distortions to catch, never legitimate forms)
    Fails loud when the file is missing — callers gate on existence for the skip path."""
    path = Path(roster_path or DEFAULT_ROSTER_PATH)
    roster = json.loads(path.read_text())
    canon_by_id = {p["id"]: p["canonical"] for p in roster["people"]}
    name_forms, canon_forms = [], set()
    code_to_names: dict[str, set] = {}
    for p in roster["people"]:
        canon = p["canonical"]
        name_forms.append(canon)
        canon_forms.add(clean(canon))
        canon_codes = codes(clean(canon))
        for c in canon_codes:
            code_to_names.setdefault(c, set()).add(canon)
        for nick in p.get("nicknames", []):
            name_forms.append(nick)
            canon_forms.add(clean(nick))
            for c in codes(clean(nick)) - canon_codes:   # canonical-preference
                code_to_names.setdefault(c, set()).add(nick)
    name_forms = sorted(name_forms)
    alias_to_name = {}
    for entry in roster["aliases"]:
        ct = clean(entry["token"])
        pid = entry["person_id"]
        if pid not in canon_by_id:
            raise ValueError(f"Alias {entry['token']!r} references unknown person_id {pid!r}.")
        if codes(ct) & set(code_to_names):
            raise ValueError(f"Redundant alias {entry['token']!r}: its Double Metaphone "
                             "codes already reach a canonical or nickname via the "
                             "phonetic layer.")
        alias_to_name[ct] = canon_by_id[pid]
    return name_forms, canon_forms, code_to_names, alias_to_name


def family_candidates(transcript: dict, roster, home_forms=frozenset()) -> list[dict]:
    """Every word the frozen detector would flag, with its roster match riding along —
    the detector's exact semantics: phonetic layer first, then the alias layer, then
    the capitalization gate (checked AFTER matching, so lowercase homophones drop and
    capitalized common-word disguises survive). No dictionary gate, ever (M9a rule).

    `home_forms` (TMAS-57 stage 1): cleaned canonicals of the family's OWN registered
    characters — a correctly-spelled home character is never a candidate, however much
    it sounds like a real person (the story-character→real-person sound-alike class
    dies here)."""
    _, canon_forms, code_to_names, alias_to_name = roster
    cands = []
    for seg in transcript["segments"]:
        for wi, w in enumerate(seg.get("words", [])):   # gap segments carry no words
            c = clean(w["word"])
            if not c or c in canon_forms or c in home_forms:
                continue
            shared = codes(c) & set(code_to_names)
            if shared:
                matched = sorted({n for code in shared for n in code_to_names[code]})
            elif c in alias_to_name:
                matched = [alias_to_name[c]]
            else:
                continue
            if not is_capitalized(w["word"].strip()):
                continue
            cands.append({"story": FIXER, "start": w.get("start"),
                          "blind": w["word"].strip(), "seg_id": seg["id"], "wi": wi,
                          "matched": matched})
    return cands


def _fallback_decision(cand: dict, canonical: str) -> dict:
    """The coverage-first question for a spot the re-listen couldn't help with: the
    suggestion is the detector's own match; the label computes the normal way."""
    return {"story": FIXER, "world": WORLD, "blind": cand["blind"],
            "canonical": canonical, "how": "roster",
            "action": "queued" if is_common(clean(cand["blind"])) else "auto",
            "seg_id": cand["seg_id"], "wi": cand["wi"], "start": cand["start"]}


def run_familyfix(transcript: dict, audio_path: str, roster_path=None,
                  redecode=None, timeout: int = 3600, home_path=None) -> dict:
    """The full stage. Returns {"groups": [...]}. `redecode` is injectable for tests;
    production runs namefix's Whisper worker in a subprocess (ONE model at a time).
    Reads no Name Book — rulings act at view time and the apply pass, never in a run.
    Reads the home cast (a "who exists" source, like the roster) for the guard."""
    import home_canon
    roster = load_roster(roster_path)
    name_forms, _, _, alias_to_name = roster
    cands = family_candidates(transcript, roster,
                              home_forms=frozenset(home_canon.canonical_forms(home_path)))
    if not cands:
        return {"groups": []}

    # The re-decode vocabulary: every legitimate name form (canonicals AND nicknames,
    # each mapping to ITSELF — a re-heard "Kiki" is fixed toward "Kiki", never silently
    # promoted to the canonical), plus the aliases mapping back to their person's
    # canonical (an alias IS a known distortion — a re-listen that lands on it still
    # identifies the person). The PROMPT carries the real names, nicknames included:
    # Whisper can only hear a form it has been primed with.
    cast_clean = {clean(n): n for n in name_forms if clean(n)}
    cast_clean.update(alias_to_name)
    prompt = PROMPT_TEMPLATE.format(names=", ".join(name_forms))
    singles = proper_name_candidates(transcript["segments"])

    jobs = [{"cand": {k: c[k] for k in ("story", "start", "blind", "seg_id", "wi")},
             "world": WORLD, "prompt": prompt, "cast_clean": cast_clean}
            for c in cands if c["start"] is not None]
    if redecode is None:
        from model_runner import run_model
        decisions = run_model(namefix._redecode_worker, audio_path, jobs,
                              sorted(singles), timeout=timeout)
    else:
        decisions = redecode(audio_path, jobs, sorted(singles))

    # Coverage-first: every candidate spot the gate produced nothing for still becomes
    # a question, suggested from the detector's match (one per matched person — a DM
    # code shared by two family members yields two questions with distinct gids).
    decided = {(d["seg_id"], d["wi"]) for d in decisions}
    fallback = [_fallback_decision(c, name)
                for c in cands if (c["seg_id"], c["wi"]) not in decided
                for name in c["matched"]]

    groups = namefix.annotate_signals(
        namefix.group_decisions(decisions + fallback), transcript)
    for g in groups:
        # the audio-evidence axis his verdicts will judge: did the re-listen produce a
        # usable name at ANY of this group's spots (False = pure detector suspicion)
        g["redecode_heard"] = any((o["segment_id"], o["word_index"]) in decided
                                  for o in g["occurrences"])
    return {"groups": groups}
