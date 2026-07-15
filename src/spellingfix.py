#!/usr/bin/env python3
"""spellingfix — the invented-name spelling picker (the M9b fixer), SHADOW like every fixer.

An improvised character with no canon anywhere (the child's own engines: Jiraki, Jammus)
can come out of Whisper spelled several ways in one night, which makes the character
impossible to follow across the transcript. The frozen m9b detector FLAGS the
inconsistency; this stage PROPOSES the one spelling to follow the character by and asks —
a labeled question (label "inconsistent-spelling") in the shared shadow queue. Calibration
rule as always: only add labeled questions; never write a transcript or the Name Book.

Mechanism (pure code, no model — milliseconds):
  1. CLUSTER the recording's capitalized >=MIN_NAME_LEN tokens by shared Double Metaphone
     code (union-find over distinct cleaned spellings) — the m9b detector's exact
     semantics, re-implemented against the IN-MEMORY transcript (the frozen detector's
     run() reads from disk and can't serve mid-enrichment; its module-level pieces are
     imported, never copied). Family-roster names excluded (m9a/familyfix own those);
     all-ordinary-word clusters dropped by the dictionary filter with the trailing-s rule.
     KNOWN BOUNDARY (deferred): an invented name that is also a dictionary word in every
     spelling (Bibi) is dropped here — recovering it needs the offline LLM judge.
  2. PROPOSE: the most-frequent spelling wins; tie -> alphabetical. Deterministic — the
     proposal is a starting point for the human, not a truth claim (his verdict, or an
     override-then-yes with a different spelling, is the ruling).
  3. ASK: one question per DEVIANT spelling (the spots that would change), all carrying
     the cluster's id — so one verdict on any card rules the WHOLE cluster (the standing
     pick folds the sibling cards at view time). gid = spelling:{cluster_id}:{heard}.

World: the story's recognized world when every occurrence of the cluster sits in stories
agreeing on one world; else "" (a verdict on a world-less question records as audit-only —
the cross-night "home canon" book is deliberately not built; Choksi's design call).
Verdicts route to the Name Book's SPELLINGS slot: yes = unify to this spelling, no =
"don't unify these" (null pick) — never a word-level ruling.
"""
import hashlib
from collections import Counter
from pathlib import Path

from detectors.name_consistency import (CONTRACTIONS, DEFAULT_WORDLIST, EXTRA_COMMON,
                                        MIN_NAME_LEN, _UnionFind)
from detectors.phonetics import clean, codes, is_capitalized
from detectors.roster import DEFAULT_ROSTER_PATH, RosterMatcher
from detectors.story_names._worker import build_regions
from namefix import SPELLING_LABEL

FIXER = "spelling"
VERSION = "1.2.0"   # 1.1.0: defer-to-namefix — a cluster touching words namefix asks
                    # about is namefix's territory (the detector-era canon-dedup
                    # lesson, applied to the fixers; see run_spellingfix)
                    # 1.2.0: home canon (TMAS-57) — the registered spelling outranks
                    # the vote, and world-less clusters file under "Home"


def config_fingerprint(wordlist_path=DEFAULT_WORDLIST, roster_path=None,
                       home_path=None) -> str:
    """Ties a run to the clustering config + all three word sources (a wordlist edit,
    a roster edit, or a character registration re-asks; constants ride in VERSION)."""
    import home_canon
    def _hash(p):
        p = Path(p) if p else None
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p and p.exists() else "absent"
    blob = (f"{VERSION}|{MIN_NAME_LEN}|{sorted(EXTRA_COMMON)}|{sorted(CONTRACTIONS)}|"
            f"{_hash(wordlist_path)}|{_hash(roster_path or DEFAULT_ROSTER_PATH)}|"
            f"{home_canon.fingerprint(home_path)}")
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _common_words(wordlist_path) -> set[str]:
    """The m9b detector's ordinary-word set: dictionary + fillers + cleaned contractions.
    wordlist_path=None -> stoplist-only (the detector's degraded/test mode)."""
    common = set(EXTRA_COMMON) | CONTRACTIONS
    if wordlist_path and Path(wordlist_path).exists():
        common |= {line.strip().lower()
                   for line in Path(wordlist_path).read_text(errors="ignore").splitlines()
                   if line.strip()}
    return common


def _is_common(c: str, common: set[str]) -> bool:
    return c in common or (c.endswith("s") and c[:-1] in common)


def find_clusters(transcript: dict, wordlist_path=DEFAULT_WORDLIST,
                  roster_path=None) -> list[dict]:
    """The m9b clustering over the in-memory transcript. Returns one dict per
    INCONSISTENT, dictionary-passing cluster:
      {cluster_id, forms: {cleaned: {"surface": str, "count": int, "occurrences": [...]}}}
    cluster_id = the cluster's min cleaned spelling (the stable key picks fold on)."""
    roster = RosterMatcher(roster_path)   # absent roster -> matches nothing (graceful)
    common = _common_words(wordlist_path)

    occs = []          # (seg_id, word_index, start, raw, cleaned)
    form_codes = {}    # cleaned form -> DM codes
    for seg in transcript["segments"]:
        for wi, w in enumerate(seg.get("words", [])):   # gap segments carry no words
            raw = w["word"].strip()
            c = clean(raw)
            if not c or not is_capitalized(raw) or len(c) < MIN_NAME_LEN:
                continue
            if roster.is_roster_name(c):
                continue   # family name — familyfix owns it
            occs.append((seg["id"], wi, w.get("start"), raw, c))
            form_codes.setdefault(c, codes(c))

    uf = _UnionFind()
    for c in form_codes:
        uf.add(c)
    code_to_forms: dict[str, list] = {}
    for form, fcodes in form_codes.items():
        for code in fcodes:
            code_to_forms.setdefault(code, []).append(form)
    for forms in code_to_forms.values():
        for other in forms[1:]:
            uf.union(forms[0], other)

    by_root: dict[str, dict] = {}
    for sid, wi, start, raw, c in occs:
        root = uf.find(c)
        forms = by_root.setdefault(root, {})
        slot = forms.setdefault(c, {"surface": raw.rstrip(" .,?!…\"'"), "count": 0,
                                    "occurrences": []})
        slot["count"] += 1
        slot["occurrences"].append({"segment_id": sid, "word_index": wi,
                                    "start": start, "token": raw})

    clusters = []
    for forms in by_root.values():
        if len(forms) <= 1:
            continue   # one consistent spelling — never a question
        if all(_is_common(f, common) for f in forms):
            continue   # all-ordinary cluster (What/Whats) — the Bibi-class boundary
        clusters.append({"cluster_id": min(forms), "forms": forms})
    return clusters


def propose_spelling(forms: dict) -> str:
    """Most-frequent cleaned spelling wins; tie -> alphabetical. Deterministic."""
    counts = Counter({f: info["count"] for f, info in forms.items()})
    top = max(counts.values())
    return sorted(f for f, n in counts.items() if n == top)[0]


def _segment_worlds(transcript: dict, story_worlds: dict) -> dict:
    """segment id -> recognized world, from the story regions. Segments outside any
    story map to ""."""
    segments = transcript["segments"]
    pos_of = {s["id"]: i for i, s in enumerate(segments)}
    seg_world = {}
    for r in build_regions(transcript.get("_stories") or [], pos_of):
        world = story_worlds.get(r["idx"], "")
        for pos in range(r["start_pos"], r["end_pos"] + 1):
            seg_world[segments[pos]["id"]] = world
    return seg_world


def defer_words_from(namefix_groups: list) -> set[str]:
    """The words namefix already asks about in this session — its heard forms AND its
    suggested canonicals (cleaned). Any cluster touching one is namefix's territory."""
    out = set()
    for g in namefix_groups:
        out.add(g["heard_cleaned"])
        c = clean(g["canonical"])
        if c:
            out.add(c)
    return out


def run_spellingfix(transcript: dict, story_worlds: dict | None = None,
                    wordlist_path=DEFAULT_WORDLIST, roster_path=None,
                    defer_words: set | None = None, home_path=None) -> dict:
    """The full stage. Returns {"groups": [...]} — one group per deviant spelling, all
    of a cluster's groups sharing cluster_id (one verdict rules the cluster; the pick
    folds the siblings at view time). `story_worlds` maps story idx -> recognized world
    (namefix's report); None/missing -> world "" (verdicts record audit-only).

    `defer_words` (v1.1.0): cleaned words namefix asks about this session (build with
    defer_words_from). A cluster containing ANY of them is dropped — it's a canon-name
    garble family, namefix's territory, and a majority vote there actively proposes the
    wrong thing (the baseline run proposed rewriting a correct canon spelling toward
    its more-frequent garble, and unifying an invented engine into the canon name it
    sounds like). Same lesson as the detector era's M9b-defers-to-M9c dedup.

    Home canon (v1.2.0, TMAS-57 — defer-to-namefix stays senior to both effects):
    a cluster containing a REGISTERED character's spelling proposes that spelling
    (method "home-canon" — the registry outranks the vote, and the registered form is
    never a deviant); and a cluster whose spots all sit outside recognized story worlds
    files under world "Home", so its picks land durably in name-book/home.json."""
    import home_canon
    seg_world = _segment_worlds(transcript, story_worlds or {})
    defer = defer_words or set()
    home_forms = home_canon.canonical_forms(home_path)
    groups = []
    for cl in find_clusters(transcript, wordlist_path, roster_path):
        if any(f in defer for f in cl["forms"]):
            continue   # namefix already asks about this word family
        # the cluster's world: unanimous across ALL its spots. Unanimously OUTSIDE any
        # recognized world -> "Home" (the family's own canon is the ruling home for
        # world-less clusters — his call, 2026-07-03). Genuinely mixed -> "" (no home).
        worlds = {seg_world.get(o["segment_id"], "")
                  for info in cl["forms"].values() for o in info["occurrences"]}
        world = worlds.pop() if len(worlds) == 1 else None
        world = home_canon.WORLD if world == "" else (world or "")
        home_in = sorted((f for f in cl["forms"] if f in home_forms),
                         key=lambda f: (-cl["forms"][f]["count"], f))
        if home_in:
            proposed = home_in[0]
            proposed_surface = home_forms[proposed]   # the REGISTERED spelling
            method = "home-canon"
        else:
            proposed = propose_spelling(cl["forms"])
            proposed_surface = cl["forms"][proposed]["surface"]
            method = "majority"
        spellings = sorted(info["surface"] for info in cl["forms"].values())
        for form, info in sorted(cl["forms"].items()):
            if form == proposed:
                continue   # the winning spelling's spots don't change — no question
            groups.append({
                "gid": f"spelling:{cl['cluster_id']}:{form}",
                "world": world,
                "story_id": None,   # clusters span the whole recording, not one story
                "heard": info["surface"],
                "heard_cleaned": form,
                "canonical": proposed_surface,
                "method": method,
                "label": SPELLING_LABEL,
                "dm_share": bool(codes(form) & codes(proposed)),
                "cluster_id": cl["cluster_id"],
                "cluster_spellings": spellings,
                "majority_count": cl["forms"][proposed]["count"],
                # namefix's suspicion signals don't apply here; neutral values keep the
                # UI chips quiet (>=1 sibling by construction — the cluster IS wobble)
                "echo_in_sentence": False,
                "n_sibling_spellings": len(cl["forms"]) - 1,
                "occurrences": info["occurrences"],
            })
    return {"groups": groups}
