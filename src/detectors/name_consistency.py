"""M9b — inconsistent-name detector.

Catches an improvised story name spelled several ways within one recording —
the engines Jiraki/Jameis/Jammus, or a name Whisper renders one way here and
another there. Unlike the M9a detector there is no roster and no truth source:
the *only* defect is inconsistency, so the detector flags it and leaves the
choice of a winning spelling to a separate fix.

Mechanism (code-only, no model): group the recording's capitalized tokens into
phonetic clusters by shared Double Metaphone code; a cluster spelled more than one
way is inconsistent. We flag only the *deviating* occurrences: if one spelling is
the recording's majority (used in more than half the occurrences), it is treated as
the intended one and its occurrences are NOT flagged — only the odd spellings are.
With no clear majority (spellings scattered, no winner), every occurrence is flagged
so the unsettled name can be reviewed. A name spelled one consistent way is never
flagged. The majority is a *within-recording* call about what to surface, never a
global canonical — settling one spelling per character across recordings is a
separate concern (cross-session consistency).

Family-roster names are EXCLUDED (deferred to the m9a detector): a roster match —
the child's or a parent's name and its misspellings — is dropped before clustering,
so m9b reports only *improvised* inconsistency and never double-counts a family name
m9a already owns. The exclusion is graceful: with no roster file, m9b falls back to
its original roster-agnostic behavior. Flags can still echo improvised story names,
so the per-session detections.json stays under gitignored sessions/.
"""

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from detectors.base import Detector, load_transcript
from detectors.phonetics import clean, codes, is_capitalized
from detectors.roster import RosterMatcher, DEFAULT_ROSTER_PATH

# Precision layer (see the score_m9b validation: code-only clustering had 0.94
# recall but 0.06 precision — it drowned in capitalized common words and
# interjections sharing metaphone codes). Two cheap deterministic gates:
DEFAULT_WORDLIST = Path("/usr/share/dict/words")  # the system English dictionary
MIN_NAME_LEN = 4   # improvised names are >=4 chars; I/Oh/We/No/Uh are not
# Fillers a formal wordlist may omit (the <4-char ones are already dropped by
# MIN_NAME_LEN; these are the >=4 stragglers).
EXTRA_COMMON = {"yeah", "whoa", "woah", "haha", "hehe", "uhhuh", "mmhmm", "okay"}
# Contractions: clean() strips the apostrophe, so "You're"->"youre", "We've"->"weve"
# — forms the dictionary lacks, which otherwise drag a whole cluster (incl. ordinary
# words like "where") past the all-common filter and get flagged as names. Only the
# 're/'ve/'ll/'d/'m and n't forms are listed; the 's forms ("what's"->"whats") are
# already caught by the trailing-s stem check, and that same check keeps real name
# possessives ("Martie's"->"marties") OUT of this set, so they stay name candidates.
CONTRACTIONS = {
    "youre", "theyre",
    "weve", "youve", "theyve", "couldve", "wouldve", "shouldve", "mightve",
    "youll", "theyll", "itll", "thatll", "wholl", "whatll",
    "youd", "theyd", "hed", "itd", "thatd", "whod",
    "im",
    "dont", "cant", "wont", "didnt", "doesnt", "isnt", "arent", "wasnt", "werent",
    "hasnt", "havent", "hadnt", "wouldnt", "couldnt", "shouldnt", "mustnt", "aint",
}


class _UnionFind:
    """Minimal union-find over hashable items (the distinct cleaned spellings)."""

    def __init__(self):
        self.parent = {}

    def add(self, x):
        self.parent.setdefault(x, x)

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


class NameConsistencyDetector(Detector):
    id = "m9b-name-consistency"
    label = "Inconsistent name spelling"
    failure_mode = "M9b"
    version = "1.2.0"  # 1.2.0: flag only deviating spellings (leave the majority alone)
    accepts_judge = True  # run(session_dir, judge=...) enables the offline LLM layer

    def __init__(self, wordlist_path=DEFAULT_WORDLIST, roster_path=None):
        # wordlist_path=None → stoplist-only (degraded, used by tests).
        self.wordlist_path = Path(wordlist_path) if wordlist_path else None
        # Family-roster matcher excludes m9a's names so m9b reports only improvised
        # inconsistency. Graceful — no roster file → matches nothing (roster-agnostic).
        self.roster_path = Path(roster_path) if roster_path else DEFAULT_ROSTER_PATH
        self._roster = RosterMatcher(self.roster_path)
        self._common = None  # loaded lazily

    def config_fingerprint(self) -> str | None:
        """Hash of the family roster — now an input, since m9b excludes roster names.
        None when the roster is absent (roster-agnostic fallback). Mirrors m9a so a
        roster edit re-runs m9b; it also marks pre-1.1.0 sections (fingerprint None)
        stale, so the scan picks up this precision change."""
        if self.roster_path.exists():
            return "sha256:" + hashlib.sha256(self.roster_path.read_bytes()).hexdigest()
        return None

    def _is_common(self, c: str) -> bool:
        """Is this cleaned token an ordinary English word? (in the dictionary, a known
        filler/contraction, or a plural/possessive/3sg of a dictionary word — the
        trailing-s strip catches cleaned 's contractions like 'whats')."""
        if self._common is None:
            common = set(EXTRA_COMMON) | CONTRACTIONS
            if self.wordlist_path and self.wordlist_path.exists():
                common |= {
                    line.strip().lower()
                    for line in self.wordlist_path.read_text(errors="ignore").splitlines()
                    if line.strip()
                }
            self._common = common
        return c in self._common or (c.endswith("s") and c[:-1] in self._common)

    def run(self, session_dir: Path, judge=None) -> dict:
        """Detect inconsistently-spelled names.

        Code-only by default. `judge` is an optional callable for the offline
        LLM layer: it receives the ambiguous all-common clusters (the residue
        the dictionary filter would drop — improvised names that are also
        dictionary words, like Bibi/Bacchus) and returns the set of cluster_ids
        to keep. It is never invoked from the live API (too slow); only the
        `detect.py --judge` path supplies it.
        """
        data = load_transcript(session_dir)
        seg_text = {seg["id"]: (seg.get("text") or "").strip() for seg in data["segments"]}

        # 1. Collect name-candidate occurrences: capitalized (name-shaped) and at
        #    least MIN_NAME_LEN chars (drops short function words / interjections).
        occurrences = []  # (segment_id, word_index, start, end, raw, cleaned)
        form_codes = {}   # cleaned form -> set of Double Metaphone codes
        n_tokens = 0
        for seg in data["segments"]:
            for wi, w in enumerate(seg.get("words", [])):  # gap segments have no words
                raw = w["word"].strip()
                c = clean(raw)
                if not c:
                    continue
                n_tokens += 1
                if not is_capitalized(raw) or len(c) < MIN_NAME_LEN:
                    continue
                if self._roster.is_roster_name(c):
                    continue  # family name — m9a owns it; don't double-count in m9b
                occurrences.append((seg["id"], wi, w.get("start"), w.get("end"), raw, c))
                form_codes.setdefault(c, codes(c))

        # 2. Cluster the distinct spellings: union any two that share a DM code.
        uf = _UnionFind()
        for c in form_codes:
            uf.add(c)
        code_to_forms = defaultdict(list)
        for form, fcodes in form_codes.items():
            for code in fcodes:
                code_to_forms[code].append(form)
        for forms in code_to_forms.values():
            for other in forms[1:]:
                uf.union(forms[0], other)

        # 3. A cluster is inconsistent iff it holds >1 distinct cleaned spelling.
        cluster_forms = defaultdict(set)            # root -> {cleaned forms}
        cluster_surface = defaultdict(set)          # root -> {raw surface forms}
        cluster_count = defaultdict(int)            # root -> total occurrences
        cluster_segs = defaultdict(list)            # root -> [segment ids]
        cluster_form_counts = defaultdict(Counter)  # root -> Counter(cleaned form)
        cluster_form_surface = defaultdict(dict)    # root -> {cleaned form: a surface}
        for sid, wi, start, end, raw, c in occurrences:
            root = uf.find(c)
            cluster_forms[root].add(c)
            cluster_surface[root].add(raw)
            cluster_count[root] += 1
            cluster_segs[root].append(sid)
            cluster_form_counts[root][c] += 1
            cluster_form_surface[root].setdefault(c, raw)
        # Split the inconsistent clusters by the dictionary filter. A cluster with
        # at least one non-ordinary spelling is a clear name (Pataki/Bacchus
        # survives via "pataki" — the misspelling is the signal). A cluster that is
        # *all* common words (What/Whats, Yeah/Whoa) is ambiguous: usually junk,
        # but sometimes an invented name that is also a dictionary word (Bibi).
        auto_keep, candidate_roots = set(), []
        for r, forms in cluster_forms.items():
            if len(forms) <= 1:
                continue
            if any(not self._is_common(f) for f in forms):
                auto_keep.add(r)
            else:
                candidate_roots.append(r)

        inconsistent = set(auto_keep)
        if judge and candidate_roots:
            candidates = [{
                "cluster_id": min(cluster_forms[r]),
                "spellings": sorted(cluster_surface[r]),
                "examples": [seg_text[s] for s in dict.fromkeys(cluster_segs[r])
                             if seg_text.get(s)][:3],
            } for r in candidate_roots]
            kept_ids = set(judge(candidates))
            inconsistent |= {r for r in candidate_roots
                             if min(cluster_forms[r]) in kept_ids}

        # 4. Per inconsistent cluster, find the in-recording majority spelling — the one
        #    used in more than half the occurrences. It is treated as the intended spelling
        #    *for this recording* (NOT a global canonical), so its occurrences are not
        #    flagged. With no majority, the name is unsettled and every occurrence is.
        cluster_majority = {}  # root -> (cleaned_form, surface, count) or None
        for r in inconsistent:
            top_form, top_n = cluster_form_counts[r].most_common(1)[0]
            if top_n * 2 > cluster_count[r]:  # strict majority
                surface = cluster_form_surface[r][top_form].rstrip(" .,?!…\"'")
                cluster_majority[r] = (top_form, surface, top_n)
            else:
                cluster_majority[r] = None

        # 5. Flag the deviating occurrences (everything except the majority spelling).
        flags = []
        for sid, wi, start, end, raw, c in occurrences:
            root = uf.find(c)
            if root not in inconsistent:
                continue
            maj = cluster_majority[root]
            if maj is not None and c == maj[0]:
                continue  # the in-recording majority (intended) spelling — not an error
            flags.append({
                "segment_id": sid,
                "word_index": wi,
                "start": start,
                "end": end,
                "token": raw,
                "cleaned": c,
                "dm_codes": sorted(form_codes[c]),
                "cluster_id": min(cluster_forms[root]),
                "cluster_spellings": sorted(cluster_surface[root]),
                "n_cluster_occurrences": cluster_count[root],
                # The recording's majority spelling this token deviates from, if any —
                # None when no spelling dominates. A within-recording reference, NOT the
                # character's global canonical (that is the cross-session bible's job).
                "majority_spelling": maj[1] if maj else None,
                "majority_count": maj[2] if maj else None,
            })
        return {"n_word_tokens": n_tokens, "flags": flags}
