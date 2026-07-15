#!/usr/bin/env python3
"""Name-candidate generation + story-region helpers for the per-story name auditor.

PORTED from the sealed EMP probes: `proper_name_candidates` / `detect_phrases` /
`PHRASE_STOP` / `MIN_NAME_LEN` / `SENT_END` come from emp/src/name_truth.py, and
`seg_word_text` / `story_segments` from emp/src/audit_common.py. Only the functions
the v2 audit path actually reaches are copied — the review-tool half of name_truth.py
(`collect_flags`, the HTML/audio server) and its `FamilyNameDetector` import are
deliberately NOT ported, so this module carries no dependency on the gitignored
family-name roster.

Non-negotiable encoded in seg_word_text (see emp.md "Stage 1, step 1"): read the
DELIVERED word tokens, never the segment `text`. LLM-normalization rewrites the
words but not `text` (a known pipeline bug), so `text` can still show a
pre-correction mishearing while the words show the corrected spelling.
"""
from collections import Counter, defaultdict

from detectors.phonetics import clean, is_capitalized

MIN_NAME_LEN = 4  # matches the M9b detector; raise the recall floor here if a short name is missed
SENT_END = (".", "!", "?", "…", '."', '?"', '!"')


def seg_word_text(seg):
    """The segment's sentence built from the (corrected) WORD tokens — NOT seg['text'],
    which the normalizer leaves stale. Empty string if a gap segment with no words."""
    ws = seg.get("words") or []
    return " ".join(w["word"].strip() for w in ws).strip() if ws else (seg.get("text") or "").strip()


def story_segments(rich, region, pos_of):
    """The rich segment dicts that fall inside one story region, in order."""
    lo, hi = region["start_pos"], region["end_pos"]
    return [s for s in rich["segments"]
            if lo <= pos_of.get(s["id"], -1) <= hi]


def proper_name_candidates(segments, min_len=MIN_NAME_LEN):
    """Cleaned tokens that appear Capitalized in a NON-sentence-initial position at
    least once — a strong proper-noun signal, since English only capitalizes
    mid-sentence for names. Dictionary-agnostic on purpose: Thomas and Bacchus are
    dictionary words but still names, and the point of the sweep is to miss nothing
    (the dictionary filter is exactly what hides them from the detector)."""
    cands = set()
    for s in segments:
        words = s.get("words", [])
        for i, w in enumerate(words):
            raw = w["word"].strip()
            c = clean(raw)
            prev = words[i - 1]["word"].strip() if i > 0 else ""
            sentence_initial = i == 0 or prev.endswith(SENT_END)
            if c and len(c) >= min_len and is_capitalized(raw) and not sentence_initial:
                cands.add(c)
    return cands


# Function words + discourse fillers that should never begin or end a NAME phrase — kills
# "the ducky", "ducky said", and narrated kid-speech fillers like "like Urjun" / "exactly
# Arthi", while keeping content-modifier names like "rubber ducky" / "cruel kid". The rule:
# filler/function words ONLY — never a descriptive modifier, which carries real name meaning
# (so "little", "old", "rubber" stay OUT). Stripping a glued filler also un-blinds the phonetic
# canon matcher: "Urjun" alone sounds like Arjuna, but "like Urjun" matches nothing.
PHRASE_STOP = {"the", "a", "an", "my", "your", "his", "her", "their", "our", "its", "that",
               "this", "these", "those", "said", "says", "and", "but", "or", "to", "of", "in",
               "on", "at", "with", "is", "was", "are", "were", "it", "he", "she", "they", "you",
               "i", "we", "me", "him", "them", "there", "here", "what", "who", "when", "then",
               "so", "no", "not", "do", "does", "did", "had", "have", "for", "from", "be",
               "like", "just", "really", "very", "actually", "exactly", "okay", "well",
               "yeah", "oh", "um", "uh", "now", "also", "maybe"}


def detect_phrases(segments, single_cands, min_count=2, dominance=0.5):
    """Recurring within-segment BIGRAMS that form a multi-word name: a non-function-word
    MODIFIER followed by a name-candidate HEAD that is *dominantly* preceded by that
    modifier (>= `dominance` of the head's occurrences). The collocation test is what keeps
    'rubber ducky' / 'cruel kid' (the pair sticks together) while dropping loose fragments
    like 'magic rubber' or 'rosie mom' (rubber/mom appear in many other contexts). Returns
    {phrase_cleaned: [occ dicts]}."""
    unigram, bigram = Counter(), Counter()
    for s in segments:
        cl = [clean(w["word"].strip()) for w in s.get("words", [])]
        for c in cl:
            if c:
                unigram[c] += 1
        for i in range(len(cl) - 1):
            a, b = cl[i], cl[i + 1]
            if a and b and a not in PHRASE_STOP and b in single_cands:
                bigram[(a, b)] += 1
    keep = {bg for bg, n in bigram.items()
            if n >= min_count and unigram[bg[1]] and n / unigram[bg[1]] >= dominance}
    out = defaultdict(list)
    for s in segments:
        ws = s.get("words", [])
        cl = [clean(w["word"].strip()) for w in ws]
        for i in range(len(ws) - 1):
            if (cl[i], cl[i + 1]) in keep:
                w0, w1 = ws[i], ws[i + 1]
                out[f"{cl[i]} {cl[i + 1]}"].append({
                    "seg_id": s["id"], "wi": i,
                    "surface": (w0["word"].strip() + " " + w1["word"].strip()).strip(),
                    "w_start": w0.get("start"), "w_end": w1.get("end"),
                    "cap": is_capitalized(w0["word"].strip()) or is_capitalized(w1["word"].strip()),
                })
    return out
