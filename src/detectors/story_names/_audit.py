#!/usr/bin/env python3
"""Stage-1 per-story name auditor — the v2 architecture only.

PORTED from the sealed EMP probe emp/src/audit_names.py. Only the validated v2
path is copied: the worksheet over RECALL-RECOVERED cards, then a CANON SHIELD that
removes correctly-spelled real names from the inconsistency (M9b) flags only. The
three eval-only architectures (worksheet/fulltext/hybrid), the sweep orchestration
(`audit_session`), and the CLI (`main`) are dropped — this module is imported by the
worker, never run standalone.

Detects, per story, in one local-LLM pass:
  - M9c: a real canon name (Thomas & Friends, Mahabharata, …) SPELLED WRONG.
  - M9b: ONE improvised name spelled SEVERAL different ways.

FIREWALL: reads the transcript + the live Stage-0 regions ONLY. Never the answer key.
LOCAL-ONLY: Gemma-4 E4B via MLX in a fresh subprocess; family content never leaves the machine.
"""
from collections import defaultdict

from detectors.story_names._names import (proper_name_candidates, detect_phrases,
                                          seg_word_text)
from story_segment import extract_json
from detectors.phonetics import codes, clean
from detectors.name_consistency import NameConsistencyDetector, MIN_NAME_LEN

CARD_CHUNK = 16        # worksheet cards per model call (keeps each generation bounded)

# One shared dictionary instance for the v2 recall recovery (wordlist loads lazily, once).
_NCD = NameConsistencyDetector()


def _plain_word(c):
    """Exact dictionary membership — recovery's filter. The M9b detector's _is_common
    also strips a trailing 's', which wrongly reads 'jamis' as a plural of the archaic
    'jami' and would drop a real misspelled name. For folding name-spellings we want
    EXACT membership only: 'jamis' isn't a word (kept); 'today'/'work'/'that' are (dropped)."""
    _NCD._is_common("_warmup_")          # force the wordlist to load onto _NCD._common
    return c in (_NCD._common or set())


# ============================ candidate worksheet =============================
def story_name_cards(segs, recover=False):
    """Per-story name cards (transcript-only). A card is a phonetic cluster of the
    capitalized name candidates (Double Metaphone union-find) OR a multi-word phrase
    card, each with its distinct spellings, occurrence count, up to 3 example lines,
    and occurrence positions. Phrase members are folded out of the single-word cards
    (the 'names are spans' rule).

    recover=True (v2): also fold in tokens that the capitalization gate skips — a
    misspelling that only ever appears lowercase or sentence-initial (e.g. "Jamis") —
    but ONLY when it sounds like an existing name cluster AND is not an ordinary
    dictionary word. That dictionary filter is what lets "Jamis" in while keeping
    "today"/"work" out (they share a sound-code with a name but are real words)."""
    singles = proper_name_candidates(segs)
    phrases = detect_phrases(segs, singles)
    phrase_members = {tok for ph in phrases for tok in ph.split(" ")}
    cluster_singles = {c for c in singles if c not in phrase_members}
    seg_sent = {s["id"]: seg_word_text(s) for s in segs}

    # union-find on shared DM codes
    parent = {c: c for c in cluster_singles}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    code_to_forms = defaultdict(list)
    for c in cluster_singles:
        for code in codes(c):
            code_to_forms[code].append(c)
    for forms in code_to_forms.values():
        for o in forms[1:]:
            union(forms[0], o)

    if recover:  # fold sound-alike, non-dictionary tokens the cap-gate skipped into their cluster
        code_root = {}
        for c in cluster_singles:
            r = find(c)
            for code in codes(c):
                code_root.setdefault(code, r)
        for s in segs:
            for w in s.get("words", []):
                c = clean(w["word"].strip())
                if (not c) or len(c) < MIN_NAME_LEN or c in cluster_singles or c in phrase_members:
                    continue
                if _plain_word(c):          # an ordinary word that merely rhymes — skip
                    continue
                root = next((code_root[code] for code in codes(c) if code in code_root), None)
                if root is not None:
                    parent[c] = root         # join the name cluster it sounds like
                    cluster_singles.add(c)

    surf, cln, cnt, occ = defaultdict(set), defaultdict(set), defaultdict(int), defaultdict(list)
    ex = defaultdict(list)
    for s in segs:
        sent = seg_sent[s["id"]]
        for wi, w in enumerate(s.get("words", [])):
            raw = w["word"].strip()
            c = clean(raw)
            if c in cluster_singles:
                r = find(c)
                surf[r].add(raw); cln[r].add(c); cnt[r] += 1
                occ[r].append({"seg_id": s["id"], "wi": wi})
                if sent and sent not in ex[r] and len(ex[r]) < 3:
                    ex[r].append(sent)

    cards = []
    for r in cln:
        cards.append({"clean": sorted(cln[r]), "surface": sorted(surf[r]), "n": cnt[r],
                      "examples": ex[r], "is_phrase": False, "occ": occ[r]})
    for ph, occs in phrases.items():
        sents, seen = [], set()
        for o in occs:
            t = seg_sent.get(o["seg_id"], "")
            if t and t not in seen:
                seen.add(t); sents.append(t)
        cards.append({"clean": [ph], "surface": sorted({o["surface"] for o in occs}),
                      "n": len(occs), "examples": sents[:3], "is_phrase": True,
                      "occ": [{"seg_id": o["seg_id"], "wi": o["wi"]} for o in occs]})

    cards.sort(key=lambda c: -c["n"])
    for i, c in enumerate(cards):
        c["id"] = i + 1
    return cards


# ================================== prompts ===================================
WORKSHEET_PROMPT = """You are checking the NAMES in one bedtime story a parent told a young child.
This story's world is: {world}.
{context}
Below are the names found in the story — each with the spelling(s) the transcriber used and example lines. For EACH numbered entry, choose ONE verdict:
- "canon_wrong": a real name from this world (or other well-known canon, e.g. Thomas & Friends, Mahabharata) that is SPELLED WRONG. Give the correct spelling in "canonical".
- "inconsistent": ONE made-up name the transcriber spelled SEVERAL different ways. Give the spelling to standardize on in "canonical".
- "substitution": it may be a made-up name written as a DIFFERENT real word/name (you cannot be sure from text alone).
- "ok": a name spelled correctly — a canon name spelled right, OR a made-up name spelled the same way every time.
- "not_name": an ordinary word, not a name.

Names:
{cards}

Return JSON only, no other text:
{{"verdicts": [{{"id": <number>, "verdict": "canon_wrong|inconsistent|substitution|ok|not_name", "canonical": "<correct spelling or empty>", "wrong_spellings": ["<the misspelled form(s)>"]}}]}}
"""

_VERDICT_CASE = {"canon_wrong": "M9c", "inconsistent": "M9b", "substitution": "M9d-suspect"}


# ================================ renderers ===================================
def render_cards(cards):
    out = []
    for c in cards:
        forms = ", ".join(f'"{s}"' for s in c["surface"])
        kind = " [multi-word]" if c["is_phrase"] else ""
        out.append(f'[{c["id"]}] spellings: {forms}  ({c["n"]} uses){kind}')
        for e in c["examples"][:3]:
            out.append(f'    - "{e[:130]}"')
    return "\n".join(out)


# ============================ per-architecture run ============================
def _flag_from_verdict(card, verdict, canonical, wrong):
    case = _VERDICT_CASE.get(verdict)
    if not case:
        return None
    wrong_surface = [w for w in wrong if w] or [s for s in card["surface"] if clean(s) != clean(canonical)]
    if not wrong_surface:
        wrong_surface = card["surface"]
    return {"case": case, "canonical": canonical,
            "wrong_surface": sorted(set(wrong_surface)),
            "wrong_cleaned": sorted({clean(s) for s in wrong_surface if clean(s)}),
            "all_spellings": card["surface"], "card_id": card["id"],
            "evidence": card["examples"][0] if card["examples"] else ""}


def _run_card_pass(gen, world, cards, raw_log, context=""):
    """Worksheet: classify the cards (chunked). Returns (flags, verdicts)."""
    flags, verdicts = [], []
    by_id = {c["id"]: c for c in cards}
    for i in range(0, len(cards), CARD_CHUNK):
        chunk = cards[i:i + CARD_CHUNK]
        prompt = WORKSHEET_PROMPT.format(world=world, context=context,
                                         cards=render_cards(chunk))
        raw = gen(prompt, max_tokens=1100)
        raw_log.append({"pass": "cards", "chunk": [c["id"] for c in chunk], "raw": raw})
        obj = extract_json(raw)
        arr = obj.get("verdicts") if isinstance(obj, dict) else None
        if not isinstance(arr, list):
            continue
        for v in arr:
            if not isinstance(v, dict):
                continue
            try:
                cid = int(str(v.get("id")).strip().strip("[]"))
            except (ValueError, TypeError):
                continue
            card = by_id.get(cid)
            if not card:
                continue
            verdict = str(v.get("verdict", "")).strip().lower()
            canonical = str(v.get("canonical", "")).strip()
            wrong = [str(x).strip() for x in (v.get("wrong_spellings") or []) if str(x).strip()]
            verdicts.append({"id": cid, "spellings": card["surface"],
                             "cleaned": card["clean"], "verdict": verdict,
                             "canonical": canonical})
            f = _flag_from_verdict(card, verdict, canonical, wrong)
            if f:
                flags.append(f)
    return flags, verdicts


# ----------------------- canon dictionary gate (M9c only) ---------------------
# PRODUCTION-ONLY hardening (not in the sealed eval audit_names.py): the worksheet
# reader sometimes dresses an ordinary word up as a misspelled canon name — on the
# held-out Mahabharata it clustered "arrows"/"wars"/"Urzi" and called them canon_wrong
# -> "Arrow" (24 flags, ~22 false). This gate drops those, and is recall-guarded so it
# never removes a real canon name that merely happens to be a dictionary entry.
def _is_ordinary_word(cleaned, singles):
    """True when a cleaned token/canonical is an ordinary common word: the dictionary
    knows it (trailing-'s' tolerated, via the M9b detector's _is_common) AND the story
    never uses it as a capitalized proper name. The proper-name guard (`singles`) is the
    whole point — names like Krishna, Drona, Dhritarashtra ARE dictionary entries, so a
    bare dictionary test would delete real catches; but in their own story they appear
    capitalized mid-sentence, so they sit in `singles` and are protected here."""
    if not cleaned or cleaned in singles:
        return False
    return _NCD._is_common(cleaned)


def gate_canon_flags(flags, singles):
    """Dictionary-gate the canon (M9c) flags so an ordinary English word can't be flagged
    as a misspelled canon name. Recall-guarded cuts (M9b/M9d pass straight through):
      (1) WHOLE-FLAG canonical cut — drop a flag whose proposed canonical is an ordinary
          word AND whose cluster also contains an ordinary word ("Arrow", clustered with
          "arrows"/"wars"). The "cluster also contains an ordinary word" condition is the
          recall guard: a real canon name spelled wrong produces NON-words
          (Dhrashtra->Dhritarashtra), so a canon name the dictionary merely happens to list
          is never dropped on the canonical alone.
      (2) WRONG-SPELLING cut — drop any wrong spelling that is itself an ordinary word
          ("arrows", "wars"); if none survive, drop the flag.
    `dict_gated` records what each surviving flag lost, mirroring the M9b shield's
    `shielded` audit trail."""
    kept = []
    for f in flags:
        if f["case"] != "M9c":
            kept.append(f)
            continue
        ordinary_wrong = [c for c in f["wrong_cleaned"] if _is_ordinary_word(c, singles)]
        if _is_ordinary_word(clean(f["canonical"]), singles) and ordinary_wrong:
            continue                                   # ordinary-word cluster dressed as canon
        survive = [c for c in f["wrong_cleaned"] if not _is_ordinary_word(c, singles)]
        if not survive:
            continue                                   # every wrong spelling was a common word
        gated = sorted(set(f["wrong_cleaned"]) - set(survive))
        if gated:
            keep_set = set(survive)
            f["dict_gated"] = gated
            f["wrong_cleaned"] = sorted(keep_set)
            f["wrong_surface"] = sorted(s for s in f["wrong_surface"] if clean(s) in keep_set)
        kept.append(f)
    return kept


# ---- v2: worksheet (with recall recovery) + a canon shield over the M9b flags ----
SHIELD_PROMPT = """This story is set in the world of "{world}". Here are some name spellings the checker thinks might be one made-up name spelled inconsistently:
{names}

Which of these (if any) are actually a CORRECTLY-SPELLED, real, well-known character or place from that world — a real name spelled right, NOT a made-up name and NOT a misspelling? (For a Thomas & Friends story, real engines like James, Thomas, Percy, Gordon count; an invented engine does not.)

Return JSON only, no other text:
{{"real_correct": ["<spelling>", ...]}}
If none are real correctly-spelled names, return {{"real_correct": []}}.
"""


def recognize_real_names(gen, world, surfaces, raw_log):
    """Ask the model which of these spellings are correctly-spelled real canon names —
    the ones to PROTECT from an inconsistency flag (so 'James' isn't called an error)."""
    if not surfaces:
        return set()
    raw = gen(SHIELD_PROMPT.format(world=world, names=", ".join(f'"{s}"' for s in sorted(surfaces))),
              max_tokens=160)
    raw_log.append({"pass": "shield", "raw": raw})
    obj = extract_json(raw)
    out = set()
    if isinstance(obj, dict) and isinstance(obj.get("real_correct"), list):
        for x in obj["real_correct"]:
            c = clean(str(x))
            if c:
                out.add(c)
    return out


def run_v2(gen, world, segs, cards, raw_log):
    """The shipping candidate: the worksheet over recall-recovered cards, then a canon
    shield that removes correctly-spelled real names from the INCONSISTENCY (M9b) flags
    only — never from canon-wrong (M9c) flags, so a misspelled canon name stays caught."""
    flags, verdicts = _run_card_pass(gen, world, cards, raw_log, context="")
    flags = gate_canon_flags(flags, proper_name_candidates(segs))   # M9c dictionary gate
    m9b_surfaces = {s for f in flags if f["case"] == "M9b" for s in f["wrong_surface"]}
    canon = recognize_real_names(gen, world, m9b_surfaces, raw_log)
    if not canon:
        return flags, verdicts
    kept = []
    for f in flags:
        if f["case"] != "M9b":
            kept.append(f); continue
        ws = [s for s in f["wrong_surface"] if clean(s) not in canon]
        if not ws:
            continue  # the whole flag was a protected real name -> drop it
        shielded = sorted({clean(s) for s in f["wrong_surface"]} & canon)
        f.update({"wrong_surface": sorted(set(ws)),
                  "wrong_cleaned": sorted({clean(s) for s in ws if clean(s)}),
                  "shielded": shielded})
        kept.append(f)
    return kept, verdicts
