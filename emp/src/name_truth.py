#!/usr/bin/env python3
"""Human ground-truth review of the flagged names in one session.  CO-EDITABLE.

Lists every M9a/M9b flagged name (one card per distinct spelling), in context and
with per-occurrence audio, and lets a human record the truth:

  - per spelling: what KIND of name it is — Roster (family), TV canon (Thomas &
    Friends), Improvised (no canon), Place/other real word, Not a name — and the
    correct spelling.
  - per occurrence: a "true name", because a phonetically-collided cluster (James
    the TV engine vs the improvised Jammus, both Double Metaphone JMS) mixes two
    real names under one spelling. The spelling can't separate them; the ear can,
    so each occurrence plays the audio around it.

Judgments persist to a gitignored JSON sidecar both the human and Claude edit
between turns. Same shape as emp/src/count.py: a generator that bakes the current
labels into the HTML (still reads on file://) plus a --serve bridge that writes
edits back one field at a time (a browser can't write disk itself) and streams the
session audio with HTTP range support so clips can seek.

    python emp/src/name_truth.py 20260129-204404            # write the static HTML
    python emp/src/name_truth.py 20260129-204404 --serve    # serve with audio + saving

PRIVACY: rows echo real roster names and improvised story names, and the audio is
family content — all of it stays under the gitignored emp/results/visuals/<id>/ and
the local-only session audio; nothing is committed.
"""
import argparse
import fcntl
import html
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metaphone import doublemetaphone  # noqa: E402

from detectors.family_names import FamilyNameDetector  # noqa: E402
from detectors.name_consistency import NameConsistencyDetector  # noqa: E402
from detectors.phonetics import clean, codes, is_capitalized  # noqa: E402
from detectors.roster import RosterMatcher  # noqa: E402
import home_canon  # noqa: E402

# (value, label) — value is what is stored; "" is the unset state.
CATEGORIES = [
    ("", "— unset —"),
    ("roster", "Roster (family)"),
    # value stays "tv-canon" for back-compat (downstream treats it as generic M9c canon);
    # label is source-neutral so it fits any external canon (Thomas, Mahabharata, …)
    ("tv-canon", "Sourced canon (external — Thomas, Mahabharata, …)"),
    ("improvised", "Improvised (no canon)"),
    ("place-other", "Place / other real word"),
    ("not-a-name", "Not a name"),
]
ITEM_DEFAULT = {"category": "", "true_spelling": "", "notes": "", "provenance": ""}
OCC_DEFAULT = {"true_name": "", "scope": "", "kind": ""}
# `kind` is only needed when a spot's true_name names a character that has NO item of its
# own — because `items` is keyed by the spelling Whisper DELIVERED, a name it never once
# spelled right has nowhere else to record its name-kind ("Japanese" -> Jumping Portal;
# "The..." -> Dhi). Same vocabulary as an item's category. Blank everywhere else: the
# scorer takes the kind from the item the gold names.
# `scope` says WHICH JOB a per-spot row records — the key does three at once:
#   "name"   Job 1: the true NAME at this spot. A name-fixer must produce it, so the
#            transformation scorer grades it ("Cool," -> Cruel; "things" -> James).
#   "speech" Job 2: the true SPEECH at this spot — a hallucination ([nothing heard]),
#            unintelligible audio, the wrong person ("Mommy!" -> Dad), or words Whisper
#            dropped ("Krishna," -> Jai Sri Krishna). NOT a name-fixer's job: it gets no
#            credit for producing these and is never marked down for missing them. The
#            spot still counts as must-not-touch — mangling it is real damage.
#   "tail"   Job 3: NO decision of its own — this spot is part of the name that starts at
#            an earlier spot. Whisper splits a name across several word-tokens ("you" +
#            "dished" + "there." = Yudhishthira), and without this the key can only rule
#            the FIRST of them: a fixer that writes Yudhishthira at the head and leaves
#            "dished there." standing scores FIXED for producing "Yudhishthira dished
#            there." A tail carries no true_name — it hands its token to the head's span,
#            so the whole span must READ the name to pass, leftovers and all. Ruled by
#            Choksi 2026-07-14 (TMAS-101) after he heard the split by ear mid-pass.
#   ""       Unstamped: score_transform.py infers (bracketed/quoted -> speech, else name)
#            and PRINTS the inference. Stamp anything the inference gets wrong. A tail is
#            never inferred — nothing in the text of a spot says it belongs to its
#            neighbour; only the ear knows.
# Ruled by Choksi 2026-07-13 (TMAS-100) after the scorer surfaced the two jobs sharing
# one field; the 15 speech spots in the 7 keys are stamped.
OCC_SCOPES = ("name", "speech", "tail")
# provenance records HOW true_spelling got here, so a draft-assisted key stays honest
# about itself: "ear-cold" (typed with no suggestion), "suggestion-confirmed" (an Opus
# draft the ear let stand), "ear-corrected" (an Opus draft the ear overrode). The ruler
# still IS true_spelling — provenance only lets a later LLM-method score weight or set
# aside the confirmed-from-draft ones (TMAS-97 refinement).

# Every unflagged proper name is swept in automatically (see proper_name_candidates):
# a silent substitution of an improvised name onto a valid one (the made-up Jammus
# heard as the canon "Thomas") shares no code with its cluster and never flags, so
# the review must start from ALL names, not a hand-list. `_watch` in the sidecar is
# only a manual escape-hatch — add a cleaned token the sweep can't reach (e.g. a name
# that only ever appears at a sentence start, or shorter than the length floor).
MANUAL_WATCH_DEFAULT = []
MIN_NAME_LEN = 4  # matches the M9b detector; raise the recall floor here if a short name is missed
SENT_END = (".", "!", "?", "…", '."', '?"', '!"')

# Clip window around an occurrence (seconds), mirroring the Monitor's SessionDetections.
PREROLL, TAIL, MAX_SEGMENT_CLIP = 0.5, 0.4, 12.0

NOTES_ABOUT = ("Human ground-truth review of the names flagged in this session. `items` "
               "is per distinct spelling (category + true_spelling); `occurrences` is "
               "per word occurrence (true_name + scope) for clusters that mix two real "
               "names. `scope` says whether a spot's true_name is the true NAME there "
               "(scope 'name' — a name-fixer must produce it), the true SPEECH there "
               "(scope 'speech' — a hallucination, unintelligible audio, or words Whisper "
               "dropped; not a name-fixer's job), or no decision of its own (scope 'tail' "
               "— this word is part of the name that starts before it, so it joins that "
               "name's span and the fixer must clear it). "
               "Edited live in name_truth.py --serve, or by hand. Private (real names) — "
               "gitignored with the rest of visuals/<id>/.")


def sidecar_path(sid):
    return ROOT / "emp" / "results" / "visuals" / sid / "name-truth.json"


def suggestions_path(sid):
    """Opus's draft spellings for a session — machine scaffolding, NEVER the key.
    Read by the name-pass GET and rendered inline; the key (sidecar_path) holds only
    what the ear confirms or corrects. Kept a separate file so the ruler is never a
    machine guess (the firewall), only honestly tagged as one via `provenance`."""
    return ROOT / "emp" / "results" / "visuals" / sid / "name-suggestions.json"


def html_path(sid):
    return ROOT / "emp" / "results" / "visuals" / sid / "name-truth.html"


def audio_file(sid):
    return next((ROOT / "sessions" / sid).glob("audio.*"), None)


def load_notes(sid):
    """{_about, _session, [_*], items:{spelling->...}, occurrences:{occ_id->...}}.
    Preserves top-level _-metadata and every saved item/occurrence across a save."""
    path = sidecar_path(sid)
    raw = json.loads(path.read_text()) if path.exists() else {}
    data = {k: v for k, v in raw.items() if k.startswith("_")}
    data.setdefault("_about", NOTES_ABOUT)
    data["_session"] = sid
    src_i = raw.get("items", {}) if isinstance(raw.get("items"), dict) else {}
    src_o = raw.get("occurrences", {}) if isinstance(raw.get("occurrences"), dict) else {}
    data["items"] = {k: {**ITEM_DEFAULT, **(v if isinstance(v, dict) else {})}
                     for k, v in src_i.items()}
    data["occurrences"] = {k: {**OCC_DEFAULT, **(v if isinstance(v, dict) else {})}
                           for k, v in src_o.items()}
    if not isinstance(data.get("_watch"), list):
        data["_watch"] = list(MANUAL_WATCH_DEFAULT)
    return data


def transcript_fingerprint(sid):
    """Word-level fingerprint of the session's CURRENT transcript (matches the detections /
    namefix convention). A key is only an honest ruler for the transcript state it was
    marked against — the stamp lets every scorer check before grading."""
    import hashlib
    rich = json.loads((ROOT / "sessions" / sid / "transcript-rich.json").read_text())
    parts = [w.get("word", "") for seg in rich.get("segments", []) for w in seg.get("words", [])]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def _save_notes_unlocked(sid, data):
    # PRIVATE — the unlocked half of the write pair. Every outside write goes through
    # update_notes (the flock), so the lost-update race (TMAS-97) can't be reintroduced
    # by a future one-off script reaching for a bare save.
    # Stamp every save with the transcript state the human is marking against (settled
    # 2026-07-02: keys carry a transcript fingerprint; scorers warn on mismatch).
    data["_transcript_fingerprint"] = transcript_fingerprint(sid)
    data["_marked_at"] = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).isoformat()
    path = sidecar_path(sid)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: a concurrent reader or writer never sees a torn file (the race that
    # corrupted 20260304's key mid-write, TMAS-97). Write a sibling temp file, flush +
    # fsync, then os.replace (atomic rename) into place — never a partial name-truth.json.
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".name-truth-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def update_notes(sid, mutate):
    """Serialized, atomic read-modify-write of the key (TMAS-97). An flock on a
    per-session lock file makes concurrent per-edit saves (Flask threads, or a second
    process) take turns, and the private _save_notes_unlocked writes atomically — so racing saves can neither
    lose each other's updates nor tear the file (the load->modify->save race that
    corrupted 20260304's key). `mutate(data)` edits the loaded dict in place. Returns
    the saved data."""
    lock_path = sidecar_path(sid).with_name("name-truth.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            data = load_notes(sid)
            mutate(data)
            _save_notes_unlocked(sid, data)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return data


def clip_window(seg_start, seg_end, w_start, w_end):
    """[start, end] audio window for an occurrence, or None if untimed."""
    if seg_start is not None and seg_end is not None and seg_end - seg_start <= MAX_SEGMENT_CLIP:
        return [seg_start, seg_end]
    if w_start is not None and w_end is not None:
        return [w_start, w_end]
    if seg_start is not None and seg_end is not None and w_start is not None:
        return [w_start, min(w_start + 4, seg_end)]
    return None


def m9_coded_segments(sd, segments):
    """Segment ids (rich 'id' space) the human coded M9* during axial coding — the
    verified FLOOR of 'a name error is here'. Used to guarantee the review covers every
    hand-marked name error, and to badge those occurrences. Empty if uncoded."""
    p = sd / "axial-labels.json"
    if not p.exists():
        return set()
    by_id = {s["id"] for s in segments}
    out = set()
    for lab in json.loads(p.read_text()).get("labels", []):
        if any(str(c).startswith("M9") for c in (lab.get("codes") or [])):
            x = lab.get("segmentId")
            if x in by_id:
                out.add(x)
            elif isinstance(x, int) and 0 <= x < len(segments):  # axial id may be a list index
                out.add(segments[x]["id"])
    return out


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


# Function words that should never begin or end a NAME phrase — kills "the ducky",
# "ducky said" while keeping content-modifier names like "rubber ducky" / "cruel kid".
PHRASE_STOP = {"the", "a", "an", "my", "your", "his", "her", "their", "our", "its", "that",
               "this", "these", "those", "said", "says", "and", "but", "or", "to", "of", "in",
               "on", "at", "with", "is", "was", "are", "were", "it", "he", "she", "they", "you",
               "i", "we", "me", "him", "them", "there", "here", "what", "who", "when", "then",
               "so", "no", "not", "do", "does", "did", "had", "have", "for", "from", "be"}


def detect_phrases(segments, single_cands, min_count=2, dominance=0.5):
    """Recurring within-segment BIGRAMS that form a multi-word name: a non-function-word
    MODIFIER followed by a name-candidate HEAD that is *dominantly* preceded by that
    modifier (>= `dominance` of the head's occurrences). The collocation test is what keeps
    'rubber ducky' / 'cruel kid' (the pair sticks together) while dropping loose fragments
    like 'magic rubber' or 'martie mom' (rubber/mom appear in many other contexts). Returns
    {phrase_cleaned: [occ dicts]}.

    BIGRAMS ONLY is a cross-process invariant: the name-pass UI (NamePassPage.tsx
    `phraseTails`) re-derives each phrase's absorbed tail as exactly wi+1. Growing this
    past bigrams requires shipping the span length through the API, not just changing
    this function."""
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


def collect_flags(sid, watch=(), roster_path=None):
    """Per-spelling records with every occurrence (timed, with its sentence),
    grouped: each M9b cluster, the roster names M9b never clustered, then EVERY
    other proper name no detector flags (swept from the transcript + manual
    `watch`) so a silent substitution onto a valid name can't hide from review.

    `roster_path` reroutes every roster read in here (M9a, M9b, the lowercase-sound
    net) — the constructors already accept it, and None means each one's default
    (data/name_roster.json). The API route passes its app's ROSTER_PATH through,
    so tests and the publish export never need the real gitignored roster."""
    sd = ROOT / "sessions" / sid
    rich = json.loads((sd / "transcript-rich.json").read_text())
    seg = {s["id"]: s for s in rich["segments"]}
    m9_segs = m9_coded_segments(sd, rich["segments"])  # human's hand-coded M9 segments

    m9a = FamilyNameDetector(roster_path=roster_path).run(sd)
    m9b_detector = NameConsistencyDetector(roster_path=roster_path)
    m9b = m9b_detector.run(sd)  # code-only

    rec = {}

    def get(cleaned):
        return rec.setdefault(cleaned, {
            "cleaned": cleaned, "surface": set(), "detectors": set(),
            "cluster": None, "watch": False, "occ": {}, "source": set(),
            "primary": doublemetaphone(cleaned)[0],
        })

    def add_occ(r, seg_id, wi, raw, w_start, w_end, det):
        r["surface"].add(raw)
        if det:
            r["detectors"].add(det)
            r["source"].add(det)
        s = seg.get(seg_id, {})
        # Build the context sentence from the (corrected) WORD tokens, not the segment
        # `text` field: LLM-normalization rewrites the words but does NOT rebuild `text`,
        # so `text` can still show the pre-correction mishearing (Whisper's "fondos")
        # while the words — and this card — show the corrected "Pandavas". Joining the
        # words keeps the context line consistent with the card and the audio.
        sw = s.get("words") or []
        sent = " ".join(w["word"].strip() for w in sw) if sw else (s.get("text") or "")
        r["occ"][(seg_id, wi)] = {
            "occ_id": f"{seg_id}:{wi}",
            "start": w_start,
            "text": sent.strip(),
            "cap": is_capitalized(raw),       # False = a lowercase rendering (M9d hiding spot)
            "m9_coded": seg_id in m9_segs,     # the human coded a name error in this segment
            "window": clip_window(s.get("start"), s.get("end"), w_start, w_end),
        }

    for f in m9a["flags"]:
        add_occ(get(f["cleaned"]), f["segment_id"], f["word_index"], f["token"],
                f.get("start"), f.get("end"), "M9a")
    for f in m9b["flags"]:
        r = get(f["cleaned"]); r["cluster"] = f["cluster_id"]
        add_occ(r, f["segment_id"], f["word_index"], f["token"],
                f.get("start"), f.get("end"), "M9b")

    # sweep EVERY unflagged proper name (mid-sentence-capitalized) plus any manual
    # extras, minus what the detectors already flag. A silent substitution onto a
    # valid name (Jammus -> Thomas) never flags, so the review must start from all
    # names — surface every occurrence here for an ear check.
    cap_sweep = proper_name_candidates(rich["segments"]) | set(watch)
    # NEW — the lowercase-sound net: a name Whisper rendered lowercase AND garbled
    # (viasa for Vyasa) shares no spelling with the real name and is never capitalized,
    # so neither the cap-sweep nor the detectors reach it. Surface every lowercase token
    # whose Double Metaphone SOUNDS like a name we already know — this session's own
    # capitalized names, the family roster, and the home cast — for an ear check. Pure
    # code, no model; the residue (a garble that sounds like nothing known) is left to
    # the full listen. Roster/home reads are graceful (empty when the file is absent).
    name_source_codes = set()
    for nm in cap_sweep:
        name_source_codes |= codes(nm)
    name_source_codes |= RosterMatcher(roster_path)._roster_codes
    for canon in home_canon.canonical_forms().values():
        name_source_codes |= codes(clean(canon))  # clean() first — transcript tokens are cleaned too
    lowercase_net = set()
    if name_source_codes:
        already = set(rec) | cap_sweep
        for s in rich["segments"]:
            for w in s.get("words", []):
                raw = w["word"].strip()
                c = clean(raw)
                if (c and len(c) >= MIN_NAME_LEN and not is_capitalized(raw)
                        and c not in already and not m9b_detector._is_common(c)
                        and (codes(c) & name_source_codes)):
                    lowercase_net.add(c)

    watch_set = (cap_sweep | lowercase_net) - set(rec)
    # Surface EVERY occurrence of every name of interest — capitalized AND lowercase — so a
    # substitution or inconsistency hiding in a lowercase rendering (the worst-case M9d) can't
    # escape the ear. Identification still needs capitalization (proper_name_candidates above);
    # occurrence collection does not. Re-adding an already-flagged occurrence is idempotent.
    interest = set(rec) | watch_set
    for s in rich["segments"]:
        for wi, w in enumerate(s.get("words", [])):
            raw = w["word"].strip()
            c = clean(raw)
            if c in interest:
                r = get(c)
                if c in watch_set:
                    r["watch"] = True
                    r["source"].add("lowercase-sound" if c in lowercase_net else "cap-sweep")
                add_occ(r, s["id"], wi, raw, w.get("start"), w.get("end"), None)

    # ---- multi-word name phrases: surface recurring name bigrams as their own cards, then
    # HARD-dedup their member tokens out of the single-word cards, so 'rubber'+'ducky' collapse
    # into one clean 'rubber ducky' card instead of confusing per-word cards. ----
    single_cands = proper_name_candidates(rich["segments"]) | set(watch)
    covered = {}  # (seg_id, wi) -> the phrase cleaned-key covering it
    for phrase, occs in detect_phrases(rich["segments"], single_cands).items():
        r = get(phrase); r["phrase"] = True; r["source"].add("phrase")
        for o in occs:
            r["surface"].add(o["surface"])
            s = seg.get(o["seg_id"], {})
            sw = s.get("words") or []
            sent = " ".join(w["word"].strip() for w in sw) if sw else (s.get("text") or "")
            r["occ"][(o["seg_id"], o["wi"])] = {
                "occ_id": f'{o["seg_id"]}:{o["wi"]}',
                "start": o["w_start"], "text": sent.strip(),
                "cap": o["cap"], "m9_coded": o["seg_id"] in m9_segs,
                "window": clip_window(s.get("start"), s.get("end"), o["w_start"], o["w_end"]),
            }
            covered[(o["seg_id"], o["wi"])] = phrase
            covered[(o["seg_id"], o["wi"] + 1)] = phrase
    for cleaned, r in list(rec.items()):
        if r.get("phrase"):
            continue
        for key in list(r["occ"]):
            if key in covered:
                rec[covered[key]]["detectors"] |= r["detectors"]  # carry M9a/M9b provenance to the phrase
                rec[covered[key]]["source"] |= r["source"]
                del r["occ"][key]
        if not r["occ"]:  # token fully absorbed into phrase(s) -> drop the confusing per-word card
            del rec[cleaned]

    clusters, roster_only, watch_rows, phrase_rows = defaultdict(list), [], [], []
    for cleaned, r in rec.items():
        r["occ_list"] = sorted(r["occ"].values(), key=lambda o: (o["start"] is None, o["start"]))
        if r.get("phrase"):
            phrase_rows.append(r)
        elif r["watch"]:
            watch_rows.append(r)
        elif r["cluster"] is not None:
            clusters[r["cluster"]].append(r)
        else:
            roster_only.append(r)

    groups = []
    if phrase_rows:
        groups.append(("Multi-word names — review the whole phrase (one card per name, not per word)",
                       sorted(phrase_rows, key=lambda r: -len(r["occ"]))))
    for cid in sorted(clusters):
        groups.append((f"M9b cluster · {cid}", sorted(clusters[cid], key=lambda r: r["cleaned"])))
    if roster_only:
        groups.append(("Roster names — M9a only (M9b never clustered these)",
                       sorted(roster_only, key=lambda r: r["cleaned"])))
    if watch_rows:
        groups.append(("Unflagged proper names — review by ear (every name no detector checks)",
                       sorted(watch_rows, key=lambda r: -len(r["occ"]))))
    # M9-coverage guarantee: which hand-coded M9 segments did NO surfaced name land in?
    covered = {sid_ for r in rec.values() for (sid_, _wi) in r["occ"]}
    m9_meta = {"coded": m9_segs,
               "uncovered": sorted((s for s in m9_segs if s not in covered), key=str)}
    return groups, m9_meta


def highlight(text, surfaces):
    """HTML-escape, then bold any run of letters matching a surface form."""
    esc = html.escape(text)
    for let in sorted({re.sub(r"[^a-zA-Z]", "", s) for s in surfaces}, key=len, reverse=True):
        if let:
            esc = re.sub(rf"(?i)\b({re.escape(let)}\w*)", r"<b>\1</b>", esc)
    return esc


def fmt_time(s):
    if s is None:
        return "—"
    return f"{int(s // 60)}:{int(s % 60):02d}"


def render_html(sid):
    nd = load_notes(sid)
    notes, occ_notes = nd["items"], nd["occurrences"]
    groups, m9_meta = collect_flags(sid, nd.get("_watch", []))
    has_audio = audio_file(sid) is not None

    n_items = sum(len(rows) for _, rows in groups)
    labeled = sum(1 for _, rows in groups for r in rows if notes.get(r["cleaned"], {}).get("category"))

    blocks = []
    for title, rows in groups:
        cards = []
        for r in rows:
            saved = notes.get(r["cleaned"], ITEM_DEFAULT)
            prov = " + ".join(sorted(r["detectors"])) if r["detectors"] else "unflagged"
            surface = " · ".join(sorted(r["surface"]))
            cid = html.escape(r["cleaned"])

            occ_html = []
            for o in r["occ_list"]:
                win = o["window"]
                play = (f'<button class="play" data-start="{win[0]}" data-end="{win[1]}"'
                        f' title="Play this occurrence">▶</button>' if (win and has_audio)
                        else '<span class="play disabled" title="no audio">▷</span>')
                oid = html.escape(o["occ_id"])
                tn = html.escape(occ_notes.get(o["occ_id"], {}).get("true_name", ""))
                low = not o.get("cap", True)
                badges = ""
                if o.get("m9_coded"):
                    badges += ('<span style="font:600 .58rem monospace;color:#c97;border:1px solid #c97;'
                               'border-radius:3px;padding:0 .25rem;margin-left:.3rem">M9 coded</span>')
                if low:
                    badges += ('<span style="font:600 .58rem monospace;color:#88a;border:1px solid #88a;'
                               'border-radius:3px;padding:0 .25rem;margin-left:.3rem">lowercase</span>')
                occ_html.append(f"""
                <div class="occ"{' style="opacity:.72"' if low else ''}>
                  {play}
                  <span class="ts">{fmt_time(o['start'])}</span>
                  <span class="line">“{highlight(o['text'], r['surface'])}”{badges}</span>
                  <input class="tn" data-id="{oid}" placeholder="this one is…" value="{tn}">
                  <span class="stat occstat" data-id="{oid}"></span>
                </div>""")

            opts = "".join(
                f'<option value="{v}"{" selected" if saved.get("category") == v else ""}>{html.escape(lab)}</option>'
                for v, lab in CATEGORIES)
            cards.append(f"""
            <div class="item{' watch' if r.get('watch') else ''}" data-id="{cid}">
              <div class="ihead">
                <span class="spelling">{html.escape(sorted(r['surface'])[0])}</span>
                <span class="code">{html.escape(r['primary'])}</span>
                <span class="prov">{html.escape(prov)}</span>
                <span class="count">{len(r['occ_list'])}×</span>
              </div>
              <div class="surface">forms: {html.escape(surface)}</div>
              <div class="controls">
                <select class="cat" data-id="{cid}">{opts}</select>
                <input class="truth" data-id="{cid}" placeholder="true spelling"
                       value="{html.escape(saved.get('true_spelling', ''))}">
                <input class="note" data-id="{cid}" placeholder="notes (e.g. mixes two engines)"
                       value="{html.escape(saved.get('notes', ''))}">
                <span class="stat" data-id="{cid}"></span>
              </div>
              <div class="occs">{''.join(occ_html)}</div>
            </div>""")
        blocks.append(f'<section class="group"><h2>{html.escape(title)}</h2>{"".join(cards)}</section>')

    if m9_meta["uncovered"]:  # hand-coded M9 segments with no detected name — the recall guarantee
        ids = ", ".join(str(x) for x in m9_meta["uncovered"][:25])
        blocks.insert(0, f'<p class="tally" style="color:#e89">⚠ {len(m9_meta["uncovered"])} segment(s) '
                         f'you coded M9 have no detected name — likely a lowercase or dropped name not in the '
                         f'sweep; check by ear: seg {ids}.</p>')

    audio_el = '<audio id="player" src="/audio" preload="metadata"></audio>' if has_audio else ''
    audio_note = ('' if has_audio else
                  '<p class="tally" style="color:var(--place)">⚠ no audio file for this session — '
                  'play buttons disabled.</p>')
    return PAGE.format(sid=html.escape(sid), n_items=n_items, labeled=labeled,
                       groups="".join(blocks), script=SCRIPT, audio_el=audio_el,
                       audio_note=audio_note)


SCRIPT = r"""
(function(){
  var served = location.protocol.indexOf('http')===0;
  var timers = {};
  function save(kind, id, field, value, stat){
    if(!served){ stat.textContent='⚠ run --serve to save'; stat.className=stat.className.replace(/ (ok|warn)/g,'')+' warn'; return; }
    stat.textContent='saving…';
    fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
      body: JSON.stringify({kind:kind, id:id, field:field, value:value})})
      .then(function(r){ stat.textContent = r.ok?'saved ✓':'save failed';
        stat.className = stat.className.replace(/ (ok|warn)/g,'')+(r.ok?' ok':' warn'); })
      .catch(function(){ stat.textContent='save failed'; });
  }
  function debounced(kind, el, field, stat){
    el.addEventListener('input', function(){
      if(!served){ stat.textContent='⚠ run --serve to save'; return; }
      stat.textContent='saving…';
      var k = el.getAttribute('data-id')+field;
      clearTimeout(timers[k]);
      timers[k] = setTimeout(function(){ save(kind, el.getAttribute('data-id'), field, el.value, stat); }, 500);
    });
  }
  function statFor(sel, id){ return document.querySelector(sel+'[data-id="'+CSS.escape(id)+'"]'); }

  document.querySelectorAll('.cat').forEach(function(sel){
    sel.addEventListener('change', function(){
      var id = sel.getAttribute('data-id');
      save('item', id, 'category', sel.value, statFor('.stat', id));
      sel.closest('.item').setAttribute('data-cat', sel.value);
    });
    sel.closest('.item').setAttribute('data-cat', sel.value);
  });
  document.querySelectorAll('.truth').forEach(function(el){ debounced('item', el, 'true_spelling', statFor('.stat', el.getAttribute('data-id'))); });
  document.querySelectorAll('.note').forEach(function(el){ debounced('item', el, 'notes', statFor('.stat', el.getAttribute('data-id'))); });
  document.querySelectorAll('.tn').forEach(function(el){ debounced('occ', el, 'true_name', el.parentElement.querySelector('.occstat')); });

  /* audio: one shared element; each play button plays a windowed clip and auto-stops */
  var player = document.getElementById('player');
  var stopAt = null, current = null;
  if(player){
    player.addEventListener('timeupdate', function(){ if(stopAt!=null && player.currentTime>=stopAt) player.pause(); });
    function reset(){ stopAt=null; if(current){ current.textContent='▶'; current=null; } }
    player.addEventListener('pause', reset);
    player.addEventListener('ended', reset);
    document.querySelectorAll('.play[data-start]').forEach(function(btn){
      btn.addEventListener('click', function(){
        var s = parseFloat(btn.getAttribute('data-start')), e = parseFloat(btn.getAttribute('data-end'));
        if(current===btn){ player.pause(); return; }
        if(current) current.textContent='▶';
        stopAt = e + %TAIL%; player.currentTime = Math.max(0, s - %PREROLL%);
        player.play().then(function(){ current=btn; btn.textContent='❚❚'; }).catch(function(){ reset(); });
      });
    });
  }
})();
""".replace("%TAIL%", str(TAIL)).replace("%PREROLL%", str(PREROLL))


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Name truth · {sid}</title>
<style>
  :root {{ --ink:#1a1d24; --muted:#6b7280; --line:#e6e8ec; --bg:#fbfcfd;
    --roster:#7c5cff; --canon:#0369a1; --improvised:#0d7a4f; --place:#c2410c; --nota:#9aa0ab; }}
  * {{ box-sizing:border-box; }}
  body {{ font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink);
    background:var(--bg); margin:0; padding:2rem clamp(1rem,5vw,4rem); max-width:64rem; }}
  h1 {{ font-size:1.45rem; margin:0 0 .2rem; letter-spacing:-.02em; }}
  .lede {{ color:var(--muted); margin:0 0 .4rem; font-size:.92rem; }}
  .tally {{ font-size:.85rem; color:var(--muted); margin:0 0 1.4rem; }}
  .tally b {{ color:var(--ink); }}
  h2 {{ font-size:.76rem; text-transform:uppercase; letter-spacing:.07em; color:var(--muted);
    margin:2rem 0 .8rem; border-bottom:1px solid var(--line); padding-bottom:.35rem; }}
  .item {{ background:#fff; border:1px solid var(--line); border-left:3px solid var(--line);
    border-radius:9px; padding:.85rem 1rem; margin-bottom:.7rem; }}
  .item[data-cat="roster"] {{ border-left-color:var(--roster); }}
  .item[data-cat="tv-canon"] {{ border-left-color:var(--canon); }}
  .item[data-cat="improvised"] {{ border-left-color:var(--improvised); }}
  .item[data-cat="place-other"] {{ border-left-color:var(--place); }}
  .item[data-cat="not-a-name"] {{ border-left-color:var(--nota); }}
  .item.watch {{ border-left-style:dashed; border-left-color:var(--place); background:#fffdf9; }}
  .item.watch .prov {{ color:var(--place); }}
  .ihead {{ display:flex; align-items:center; gap:.65rem; flex-wrap:wrap; }}
  .spelling {{ font-weight:700; font-size:1.05rem; }}
  .code {{ font-family:ui-monospace,Menlo,monospace; font-size:.72rem; color:#9aa0ab;
    background:var(--bg); padding:.05rem .4rem; border-radius:4px; }}
  .prov {{ font-size:.72rem; font-weight:600; color:var(--muted); }}
  .count {{ margin-left:auto; color:var(--muted); font-size:.82rem; font-variant-numeric:tabular-nums; }}
  .surface {{ font-size:.78rem; color:var(--muted); margin:.3rem 0 .5rem; }}
  .controls {{ display:flex; align-items:center; gap:.5rem; flex-wrap:wrap; margin-bottom:.5rem; }}
  select, input {{ font:inherit; font-size:.86rem; padding:.32rem .5rem; border:1px solid var(--line);
    border-radius:6px; background:#fff; }}
  select.cat {{ font-weight:600; }}
  input.truth {{ width:9rem; }}
  input.note {{ flex:1; min-width:11rem; }}
  .stat {{ font-size:.74rem; color:var(--muted); min-width:4.5rem; }}
  .stat.ok {{ color:var(--improvised); }}
  .stat.warn {{ color:var(--place); }}
  .occs {{ border-top:1px dashed var(--line); padding-top:.45rem; display:flex; flex-direction:column; gap:.3rem; }}
  .occ {{ display:flex; align-items:center; gap:.5rem; font-size:.84rem; }}
  .play {{ flex:none; width:1.7rem; height:1.7rem; border-radius:50%; border:1px solid var(--line);
    background:#fff; color:var(--canon); cursor:pointer; font-size:.7rem; line-height:1; }}
  .play:hover {{ background:#f0f7ff; }}
  .play.disabled {{ color:#cbd0d6; cursor:default; border-style:dashed; }}
  .ts {{ flex:none; font-variant-numeric:tabular-nums; color:var(--muted); font-size:.76rem; min-width:2.6rem; }}
  .line {{ flex:1; color:#4b5563; font-style:italic; }}
  .line b {{ font-style:normal; color:var(--ink); background:#fff3cd; padding:0 .1rem; border-radius:2px; }}
  input.tn {{ flex:none; width:8rem; }}
  .occstat {{ min-width:3.5rem; }}
  .hint {{ position:sticky; top:0; z-index:5; background:linear-gradient(var(--bg),var(--bg) 70%,transparent);
    padding:.4rem 0; font-size:.78rem; color:var(--muted); }}
</style></head>
<body>
  {audio_el}
  <h1>Name truth — session {sid}</h1>
  <p class="lede">Per spelling: set the category + true spelling. Per occurrence: ▶ to hear it, then
    type which name it really is — for the James/Jammus blob, the ear separates what the spelling can't.</p>
  <p class="tally"><b>{labeled}</b> of <b>{n_items}</b> spellings categorized · grouped by the detector's clusters</p>
  {audio_note}
  <div class="hint">Edits auto-save to <code>name-truth.json</code> when run with <code>--serve</code>.</div>
  {groups}
  <script>{script}</script>
</body></html>"""


def serve(sid, port):
    """Serve the review page, stream the session audio (range-aware so clips seek),
    and persist edits to the sidecar. The browser can't write files itself."""
    import http.server

    audio_path = audio_file(sid)

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="text/html; charset=utf-8", extra=None):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")  # always re-fetch as we iterate
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def _send_audio(self):
            if not audio_path or not audio_path.exists():
                self._send(404, "no audio", "text/plain"); return
            size = audio_path.stat().st_size
            rng = self.headers.get("Range", "")
            ctype = "audio/mp4" if audio_path.suffix == ".m4a" else "audio/mpeg"
            if rng.startswith("bytes="):
                s_str, _, e_str = rng[6:].partition("-")
                start = int(s_str) if s_str else 0
                end = int(e_str) if e_str else size - 1
                end = min(end, size - 1)
                with open(audio_path, "rb") as f:
                    f.seek(start); chunk = f.read(end - start + 1)
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers(); self.wfile.write(chunk)
            else:
                self._send(200, audio_path.read_bytes(), ctype, {"Accept-Ranges": "bytes"})

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html", "/name-truth.html"):
                self._send(200, render_html(sid))
            elif path == "/audio":
                self._send_audio()
            elif path == "/name-truth.json":
                self._send(200, json.dumps(load_notes(sid), indent=2, ensure_ascii=False),
                           "application/json; charset=utf-8")
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            if self.path != "/save":
                self._send(404, "not found", "text/plain"); return
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._send(400, '{"error":"bad json"}', "application/json"); return
            kind, item_id, field = body.get("kind"), body.get("id"), body.get("field")
            bucket = {"item": ("items", ITEM_DEFAULT), "occ": ("occurrences", OCC_DEFAULT)}.get(kind)
            if not bucket or not item_id or field not in bucket[1]:
                self._send(400, '{"error":"need {kind:item|occ, id, field, value}"}',
                           "application/json"); return
            key, default = bucket

            def mutate(data):
                # field-routed: only the changed field is written, so one save never
                # wipes a sibling field (the count.py lesson).
                data[key].setdefault(item_id, dict(default))[field] = body.get("value") or ""

            update_notes(sid, mutate)  # locked + atomic (TMAS-97), same path as the API
            self._send(200, '{"ok":true}', "application/json")

        def log_message(self, *a):
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", port), Handler)
    print(f"name truth  ->  http://127.0.0.1:{port}/")
    print(f"audio       ->  {'(' + audio_path.name + ')' if audio_path else 'NONE — play disabled'}")
    print(f"sidecar     ->  {sidecar_path(sid).relative_to(ROOT)}   (edits auto-save; Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


def main():
    ap = argparse.ArgumentParser(description="Human ground-truth review of flagged names")
    ap.add_argument("session_id")
    ap.add_argument("--serve", action="store_true", help="serve with audio + live, file-backed saving")
    ap.add_argument("--port", type=int, default=5056, help="port for --serve (default 5056)")
    args = ap.parse_args()
    if args.serve:
        serve(args.session_id, args.port)
    else:
        out = html_path(args.session_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_html(args.session_id))
        print(f"wrote {out}  (open with --serve for audio + saving)")


if __name__ == "__main__":
    main()
