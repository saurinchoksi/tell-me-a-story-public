"""Name-question endpoints — the labeled shadow queue + record-only verdicts (calibration).

The namefix stage runs in SHADOW: it applies nothing and writes every opinion as a labeled
question to `sessions/<id>/pending-name-corrections.json`. These routes serve those
questions and record the human's verdicts. NOTHING here touches a transcript — verdicts are
recorded (Name Book + audit lists) and everything applies later, in one deliberate apply
pass, when classes graduate (docs/name-architecture.md).

Identity: every group carries a stable `gid` minted by namefix at grouping time — one heard
word can carry two different suggestions in one night (Jammus→Thomas AND Jammus→James), so
heard alone is never an identity. All verdict endpoints key on gid.

  GET  /api/name-corrections
        Rollup across all sessions, organized by LABEL (why the machine believed it), then
        world, then name. Standing Name Book rulings partition questions into two view-time
        folds — `covered_by_yes` and `suppressed` — computed live against the book (cleaned
        tokens), so a new ruling folds instantly and lifting one restores instantly; nothing
        is rewritten on disk. A THIRD fold, `spot_answered`, covers regenerated questions
        whose every spot the human already ruled without writing a standing rule
        (per-spot verdicts, world-less picks) — strict same-word spot coverage, same
        transcript version only (_partition_spot_answered). Also serves `ruled` (the
        verdict history, undoable), `stale_sessions` (queues whose transcript changed
        since the fixers ran), and `family_fixer` (did the family side run — checked /
        skipped-with-reason / stale / unknown per session, plus the live family
        question count).
  POST /api/sessions/<id>/name-corrections/bless   {gid, canonical?, occurrences?}
        RECORD a yes. Whole-name: the ruling enters the world's Name Book (deterministic at
        the future apply pass) + the _blessed audit. Per-occurrence: audit only — a partial
        yes never writes the book (the Bheem ambiguity guard). The transcript is untouched.
  POST /api/sessions/<id>/name-corrections/reject  {gid}
        RECORD a no on the WORD in the world's Name Book (never rewritten at the apply
        pass, never asked again) + the _rejected audit. The transcript is untouched.
  POST /api/sessions/<id>/name-corrections/unrule  {verdict, gid, at?}
        Undo a recorded verdict: pop the audit entry, reverse its Name Book write, restore
        the question from the snapshot the verdict carried.
  POST /api/name-book/unreject  {world, heard_cleaned}
        Lift a standing no — matching questions return to the live queue on the next GET.
  POST /api/name-book/unpick-spelling  {world, cluster}
        Lift a standing spelling pick — the cluster's questions return to the live queue.

Spelling questions (label "inconsistent-spelling", from the spellingfix fixer) route to
the Name Book's SPELLINGS slot instead of word rulings: yes = unify the cluster to this
spelling (pick), no = "don't unify these" (null pick) — narrow by design, never a
word-level no, so a spelling opinion can never mute another fixer's questions about the
same word. Either pick folds the cluster's questions at view time, liftable.

Verdicts against a STALE queue are refused (409): if the transcript changed since the
namefix run, occurrence positions may have drifted and a recorded verdict would corrupt
the calibration dataset — re-run namefix first (namefix_session.py <sid> --write).

Only Choksi's taps write the Name Book; the audit lists on each queue file are the
calibration dataset (verdict + label + signals per group).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from api.helpers import get_session_dir, validate_session_id
from familyfix import FIXER as FAMILY_FIXER
from jsonio import atomic_write_json
from namefix import (LABELS, SPELLING_LABEL, pre_correction_fingerprint,
                     transcript_fingerprint)

bp = Blueprint("name_corrections", __name__)

PENDING_FILE = "pending-name-corrections.json"


def _book_dir() -> Path:
    return current_app.config["NAME_BOOK_DIR"]


def _read_pending(session_dir: Path) -> dict | None:
    path = session_dir / PENDING_FILE
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    for g in data.get("groups", []):
        g.setdefault("fixer", "namefix")   # legacy single-fixer files predate the tag
    return data


def _write_pending(session_dir: Path, data: dict) -> None:
    atomic_write_json(session_dir / PENDING_FILE, data)


def _read_transcript(session_dir: Path) -> dict | None:
    rich_path = session_dir / "transcript-rich.json"
    if not rich_path.exists():
        return None
    return json.loads(rich_path.read_text())


def _is_stale(data: dict, transcript: dict | None) -> bool:
    """The queue's stamped transcript fingerprint vs the transcript as it is NOW. A stale
    queue's occurrence positions may have drifted — verdicts against it are refused.

    One sanctioned mismatch is NOT staleness: dictionary normalization (Pass 5)
    runs AFTER the fixers stamp the queue and rewrites word tokens 1:1 in place
    (apply_corrections — positions stable, each corrected word keeping its `_original`).
    The fixers truthfully stamped the pre-dictionary words at the very positions the
    queue references, so a stamp matching the restored-originals stream
    (pre_correction_fingerprint) is fresh. Any other difference still reads stale."""
    if transcript is None:
        return False
    stamp = data.get("transcript_fingerprint")
    return (transcript_fingerprint(transcript) != stamp
            and pre_correction_fingerprint(transcript) != stamp)


def _partition_rulings(groups: list, rulings_memo: dict) -> tuple[list, list, list]:
    """Split groups into (live, covered_by_yes, suppressed) against the Name Book — the
    view-time rule (same pattern as the detector verdicts): files on disk are untouched,
    so a new ruling folds a question instantly and lifting it restores instantly. Word
    rulings (yes/no) cover the WORD, whatever fixer asked; a spelling PICK covers its
    CLUSTER and folds only spelling questions (a unify pick reads as a yes-fold, a
    "don't unify" as a no-fold). Each folded group is annotated `folded_by` so the UI's
    lift button knows which ruling to lift. Book keys are clean()'d to match
    heard_cleaned normalization. `rulings_memo` caches per-world book reads."""
    import namebook  # src/ on sys.path via api.app
    from detectors.phonetics import clean
    live, covered_yes, suppressed = [], [], []
    for g in groups:
        world = g.get("world") or ""
        if world not in rulings_memo:
            if world:
                rulings_memo[world] = (
                    {clean(k) for k in namebook.load_yes_map(world, base=_book_dir())
                     if clean(k)},
                    {clean(w) for w in namebook.load_no_words(world, base=_book_dir())
                     if clean(w)},
                    namebook.load_spellings(world, base=_book_dir()),
                )
            else:
                rulings_memo[world] = (set(), set(), {})
        yes_words, no_words, picks = rulings_memo[world]
        word = g.get("heard_cleaned") or ""
        pick = (picks.get(g.get("cluster_id"))
                if g.get("label") == SPELLING_LABEL else None)
        if word in no_words:
            suppressed.append({**g, "folded_by": "no"})
        elif word in yes_words:
            covered_yes.append({**g, "folded_by": "yes"})
        elif pick is not None:
            if pick["spelling"] is not None:
                covered_yes.append({**g, "folded_by": "pick-unify",
                                    "canonical": pick["spelling"]})
            else:
                suppressed.append({**g, "folded_by": "pick-leave"})
        else:
            live.append(g)
    return live, covered_yes, suppressed


def _annotate_frequency(transcript: dict | None, groups: list) -> None:
    """The frequency-asymmetry signal (the Marley lesson, 2026-07-03): garbles are
    USUALLY the minority form — a suspect said 68 times next to a person named once is
    probably its own name. Logged, never acted on: the record already holds a real
    counter-example (a garble outnumbering its correct form 12-to-3 on the Mahabharata
    night), which is exactly why this is calibration data for his verdicts to judge,
    not a gate. Computed at VIEW time from the current transcript (and stamped into the
    audit at verdict time), so every question — old queues included — carries it with
    no re-runs. Adds `n_heard_in_transcript` / `n_suggested_in_transcript` per group.

    Frequency honesty for the dictionary-corrected case: when the dictionary
    rewrote every heard spot in place, `n_heard_in_transcript` reads 0 purely because the
    live tokens no longer say the heard form — a bare "0×" would contradict a card whose
    own spots were the heard word. Also expose `n_heard_dictionary_corrected`: how many of
    this group's spots the dictionary has since rewritten. Computed HERE from the
    transcript itself (each occurrence joined to its live word — never read off the
    display markers _enrich_occurrences sets), so the count is right on EVERY path: the
    POST verdict paths stamp audit snapshots without ever enriching, and no call-order
    dependency exists between the two annotators."""
    if transcript is None:
        return
    from collections import Counter
    from detectors.phonetics import clean
    counts = Counter(clean(w["word"]) for s in transcript.get("segments", [])
                     for w in s.get("words", []) if clean(w["word"]))
    seg_by_id = {s["id"]: s for s in transcript.get("segments", [])}
    for g in groups:
        g["n_heard_in_transcript"] = counts[g.get("heard_cleaned") or ""]
        g["n_suggested_in_transcript"] = counts[clean(g.get("canonical") or "")]
        n_corrected = 0
        for o in g.get("occurrences", []):
            seg = seg_by_id.get(o.get("segment_id"))
            words = (seg or {}).get("words") or []
            wi = o.get("word_index")
            if (seg is None or wi is None or wi >= len(words)
                    or "token" not in o):   # unjoinable/legacy spot: no basis, no count
                continue
            if _dictionary_corrected(words[wi], o["token"]):
                n_corrected += 1
        g["n_heard_dictionary_corrected"] = n_corrected


def _dictionary_corrected(word: dict, token: str) -> bool:
    """The one "was this spot rewritten by the dictionary?" test, shared by
    _enrich_occurrences (the display marker) and _annotate_frequency (the honest count)
    so the two can never drift: the live word keeps a pre-rewrite `_original` whose
    clean form equals the occurrence's recorded heard token, the surface now differs,
    and the rewrite's recorded stage is the DICTIONARY's. The stage check is provenance:
    apply_corrections stamps `_original` for ANY stage, and the future apply pass will
    write namefix-class stages — those spots were not corrected by the dictionary and
    must not read as such. `_corrections` is read fail-loud: apply_corrections always
    writes it alongside `_original`."""
    from detectors.phonetics import clean
    if "_original" not in word:
        return False
    heard = clean(token)
    return (clean(word["_original"]) == heard
            and clean(word["word"]) != heard
            and any(c["stage"] == "dictionary" for c in word["_corrections"]))


def _enrich_occurrences(transcript: dict | None, groups: list) -> None:
    """Join each occurrence to its segment's CURRENT sentence (built from words, the source
    of truth the offsets index into) so the reviewer sees the word in context. Adds
    `segment_text`, `word_offset`/`word_len` (char span of the judged word inside it).

    Also annotates the dictionary-corrected spot. The opt-in dictionary pass
    (Pass 5, --library) can rewrite the very token a question points at, in place, AFTER
    the fixer stamped the queue (apply_corrections keeps each corrected word's `_original`
    = its pre-rewrite stripped token). Since the staleness fix that queue no longer reads stale, so
    the card is live — but its highlight shows the ALREADY-CORRECTED surface and the heard
    form counts 0, a card contradicting its own premise. When the live word at the spot
    carries an `_original` that cleans to this occurrence's recorded heard token and the
    current surface differs, mark the occurrence honestly: `corrected_by_dictionary` +
    `now_reads` (the current display token). GET-time, read-only — the queue is untouched;
    nothing but the fixers ever writes it."""
    if transcript is None:
        return
    seg_by_id = {s["id"]: s for s in transcript.get("segments", [])}
    for g in groups:
        for o in g.get("occurrences", []):
            seg = seg_by_id.get(o.get("segment_id"))
            words = (seg or {}).get("words") or []
            wi = o.get("word_index")
            if seg is None or wi is None or wi >= len(words):
                continue
            parts = [w["word"] for w in words]
            text = "".join(parts)
            off = len("".join(parts[:wi]))
            tok = parts[wi]
            lead_ws = len(tok) - len(tok.lstrip())
            o["segment_text"] = text.strip()
            # offset within the STRIPPED text: subtract the segment's own leading whitespace
            seg_lead = len(text) - len(text.lstrip())
            o["word_offset"] = off + lead_ws - seg_lead
            o["word_len"] = len(tok.strip())
            # dictionary-corrected spot (the shared predicate — see _dictionary_corrected).
            # A legacy/hand-edited occurrence without `token` simply gets no marker rather
            # than crashing the whole rollup (the _read_pending backfill precedent).
            if "token" in o and _dictionary_corrected(words[wi], o["token"]):
                o["corrected_by_dictionary"] = True
                o["now_reads"] = tok.strip()


def _rollup_add(rollup: dict, g: dict, session_id: str, stale: bool) -> None:
    wkey = g.get("world") or "(unrecognized)"
    nkey = f"{g.get('heard_cleaned')}→{g.get('canonical')}"
    slot = rollup.setdefault(wkey, {}).setdefault(nkey, {
        "heard": g.get("heard"),
        "heard_cleaned": g.get("heard_cleaned"),
        "canonical": g.get("canonical"),
        "method": g.get("method"),
        "label": g.get("label"),
        "fixer": g.get("fixer"),
        "cluster_id": g.get("cluster_id"),
        "folded_by": g.get("folded_by"),
        "dm_share": g.get("dm_share"),
        "echo_in_sentence": False,
        "n_sibling_spellings": 0,
        "sessions": [],
    })
    # a flag seen in ANY session rides on the rolled-up item; siblings take the max
    slot["echo_in_sentence"] = slot["echo_in_sentence"] or bool(g.get("echo_in_sentence"))
    slot["n_sibling_spellings"] = max(slot["n_sibling_spellings"],
                                      g.get("n_sibling_spellings", 0))
    # frequency-asymmetry counts sum across the item's sessions (a night-level signal
    # rolled up to the card the reviewer actually sees)
    if "n_heard_in_transcript" in g:
        slot["n_heard_in_transcript"] = (slot.get("n_heard_in_transcript", 0)
                                         + g["n_heard_in_transcript"])
        slot["n_suggested_in_transcript"] = (slot.get("n_suggested_in_transcript", 0)
                                             + g["n_suggested_in_transcript"])
        # legitimately optional, not a silent fallback: an older verdict snapshot
        # (restorable via unrule) carries the two older frequency fields but not this
        # one, and with the transcript unavailable nothing recomputes it — 0 is honest
        slot["n_heard_dictionary_corrected"] = (
            slot.get("n_heard_dictionary_corrected", 0)
            + g.get("n_heard_dictionary_corrected", 0))
    entry = {
        "session_id": session_id,
        "story_id": g.get("story_id"),
        "gid": g.get("gid"),
        "stale": stale,
        "occurrences": g.get("occurrences", []),
    }
    if g.get("covering"):                      # spot-answer fold: the inline-undo rows
        entry["covering"] = g["covering"]
    if g.get("n_spot_answered_occ"):           # live card with some spots already ruled
        entry["n_spot_answered_occ"] = g["n_spot_answered_occ"]
    slot["sessions"].append(entry)


def _rollup_payload(rollup: dict) -> list:
    return [
        {"world": w, "names": sorted(names.values(), key=lambda x: x["heard_cleaned"] or "")}
        for w, names in sorted(rollup.items())
    ]


def _ruled_rows(data: dict, session_id: str) -> list:
    """The session's verdict history — each entry undoable if it carries a snapshot."""
    rows = []
    for key, verdict in (("_blessed", "yes"), ("_rejected", "no")):
        for e in data.get(key, []):
            rows.append({
                "session_id": session_id,
                "verdict": verdict,
                **{k: e.get(k) for k in ("gid", "heard", "heard_cleaned", "canonical",
                                         "world", "label", "fixer", "partial",
                                         "n_occurrences", "at")},
                "undoable": "group" in e,
            })
    return rows


@bp.route("/name-corrections", methods=["GET"])
def name_corrections_rollup():
    """The consolidated labeled queue: every session's questions, organized by label ->
    world -> name, plus the two Name Book folds, the verdict history, and staleness."""
    sessions_dir = current_app.config["SESSIONS_DIR"]
    by_label: dict = {}
    yes_worlds: dict = {}
    no_worlds: dict = {}
    spot_worlds: dict = {}
    rulings_memo: dict = {}
    ruled: list = []
    stale_sessions: list = []
    n_live = n_yes = n_no = n_spot = 0
    # Did the family side run? — per session: checked (a metadata entry with
    # no status), skipped (with reason), stale (the record — ran or skipped — is for an
    # older transcript), or unknown (no queue file / no family entry: never stamped).
    # `questions` counts LIVE family cards on non-stale sessions, deduped the way the
    # page merges them (world + heard→canonical), so the strip's number matches the
    # cards the reviewer can actually count below it.
    family_fixer = {"checked": 0, "skipped": 0, "stale": 0, "unknown": 0,
                    "questions": 0, "skip_reasons": []}
    family_live_cards: set = set()
    if sessions_dir.exists():
        for entry in sorted(sessions_dir.iterdir()):
            if not entry.is_dir() or not validate_session_id(entry.name):
                continue
            data = _read_pending(entry)
            if data is None:
                # a processed session the fixers never stamped (older sessions,
                # or a pipeline that died before any queue write) — these being
                # invisible was the original hole, so they count as unknown
                if (entry / "transcript-rich.json").exists():
                    family_fixer["unknown"] += 1
                continue
            groups = data["groups"]   # fail loud: a queue file without groups is corrupt
            transcript = _read_transcript(entry)
            stale = _is_stale(data, transcript)
            if stale:
                stale_sessions.append(entry.name)
            fam = (data.get("fixers") or {}).get(FAMILY_FIXER)
            if fam is None:
                family_fixer["unknown"] += 1
            elif stale:
                family_fixer["stale"] += 1   # whatever it says, it's about older text
            elif fam.get("status") == "skipped":
                family_fixer["skipped"] += 1
                if fam.get("reason") and fam["reason"] not in family_fixer["skip_reasons"]:
                    family_fixer["skip_reasons"].append(fam["reason"])
            else:
                family_fixer["checked"] += 1
            live, covered_yes, suppressed = _partition_rulings(groups, rulings_memo)
            live, spot_answered = _partition_spot_answered(live, data)
            if not stale:
                family_live_cards.update(
                    (g.get("world"), g.get("heard_cleaned"), g.get("canonical"))
                    for g in live if g["fixer"] == FAMILY_FIXER)
            _enrich_occurrences(transcript,
                                live + covered_yes + suppressed + spot_answered)
            _annotate_frequency(transcript,
                                live + covered_yes + suppressed + spot_answered)
            for g in live:
                n_live += 1
                _rollup_add(by_label.setdefault(g.get("label") or "unlabeled", {}),
                            g, entry.name, stale)
            for g in covered_yes:
                n_yes += 1
                _rollup_add(yes_worlds, g, entry.name, stale)
            for g in suppressed:
                n_no += 1
                _rollup_add(no_worlds, g, entry.name, stale)
            for g in spot_answered:
                n_spot += 1
                _rollup_add(spot_worlds, g, entry.name, stale)
            ruled.extend(_ruled_rows(data, entry.name))
    family_fixer["questions"] = len(family_live_cards)
    ruled.sort(key=lambda r: r.get("at") or "", reverse=True)
    labels = [{"label": label, "worlds": _rollup_payload(by_label[label])}
              for label in LABELS if label in by_label]
    for label in sorted(by_label):            # future labels this route doesn't know yet
        if label not in LABELS:
            labels.append({"label": label, "worlds": _rollup_payload(by_label[label])})
    payload = {
        "n_pending_groups": n_live,
        "n_covered_by_yes": n_yes,
        "n_suppressed_groups": n_no,
        "n_spot_answered": n_spot,
        "n_ruled": len(ruled),
        "labels": labels,
        "covered_by_yes": _rollup_payload(yes_worlds),
        "suppressed": _rollup_payload(no_worlds),
        "spot_answered": _rollup_payload(spot_worlds),
        "ruled": ruled,
        "stale_sessions": stale_sessions,
        "family_fixer": family_fixer,
    }
    return jsonify(payload)


@bp.route("/sessions/<session_id>/name-corrections", methods=["GET"])
def session_name_corrections(session_id):
    """One session's labeled queue, partitioned at view time (404 if the shadow stage
    never wrote a file)."""
    if not validate_session_id(session_id):
        return jsonify({"error": "Invalid session ID"}), 400
    session_dir = get_session_dir(current_app.config["SESSIONS_DIR"], session_id)
    if not session_dir.exists():
        return jsonify({"error": "Session not found"}), 404
    data = _read_pending(session_dir)
    if data is None:
        return jsonify({"error": "No name questions for this session"}), 404
    groups = data["groups"]   # fail loud on a corrupt/legacy file
    transcript = _read_transcript(session_dir)
    live, covered_yes, suppressed = _partition_rulings(groups, {})
    live, spot_answered = _partition_spot_answered(live, data)
    _enrich_occurrences(transcript, live + covered_yes + suppressed + spot_answered)
    _annotate_frequency(transcript, live + covered_yes + suppressed + spot_answered)
    data["groups"] = live
    data["covered_by_yes"] = covered_yes   # view-time only — the file is untouched
    data["suppressed"] = suppressed
    data["spot_answered"] = spot_answered
    data["stale"] = _is_stale(data, transcript)
    return jsonify(data)


def _pop_group(data: dict, gid: str) -> dict | None:
    for i, g in enumerate(data["groups"]):
        if g.get("gid") == gid:
            return data["groups"].pop(i)
    return None


def _stale_guard(session_dir: Path, data: dict):
    """Refuse verdicts against a drifted queue — a tap on stale positions would record a
    wrong datapoint. Returns an error response or None."""
    if _is_stale(data, _read_transcript(session_dir)):
        return jsonify({"error": "queue is stale — the transcript changed since the "
                                 "fixers ran; re-run: python src/fix_session.py "
                                 f"{session_dir.name} --write"}), 409
    return None


def _verdict_record(group: dict, data: dict, extra: dict) -> dict:
    """One audit entry = one labeled calibration datapoint: the group's label + signals +
    the human's verdict details. Carries the full `group` snapshot so a verdict is
    undoable (the unrule endpoint restores it to the queue verbatim), and the transcript
    fingerprint it was recorded under (the stale guard guarantees it matches the live
    transcript at record time) — the spot-answer fold only trusts a verdict against the
    same transcript version (2026-07-03 grill: drifted positions never act)."""
    return {
        **{k: group.get(k) for k in ("gid", "heard", "heard_cleaned", "canonical", "world",
                                     "label", "fixer", "dm_share", "echo_in_sentence",
                                     "n_sibling_spellings", "method")},
        "n_occurrences": len(group.get("occurrences", [])),
        "group": group,
        "transcript_fingerprint": data["transcript_fingerprint"],
        **extra,
        "at": datetime.now(timezone.utc).isoformat(),
    }


def _partition_spot_answered(live: list, data: dict) -> tuple[list, list]:
    """The THIRD fold (2026-07-03 grill, all six calls Choksi's): fixer re-runs
    regenerate questions from scratch, so verdicts that wrote NO standing rule
    (spot-by-spot answers, world-less picks, partial remainders) would be re-asked on
    every config change. Here they fold instead — strictly:

    - a ruled spot settles the WORD at that position, not one card: a no means "right
      as heard here" (answers any suggestion), a yes names what the word is there — so
      coverage matches on (heard_cleaned, segment_id, word_index) across BOTH audit
      lists, never on gid;
    - only verdicts stamped with the queue's CURRENT transcript fingerprint count
      (drifted positions never act — the stale guard's instinct, applied to history);
    - every spot on the card must be covered; a card with any un-ruled spot stays live
      showing only the un-ruled spots (`n_spot_answered_occ` counts the folded ones).

    Folded groups carry `covering` — the verdicts that answered them, one row each,
    which is exactly what the UI's inline undo needs. Runs AFTER the book partition:
    book folds state durable promises; this one states only "you already ruled these
    exact spots"."""
    tf = data["transcript_fingerprint"]
    ruled: dict = {}   # (heard_cleaned, segment_id, word_index) -> [audit entries]
    for key, verdict in (("_blessed", "yes"), ("_rejected", "no")):
        for e in data.get(key, []):
            if e.get("transcript_fingerprint") != tf:
                continue   # recorded against a different transcript version
            spots = e.get("ruled_occurrences")
            if spots is None:   # oldest entries: the snapshot carried the spots
                spots = (e.get("group") or {}).get("occurrences", [])
            for o in spots:
                k = (e.get("heard_cleaned"), o.get("segment_id"), o.get("word_index"))
                ruled.setdefault(k, []).append({**e, "verdict": verdict})
    still_live, spot_answered = [], []
    for g in live:
        covered, uncovered, covering = 0, [], {}
        for o in g["occurrences"]:
            entries = ruled.get((g.get("heard_cleaned"), o["segment_id"],
                                 o["word_index"]))
            if entries:
                covered += 1
                for e in entries:
                    ck = (e["verdict"], e.get("gid"), e.get("at"))
                    slot = covering.setdefault(ck, {
                        "verdict": e["verdict"], "gid": e.get("gid"),
                        "canonical": e.get("canonical"), "at": e.get("at"),
                        "n_spots": 0})
                    slot["n_spots"] += 1
            else:
                uncovered.append(o)
        if covered and not uncovered:
            spot_answered.append({**g, "folded_by": "spot-answers",
                                  "covering": list(covering.values())})
        elif covered:
            still_live.append({**g, "occurrences": uncovered,
                               "n_spot_answered_occ": covered})
        else:
            still_live.append(g)
    return still_live, spot_answered


def _word_is_ambiguous(data: dict, group: dict, verdict: str) -> bool:
    """The Bheem guard, generalized to the WORD (2026-07-03, found live on `arrows`):
    a forever-rule may be written to the Name Book only when this verdict covers the
    word's whole story in this session — no sibling group for the same heard word still
    queued (same word, different suggestion = ambiguity evidence), and no prior verdict
    on the word pointing the other way. Ambiguous -> the verdict is recorded as per-spot
    calibration data and deliberately teaches nothing."""
    from detectors.phonetics import clean
    hc = group.get("heard_cleaned")
    # a sibling question for the same word is still live (the acted-on group is popped)
    if any(g.get("heard_cleaned") == hc for g in data["groups"]):
        return True
    canon = clean(group.get("canonical") or "")
    for e in data.get("_blessed", []):
        if e.get("heard_cleaned") == hc and clean(e.get("canonical") or "") != canon:
            return True                     # already ruled a DIFFERENT name for this word
    if verdict == "yes":
        if any(e.get("heard_cleaned") == hc for e in data.get("_rejected", [])):
            return True                     # some spots were ruled "really the word"
    else:
        if any(e.get("heard_cleaned") == hc for e in data.get("_blessed", [])):
            return True                     # some spots were ruled a real name
    return False


@bp.route("/sessions/<session_id>/name-corrections/bless", methods=["POST"])
def bless_name_correction(session_id):
    """RECORD a yes (calibration: nothing applies — the transcript is untouched).

    Whole-name yes -> the world's Name Book (deterministic at the apply pass) + audit.
    Optional `occurrences` (a list of {segment_id, word_index}): a PER-OCCURRENCE yes for
    the same-sound-two-referents case — audit only, the book is never written from a
    partial (the Bheem guard); unselected occurrences stay in the queue."""
    if not validate_session_id(session_id):
        return jsonify({"error": "Invalid session ID"}), 400
    session_dir = get_session_dir(current_app.config["SESSIONS_DIR"], session_id)
    body = request.get_json(silent=True) or {}
    gid = (body.get("gid") or "").strip()
    if not gid:
        return jsonify({"error": "gid is required"}), 400

    data = _read_pending(session_dir)
    if data is None:
        return jsonify({"error": "No name questions for this session"}), 404
    stale = _stale_guard(session_dir, data)
    if stale is not None:
        return stale
    group = _pop_group(data, gid)
    if group is None:
        return jsonify({"error": f"No question with gid {gid!r}"}), 404
    # allow the human to override the suggested spelling in the same gesture
    canonical = (body.get("canonical") or group["canonical"]).strip()
    group["canonical"] = canonical

    # per-occurrence selection: split the group; the remainder goes back to the queue
    wanted = body.get("occurrences")
    partial = False
    if isinstance(wanted, list) and wanted:
        keys = {(o.get("segment_id"), o.get("word_index")) for o in wanted}
        selected = [o for o in group["occurrences"]
                    if (o["segment_id"], o["word_index"]) in keys]
        rest = [o for o in group["occurrences"]
                if (o["segment_id"], o["word_index"]) not in keys]
        if not selected:
            data["groups"].append(group)  # nothing matched — put it back untouched
            _write_pending(session_dir, data)
            return jsonify({"error": "no matching occurrences"}), 400
        if rest:
            partial = True
            data["groups"].append({**group, "occurrences": rest})
        group = {**group, "occurrences": selected}

    # stamp the frequency signal into the verdict record (the group snapshot carries it)
    _annotate_frequency(_read_transcript(session_dir), [group])

    # The word-level Bheem guard: only an unambiguous whole-word yes teaches the book.
    # A spelling question is CLUSTER-scoped, not word-scoped: its yes records the pick
    # (unify the cluster to this spelling at the apply pass) in the spellings slot and
    # never touches word rulings, so the word guard doesn't apply — partial is the only
    # ambiguity (some spots excluded -> audit-only, teaches nothing).
    is_spelling = group.get("label") == SPELLING_LABEL
    ambiguous = partial or (not is_spelling and _word_is_ambiguous(data, group, "yes"))
    book_written = bool(group.get("world")) and not ambiguous
    if book_written:
        import namebook  # src/ on sys.path via api.app
        if is_spelling:
            namebook.pick_spelling(group["world"], group["cluster_id"], canonical,
                                   provenance=f"bless:{session_id}", base=_book_dir())
        else:
            namebook.bless(group["world"], group.get("heard") or group["heard_cleaned"],
                           canonical, provenance=f"bless:{session_id}", base=_book_dir())

    data.setdefault("_blessed", []).append(_verdict_record(group, data, {
        "canonical": canonical,
        "partial": partial,
        "book_written": book_written,
        "ruled_occurrences": [
            {"segment_id": o["segment_id"], "word_index": o["word_index"]}
            for o in group["occurrences"]],
    }))
    _write_pending(session_dir, data)
    return jsonify({"status": "recorded", "verdict": "yes", "canonical": canonical,
                    "applied_occurrences": 0,   # calibration: nothing applies
                    "book_written": book_written,
                    "n_pending": len(data["groups"]),
                    "note": ("recorded — applies at the apply pass" if book_written else
                             "recorded for these spots only — the word is ambiguous here, "
                             "so no forever-rule was written")})


@bp.route("/sessions/<session_id>/name-corrections/reject", methods=["POST"])
def reject_name_correction(session_id):
    """RECORD a no (calibration: the transcript was never going to change for a queued
    item). An unambiguous whole-word no enters the Name Book (never rewritten at the
    apply pass, never asked again); a per-occurrence no — or a no on a word with sibling
    questions / prior yeses — is recorded as per-spot calibration data and teaches
    nothing (the word-level Bheem guard). Optional `occurrences` selects spots; the
    unselected remainder stays queued."""
    if not validate_session_id(session_id):
        return jsonify({"error": "Invalid session ID"}), 400
    session_dir = get_session_dir(current_app.config["SESSIONS_DIR"], session_id)
    body = request.get_json(silent=True) or {}
    gid = (body.get("gid") or "").strip()
    if not gid:
        return jsonify({"error": "gid is required"}), 400

    data = _read_pending(session_dir)
    if data is None:
        return jsonify({"error": "No name questions for this session"}), 404
    stale = _stale_guard(session_dir, data)
    if stale is not None:
        return stale
    group = _pop_group(data, gid)
    if group is None:
        return jsonify({"error": f"No question with gid {gid!r}"}), 404

    # per-occurrence selection: split the group; the remainder goes back to the queue
    wanted = body.get("occurrences")
    partial = False
    if isinstance(wanted, list) and wanted:
        keys = {(o.get("segment_id"), o.get("word_index")) for o in wanted}
        selected = [o for o in group["occurrences"]
                    if (o["segment_id"], o["word_index"]) in keys]
        rest = [o for o in group["occurrences"]
                if (o["segment_id"], o["word_index"]) not in keys]
        if not selected:
            data["groups"].append(group)  # nothing matched — put it back untouched
            _write_pending(session_dir, data)
            return jsonify({"error": "no matching occurrences"}), 400
        if rest:
            partial = True
            data["groups"].append({**group, "occurrences": rest})
        group = {**group, "occurrences": selected}

    # stamp the frequency signal into the verdict record (the group snapshot carries it)
    _annotate_frequency(_read_transcript(session_dir), [group])

    # A spelling-card no means "don't unify these" — NARROWLY (Choksi's call, 2026-07-03):
    # it records a null pick for the CLUSTER (folds this question forever, liftable) and
    # writes NO word-level no, so it can never mute another fixer's questions about the
    # same word (the arrows lesson). Word questions keep the word-level no + Bheem guard.
    is_spelling = group.get("label") == SPELLING_LABEL
    ambiguous = partial or (not is_spelling and _word_is_ambiguous(data, group, "no"))
    book_written = bool(group.get("world")) and not ambiguous
    if book_written:
        import namebook  # src/ on sys.path via api.app
        if is_spelling:
            namebook.pick_spelling(group["world"], group["cluster_id"], None,
                                   provenance=f"reject:{session_id}", base=_book_dir())
        else:
            namebook.reject(group["world"], group["heard_cleaned"],
                            provenance=f"reject:{session_id}", base=_book_dir())

    data.setdefault("_rejected", []).append(_verdict_record(group, data, {
        "partial": partial,
        "book_written": book_written,
        "ruled_occurrences": [
            {"segment_id": o["segment_id"], "word_index": o["word_index"]}
            for o in group["occurrences"]],
    }))
    _write_pending(session_dir, data)
    return jsonify({"status": "recorded", "verdict": "no",
                    "book_written": book_written,
                    "n_pending": len(data["groups"]),
                    "note": ("recorded — this word is never asked about again"
                             if book_written else
                             "recorded for these spots only — the word is ambiguous here, "
                             "so no forever-rule was written")})


@bp.route("/sessions/<session_id>/name-corrections/unrule", methods=["POST"])
def unrule_name_correction(session_id):
    """Undo a recorded verdict (the 'Your verdicts' section's undo button): pop the audit
    entry, reverse its Name Book write, and restore the question to the live queue from
    the snapshot the verdict carried. Body: {verdict: "yes"|"no", gid, at?} — `at` pins
    one entry when the same question was ruled more than once (default: newest)."""
    if not validate_session_id(session_id):
        return jsonify({"error": "Invalid session ID"}), 400
    session_dir = get_session_dir(current_app.config["SESSIONS_DIR"], session_id)
    body = request.get_json(silent=True) or {}
    verdict = (body.get("verdict") or "").strip()
    gid = (body.get("gid") or "").strip()
    at = (body.get("at") or "").strip()
    if verdict not in ("yes", "no") or not gid:
        return jsonify({"error": "verdict ('yes'|'no') and gid are required"}), 400

    data = _read_pending(session_dir)
    if data is None:
        return jsonify({"error": "No name questions for this session"}), 404
    key = "_blessed" if verdict == "yes" else "_rejected"
    entries = data.get(key, [])
    idx = next((i for i in range(len(entries) - 1, -1, -1)
                if entries[i].get("gid") == gid
                and (not at or entries[i].get("at") == at)), None)
    if idx is None:
        return jsonify({"error": f"no recorded {verdict} for gid {gid!r}"}), 404
    entry = entries[idx]
    snapshot = entry.get("group")
    if snapshot is None:
        return jsonify({"error": "this verdict predates undo support"}), 409
    entries.pop(idx)

    # reverse the Name Book write — only if this verdict actually wrote one (ambiguous /
    # partial verdicts teach nothing; legacy entries without the flag follow the old rule)
    import namebook  # src/ on sys.path via api.app
    world = entry.get("world") or ""
    wrote_book = entry.get("book_written", not entry.get("partial"))
    if world and wrote_book:
        if entry.get("label") == SPELLING_LABEL:
            # a spelling verdict (either kind) wrote a cluster pick, not a word ruling
            namebook.unpick_spelling(world, snapshot["cluster_id"], base=_book_dir())
        elif verdict == "yes":
            namebook.unbless(world, snapshot.get("heard") or snapshot.get("heard_cleaned"),
                             base=_book_dir())
        else:
            namebook.unreject(world, entry.get("heard_cleaned") or "", base=_book_dir())

    # restore the question: merge into the surviving same-gid group (a partial yes left
    # the remainder queued) or append the snapshot whole
    sibling = next((g for g in data["groups"] if g.get("gid") == gid), None)
    if sibling is not None:
        have = {(o["segment_id"], o["word_index"]) for o in sibling["occurrences"]}
        sibling["occurrences"].extend(
            o for o in snapshot.get("occurrences", [])
            if (o["segment_id"], o["word_index"]) not in have)
    else:
        data["groups"].append(snapshot)

    _write_pending(session_dir, data)
    return jsonify({"status": "unruled", "verdict": verdict, "gid": gid,
                    "n_pending": len(data["groups"])})


@bp.route("/name-book/unpick-spelling", methods=["POST"])
def unpick_spelling():
    """Lift a standing spelling pick (the fold's lift button on a spelling question).
    The cluster's questions return to the live queue on the next GET — the partition is
    view-time, so nothing else needs rewriting."""
    body = request.get_json(silent=True) or {}
    world = (body.get("world") or "").strip()
    cluster = (body.get("cluster") or "").strip()
    if not world or not cluster:
        return jsonify({"error": "world and cluster are required"}), 400
    import namebook  # src/ on sys.path via api.app
    removed = namebook.unpick_spelling(world, cluster, base=_book_dir())
    if not removed:
        return jsonify({"error": "no matching spelling pick in the Name Book"}), 404
    return jsonify({"status": "unpicked", "world": world, "cluster": cluster})


@bp.route("/name-book/unreject", methods=["POST"])
def unreject_name():
    """Lift a standing no (the fold's "remove this no" button). Matching questions return
    to the live queue on the next GET — the partition is view-time, so nothing else needs
    rewriting."""
    body = request.get_json(silent=True) or {}
    world = (body.get("world") or "").strip()
    heard_cleaned = (body.get("heard_cleaned") or "").strip()
    if not world or not heard_cleaned:
        return jsonify({"error": "world and heard_cleaned are required"}), 400
    import namebook  # src/ on sys.path via api.app
    removed = namebook.unreject(world, heard_cleaned, base=_book_dir())
    if not removed:
        return jsonify({"error": "no matching no in the Name Book"}), 404
    return jsonify({"status": "unrejected", "world": world, "heard_cleaned": heard_cleaned})
