#!/usr/bin/env python3
"""Offline worker for the per-story name auditor — runs in a fresh subprocess.

`run(session_dir)` is invoked via model_runner.run_model from CanonNameDetector: a
spawned subprocess of this venv, so its Gemma load gets a clean GPU process and frees
it on exit. In ONE model load it (1) takes the pipeline's saved story regions
(`_stories`), or segments live as a fallback for older sessions, (2) per story builds
recall-recovered name cards and runs the v2 audit (+ canon shield), (3) EXPANDS each
per-spelling verdict to the standard per-occurrence flag schema (segment_id/word_index/
...), and RETURNS {n_word_tokens, flags}. Diagnostics go to stderr; a __main__ block
prints the JSON for standalone debugging.

The flag-expansion (the one production-specific adaptation) and the region adapter live
here; the validated logic is imported from story_segment and the ported _audit/_names
modules.
"""
import json
import sys
from pathlib import Path

# This file lives at src/detectors/story_names/_worker.py — put src/ on the path so the
# `detectors` package imports resolve (the venv's sys.path[0] is this dir, not src/).
SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from story_segment import segment_session, make_reader
from detectors.story_names._audit import story_name_cards, run_v2
from detectors.story_names._names import story_segments, proper_name_candidates
from detectors.phonetics import clean, codes


def build_regions(stories, pos_of):
    """Live equivalent of the EMP's load_regions: map each Stage-0 story's start/end
    ids to transcript positions, in order. The story SOURCE is the live segmenter
    output (not a pre-baked file); the id->position bridge is unchanged."""
    out = []
    for i, st in enumerate(stories):
        sp = pos_of.get(st["start_id"])
        ep = pos_of.get(st["end_id"])
        if sp is None or ep is None:
            print(f"  WARNING: story ids {st.get('start_id')}-{st.get('end_id')} "
                  f"absent from transcript — skipped", file=sys.stderr)
            continue
        if ep < sp:
            sp, ep = ep, sp
        out.append({"idx": i, "start_id": st["start_id"], "end_id": st["end_id"],
                    "start_pos": sp, "end_pos": ep,
                    "world": st.get("world", ""), "title": st.get("title", "")})
    return out


def expand_flag(flag, card, seg_by_id, story_idx, world):
    """Expand one per-(story, spelling) v2 flag into per-occurrence Monitor flags.

    Walk the card's occurrence positions and emit one flag per occurrence whose token
    (single word, or the reconstructed bigram for a phrase card) cleans into the flag's
    wrong_cleaned set. Because run_v2's canon shield rebuilds wrong_cleaned to EXCLUDE
    shielded spellings, a protected real name (James) yields zero occurrences here with
    no special-casing."""
    wrong = set(flag["wrong_cleaned"])
    is_phrase = card.get("is_phrase", False)
    out = []
    for o in card["occ"]:
        seg = seg_by_id.get(o["seg_id"])
        if not seg:
            continue
        words = seg.get("words") or []
        wi = o["wi"]
        if wi >= len(words):
            continue
        w0 = words[wi]
        if is_phrase:
            if wi + 1 >= len(words):
                continue
            w1 = words[wi + 1]
            raw = (w0["word"].strip() + " " + w1["word"].strip()).strip()
            start, end = w0.get("start"), w1.get("end")
        else:
            raw = w0["word"].strip()
            start, end = w0.get("start"), w0.get("end")
        c = clean(raw)
        if c not in wrong:
            continue
        rec = {
            "segment_id": o["seg_id"],
            "word_index": wi,
            "start": start,
            "end": end,
            "token": raw,
            "cleaned": c,
            "dm_codes": sorted(codes(c)),
            "case": flag["case"],            # "M9b" or "M9c"
            "canonical": flag["canonical"],  # spelling to standardize on (NOT applied — detection only)
            "card_id": flag["card_id"],
            "all_spellings": flag["all_spellings"],
            "wrong_cleaned": flag["wrong_cleaned"],
            "evidence": flag.get("evidence", ""),
            "story_id": story_idx,
            "story_world": world,
        }
        if flag.get("shielded"):  # M9b only — the canon spellings the shield protected
            rec["shielded"] = flag["shielded"]
        out.append(rec)
    return out


def expand_combined(flags, cards, seg_by_id, story_idx, world):
    """Expand combined judge+phonetic (Qwen3.5) flags to per-occurrence Monitor flags.

    The judge flags carry no real card_id (they came from a name list, not a card), so we
    pair each flag with EVERY card and let expand_flag emit only the occurrences whose
    cleaned token is in the flag's wrong_cleaned set — a non-matching card yields nothing.
    Dedupe by (segment_id, word_index, case) so a spelling that lives in two cards (or is
    caught by both methods) lands once."""
    out, seen = [], set()
    for f in flags:
        for card in cards:
            f2 = dict(f)
            f2["card_id"] = card["id"]
            # Always carry the matched card's full surface set (a judge flag arrives with only
            # the one misspelling it saw — [wrong] — which would otherwise under-report the
            # spellings the Monitor shows; the phonetic path already sets this to card surface).
            f2["all_spellings"] = card["surface"]
            for rec in expand_flag(f2, card, seg_by_id, story_idx, world):
                rec["methods"] = f2["methods"]  # how it was caught (judge / sound-match) — informational
                # Confidence = does the suggested spelling SOUND like what was heard (shared Double
                # Metaphone code)? A judge-only catch can still be confident (Bishma->Bhishma); a
                # garbled guess (Urzi->Urjani) is not. This drives the "best guess" badge, not `methods`.
                rec["suggestion_confident"] = bool(codes(rec["cleaned"]) & codes(clean(rec["canonical"])))
                if "vote_count" in f2:          # order-robust judge stamped how many rounds agreed
                    rec["vote_count"], rec["vote_rounds"] = f2["vote_count"], f2["vote_rounds"]
                key = (rec["segment_id"], rec["word_index"], rec["case"])
                if key not in seen:
                    seen.add(key)
                    out.append(rec)
    return out


def count_word_tokens(rich):
    """Non-empty cleaned word tokens across all segments — matches the M9b detector's
    n_word_tokens so the Monitor's 'flags / tokens' line is comparable."""
    n = 0
    for s in rich["segments"]:
        for w in s.get("words", []):
            if clean(w["word"].strip()):
                n += 1
    return n


def run(session_dir):
    session_dir = Path(session_dir)
    rich = json.loads((session_dir / "transcript-rich.json").read_text())
    seg_by_id = {s["id"]: s for s in rich["segments"]}
    pos_of = {s["id"]: i for i, s in enumerate(rich["segments"])}

    gen = make_reader()  # one model load — the per-story audit (run_v2) needs it

    stories = rich.get("_stories")
    if stories is not None:  # pipeline already segmented (even to zero stories) — no re-split
        print(f"  using {len(stories)} saved story region(s)", file=sys.stderr)
    else:  # fallback: a session processed before story segmentation shipped (key absent)
        seg_result, _ = segment_session(session_dir, gen)
        stories = seg_result["stories"]
    regions = build_regions(stories, pos_of)
    print(f"  auditing names across {len(regions)} story region(s)", file=sys.stderr)

    flags = []
    for r in regions:
        segs = story_segments(rich, r, pos_of)
        cards = story_name_cards(segs, recover=True)      # recall recovery
        card_by_id = {c["id"]: c for c in cards}
        v2_flags, _ = run_v2(gen, r["world"], segs, cards, raw_log=[])  # + canon shield
        for f in v2_flags:
            if f["case"] == "M9d-suspect":                # dropped for v1 (low-confidence substitution)
                continue
            card = card_by_id.get(f["card_id"])
            if card is None:                              # v2 always sets a real card_id; defensive
                continue
            flags.extend(expand_flag(f, card, seg_by_id, r["idx"], r["world"]))

    return {"n_word_tokens": count_word_tokens(rich), "flags": flags}


def run_qwen35(session_dir):
    """Qwen3.5 canon pass (the production M9c detector, v0.3.0). One model load; per story
    region: read the world from the saved Qwen3.5 segmentation, generate a cast, judge the
    names, phonetic-match against the cast, UNION the two, dictionary-gate, and expand to
    per-occurrence flags. The world already lives in `_stories` (Qwen3.5 segmentation), so no
    separate world call; an older session with no `_stories` is segmented live with Qwen3.5
    (world included). Returns {n_word_tokens, flags}. The Gemma run() above stays as baseline."""
    from story_segment_qwen35 import segment_session as segment_session_q, make_reader as make_reader_q
    from detectors.story_names import _qwen35

    session_dir = Path(session_dir)
    rich = json.loads((session_dir / "transcript-rich.json").read_text())
    seg_by_id = {s["id"]: s for s in rich["segments"]}
    pos_of = {s["id"]: i for i, s in enumerate(rich["segments"])}

    gen = make_reader_q()  # one Qwen3.5 load for the whole session

    stories = rich.get("_stories")
    if stories is not None:
        print(f"  using {len(stories)} saved story region(s)", file=sys.stderr)
    else:  # older session: segment live with Qwen3.5 (world included)
        seg_result, _ = segment_session_q(session_dir, gen)
        stories = seg_result["stories"]
    regions = build_regions(stories, pos_of)
    print(f"  Qwen3.5 canon audit across {len(regions)} story region(s)", file=sys.stderr)

    flags = []
    for r in regions:
        try:  # one bad story must not lose the flags already found in the others
            segs = story_segments(rich, r, pos_of)
            cards = story_name_cards(segs, recover=True)
            singles = proper_name_candidates(segs)
            names = sorted({s for c in cards for s in c["surface"]})
            # Recognize the world from the NAME LIST, not the saved segmentation world: pass-2 over
            # a long whole-recording region abstains (Mahabharata held-out world=''), but the names
            # place it reliably. Abstains to "" on unrecognizable names -> canon check stays off.
            world = _qwen35.recognize_world(gen, names)
            if not world:
                continue  # no recognized world -> no canon check (same contract as Gemma run)
            # Split-prompt cast (characters + GROUPS, unioned; worldcast) — the single
            # "25 best-known characters/families/places" prompt reliably DROPPED the group
            # names (Pandavas, Kauravas, Crystal Gems...), which are exactly what a child's
            # garbled group-word needs to sound-match against. Found in the 2026-07-01
            # namefix validation (emp/emp.md); shared with the namefix correction stage.
            import worldcast
            cast = worldcast.correction_cast(worldcast.cached_cast_split(gen, world))
            judge_flags = _qwen35.judge_names_voted(gen, world, names, singles)  # order-robust vote
            phon_flags = _qwen35.phonetic_flags(cards, singles, cast)
            combined = _qwen35.combine(judge_flags, phon_flags)
            flags.extend(expand_combined(combined, cards, seg_by_id, r["idx"], world))
        except Exception as ex:
            print(f"  WARNING: canon audit failed on story {r.get('idx')}: {repr(ex)[:200]}",
                  file=sys.stderr)

    return {"n_word_tokens": count_word_tokens(rich), "flags": flags}


if __name__ == "__main__":  # standalone debugging; production calls run() via model_runner
    if len(sys.argv) < 2:
        print("usage: _worker.py <session_dir>", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(run(sys.argv[1])))
