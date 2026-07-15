#!/usr/bin/env python3
"""Stage-0 story segmenter — Qwen3.5 backend (the production default).

Mirrors `story_segment.py`'s public surface (segment_segments / segment_session /
segment_transcript / MODEL_ID / SEGMENT_CONFIG_VERSION) so `pipeline.py` swaps backends
by changing one import. Every DETERMINISTIC piece — the candidate-boundary pre-filter,
the windows, the stateful region assembly, the consecutive-run merge, the wind-down
drop — is imported untouched from `story_segment`; only the three MODEL passes change:
Gemma+JSON -> Qwen3.5+plain-text. The Gemma module stays intact as the runnable baseline
(`story_segment.py`), exactly the family_names.py precedent.

Why Qwen3.5 here: it splits the 5 hand-marked sessions at 5/5 story counts and mean
region IoU 0.94 (vs Gemma's 5/5 / 0.86) — validated in emp/src/seg_qwen35.py, which
swept 4 plain-text boundary prompts; the `conservative` wording (default to "no
boundary", only break when sure) won. The merge prompt didn't move the score; the
`faithful` wording is kept. World/title naming (pass 2) reads the full story region.

PLAIN TEXT across the model boundary, never JSON (memory plain-text-not-json):
  pass 1 emits `START <id>` / `END <id>` / `NONE` lines;
  the merge emits one line of letters per story;
  pass 2 emits `Title:` / `World:` lines.
Qwen3.5 runtime nuances (VLM loader, enable_thinking=False) live in `qwen35.make_reader`.
"""
import hashlib
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen35 import make_reader, MODEL_ID
# Deterministic scaffolding — imported, never reimplemented (one source of truth).
from story_segment import (load_segments, load_segments_from_list, propose, cluster,        # noqa: E402
                           build_window, render_window, _mkregion, _state_line,
                           _render_regions_for_merge, _winddown_fraction, coerce_id,
                           full_region_lines, CUES, CTX, MAX_WIN, HEAD, TAIL)

# ============================ prompts (plain text) ===========================
# Ported verbatim from the winning sweep (emp/src/seg_qwen35.py): PASS1='conservative',
# merge='faithful'. Don't edit without re-scoring against the segmentation truth.
FMT_P1 = ("\n\nOutput — for THIS window only, write one line per boundary you find:\n"
          "START <id>   (a new story begins at that line)\n"
          "END <id>     (the currently-open story ends at that line)\n"
          "Use only the [id] numbers shown above. If nothing in this window begins or "
          "ends a story, write exactly:\nNONE\nWrite nothing else — no explanations, no JSON.\n\n"
          "Window:\n{window}")

PASS1_PROMPT = (
    "You are splitting a bedtime recording into its stories, reading one window at a time. The "
    "DEFAULT answer is NONE — a recording is usually a single long story, and most windows are "
    "deep inside it.\n\nCURRENT STATE: {state}\n\n"
    "Only break the default when you are SURE:\n"
    "- START: a genuinely new story opens here (a clear launch cue, or — at the very start of "
    "the recording — the first line of narration).\n"
    '- END: the open story has clearly finished ("the end", a sung wind-down, or the parent and '
    "child start choosing a different story).\n"
    "Pauses, milk breaks, spills, repeated lines, and characters talking are NOT boundaries — "
    "they are the same story. When unsure, answer NONE." + FMT_P1)

FMT_MERGE = ("\n\nOutput: one line per distinct story, listing that story's segment letters "
             "separated by spaces. Every letter from A to {last} must appear on exactly one line. "
             "Write only letters — no other words, no JSON. Example:\nA B\nC")

PASS_MERGE_PROMPT = (
    "A first pass split ONE bedtime recording into the candidate story-segments below, labelled "
    "by LETTER, in order. Group the segments that belong to the SAME story; keep different "
    "stories separate. Every segment is part of some story — your only job is grouping.\n"
    "- SAME story: same characters and plot; a later segment just CONTINUES after an "
    "interruption (milk, a spill, a tangent). A back-and-forth where parent and child build ONE "
    "story together is still one story.\n"
    "- DIFFERENT stories: the characters or world clearly change, or someone announces a switch "
    '("let\'s tell a different story"). When unsure, keep them separate.\n\nSegments:\n{segments}'
    + FMT_MERGE)

# PASS 2 — world + title over the full story region. Plain text. The two-step reasoning
# (judge garbled names by sound; name a REAL world or abstain) is the worlds-bench winner;
# Qwen3.5's recent-world knowledge is what clears the world wall Gemma hit (e.g. KPop Demon
# Hunters). "original" maps to the empty world (canon check stays off).
PASS2_PROMPT = (
    # NOTE: do NOT add a parenthetical "(a name may be misspelled or split into pieces)" after
    # this sentence — it primes the model to distrust the names and abstain (Pandavas flips
    # Mahabharata->"" deterministically, 6/6). This wording byte-matches the validated
    # `twostep_soft` winner (emp/src/tune_pass2_worlds.py): 2/2 canon, 5/5 made-up kept empty.
    "You are reading the full transcript of ONE bedtime story a parent told a young child, out "
    "loud — so it rambles and proper names may be mis-transcribed.\n\n"
    "Give it a short descriptive title, and decide what fictional WORLD it is set in. Think in "
    "two quick steps:\n"
    "1) Note the distinctive character names, place names, and signature words. Names may be "
    "garbled — judge by what they sound like.\n"
    "2) Do these characters and places come from a REAL, widely-known story world (a book, film, "
    "show, myth, or franchise) — even one never named aloud in the story? If you can place them, "
    'name that world. Write "original" ONLY when you cannot place the characters in any known '
    'world.\n\n'
    "Write exactly two lines, nothing else:\n"
    "Title: <short title>\n"
    "World: <the world, or \"original\" if made up>\n\n"
    "Story lines:\n{lines}")

_NON_WORLDS = {"", "original", "made up", "made-up", "none", "unknown", "n/a", "original / made-up"}


# ============================ plain-text parsers =============================
def parse_pass1(raw, valid_ids, by_id):
    """START/END lines -> walk events. Robust to trailing text on the line."""
    events = []
    for line in (raw or "").splitlines():
        mt = re.match(r"\s*(START|END)\b", line, re.I)
        if not mt:
            continue
        mid = re.search(r"(gap_[\d.]+|\d+)", line[mt.end():])
        if not mid:
            continue
        sid = coerce_id(mid.group(1), valid_ids)
        if sid is None:
            continue
        typ = "story_start" if mt.group(1).upper() == "START" else "story_end"
        events.append({"type": typ, "segment_id": sid, "quote": by_id.get(sid, "")})
    return events


def parse_groups(raw, n):
    """Merge groups: each non-empty line is one story's letters. A token counts only if
    it is exactly a single in-range letter (drops stray prose like 'Group')."""
    last = chr(64 + n)
    groups = []
    for line in (raw or "").splitlines():
        letts = [t.upper() for t in re.split(r"[\s,]+", line.strip())
                 if len(t) == 1 and "A" <= t.upper() <= last]
        if letts:
            groups.append([ord(c) - 65 for c in letts])
    return groups


def parse_name(raw):
    """`Title:`/`World:` lines -> (title, world). 'original'/made-up -> '' (canon off)."""
    title, world = "", ""
    for line in (raw or "").splitlines():
        m = re.match(r"\s*title\s*[:\-]\s*(.+)", line, re.I)
        if m:
            title = m.group(1).strip().strip('"')
        m = re.match(r"\s*world\s*[:\-]\s*(.+)", line, re.I)
        if m:
            world = m.group(1).strip().strip('"')
    if world.lower() in _NON_WORLDS:
        world = ""
    return title, world


# ============================ model passes ===================================
def pass1_events(gen, window_text, state_line, valid_ids, by_id, raw_log):
    raw = gen(PASS1_PROMPT.format(state=state_line, window=window_text), max_tokens=200)
    raw_log.append({"pass": 1, "raw": raw})
    return parse_pass1(raw, valid_ids, by_id)


def pass2_name(gen, lines_text, raw_log):
    raw = gen(PASS2_PROMPT.format(lines=lines_text), max_tokens=160)
    raw_log.append({"pass": 2, "raw": raw})
    return parse_name(raw)


def global_consolidate(gen, segs, regions, raw_log, winddown_thresh=0.34):
    """Global merge — wind-down drop (deterministic) then MODEL merge-only over consecutive
    same-label runs. Ported from story_segment.global_consolidate; plain-text groups. Only
    ever reduces the count; an omitted/odd group stays its own story (degrades safely)."""
    kept = [r for r in regions if _winddown_fraction(segs, r) < winddown_thresh] or regions
    n = len(kept)
    if n <= 1:
        return kept
    raw = gen(PASS_MERGE_PROMPT.format(segments=_render_regions_for_merge(segs, kept),
                                       last=chr(64 + n)), max_tokens=200)
    raw_log.append({"pass": "merge", "raw": raw})
    label = {}
    for gid, g in enumerate(parse_groups(raw, n)):
        for i in g:
            if 0 <= i < n:
                label.setdefault(i, ("g", gid))
    for i in range(n):
        label.setdefault(i, ("solo", i))  # never silently lose a region
    runs, cur = [], None
    for i in range(n):
        lab = label[i]
        if cur is not None and cur["lab"] == lab:
            cur["last"] = i
        else:
            cur = {"lab": lab, "first": i, "last": i}
            runs.append(cur)
    return [{"start_id": kept[c["first"]]["start_id"], "end_id": kept[c["last"]]["end_id"],
             "start_pos": kept[c["first"]]["start_pos"], "end_pos": kept[c["last"]]["end_pos"],
             "start_quote": kept[c["first"]].get("start_quote", ""),
             "end_quote": kept[c["last"]].get("end_quote", "")} for c in runs]


# ============================ orchestration ==================================
def segment_segments(segs, gen, name="session", use_global=True):
    """Stateful left-to-right walk -> story regions -> world/title naming. The assembly
    state machine is identical to story_segment.segment_segments; only the model passes are
    plain-text Qwen3.5. Returns (result_dict, raw_log)."""
    order = [s["id"] for s in segs]
    pos_of = {x: i for i, x in enumerate(order)}
    by_id = {s["id"]: s["text"] for s in segs}
    valid_ids = set(order)
    zones = cluster(propose(segs))
    raw_log = []

    regions, open_story, n_events = [], None, 0
    for z in zones:
        positions, elide = build_window(segs, z)
        window_text = render_window(segs, positions, z["sig_at"], elide)
        try:
            evs = pass1_events(gen, window_text, _state_line(open_story), valid_ids, by_id, raw_log)
        except Exception as ex:  # one bad zone must not kill the session
            raw_log.append({"pass": 1, "error": repr(ex)[:200], "zone": z["positions"]})
            evs = []
        n_events += len(evs)
        for e in sorted(evs, key=lambda e: (pos_of[e["segment_id"]], 0 if e["type"] == "story_end" else 1)):
            ep = pos_of[e["segment_id"]]
            if e["type"] == "story_start":
                if open_story is None:
                    open_story = e
                elif ep - 1 >= pos_of[open_story["segment_id"]]:
                    regions.append(_mkregion(open_story, order[ep - 1], "<next story starts>", pos_of))
                    open_story = e
                else:  # spurious backward start — keep the open story, never lose it
                    raw_log.append({"pass": 1, "note": "skipped backward story_start"})
            else:  # story_end
                if open_story is None:  # recording opened mid-story
                    sp = min((regions[-1]["end_pos"] + 1) if regions else 0, ep)
                    open_story = {"segment_id": order[sp], "quote": "<recording opened mid-story>"}
                regions.append(_mkregion(open_story, e["segment_id"], e.get("quote", ""), pos_of))
                open_story = None
    if open_story is not None:  # dangling open story -> runs to end of recording
        regions.append(_mkregion(open_story, order[-1], "<runs to end of recording>", pos_of))
    if not regions and order:  # no boundary detected anywhere -> the whole recording is ONE story
        regions.append(_mkregion({"segment_id": order[0], "quote": "<whole recording>"},
                                 order[-1], "<whole recording>", pos_of))

    n_pre = len(regions)
    if use_global:
        regions = global_consolidate(gen, segs, regions, raw_log)

    stories = []
    for r in regions:
        try:
            title, world = pass2_name(gen, full_region_lines(segs, r), raw_log)
        except Exception as ex:
            title, world = "", ""
            raw_log.append({"pass": 2, "error": repr(ex)[:200]})
        stories.append({"start_id": r["start_id"], "end_id": r["end_id"],
                        "title": title, "world": world,
                        "evidence": {"start_quote": r["start_quote"], "end_quote": r["end_quote"]}})
    merge_note = f"{n_pre}->{len(stories)} after merge" if use_global else f"{len(stories)} (no merge)"
    print(f"  {name:12s}: {len(zones):2d} zones -> {n_events:2d} events -> {merge_note} stories  "
          + " | ".join(f'{s["start_id"]}-{s["end_id"]} [{s["world"]}]' for s in stories),
          file=sys.stderr)
    return {"name": name, "n_zones": len(zones), "n_events": n_events,
            "n_regions_pre_merge": n_pre, "stories": stories}, raw_log


def segment_session(session_dir, gen, use_global=True):
    """Disk-based entry (the detector's fallback path): read the session's transcript from
    disk, then run the Qwen3.5 segmentation walk. Returns (result_dict, raw_log)."""
    segs = load_segments(session_dir)
    return segment_segments(segs, gen, name=Path(session_dir).name, use_global=use_global)


# ============================ in-memory caller (pipeline) ====================
def _segment_in_subprocess(transcript):
    """Module-level (picklable for the spawned subprocess). Segment an IN-MEMORY transcript
    and return its stories list. Loads Qwen3.5 once; the process exits after, freeing GPU."""
    segs = load_segments_from_list(transcript["segments"])
    gen = make_reader()
    result, _ = segment_segments(segs, gen)
    return result["stories"]


def segment_transcript(transcript, timeout=1800):
    """Segment the given in-memory transcript via a fresh subprocess (model_runner),
    returning the stories list [{start_id, end_id, title, world, evidence}]."""
    from model_runner import run_model
    return run_model(_segment_in_subprocess, transcript, timeout=timeout)


# Cache config fingerprint — a hash of everything that drives the segmenter's output, so
# editing any prompt / cue / window constant invalidates cached stories on the next
# re-enrich automatically. Switching from the Gemma module to this one changes MODEL_ID and
# the prompts, so every session's cached Gemma segmentation is naturally considered stale.
SEGMENT_CONFIG_VERSION = MODEL_ID + "|" + hashlib.sha256("".join([
    "q35-v1", PASS1_PROMPT, PASS2_PROMPT, PASS_MERGE_PROMPT,
    repr(sorted(CUES.items())), repr((CTX, MAX_WIN, HEAD, TAIL)),
]).encode()).hexdigest()[:16]
