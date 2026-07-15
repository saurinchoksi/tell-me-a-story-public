#!/usr/bin/env python3
"""Stage-0 story segmenter — split a recording into its stories.

PORTED VERBATIM from the sealed EMP probe emp/src/segment.py (the family_names.py
precedent: production copies the validated eval logic, emp/ stays the eval of
record). The only changes from the original are production de-hardcodings, marked
PROD: load_segments/segment_session take a session_dir instead of an EMP session
id; the 5-session `SESSIONS` name lookup is gone (any session works now); and the
one summary print goes to stderr so the worker's stdout stays clean JSON. A later
PROD change generalizes pass 2 (world naming): it reads the FULL story region (not a
12-line sample) and uses a general world prompt with no dataset-specific examples —
validated on emp/results/worlds-bench (the held-out Mahabharata, missed under
sampling, is recovered with the full story). Boundary detection is unchanged. The CLI
(`main`), the pre-baked prediction file, and the truth sidecar are dropped — this
module is imported, never run standalone.

"Code proposes, model confirms," the same shape as the name detectors:

  1. PRE-FILTER (cheap code, high recall) — verbal cues + time-gaps + structural
     markers + recording-edge sentinels propose candidate boundary zones.
  2. READER pass 1 (local LLM) — per candidate zone, extract boundary EVENTS
     (story_start / story_end / nothing) in a small window.
  3. ASSEMBLY (deterministic code) — pair start/end events into story regions.
  4. GLOBAL MERGE (local LLM, on by default) — re-read the regions together and
     merge interruption-split halves into one story.
  5. READER pass 2 (local LLM) — per region, name the world + a title.

Output per session: stories = [{start_id, end_id, title, world, evidence}].
LOCAL-ONLY: the reader is Gemma-4 E4B via MLX, run in a fresh subprocess.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

# window construction
CTX = 3        # context lines on each side of a candidate cluster
MAX_WIN = 22   # if a window exceeds this, show head+tail with an elision
HEAD, TAIL = 11, 9

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# --- cue lexicon (shared with the recall harness) -----------------------------
CUES = {
    "START": [
        r"\bstart the story\b", r"\blet'?s start\b", r"\bsay the words\b",
        r"\bonce upon a\b", r"\bhelp me tell (it|the story)\b",
    ],
    "END": [r"\bthe end\b", r"\bthat'?s the end\b"],
    "TRANSITION": [
        r"\b(a )?(different|another|new|last|next) (kind of )?stor(y|ies)\b",
        r"\bno more .{0,20}\bstor(y|ies)\b",
        r"\bwhat should we (do|tell) .{0,20}\bstor(y|ies)\b",
        r"\bdo you want (a|an)other\b",
    ],
    "WINDDOWN": [r"\bi see trees of green\b", r"\bred roses,? too\b", r"\bwhat a wonderful world\b"],
}
COMPILED = {k: [re.compile(p, re.I) for p in pats] for k, pats in CUES.items()}


# ============================ pre-filter (code) ===============================
def load_segments_from_list(rich_segments):
    """Build the segmenter's per-segment meta-list from an IN-MEMORY transcript's
    `segments` (each has id/start/end/text/words). Used by the pipeline pass, which
    holds the transcript before it's written to disk — never re-read the stale file."""
    out = []
    for i, s in enumerate(rich_segments):
        out.append({"pos": i, "id": s["id"], "start": s.get("start"), "end": s.get("end"),
                    "text": (s.get("text") or "").strip(), "is_gap": not isinstance(s["id"], int)})
    for i, s in enumerate(out):
        prev = out[i - 1] if i > 0 else None
        s["gap_before"] = (s["start"] - prev["end"]) if (prev and s["start"] is not None and prev["end"] is not None) else None
    return out


def load_segments(session_dir):
    # PROD: read the session dir we're handed (the detector fallback path).
    rich_segments = json.loads((Path(session_dir) / "transcript-rich.json").read_text())["segments"]
    return load_segments_from_list(rich_segments)


def cue_hits(text):
    hits = []
    for kind, pats in COMPILED.items():
        if any(p.search(text) for p in pats):
            hits.append(kind)
    return hits


def duplicate_run(segs, i, k=2):
    t = segs[i]["text"].lower()
    if not t:
        return False
    run, j = 1, i - 1
    while j >= 0 and segs[j]["text"].lower() == t:
        run += 1; j -= 1
    j = i + 1
    while j < len(segs) and segs[j]["text"].lower() == t:
        run += 1; j += 1
    return run >= k


def propose(segs, gap_threshold=3.0):
    cands = {}

    def add(pos, sig):
        cands.setdefault(pos, {"pos": pos, "signals": []})
        if sig not in cands[pos]["signals"]:
            cands[pos]["signals"].append(sig)

    if segs:
        add(0, "sentinel:rec-start")
        add(len(segs) - 1, "sentinel:rec-end")
    for s in segs:
        for kind in cue_hits(s["text"]):
            add(s["pos"], f"cue:{kind}")
        g = s["gap_before"]
        if g is not None and g >= gap_threshold:
            add(s["pos"], f"gap:{g:.1f}s")
        if s["is_gap"]:
            add(s["pos"], "struct:unintelligible")
        if duplicate_run(segs, s["pos"]):
            add(s["pos"], "struct:dup")
    return sorted(cands.values(), key=lambda c: c["pos"])


def cluster(cands, max_gap=3):
    zones = []
    for c in cands:
        if zones and c["pos"] - zones[-1]["positions"][-1] <= max_gap:
            zones[-1]["positions"].append(c["pos"])
            zones[-1]["signals"].update(c["signals"])
            zones[-1]["sig_at"][c["pos"]] = c["signals"]
        else:
            zones.append({"positions": [c["pos"]], "signals": set(c["signals"]),
                          "sig_at": {c["pos"]: c["signals"]}})
    return zones


# ============================ worksheet (windows) =============================
def build_window(segs, zone):
    """Positions to show for a zone's window: cluster span +/- CTX, clamped, and
    head+tail-elided if too long. Returns (positions, elided_after_index|None)."""
    lo = max(0, zone["positions"][0] - CTX)
    hi = min(len(segs) - 1, zone["positions"][-1] + CTX)
    span = list(range(lo, hi + 1))
    if len(span) <= MAX_WIN:
        return span, None
    head = span[:HEAD]
    tail = span[-TAIL:]
    return head + tail, len(head)  # elision marker goes after head


def render_window(segs, positions, sig_at, elide_after=None):
    lines = []
    for k, p in enumerate(positions):
        if elide_after is not None and k == elide_after:
            lines.append(f"   ... [{positions[elide_after] - positions[elide_after-1] - 1} lines omitted] ...")
        s = segs[p]
        sig = sig_at.get(p)
        tag = f"   (signals: {', '.join(sorted(sig))})" if sig else ""
        txt = s["text"] if s["text"] else "[no text]"
        lines.append(f'[{s["id"]}] "{txt}"{tag}')
    return "\n".join(lines)


# ============================ reader (local LLM) ==============================
PASS1_PROMPT = """You are reading a transcript of a parent telling a young child bedtime stories, working through the recording in order, one window at a time. A recording usually holds just ONE story; occasionally a parent tells two or three, with negotiation and milk breaks between them. Each line is tagged with its segment id in [brackets]; some lines were auto-flagged as possible boundaries ("signals").

CURRENT STATE: {state}

Decide what — if anything — happens in THIS window. Be conservative: MOST windows contain no boundary.
- If NO story is open and a new story clearly BEGINS here, return a story_start. Mark the launch-cue line ("start the story", "let's tell a different story", "you help me tell it", "once upon a time"); if there is no cue, mark the first real sentence of the story. (Special case: if this window is the very START of the recording and the parent is ALREADY mid-story — story narration, no preamble, no cue — mark that first narration line as the story_start.)
- If a story IS open, you are almost always STILL INSIDE it — stories are long and full of pauses, milk breaks, repeated words, and characters talking ("That's right, said the baby" is the story, not a new one). A real-life interruption (cleaning a spill, getting milk, fixing a pillow) is STILL the same story if the telling resumes afterward — do NOT end the story for an interruption. Return a story_end ONLY when the open story clearly concludes here: it says "the end", or it gives way for good to settling-down, a sung wind-down ("I see trees of green"), or negotiating which story to tell next. A different new story may begin right after such an end.
- Otherwise return no events.

Use ONLY segment ids that appear in [brackets] below. Return JSON only, no other text:
{{"events": [{{"type": "story_start" or "story_end", "segment_id": <id>, "quote": "<the line>"}}]}}
If nothing in this window begins or ends a story, return {{"events": []}}.

Window:
{window}
"""

# PROD: generalized world prompt — the `general_reasoned` winner of the worlds-bench
# (emp/src/bench_worlds.py), adapted to also emit a title. The old prompt named our
# own recordings' worlds as examples ("Thomas & Friends", "Mahabharata") and forbade
# outside knowledge, which both failed to generalize and made the model invent worlds
# on made-up stories. This version: two-step reasoning, judge garbled names by sound,
# use real world knowledge, and abstain (empty world) when it isn't a known world.
PASS2_PROMPT = """You are reading the full transcript of ONE bedtime story a parent told a young child, out loud — so it rambles and proper names may be mis-transcribed (a name may be misspelled or split into pieces).

Give it a short descriptive title, and decide what fictional WORLD it is set in. Think in two quick steps:
1) Note the distinctive character names, place names, and signature words. Names may be garbled — judge by what they sound like.
2) Do those match a REAL, widely-known story world (a book, film, show, myth, or franchise)? If yes, name that world. If it is an original, made-up story from no known world, leave the world empty. When unsure, leave it empty.

Return JSON only, no other text:
{{"title": "<short title>", "world": "<the world, or empty if made up>"}}

Story lines:
{lines}
"""

PASS_MERGE_PROMPT = """A first pass split ONE bedtime recording into the candidate story-segments below, labelled by LETTER, in order. Group the segments that belong to the SAME story; keep different stories in separate groups. (Every segment is part of some story — your only job is grouping.)

- SAME story (one group): they share the same characters and plot — a later segment just CONTINUES the story after an interruption (a milk break, a spill, cleaning up, a tangent). A collaborative back-and-forth where the parent and child build ONE story together ("you help me tell it", "what happens?", "yeah, then what?") is still that same one story, not a new one.
- DIFFERENT stories (separate groups): the characters or the world clearly change (different toys, people, or settings), or someone announces a switch ("let's tell a different story", "no more ___ stories"). When unsure, keep them separate.

Segments:
{segments}

Return JSON only, no other text — the letters for each distinct story, in order:
{{"stories": [["A", "B"], ["C"]]}}
Every letter from A to {last_letter} must appear in exactly one group.
"""

PASS_REFINE_PROMPT = """A bedtime story was detected as STARTING at this line:
  [{start_id}] "{start_quote}"

Here are the lines LEADING UP TO it (earliest first; the detected start line is last):
{lines}

Does the SAME story actually begin EARLIER than that line? Judge only whether the earlier lines are ALREADY part of THIS story — its opening question, its characters or setting being introduced, its first events, the question the story then answers. Some stories open as a question-and-answer ("why did they want to be king?" -> the parent explaining) rather than "once upon a time"; that opening question and its answer ARE the story.

The earlier lines are NOT the story if they are: chit-chat, deciding WHICH story to tell, repeated filler, or talk about something else (names, the day, getting settled). When unsure, keep the detected start.

Return JSON only: {{"start_id": <segment id of the true first line of this story>}}
Return the SAME id you were given if it is already the right start; only move earlier to a line that is clearly already this story.
"""


def make_reader(model_id=MODEL_ID):
    """Load Gemma-4 E4B once; return generate(prompt, max_tokens) -> raw string."""
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor = load(model_id)

    def gen(prompt_text, max_tokens=320):
        p = apply_chat_template(processor, model.config, prompt_text)
        res = generate(model, processor, p, max_tokens=max_tokens, temperature=0.0, verbose=False)
        return (getattr(res, "text", res) or "").strip()
    return gen


def extract_json(raw):
    raw = _THINK_RE.sub("", raw or "")
    raw = raw.replace("```json", "").replace("```", "")
    i, j = raw.find("{"), raw.rfind("}")
    if i == -1 or j == -1 or j < i:
        return None
    frag = raw[i:j + 1]
    for attempt in (frag, re.sub(r",\s*([}\]])", r"\1", frag)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    return None


def coerce_id(sid, valid_ids):
    """Map a model-returned segment_id to a real id. The 4B model formats this
    field inconsistently: 0, "0", [0], "[0]", "gap_18.425", "[gap_18.425]"."""
    if isinstance(sid, list):
        sid = sid[0] if sid else None
    if isinstance(sid, bool) or sid is None:
        return None
    if isinstance(sid, int):
        return sid if sid in valid_ids else None
    if isinstance(sid, str):
        s = sid.strip().strip("[]").strip()
        if s in valid_ids:                      # exact (incl. gap_* string ids)
            return s
        if s.lstrip("-").isdigit() and int(s) in valid_ids:
            return int(s)
    return None


def pass1_events(gen, window_text, state_line, valid_ids, raw_log):
    raw = gen(PASS1_PROMPT.format(state=state_line, window=window_text), max_tokens=320)
    raw_log.append({"pass": 1, "raw": raw})
    obj = extract_json(raw)
    events = []
    if isinstance(obj, dict) and isinstance(obj.get("events"), list):
        for e in obj["events"]:
            if not isinstance(e, dict):
                continue
            sid = coerce_id(e.get("segment_id"), valid_ids)
            if e.get("type") in ("story_start", "story_end") and sid is not None:
                events.append({"type": e["type"], "segment_id": sid, "quote": e.get("quote", "")})
    return events


def pass2_name(gen, lines_text, raw_log):
    raw = gen(PASS2_PROMPT.format(lines=lines_text), max_tokens=160)
    raw_log.append({"pass": 2, "raw": raw})
    obj = extract_json(raw)
    if isinstance(obj, dict):
        return str(obj.get("title", "")).strip(), str(obj.get("world", "")).strip()
    return "", ""


# ============================ stateful walk (code) ============================
def _state_line(open_story):
    if open_story is None:
        return "No story is open. The next story has not started yet (or you are in the preamble)."
    return (f'A story is currently being told; it started at line {open_story["segment_id"]}: '
            f'"{open_story.get("quote", "")}". You are most likely still inside it.')


def _mkregion(open_story, end_id, end_quote, pos_of):
    sp, ep = pos_of[open_story["segment_id"]], pos_of[end_id]
    if ep < sp:  # a backwards end collapses to a one-segment region; keep id and pos in sync
        end_id, ep = open_story["segment_id"], sp
    return {"start_id": open_story["segment_id"], "end_id": end_id,
            "start_pos": sp, "end_pos": ep,
            "start_quote": open_story.get("quote", ""), "end_quote": end_quote}


# ============================ orchestration ===================================
def full_region_lines(segs, region):
    """Every non-empty line of a region, in order — the whole story, not a
    head/middle/tail sample. PROD: pass 2 (world naming) reads the full story, because a
    thinned sample routinely drops the one canonical name that identifies the world (the
    held-out Mahabharata miss). Validated on emp/results/worlds-bench: with the full
    story the model recovers that world; a 12-line sample loses it. Only world-naming
    uses this; boundary-finding keeps its own head/tail windows."""
    sp, ep = region["start_pos"], region["end_pos"]
    return "\n".join(f'[{segs[p]["id"]}] "{segs[p]["text"]}"'
                     for p in range(sp, ep + 1) if segs[p]["text"])


def _render_regions_for_merge(segs, regions, head=4, tail=3):
    """Compact head+tail view of each predicted region, labelled by LETTER (so the
    label can't collide with the numeric segment ids shown in the lines)."""
    out = []
    for i, r in enumerate(regions):
        body = [p for p in range(r["start_pos"], r["end_pos"] + 1) if segs[p]["text"]]
        picks = body if len(body) <= head + tail else body[:head] + body[-tail:]
        lines = "\n".join(f'     "{segs[p]["text"]}"' for p in picks)
        out.append(f'Segment {chr(65 + i)} (ids {r["start_id"]}–{r["end_id"]}):\n{lines}')
    return "\n".join(out)


def _winddown_fraction(segs, region):
    """Share of a region's non-empty lines that are the sung wind-down lullaby."""
    body = [p for p in range(region["start_pos"], region["end_pos"] + 1) if segs[p]["text"]]
    if not body:
        return 0.0
    hits = sum(1 for p in body if any(pat.search(segs[p]["text"]) for pat in COMPILED["WINDDOWN"]))
    return hits / len(body)


def global_consolidate(gen, segs, regions, raw_log, winddown_thresh=0.34):
    """Second, GLOBAL pass — the lookahead the left-to-right walk lacks. Two safe steps:
      (1) CODE drops a region that is mostly the sung wind-down lullaby (deterministic;
          never touches story content — a real story has no 'I see trees of green' run).
      (2) the MODEL does MERGE-ONLY over the rest: it groups interruption-split halves
          into one story but can NEVER delete a story (that protects the genuinely
          distinct multi-story case). Degrades safely — an omitted/odd group stays its
          own story. Only ever reduces the count."""
    # (1) deterministic wind-down drop — but never drop the last region (a session keeps >=1 story)
    kept = [r for r in regions if _winddown_fraction(segs, r) < winddown_thresh]
    if not kept:
        kept = regions
    n = len(kept)
    if n <= 1:
        return kept

    # (2) model merge-only (letters A.. avoid colliding with numeric segment ids)
    raw = gen(PASS_MERGE_PROMPT.format(segments=_render_regions_for_merge(segs, kept),
                                       last_letter=chr(64 + n)), max_tokens=200)
    raw_log.append({"pass": "merge", "raw": raw})
    obj = extract_json(raw)

    def _letters(seq):
        out = []
        for x in (seq or []):
            s = str(x).strip().strip("[]").strip().upper()
            if len(s) == 1 and "A" <= s <= chr(64 + n):
                out.append(ord(s) - 65)
        return out

    label = {}
    if isinstance(obj, dict) and isinstance(obj.get("stories"), list):
        for gid, g in enumerate(obj["stories"]):
            if isinstance(g, list):
                for i in _letters(g):
                    label.setdefault(i, ("g", gid))
    for i in range(n):
        label.setdefault(i, ("solo", i))  # never silently lose a region

    # Walk in order; merge only CONSECUTIVE same-label regions (keeps merged spans
    # ordered and non-overlapping even if the model grouped non-contiguously).
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


def refine_starts(gen, segs, regions, raw_log, max_lookback=40):
    """Pull each story's START earlier when the lines before it are already the same
    story (the lookahead the left-to-right walk lacks). Only ever moves a start EARLIER,
    bounded below by the previous story's end — so it can't create an over-split or eat a
    neighbour. Targets the under-capture case: a dialogic opening (a question the parent
    answers) the walk skipped because it was primed for narration or a launch cue."""
    valid_ids = {s["id"] for s in segs}
    pos_of = {s["id"]: s["pos"] for s in segs}
    for i, r in enumerate(regions):
        lower = 0 if i == 0 else regions[i - 1]["end_pos"] + 1
        cur = r["start_pos"]
        if cur <= lower:
            continue  # no earlier room
        win_lo = max(lower, cur - max_lookback)
        lines = "\n".join(f'[{segs[p]["id"]}] "{segs[p]["text"]}"'
                          for p in range(win_lo, cur + 1) if segs[p]["text"])
        try:
            raw = gen(PASS_REFINE_PROMPT.format(start_id=r["start_id"],
                                                start_quote=r.get("start_quote", ""),
                                                lines=lines), max_tokens=64)
        except Exception as ex:
            raw_log.append({"pass": "refine", "error": repr(ex)[:200]})
            continue
        raw_log.append({"pass": "refine", "raw": raw})
        obj = extract_json(raw)
        new_id = coerce_id(obj.get("start_id"), valid_ids) if isinstance(obj, dict) else None
        if new_id is not None and new_id in pos_of and win_lo <= pos_of[new_id] < cur:
            np = pos_of[new_id]
            r["start_id"], r["start_pos"], r["start_quote"] = new_id, np, segs[np]["text"]
    return regions


def segment_session(session_dir, gen, show_only=False, use_global=True, use_refine=False):
    """Disk-based entry (the detector's fallback path): read the session's transcript
    from disk, then run the segmentation walk. Returns (result_dict, raw_log)."""
    segs = load_segments(session_dir)
    return segment_segments(segs, gen, name=Path(session_dir).name, show_only=show_only,
                            use_global=use_global, use_refine=use_refine)


def segment_segments(segs, gen, name="session", show_only=False, use_global=True, use_refine=False):
    # `segs` is the pre-built meta-list (from load_segments or load_segments_from_list);
    # `name` only labels the stderr log line.
    order = [s["id"] for s in segs]
    pos_of = {x: i for i, x in enumerate(order)}
    zones = cluster(propose(segs))
    valid_ids = set(order)
    raw_log = []

    if show_only:
        print(f"\n===== {name} — {len(zones)} zones =====", file=sys.stderr)
        for z in zones:
            positions, elide = build_window(segs, z)
            print(f"\n--- zone pos {z['positions'][0]}-{z['positions'][-1]}  [{', '.join(sorted(z['signals']))}] ---", file=sys.stderr)
            print(render_window(segs, positions, z["sig_at"], elide), file=sys.stderr)
        return None

    # Stateful sweep: walk zones in order, carrying whether a story is open. The
    # running state is what stops the model re-opening a story it's still inside.
    regions, open_story, n_events = [], None, 0
    for z in zones:
        positions, elide = build_window(segs, z)
        window_text = render_window(segs, positions, z["sig_at"], elide)
        try:
            evs = pass1_events(gen, window_text, _state_line(open_story), valid_ids, raw_log)
        except Exception as ex:  # one bad zone must not kill the session
            raw_log.append({"pass": 1, "error": repr(ex)[:200], "zone": z["positions"]})
            evs = []
        n_events += len(evs)
        # sort by position; at an equal position an end sorts before a start, so a same-line
        # close-then-open pairs in the right order
        for e in sorted(evs, key=lambda e: (pos_of[e["segment_id"]], 0 if e["type"] == "story_end" else 1)):
            ep = pos_of[e["segment_id"]]
            if e["type"] == "story_start":
                if open_story is None:
                    open_story = e
                elif ep - 1 >= pos_of[open_story["segment_id"]]:  # room: close the open story just before it
                    regions.append(_mkregion(open_story, order[ep - 1], "<next story starts>", pos_of))
                    open_story = e
                else:  # new start at/before the open start: spurious -> keep the open story, never lose it
                    raw_log.append({"pass": 1, "note": "skipped backward story_start",
                                    "skipped_id": e["segment_id"], "kept_open_id": open_story["segment_id"]})
            else:  # story_end
                if open_story is None:  # end with nothing open -> recording opened mid-story
                    sp = min((regions[-1]["end_pos"] + 1) if regions else 0, ep)
                    open_story = {"segment_id": order[sp], "quote": "<recording opened mid-story>"}
                regions.append(_mkregion(open_story, e["segment_id"], e.get("quote", ""), pos_of))
                open_story = None
    if open_story is not None:  # dangling open story -> runs to end of recording
        regions.append(_mkregion(open_story, order[-1], "<runs to end of recording>", pos_of))

    n_pre = len(regions)
    if use_global:  # global lookahead pass: merge interrupted halves, drop non-stories
        regions = global_consolidate(gen, segs, regions, raw_log)
    if use_refine:  # pull each story's start earlier when the lead-in is already the story
        regions = refine_starts(gen, segs, regions, raw_log)

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
    # PROD: diagnostics to stderr so the worker's stdout stays clean JSON.
    print(f"  {name:12s}: {len(zones):2d} zones -> {n_events:2d} events -> {merge_note} stories  "
          + " | ".join(f'{s["start_id"]}-{s["end_id"]} [{s["world"]}]' for s in stories),
          file=sys.stderr)
    return {"name": name, "n_zones": len(zones), "n_events": n_events,
            "n_regions_pre_merge": n_pre, "stories": stories}, raw_log


# ============================ in-memory caller (pipeline) ======================
def _segment_in_subprocess(transcript):
    """Module-level (picklable for the spawned subprocess). Segment an IN-MEMORY
    transcript and return its stories list. Loads Gemma once; the process exits after,
    freeing the GPU memory."""
    segs = load_segments_from_list(transcript["segments"])
    gen = make_reader()
    result, _ = segment_segments(segs, gen)
    return result["stories"]


def segment_transcript(transcript, timeout=1800):
    """Segment the given in-memory transcript via a fresh subprocess (model_runner),
    returning the stories list [{start_id, end_id, title, world, evidence}]."""
    from model_runner import run_model
    return run_model(_segment_in_subprocess, transcript, timeout=timeout)


# Cache config fingerprint: a hash of everything that drives the segmenter's output, so
# editing any prompt / cue / window constant invalidates cached stories on the next
# re-enrich automatically (no manual version bump to forget) — mirroring how the
# normalizer cache hashes its prompt. The few numeric thresholds that live as function
# defaults (gap_threshold, max_gap, winddown_thresh) are NOT folded in; if you change
# one, bump the literal tag below.
SEGMENT_CONFIG_VERSION = MODEL_ID + "|" + hashlib.sha256("".join([
    "v1", PASS1_PROMPT, PASS2_PROMPT, PASS_MERGE_PROMPT, PASS_REFINE_PROMPT,
    repr(sorted(CUES.items())), repr((CTX, MAX_WIN, HEAD, TAIL)),
]).encode()).hexdigest()[:16]
