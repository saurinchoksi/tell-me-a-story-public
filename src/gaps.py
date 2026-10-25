"""Canonical missed-speech gap detection — pure interval logic + coverage rules.

Extracted from emp/src/gap_analysis.py so there is exactly ONE
implementation of the missed-speech count: the EMP sweep CLI
(emp/src/gap_analysis.py) and the API endpoint
(GET /api/sessions/<id>/missed-speech) both import from here. The numbers the
validator shows must never diverge from the published EMP floor.

What a gap means: the speaker detector (diarization) heard a voice but the
transcript has NO segment of any kind for that stretch — no transcribed line,
and not even an [unintelligible] placeholder. Coverage is computed at SEGMENT
granularity and includes BOTH transcribed-word segments AND pipeline-injected
"[unintelligible]" placeholders (_source == "diarization_gap"): a placeholder
is a visible, reviewable trace (a clickable line in the validator), so a region
the pipeline already flagged that way is NOT a miss here. Sorting those
placeholder segments into real-speech-vs-not is a separate job.

Pure data transforms over already-computed JSON — no ML, no file I/O, and
nothing here ever modifies a transcript.
"""


# --- pure interval helpers ---------------------------------------------------
def merge_intervals(intervals):
    """Merge a list of (start, end) into sorted, non-overlapping spans."""
    spans = sorted((s, e) for s, e in intervals if e > s)
    if not spans:
        return []
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def subtract_intervals(seg, covered):
    """Return the parts of seg=(start, end) not inside any covered span.

    covered must be a merged, sorted list (output of merge_intervals).
    """
    ds, de = seg
    out = []
    cur = ds
    for cs, ce in covered:
        if ce <= cur:
            continue
        if cs >= de:
            break
        if cs > cur:
            out.append((cur, min(cs, de)))
        cur = max(cur, ce)
        if cur >= de:
            break
    if cur < de:
        out.append((cur, de))
    return out


# --- coverage rules ----------------------------------------------------------
def real_segments(transcript):
    """Segments with actual transcribed words (excludes [unintelligible]).

    Used for the readable before/after context and the green transcript strip —
    NOT for coverage. See covering_segments for the coverage rule.
    """
    return [s for s in transcript["segments"]
            if s.get("_source") != "diarization_gap" and s["end"] > s["start"]]


def unintelligible_segments(transcript):
    """Pipeline-injected [unintelligible] placeholder segments."""
    return [s for s in transcript["segments"]
            if s.get("_source") == "diarization_gap" and s["end"] > s["start"]]


def covering_segments(transcript):
    """Every segment that means 'a trace exists here' — both transcribed-word
    segments AND [unintelligible] placeholders.

    This is the coverage rule. A missed-speech gap is speech with NO
    segment at all: not a transcribed line, and not even an [unintelligible]
    marker. The reason [unintelligible] counts as coverage: it is a *visible,
    reviewable* trace — a clickable line in the validator that says "someone
    spoke here, couldn't decode it." This is specifically the failure that is
    invisible to review, so a region the pipeline already flagged
    [unintelligible] is NOT a miss here. Deciding whether those [unintelligible]
    segments hold real speech (vs noise) is a separate job.
    """
    return [s for s in transcript["segments"] if s["end"] > s["start"]]


# --- gap detection -----------------------------------------------------------
# The published EMP high-confidence floor (seconds). The ONE home for this
# threshold: the API endpoint's default and the sweep's DEFAULT_HC both import
# it, so recalibrating the floor moves every surface at once.
HIGH_CONFIDENCE_MIN_GAP = 1.0


def find_gaps(diarization, transcript, min_gap):
    """Find missed-speech gaps: where the detector heard a voice but NO segment exists.

    Coverage = every segment (transcribed words AND [unintelligible] markers),
    per covering_segments. A gap is an uncovered sliver of a diarization
    segment, >= min_gap seconds, tagged with that segment's speaker. So a gap
    means speech with no trace at all — not a transcribed line, not even an
    [unintelligible] placeholder; that is exactly the no-trace missed speech we want.

    Faithful to the original sweep in every other respect: per-diarization-
    segment subtraction, so a rare same-instant overlap from two speakers would
    be listed twice.
    """
    covered = merge_intervals(
        [(s["start"], s["end"]) for s in covering_segments(transcript)]
    )
    segs_by_start = sorted(real_segments(transcript), key=lambda s: s["start"])

    gaps = []
    for d in diarization["segments"]:
        ds, de, spk = d["start"], d["end"], d["speaker"]
        if de - ds <= 0:
            continue
        for gs, ge in subtract_intervals((ds, de), covered):
            if ge - gs >= min_gap:
                gaps.append({"start": gs, "end": ge, "dur": ge - gs,
                             "speaker": spk})
    gaps.sort(key=lambda g: g["start"])

    # Attach nearest preceding / following transcript segment for the table.
    for g in gaps:
        prev_seg = None
        next_seg = None
        for s in segs_by_start:
            if s["start"] < g["start"]:
                prev_seg = s
            elif s["start"] >= g["end"] and next_seg is None:
                next_seg = s
                break
        g["prev"] = prev_seg
        g["next"] = next_seg
    return gaps
