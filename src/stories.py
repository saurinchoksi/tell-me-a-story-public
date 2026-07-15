"""Pure data transform: tag a transcript with its story regions.

Stage 0 (story_segment) splits a recording into stories; this folds that result INTO the
transcript so everything downstream (the screen, the name auditor) reads it for free —
the same convention as `_speaker` on words. A short top-level `_stories` list holds each
story's boundary + inferred world/title once; each in-story segment carries a `_story`
index pointing into it. Non-story stretches (preamble, milk breaks, wind-down) are left
untagged. Never edits `text`/`words`; no ML imports.
"""
import copy
import sys


def enrich_with_stories(transcript, stories):
    """Return a deep copy of `transcript` with story metadata attached.

    Adds top-level `_stories = [{index, start_id, end_id, title, world}, ...]` (written
    once) and a per-segment `_story = <index>` for every segment inside a story's
    [start_id, end_id] span. Segments in no story keep no `_story` key. A story whose
    start_id/end_id is absent from the transcript is skipped with a warning. `stories` is
    the segmenter output: [{start_id, end_id, title, world, evidence?}, ...].
    """
    result = copy.deepcopy(transcript)
    segments = result.get("segments", [])
    # id -> position (raw id, int OR "gap_*" string — never coerced)
    pos_of = {seg["id"]: i for i, seg in enumerate(segments)}

    result["_stories"] = [
        {"index": k, "start_id": st.get("start_id"), "end_id": st.get("end_id"),
         "title": st.get("title", ""), "world": st.get("world", "")}
        for k, st in enumerate(stories)
    ]

    for k, st in enumerate(stories):
        sp = pos_of.get(st.get("start_id"))
        ep = pos_of.get(st.get("end_id"))
        if sp is None or ep is None:
            print(f"  WARNING: story {k} ids {st.get('start_id')}-{st.get('end_id')} "
                  f"absent from transcript — not tagged", file=sys.stderr)
            continue
        if ep < sp:
            sp, ep = ep, sp
        for i in range(sp, ep + 1):
            segments[i]["_story"] = k

    return result
