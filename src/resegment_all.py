#!/usr/bin/env python3
"""Step-5 rollout (phase 1): re-segment EVERY existing session with the production Qwen3.5
segmenter and persist `_stories`, so the story boundaries + worlds the Monitor and the
session screen show reflect the new model everywhere — not just freshly-ingested sessions.

Phase 2 (re-run the canon detector over the fresh segmentation) is a separate command so one
model is in memory at a time:  python src/detect.py --detector m9c-canon

Idempotent: rewrites only `_stories` (top-level) and the per-segment `_story` index inside
transcript-rich.json; old story tags are stripped first so a re-run is clean. Audio and
transcript-raw.json are never touched. One Qwen3.5 load for the whole loop.

    ./venv/bin/python src/resegment_all.py                       # all sessions with a transcript
    ./venv/bin/python src/resegment_all.py 20260211-210718 ...   # specific sessions
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from story_segment_qwen35 import make_reader, segment_segments
from story_segment import load_segments_from_list
from stories import enrich_with_stories

SESSIONS_DIR = ROOT / "sessions"


def discover(args):
    if args:
        return args
    return sorted((p.parent.name for p in SESSIONS_DIR.glob("*/transcript-rich.json")), reverse=True)


def _strip_old(rich):
    """Drop any previous segmentation so re-enrichment is clean (no stale _story tags on
    segments that fall outside the new story spans)."""
    rich.pop("_stories", None)
    for s in rich.get("segments", []):
        s.pop("_story", None)
    return rich


def main():
    sids = discover(sys.argv[1:])
    gen = make_reader()  # one model load for the whole rollout
    print(f"re-segmenting {len(sids)} session(s) with Qwen3.5 ...\n", flush=True)
    failed = []
    for i, sid in enumerate(sids, 1):
        path = SESSIONS_DIR / sid / "transcript-rich.json"
        if not path.exists():
            print(f"  [{i}/{len(sids)}] {sid}: no transcript-rich.json — skipped", flush=True)
            continue
        try:
            rich = _strip_old(json.loads(path.read_text()))
            t0 = time.monotonic()
            segs = load_segments_from_list(rich["segments"])
            result, _ = segment_segments(segs, gen, name=sid)
            enriched = enrich_with_stories(rich, result["stories"])
            # Atomic write: serialize fully first (so a serialization error never touches the
            # file), write a temp alongside, then rename over the original — the session ends up
            # either fully updated or completely untouched, never half-written.
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(enriched, indent=2))
            os.replace(tmp, path)
            worlds = [s.get("world", "") or "-" for s in result["stories"]]
            print(f"  [{i}/{len(sids)}] {sid}: {len(result['stories'])} story(ies), worlds={worlds}  "
                  f"({time.monotonic() - t0:.0f}s)", flush=True)
        except Exception as ex:  # one bad session must not abort the rollout
            failed.append(sid)
            print(f"  [{i}/{len(sids)}] {sid}: FAILED ({repr(ex)[:200]}) — file left untouched, "
                  f"continuing", flush=True)
    if failed:
        print(f"\n{len(failed)} session(s) failed and were skipped: {failed}", flush=True)
    print("\ndone — now re-run the canon detector:  python src/detect.py --detector m9c-canon", flush=True)


if __name__ == "__main__":
    main()
