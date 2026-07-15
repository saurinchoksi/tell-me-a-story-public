#!/usr/bin/env python3
"""Surgical, in-place word realignment for EXISTING sessions (TMAS-54).

Applies forced-alignment realignment (src/realign.py) to a session's
transcript-rich.json WITHOUT re-running the pipeline. A full --re-enrich is the
dangerous path on coded sessions (it re-runs LLM normalization, and the
post-2026-06-16 text-heal would rebuild `text` from the already-pruned `words`,
erasing the very M11 evidence we rescue from). This tool only rewrites per-word
timestamps + rescues dropped words, preserving every segment id/text/count/order.

Safety:
  - backs up transcript-rich.json (+ axial-labels.json / validation-notes.json)
    to *.pre-realign-<UTC> before writing,
  - asserts the segment id list + per-id text are byte-identical before/after
    (this is the single check that protects every EMP axial label),
  - atomic write (tmp + os.replace) — the file ends fully updated or untouched.

By default it targets transcript-rich.json. Pass --target raw to instead realign
transcript-raw.json in place — the one-time backfill that bakes corrected timings
(and rescued M11 words) into raw so it agrees with rich. This matters because word
realignment now runs BEFORE the raw snapshot in run_pipeline (it's transcript repair,
not enrichment), so a --re-enrich no longer realigns; an un-backfilled raw with pruned
words would let the normalization text-heal rebuild `text` from the incomplete words[],
erasing the rescued words. The (id, text) signature guard makes the backfill safe on
coded sessions too (realign never touches text).

Dry-run by default (shows stats + the safety check); pass --write to commit.

    venv/bin/python src/realign_session.py 20251207-195607            # preview (rich)
    venv/bin/python src/realign_session.py 20251207-195607 --write    # commit (rich)
    venv/bin/python src/realign_session.py --write                    # all sessions (rich)
    venv/bin/python src/realign_session.py --target raw               # preview raw backfill
    venv/bin/python src/realign_session.py --target raw --write       # commit raw backfill (all)

After a --write, re-run the regenerable derivatives (they are deliberately not
synced and safe to regenerate against the new word timings):
    venv/bin/python src/detect.py
    venv/bin/python src/resegment_all.py <session-id>
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realign import load_aligner, realign_transcript          # noqa: E402
from mlx_whisper.audio import load_audio                        # noqa: E402

SESSIONS = ROOT / "sessions"
BACKUP_SIDECARS = ("axial-labels.json", "validation-notes.json")


def discover(args, filename):
    if args:
        return args
    return sorted((p.parent.name for p in SESSIONS.glob(f"*/{filename}")),
                  reverse=True)


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def id_signature(transcript):
    """The invariant that protects axial labels: (id, text) per segment, in order."""
    return [(s.get("id"), s.get("text")) for s in transcript.get("segments", [])]


def process(sid, bundle, write, tag, filename, with_sidecars):
    sdir = SESSIONS / sid
    path = sdir / filename
    if not path.exists():
        print(f"  {sid}: no {filename} — skipped")
        return None
    audio_files = sorted(sdir.glob("audio.*"))
    if not audio_files:
        print(f"  {sid}: no audio file — skipped")
        return None

    transcript = json.loads(path.read_text())
    sig_before = id_signature(transcript)
    audio = np.array(load_audio(str(audio_files[0]))).astype(np.float32)

    new_t, stats = realign_transcript(transcript, audio, bundle)
    sig_after = id_signature(new_t)

    # The one check that protects every axial label.
    if sig_after != sig_before:
        print(f"  {sid}: ABORT — segment id/text signature changed "
              f"({len(sig_before)}→{len(sig_after)} segs). File NOT written.")
        return {"sid": sid, "aborted": True, **stats}

    print(f"  {sid}: realigned={stats['realigned']} rescued={stats['rescued']} "
          f"(+{stats['rescued_words']} words) guarded={stats['guarded']} "
          f"skipped={stats['skipped']} / {stats['segments']} segs  [ids preserved ✓]")

    if write:
        backup_names = [filename] + (list(BACKUP_SIDECARS) if with_sidecars else [])
        for name in backup_names:
            src = sdir / name
            if src.exists():
                shutil.copy2(src, sdir / f"{name}.pre-{tag}-{stamp()}")
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(new_t, indent=2))
        os.replace(tmp, path)
        print(f"        written (backup *.pre-{tag}-*)")
    return {"sid": sid, "aborted": False, **stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="*", help="session ids (default: all)")
    ap.add_argument("--write", action="store_true", help="commit (default: dry-run)")
    ap.add_argument("--target", choices=["rich", "raw"], default="rich",
                    help="which transcript to realign in place (default: rich; "
                         "'raw' is the one-time backfill so raw agrees with rich)")
    ap.add_argument("--tag", default="realign")
    args = ap.parse_args()

    filename = "transcript-raw.json" if args.target == "raw" else "transcript-rich.json"
    with_sidecars = args.target == "rich"

    sids = discover(args.sessions, filename)
    mode = "WRITE" if args.write else "DRY-RUN"
    print(f"[{mode}] realigning {filename} in {len(sids)} session(s)\nloading aligner...")
    bundle = load_aligner()

    results = []
    for sid in sids:
        r = process(sid, bundle, args.write, args.tag, filename, with_sidecars)
        if r:
            results.append(r)

    aborted = [r["sid"] for r in results if r.get("aborted")]
    if aborted:
        print(f"\n{len(aborted)} session(s) ABORTED (id/text changed): {aborted}")
    if not args.write:
        print("\nDry-run only — re-run with --write to commit.")
    else:
        print("\nDone. Re-run derivatives: venv/bin/python src/detect.py  +  "
              "venv/bin/python src/resegment_all.py <id>")


if __name__ == "__main__":
    main()
