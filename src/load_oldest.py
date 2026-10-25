#!/usr/bin/env python3
"""Process inbox audio oldest-first, up to a limit — a resumable wrapper (#16).

`process_inbox.py` inits ALL inbox files, then pipelines each. A run killed partway
therefore leaves sessions that were initialized but never pipelined, and a re-run
skips them (their folder already exists) — so the stock script does NOT cleanly
resume, and it processes in filename order, not by recording date.

This wrapper drives `process_inbox` ONE file at a time in recording-date order
(reusing `init_session.get_creation_time`, which reads the Voice Memo `creation_time`
tag). Each session is therefore init+pipeline *atomically*: stop with Ctrl-C whenever
the scorecard says you've seen enough, and a re-run simply continues — already-done
sessions are skipped by init's own dedup (content hash + Voice Memo UUID).

    python src/load_oldest.py --dry-run     # show the oldest-first order, process nothing
    python src/load_oldest.py --limit 1     # prove one session end to end
    python src/load_oldest.py --limit 20    # first wave
    python src/load_oldest.py               # everything, oldest first

Nothing here is modified in the proven per-session path — this only sequences it.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from init_session import INBOX_DIR, SESSIONS_DIR, SUPPORTED_FORMATS, get_creation_time


def oldest_first(inbox=INBOX_DIR):
    """Return [(recording_datetime, path), ...] for inbox audio, oldest first."""
    files = [f for f in inbox.iterdir()
             if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS]
    dated = [(get_creation_time(f), f) for f in files]
    dated.sort(key=lambda t: t[0])
    return dated


def _session_ids():
    if not SESSIONS_DIR.exists():
        return set()
    return {d.name for d in SESSIONS_DIR.iterdir() if d.is_dir()}


def main():
    ap = argparse.ArgumentParser(
        description="Process inbox audio oldest-first, resumable (#16)."
    )
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many NEW sessions are created (default: all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the oldest-first processing order and exit")
    ap.add_argument("--skip-monitor", action="store_true",
                    help="skip the deprioritized Monitor detector pass (#13/#16) — "
                         "saves a second model load per session; calibration doesn't need it")
    ap.add_argument("--skip-name-fixers", action="store_true",
                    help="defer the name fixers (namefix/familyfix/spellingfix) to a later "
                         "fix_session batch — skips the expensive per-name re-listens so the "
                         "churn finishes transcripts fast (transcripts-first, #26)")
    args = ap.parse_args()

    dated = oldest_first()
    if not dated:
        print("Inbox is empty — nothing to do.")
        return

    print(f"{len(dated)} file(s) in inbox, oldest recording first:")
    for i, (dt, f) in enumerate(dated, 1):
        print(f"  {i:>3}. {dt:%Y-%m-%d %H:%M}  {f.name}")

    if args.dry_run:
        print("\n(dry run — nothing processed)")
        return

    # FINISH-AND-FREE (#16): each session runs in its OWN short-lived subprocess
    # (process_inbox.py --file), so the OS reclaims all of its model/GPU memory the moment
    # it exits. This Mac only reliably frees GPU memory on process exit, so the earlier
    # single long-lived loop accumulated it session after session until the machine swapped
    # and every session slowed down. Why wrap the WHOLE session, not per-stage: the lighter
    # stages (Qwen segmentation, the re-listens) already finish-and-free via
    # model_runner.run_model, but the two HEAVIEST — Whisper (transcribe.py) and pyannote
    # (diarize.py) — load in-process in run_pipeline and are NOT wrapped, so their memory is
    # exactly what piled up. Wrapping the session frees all of it (both models + heap) in one
    # exit — a catch-all. (Deeper fix: wrap transcribe/diarize in run_model too, helping every
    # run_pipeline caller; done here at the session grain for now, lower-risk.) We
    # use subprocess.run of the CLI, not run_model, because run_model marshals a picklable
    # RETURN value via a Queue — here we run process_inbox for its side effects and want its
    # live stdout to keep flowing into the churn log. The parent stays light (no mlx import)
    # and never holds a model, so --dry-run needs only ffprobe and a killed run can strand at
    # most one half-done session (recoverable), never leak memory.
    process_inbox_py = str(Path(__file__).parent / "process_inbox.py")

    scope = "all of them" if args.limit is None else f"the oldest {args.limit} new session(s)"
    mon = "Monitor pass OFF" if args.skip_monitor else "Monitor pass on"
    nf = "name-fixers DEFERRED (transcripts-first)" if args.skip_name_fixers else "name-fixers on"
    print(f"\nProcessing {scope}, oldest first. ({mon}; {nf}) One subprocess per session (finish-and-free).")
    print("Safe to stop (Ctrl-C) and re-run later — it resumes where it left off.\n")

    made = 0
    for _dt, f in dated:
        if args.limit is not None and made >= args.limit:
            break
        before = _session_ids()
        cmd = [sys.executable, process_inbox_py, "--file", f.name]
        if args.skip_monitor:
            cmd.append("--skip-monitor")
        if args.skip_name_fixers:
            cmd.append("--skip-name-fixers")
        # Inherit stdout/stderr so the child's progress still flows to the churn log.
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"  ✗ {f.name}: session subprocess exited with {proc.returncode} "
                  f"(e.g. OOM-killed or crashed) — see its output above.")
        # Count only sessions that actually produced a transcript. init MOVES the audio into
        # the session folder before the pipeline runs (init_session, shutil.move), so a new
        # folder WITHOUT a transcript means the pipeline failed and the audio is now stranded
        # there — the inbox no longer has it. Warn loudly, don't count it toward --limit, and
        # never delete the folder (that destroys the only copy). Move its audio back to inbox/.
        for sid in sorted(_session_ids() - before):
            if (SESSIONS_DIR / sid / "transcript-rich.json").exists():
                made += 1
            else:
                print(f"  ⚠ {sid}: pipeline failed — audio is stranded in the session folder "
                      f"(no transcript). Move it back to inbox/ to retry; do NOT delete the "
                      f"folder (init moved the only copy there).")
        if made:
            print(f"\n  → {made} completed session(s) so far.")

    print(f"\n{'─' * 50}\nDone. {made} new session(s) created this run.")
    if args.limit is not None and made >= args.limit:
        print("Reached the limit — re-run (same command) to process the next wave.")


if __name__ == "__main__":
    main()
