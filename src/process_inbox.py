#!/usr/bin/env python3
"""Process all audio files in the inbox: init session → run pipeline → save artifacts."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from init_session import INBOX_DIR, SESSIONS_DIR, SUPPORTED_FORMATS, init_session
from namefix import SKIP_REASON_COPY
from pipeline import run_pipeline, save_computed
from embeddings import save_embeddings


def process_inbox(target_file: str | None = None, run_detectors: bool = True,
                  run_name_fixers: bool = True):
    if target_file:
        path = INBOX_DIR / target_file
        if not path.exists():
            raise FileNotFoundError(f"File not found in inbox: {target_file}")
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{path.suffix}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )
        audio_files = [path]
    else:
        audio_files = [f for f in INBOX_DIR.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS]

    if not audio_files:
        print("Inbox is empty — nothing to process.")
        return

    print(f"Found {len(audio_files)} file(s) in inbox.\n")

    initialized = []
    skipped = []
    failed = []
    detector_failed = []
    family_not_run = []   # (session_id, reason) — the family side silently not running
                          # is the discovery-gap hole; the summary says it loudly (TMAS-72)

    # Phase 1: init all sessions
    for audio_path in sorted(audio_files):
        print(f"Initializing: {audio_path.name}")
        try:
            result = init_session(audio_path)
            if result is None:
                skipped.append(audio_path.name)
            else:
                initialized.append(result)
        except Exception as e:
            print(f"  ✗ Init failed: {e}")
            failed.append((audio_path.name, str(e)))

    if not initialized:
        print("\nNo new sessions to process.")
        _print_summary([], skipped, failed)
        return

    # Phase 2: run pipeline on each initialized session
    created = []
    for session_id, audio_filename in initialized:
        audio_path = SESSIONS_DIR / session_id / audio_filename
        print(f"\nRunning pipeline: {session_id}")
        try:
            result = run_pipeline(str(audio_path), run_name_fixers=run_name_fixers)
            save_computed(
                str(SESSIONS_DIR / session_id),
                result["transcript_raw"],
                result["transcript"],
                result["diarization"],
            )
            if result.get("embeddings") is not None:
                save_embeddings(
                    result["embeddings"],
                    str(SESSIONS_DIR / session_id / "embeddings.json"),
                )

            # Measure file sizes and attach to _stats
            session_path = str(SESSIONS_DIR / session_id)
            file_sizes = {}
            for name in ["transcript-raw.json", "transcript-rich.json", "diarization.json"]:
                file_sizes[name] = os.path.getsize(os.path.join(session_path, name))
            if result.get("embeddings") is not None:
                file_sizes["embeddings.json"] = os.path.getsize(
                    os.path.join(session_path, "embeddings.json")
                )
            result["transcript"]["_stats"]["file_sizes"] = file_sizes
            with open(os.path.join(session_path, "transcript-rich.json"), "w") as f:
                json.dump(result["transcript"], f, indent=2)

            if result.get("embeddings") is not None:
                # Auto-identify against existing profiles
                from profiles import load_profiles
                from identify import identify_speakers, save_identifications
                profiles = load_profiles()
                if profiles.get("profiles"):
                    identifications = identify_speakers(
                        str(SESSIONS_DIR / session_id / "embeddings.json")
                    )
                    save_identifications(
                        identifications,
                        str(SESSIONS_DIR / session_id / "identifications.json"),
                    )

            # Run the full monitor pass on the new session: the code detectors, the
            # M9b LLM judge, AND the offline per-story name auditor (run_offline=True).
            # Separate try: a detector problem must not report the session as a
            # pipeline failure — the transcription artifacts are already saved and
            # usable. The judge falls back to code-only if mlx-vlm isn't installed; the
            # name auditor (also mlx-vlm) adds minutes + a second model load per session.
            # run_detectors=False skips the whole Monitor pass: it's deprioritized
            # (TMAS-64), doesn't feed the fixer questions, and a bulk calibration load
            # drops it to save that second model load per session (load_oldest
            # --skip-monitor, TMAS-80).
            if run_detectors:
                try:
                    from detectors import DETECTORS
                    from detectors.base import scan_session
                    from detectors.name_consistency_judge import make_judge
                    try:
                        judge = make_judge()
                    except FileNotFoundError:
                        judge = None
                    scan_session(SESSIONS_DIR / session_id, DETECTORS, force=True,
                                 judge=judge, run_offline=True)
                except Exception as e:
                    print(f"  ⚠ Detectors failed (transcript saved): {e}")
                    detector_failed.append((session_id, str(e)))

            # Say what the fixer stage did AND didn't do (TMAS-72) — before this,
            # the fixers ran as ~30 anonymous re-listen progress bars with no ending.
            # Fenced like the detector pass: a reporting crash must not reclassify a
            # fully-saved session as a pipeline failure.
            try:
                family_skip = _print_fixer_summary(result["transcript"]["_processing"])
                # "deferred" is by-design (transcripts-first churn, TMAS-95), not a real
                # gap — keep it out of the loud "Family names NOT checked" summary so a
                # genuine no_roster/error in the same batch still stands out (TMAS-72).
                if family_skip is not None and family_skip != "deferred":
                    family_not_run.append((session_id, family_skip))
            except Exception as e:
                print(f"  ⚠ Fixer summary failed (session saved): {e}")

            created.append(session_id)
            print(f"  ✓ Done — {session_id}")
        except Exception as e:
            print(f"  ✗ Pipeline failed: {e}")
            failed.append((session_id, str(e)))

    _print_summary(created, skipped, failed, detector_failed, family_not_run)


# stage name in _processing -> the fixer's spoken name
_FIXER_STAGES = [
    ("namefix", "universe-name-fix"),
    ("family_namefix", "family-name-fix"),
    ("spellingfix", "invented-spelling-fix"),
]


def _print_fixer_summary(processing):
    """The fix_session-style ending, lifted here (TMAS-72): one line per shadow fixer —
    ran (with question count), NOT RUN (with the reason), or FAILED. Returns the
    family-name-fix skip reason when it didn't run (None otherwise) so the run summary
    can say it loudly."""
    family_skip = None
    print("  Name fixers (shadow — no transcript touched):")
    for stage, name in _FIXER_STAGES:
        entry = next((p for p in processing if p.get("stage") == stage), None)
        if entry is None:
            continue
        if entry["status"] == "success":
            n = entry["groups"]
            cached = " (cached)" if entry.get("from_cache") else ""
            print(f"    {name}: {n} labeled question{'' if n == 1 else 's'}{cached}")
        elif entry["status"] == "skipped":
            reason = entry["reason"]
            print(f"    {name}: NOT RUN — {SKIP_REASON_COPY.get(reason, reason)}")
            if stage == "family_namefix":
                family_skip = reason
        else:
            print(f"    {name}: FAILED — {entry['error']}")
            if stage == "family_namefix":
                family_skip = f"error: {entry['error']}"
        if entry.get("queue_record_error"):
            print(f"      ⚠ outcome not recorded in the queue: "
                  f"{entry['queue_record_error']}")
    return family_skip


def _print_summary(created, skipped, failed, detector_failed=(), family_not_run=()):
    print("\n" + "─" * 50)
    if created:
        print(f"\nCreated ({len(created)}):")
        for s in created:
            print(f"  ✓ {s}")
    if skipped:
        print(f"\nSkipped — duplicates ({len(skipped)}):")
        for name in skipped:
            print(f"  ~ {name}")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for name, err in failed:
            print(f"  ✗ {name}: {err}")
    if detector_failed:
        print(f"\nDetectors failed — artifacts saved ({len(detector_failed)}):")
        for name, err in detector_failed:
            print(f"  ⚠ {name}: {err}")
    if family_not_run:
        print(f"\nFamily names NOT checked ({len(family_not_run)}):")
        for name, reason in family_not_run:
            print(f"  ! {name}: {SKIP_REASON_COPY.get(reason, reason)}")
        if any(r == "no_roster" for _, r in family_not_run):
            print("  Register your family's names at /roster, then re-check with:")
            print("    python src/fix_session.py <session-id> --write")
    if not created and not skipped and not failed and not detector_failed:
        print("\n  (nothing processed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process inbox audio files.")
    parser.add_argument(
        "--file",
        help="Process only this filename from the inbox (e.g. 'New Recording 37.m4a')",
    )
    parser.add_argument(
        "--skip-monitor", action="store_true",
        help="skip the deprioritized Monitor detector pass (TMAS-64/80) — saves a second "
             "model load per session; used by the finish-and-free churn (load_oldest.py)",
    )
    parser.add_argument(
        "--skip-name-fixers", action="store_true",
        help="defer the three name fixers (namefix/familyfix/spellingfix) — skips the "
             "expensive per-name Whisper re-listens; the transcripts-first churn (TMAS-95) "
             "sets this and generates the name questions later via fix_session.py",
    )
    args = parser.parse_args()
    process_inbox(target_file=args.file, run_detectors=not args.skip_monitor,
                  run_name_fixers=not args.skip_name_fixers)
