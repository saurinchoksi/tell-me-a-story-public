#!/usr/bin/env python3
"""Run ALL shadow fixers on an EXISTING session — namefix, familyfix, spellingfix.

THE one command that refreshes a session's shared question queue (the review page's
stale chip points here: a transcript change stales the whole queue, and write_queue
drops fixers that haven't re-run against the new transcript — so re-run them all).
SHADOW everywhere: the only thing --write touches is
sessions/<id>/pending-name-corrections.json; no transcript, no Name Book, ever.

Model runs are sequential (one at a time, the 16 GB rule) and cached per session:
namefix serves from its cache when transcript+config are unchanged (identical groups to
the last pipeline run); familyfix likewise ("family_namefix" cache); spellingfix is pure
code and just computes.

Usage:
    python src/fix_session.py <session-id>            # dry-run (prints per-fixer groups)
    python src/fix_session.py <session-id> --write    # also write the shared queue file
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # clone-is-the-instance — see init_session.py
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from api.helpers import get_session_dir  # noqa: E402
from model_cache import cached_or_run, fingerprint  # noqa: E402
import familyfix  # noqa: E402
import namefix  # noqa: E402
import spellingfix  # noqa: E402


def _print_groups(fixer: str, groups: list) -> None:
    print(f"\n  [{fixer}] {len(groups)} labeled question(s)")
    for g in sorted(groups, key=lambda x: (x["label"], -len(x["occurrences"]))):
        flags = []
        if g.get("echo_in_sentence"):
            flags.append("echo-in-sentence")
        if g.get("redecode_heard") is False:
            flags.append("redecode-heard-nothing")
        flag_s = f"  ⚠ {', '.join(flags)}" if flags else ""
        print(f"    {g['label']:22} {g['heard']:16} -> {g['canonical']:16} "
              f"x{len(g['occurrences'])}{flag_s}")


def fix_session(session_dir: Path, write: bool = False) -> dict:
    """Run every fixer that can run here; return {fixer: groups|None} (None = skipped).
    With write=True, each fixer that RAN writes its section of the shared queue; a
    skipped fixer leaves its prior groups alone but RECORDS the skip (with reason) in
    the queue's `fixers` metadata — "never ran" must stay visible."""
    rich_path = session_dir / "transcript-rich.json"
    transcript = json.loads(rich_path.read_text())
    audio_path = next(iter(session_dir.glob("audio.*")), None)
    results: dict = {}
    skip_reasons: dict = {}   # fixer key -> machine token (recorded in queue metadata)

    # namefix — needs stories + audio; cached (identical to the last reviewed run)
    if transcript.get("_stories") and audio_path is not None:
        nf_result, was_cached = cached_or_run(
            session_dir, "namefix",
            fingerprint(namefix.transcript_fingerprint(transcript)),
            fingerprint(namefix.config_fingerprint()),
            lambda: namefix.run_namefix(transcript, str(audio_path)))
        results["namefix"] = nf_result
        print(f"  namefix: {len(nf_result['groups'])} groups"
              + (" (cached)" if was_cached else ""))
    else:
        skip_reasons["namefix"] = "no_audio_path" if audio_path is None else "no_stories"
        results["namefix"] = None
        print(f"  namefix: skipped ({namefix.SKIP_REASON_COPY[skip_reasons['namefix']]})")

    # familyfix — needs the roster + audio; cached
    if familyfix.DEFAULT_ROSTER_PATH.exists() and audio_path is not None:
        ff_result, was_cached = cached_or_run(
            session_dir, "family_namefix",
            fingerprint(namefix.transcript_fingerprint(transcript)),
            fingerprint(familyfix.config_fingerprint()),
            lambda: familyfix.run_familyfix(transcript, str(audio_path)))
        results["family"] = ff_result
        print(f"  familyfix: {len(ff_result['groups'])} groups"
              + (" (cached)" if was_cached else ""))
    else:
        skip_reasons["family"] = "no_audio_path" if audio_path is None else "no_roster"
        results["family"] = None
        print(f"  familyfix: skipped ({namefix.SKIP_REASON_COPY[skip_reasons['family']]})")

    # spellingfix — pure code; worlds + the defer set from namefix's run when it ran
    story_worlds = ({w["story_id"]: w["recognized_world"]
                     for w in results["namefix"]["worlds"]}
                    if results["namefix"] else {})
    defer = spellingfix.defer_words_from(
        results["namefix"]["groups"] if results["namefix"] else [])
    sf_result = spellingfix.run_spellingfix(transcript, story_worlds,
                                            defer_words=defer)
    results["spelling"] = sf_result
    print(f"  spellingfix: {len(sf_result['groups'])} groups")

    for fixer in ("namefix", "family", "spelling"):
        if results[fixer] is not None:
            _print_groups(fixer, results[fixer]["groups"])

    if not write:
        print("\n  dry-run: nothing written (shadow — no file was touched).")
        return results

    if results["namefix"] is not None:
        namefix.write_pending(session_dir, results["namefix"], transcript)
    else:
        namefix.write_skip(session_dir, namefix.FIXER, namefix.VERSION, transcript,
                           skip_reasons["namefix"])
    if results["family"] is not None:
        namefix.write_queue(session_dir, familyfix.FIXER, results["family"]["groups"],
                            transcript, familyfix.VERSION, familyfix.config_fingerprint())
    else:
        namefix.write_skip(session_dir, familyfix.FIXER, familyfix.VERSION, transcript,
                           skip_reasons["family"])
    namefix.write_queue(session_dir, spellingfix.FIXER, results["spelling"]["groups"],
                        transcript, spellingfix.VERSION,
                        spellingfix.config_fingerprint())
    n = sum(len(r["groups"]) for r in results.values() if r is not None)
    print(f"\n  wrote pending-name-corrections.json ({n} groups across fixers). "
          "Transcripts untouched (shadow).")
    if skip_reasons:
        skips = ", ".join(f"{f} ({r})" for f, r in skip_reasons.items())
        print(f"  skips noted in the queue's fixer metadata — an existing completed "
              f"check of this same transcript, if any, is kept: {skips}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id")
    ap.add_argument("--write", action="store_true",
                    help="write the shared queue file (default: print only)")
    args = ap.parse_args()
    session_dir = get_session_dir(ROOT / "sessions", args.session_id)
    print(f"[fix-session] {args.session_id}  SHADOW  "
          f"mode={'write queue' if args.write else 'dry-run'}")
    fix_session(session_dir, write=args.write)


if __name__ == "__main__":
    main()
