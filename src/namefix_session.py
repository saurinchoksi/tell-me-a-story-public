#!/usr/bin/env python3
"""Run the SHADOW namefix stage on an EXISTING session — labeled questions, nothing applied.

Calibration semantics (2026-07-02, docs/name-architecture.md): the stage forms opinions and
labels them; it never touches a transcript. A dry-run prints the labeled groups; --write
persists them to sessions/<id>/pending-name-corrections.json (the shadow queue the review
page serves). The transcript is untouched in both modes — the old --demote flag is gone
because there is nothing to demote when nothing applies.

Usage:
    python src/namefix_session.py <session-id>            # dry-run (prints labeled groups)
    python src/namefix_session.py <session-id> --write    # also write the shadow queue file
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from api.helpers import get_session_dir  # noqa: E402
from model_cache import cached_or_run, fingerprint  # noqa: E402
import namefix  # noqa: E402
from namefix import LABELS as LABEL_ORDER  # noqa: E402  — the single label source


def score_vs_key(session_dir: Path, result: dict) -> None:
    """If the session has a by-ear key in validation-notes.json, score every labeled group
    against it (join by segment_id+word_index) so the label classes can be read against
    ground truth. Sessions without a key just skip this."""
    notes_path = session_dir / "validation-notes.json"
    if not notes_path.exists():
        return
    from detectors.phonetics import clean, codes
    key = {}
    for n in json.loads(notes_path.read_text()).get("notes", []):
        ans = (n.get("text") or "").strip()
        if ans and n.get("segmentId") is not None and n.get("wordIndex") is not None:
            key[(n["segmentId"], n["wordIndex"])] = ans
    if not key:
        return
    tally: dict = {}
    for g in result["groups"]:
        for o in g["occurrences"]:
            ans = key.get((o["segment_id"], o["word_index"]))
            if ans is None:
                continue
            right = (clean(g["canonical"]) == clean(ans)
                     or bool(codes(clean(g["canonical"])) & codes(clean(ans))))
            slot = tally.setdefault(g["label"], {"right": 0, "wrong": 0})
            slot["right" if right else "wrong"] += 1
    print("\n  SCORED vs by-ear key (suggestion vs the key's answer, per label):")
    for label in LABEL_ORDER:
        if label in tally:
            t = tally[label]
            print(f"    {label:14} right {t['right']} | wrong {t['wrong']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id")
    ap.add_argument("--write", action="store_true",
                    help="write the shadow queue file (default: print only)")
    args = ap.parse_args()

    session_dir = get_session_dir(ROOT / "sessions", args.session_id)
    rich_path = session_dir / "transcript-rich.json"
    audio_path = next(iter(session_dir.glob("audio.*")), None)
    if audio_path is None:
        raise SystemExit(f"no audio file in {session_dir}")
    transcript = json.loads(rich_path.read_text())
    if not transcript.get("_stories"):
        raise SystemExit("no _stories on this transcript — run segmentation first")

    print(f"[namefix-session] {args.session_id}  SHADOW  "
          f"mode={'write queue' if args.write else 'dry-run'}")
    # Same cache as the pipeline stage: a dry-run computes and caches; a subsequent --write
    # persists EXACTLY the reviewed groups. A transcript/config change recomputes.
    result, was_cached = cached_or_run(
        session_dir, "namefix",
        fingerprint(namefix.transcript_fingerprint(transcript)),
        fingerprint(namefix.config_fingerprint()),
        lambda: namefix.run_namefix(transcript, str(audio_path)))
    if was_cached:
        print("  (groups served from the namefix cache — identical to the reviewed run)")

    for w in result["worlds"]:
        print(f"  story {w['story_id']} \"{w['title']}\": recognized={w['recognized_world']!r} "
              f"({w['n_candidates']} candidates)")
    by_label: dict = {}
    for g in result["groups"]:
        by_label.setdefault(g["label"], []).append(g)
    for label in LABEL_ORDER:
        groups = by_label.pop(label, [])
        print(f"\n  {label.upper()} ({len(groups)} names):")
        for g in sorted(groups, key=lambda x: -len(x["occurrences"])):
            flags = []
            if g.get("echo_in_sentence"):
                flags.append("echo-in-sentence")
            if g.get("n_sibling_spellings", 0) == 0 and len(g["occurrences"]) > 1:
                flags.append("stable-spelling")
            flag_s = f"  ⚠ {', '.join(flags)}" if flags else ""
            print(f"    {g['heard']:16} -> {g['canonical']:16} [{g['method']}] "
                  f"x{len(g['occurrences'])}{flag_s}")
    for label, groups in by_label.items():   # any label this script doesn't know yet
        print(f"\n  {label.upper()} ({len(groups)} names) [unrecognized label]")

    score_vs_key(session_dir, result)

    if not args.write:
        print("\n  dry-run: nothing written (shadow — the transcript is never touched).")
        return

    path = namefix.write_pending(session_dir, result, transcript)
    print(f"\n  wrote {path.name} ({len(result['groups'])} labeled groups). "
          f"Transcript untouched (shadow).")


if __name__ == "__main__":
    main()
