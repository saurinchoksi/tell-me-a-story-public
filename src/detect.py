#!/usr/bin/env python3
"""Run failure-mode detectors over sessions, writing sessions/<id>/detections.json.

Detection only — transcripts are never modified. Console output is counts only;
flagged tokens (which can echo mis-rendered names) live in the gitignored
detections.json and the Monitor screen.

Usage:
    python src/detect.py                          # all sessions with a transcript
    python src/detect.py 20260414-213156 ...      # specific sessions
    python src/detect.py --detector m9a-family-names
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from detectors import DETECTORS, get_detector
from detectors.base import scan_session

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SESSIONS_DIR = PROJECT_ROOT / "sessions"
SESSION_ID_RE = re.compile(r"^\d{8}-\d{6}$")


def discover_session_ids() -> list[str]:
    """All session dirs with a transcript, newest first. Sessions without a
    transcript aren't an error in batch mode — there's nothing to scan yet."""
    return sorted(
        (
            d.name
            for d in SESSIONS_DIR.iterdir()
            if d.is_dir()
            and SESSION_ID_RE.match(d.name)
            and (d / "transcript-rich.json").exists()
        ),
        reverse=True,
    )


def main():
    ap = argparse.ArgumentParser(description="Run failure-mode detectors over sessions.")
    ap.add_argument("sessions", nargs="*", help="session ids under sessions/ (default: all)")
    ap.add_argument("--detector", help="run only this detector id")
    ap.add_argument("--judge", action="store_true",
                    help="apply the offline LLM judge to m9b-name-consistency "
                         "(recovers dictionary-word names; loads Gemma, slow)")
    ap.add_argument("--story-names", action="store_true",
                    help="run the offline canon-name auditor (m9c-canon): segments each "
                         "session then audits names with local Gemma-4, surfacing M9c "
                         "(loads Gemma locally; slow — minutes/session). Skipped by "
                         "default and never in a web request. Loads the model separately "
                         "from --judge if both are given.")
    args = ap.parse_args()

    detectors = [get_detector(args.detector)] if args.detector else DETECTORS

    # The judge is an offline upgrade for the M9b detector only.
    m9b_judge = None
    if args.judge:
        from detectors.name_consistency_judge import make_judge
        m9b_judge = make_judge()

    # Offline detectors (the per-story name auditor) run only when asked: the
    # --story-names flag, or an explicit --detector that names one. A bare run stays fast.
    run_offline = args.story_names or (bool(args.detector) and detectors[0].offline_only)

    if args.sessions:
        session_ids = args.sessions
        for sid in session_ids:
            if not (SESSIONS_DIR / sid).is_dir():
                raise FileNotFoundError(f"No session directory: sessions/{sid}")
    else:
        session_ids = discover_session_ids()
        if not session_ids:
            print("No sessions with a transcript found.")
            return

    print(f"Running {len(detectors)} detector(s) over {len(session_ids)} session(s):\n")
    for sid in session_ids:
        session_dir = SESSIONS_DIR / sid
        data = scan_session(session_dir, detectors, force=True, judge=m9b_judge,
                            run_offline=run_offline)
        for det in detectors:
            sec = data["detectors"].get(det.id)
            if sec is None:
                continue  # e.g. an offline_only detector skipped without --story-names
            judged = "  +judge" if sec.get("judge_applied") else ""
            print(
                f"  {sid}  {det.id:<20} "
                f"{sec['n_flags']:>3} flags / {sec['n_word_tokens']} tokens{judged}"
            )


if __name__ == "__main__":
    main()
