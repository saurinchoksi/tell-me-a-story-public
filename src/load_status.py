#!/usr/bin/env python3
"""Live status for the oldest-first churn (TMAS-80) — read-only, run from any terminal.

    python src/load_status.py             # print once
    python src/load_status.py --watch     # auto-refresh every 10s (Ctrl-C to stop)
    python src/load_status.py --watch 5   # auto-refresh every 5s

Infers progress from the session folders on disk, so it never touches the running
churn. Shows overall progress, the in-flight session and how long it's been going,
whether the churn process is alive, recent pace, and a rough ETA. The per-session
Whisper/stage detail lives in the churn's own stdout, not here.
"""

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from init_session import INBOX_DIR, SESSIONS_DIR, SUPPORTED_FORMATS


def _audio(d):
    return [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS] \
        if d.exists() else []


def _snapshot():
    done_files = sorted(SESSIONS_DIR.glob("*/transcript-rich.json"),
                        key=lambda p: p.stat().st_mtime) if SESSIONS_DIR.exists() else []
    inbox = len(_audio(INBOX_DIR))
    dupes = len(_audio(INBOX_DIR / "duplicates"))

    inflight = None
    if SESSIONS_DIR.exists():
        for sd in sorted(SESSIONS_DIR.iterdir()):
            if sd.is_dir() and (sd / "audio.m4a").exists() \
                    and not (sd / "transcript-rich.json").exists():
                inflight = sd  # sequential churn → at most one

    pace = None  # minutes/session, median gap between recent completions
    if len(done_files) >= 2:
        ts = [p.stat().st_mtime for p in done_files[-6:]]
        gaps = [b - a for a, b in zip(ts, ts[1:]) if 0 < b - a < 3 * 3600]
        if gaps:
            pace = sorted(gaps)[len(gaps) // 2] / 60.0

    try:
        alive = subprocess.run(["pgrep", "-f", "load_oldest.py"],
                               capture_output=True, timeout=5).returncode == 0
    except Exception:
        alive = None

    return dict(done=len(done_files), inbox=inbox, dupes=dupes,
                inflight=inflight, pace=pace, alive=alive)


def _bar(frac, width=24):
    frac = max(0.0, min(1.0, frac))
    fill = int(round(frac * width))
    return "█" * fill + "░" * (width - fill)


def _age_min(path):
    st = path.stat()
    born = getattr(st, "st_birthtime", st.st_mtime)
    return (time.time() - born) / 60.0


def render():
    s = _snapshot()
    # init MOVES processed files out of the inbox, so as sessions complete `inbox` shrinks
    # and `done` grows — the denominator must include `done` or frac would climb past 100%.
    total = s["done"] + s["inbox"] + s["dupes"]
    frac = s["done"] / total if total else 0.0
    out = [
        f"Tell Me a Story — churn status        (updated {time.strftime('%H:%M:%S')})",
        "",
        f"  [{_bar(frac)}]  {s['done']} / {total} recordings  ({frac*100:.0f}%)",
        f"  duplicates skipped: {s['dupes']}     churn process: "
        + ("running ✓" if s["alive"] else "NOT running ✗" if s["alive"] is False else "?"),
        "",
    ]
    if s["inflight"] is not None and s["alive"]:
        out.append(f"  in-flight: {s['inflight'].name}  (~{_age_min(s['inflight']):.0f} min in)")
    elif s["inflight"] is not None:
        out.append(f"  incomplete: {s['inflight'].name} — churn not running; audio is here with "
                   f"no transcript (pipeline may have failed). Needs recovery — do NOT delete it.")
    elif s["alive"]:
        out.append("  in-flight: (between sessions / starting next)")
    else:
        out.append("  in-flight: none")
    if s["pace"]:
        remaining = s["inbox"]  # files still waiting = what's left in the inbox
        out.append(f"  recent pace: ~{s['pace']:.0f} min/session   "
                   f"est. remaining: ~{remaining * s['pace'] / 60.0:.1f} h ({remaining} to go, rough)")
    return "\n".join(out)


def main():
    args = sys.argv[1:]
    watch = "--watch" in args
    interval = next((int(a) for a in args if a.isdigit()), 10)
    if not watch:
        print(render())
        return
    try:
        while True:
            print("\033[2J\033[H", end="")  # clear + home
            print(render())
            print(f"\n  (auto-refresh every {interval}s — Ctrl-C to stop)")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
