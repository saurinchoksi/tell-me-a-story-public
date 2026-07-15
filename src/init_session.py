"""Initialize session folders from inbox audio files."""

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Clone-is-the-instance (TMAS-73, settled): CLI modules hang their paths off
# their own __file__ within this checkout — no env var, no --data-root flag, on
# purpose. A blank or test instance is a fresh `git clone` (data/ contents,
# sessions/, and inbox/ are gitignored, so a clone arrives empty); path
# injection is the API layer's job (create_app kwargs). Recipe + rationale:
# docs/stranger-walk.md.
SCRIPT_DIR = Path(__file__).parent.resolve()  # src/
PROJECT_DIR = SCRIPT_DIR.parent               # tell-me-a-story/
SESSIONS_DIR = PROJECT_DIR / "sessions"
INBOX_DIR = PROJECT_DIR / "inbox"
DUPLICATES_DIR = INBOX_DIR / "duplicates"
SUPPORTED_FORMATS = {".m4a", ".mp3", ".wav"}


def _run_ffprobe(filepath: Path) -> dict | None:
    """Run ffprobe and return parsed JSON, or None on any failure."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
             str(filepath)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def _get_ffprobe_tags(filepath: Path) -> dict:
    """Extract format-level tags from ffprobe, or empty dict on failure."""
    probe = _run_ffprobe(filepath)
    if probe is None:
        return {}
    return probe.get("format", {}).get("tags", {})


def _get_metadata_creation_time(filepath: Path) -> datetime | None:
    """Extract recording date from ffprobe creation_time tag.

    Apple Voice Memos (when dragged from the Mac Voice Memos app) stores
    the original recording timestamp in the container's creation_time tag.
    Returns a naive local datetime, or None if unavailable.
    """
    tags = _get_ffprobe_tags(filepath)
    creation_time = tags.get("creation_time")
    if not creation_time:
        return None
    try:
        dt = datetime.fromisoformat(creation_time.replace("Z", "+00:00"))
        return dt.astimezone().replace(tzinfo=None)
    except (ValueError, OSError):
        return None


def get_creation_time(filepath: Path) -> datetime:
    """Get file creation time, preferring embedded metadata over filesystem.

    Checks ffprobe creation_time first (accurate for Voice Memos dragged
    from the Mac app). Falls back to filesystem birth time / mtime.
    Prints a diagnostic note when metadata and filesystem dates diverge.
    """
    metadata_time = _get_metadata_creation_time(filepath)

    stat = filepath.stat()
    timestamp = getattr(stat, "st_birthtime", stat.st_mtime)
    filesystem_time = datetime.fromtimestamp(timestamp)

    if metadata_time and metadata_time != filesystem_time:
        delta = abs((metadata_time - filesystem_time).total_seconds())
        if delta > 3600:
            print(f"  note: metadata date {metadata_time:%Y-%m-%d %H:%M} "
                  f"differs from filesystem {filesystem_time:%Y-%m-%d %H:%M} "
                  f"(using metadata)")

    return metadata_time if metadata_time else filesystem_time


def generate_session_id(dt: datetime) -> str:
    """Format datetime as YYYYMMDD-HHMMSS."""
    return dt.strftime("%Y%m%d-%H%M%S")


def file_hash(filepath: Path) -> str:
    """Compute MD5 hash of file contents."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_uuid_duplicate(uuid: str) -> str | None:
    """Check existing sessions for a matching voice-memo-uuid.

    Scans each session's audio.m4a via ffprobe for the voice-memo-uuid tag.
    Returns the session_id if a duplicate is found, None otherwise.
    """
    if not SESSIONS_DIR.exists():
        return None
    for session_dir in SESSIONS_DIR.iterdir():
        if not session_dir.is_dir():
            continue
        audio_path = session_dir / "audio.m4a"
        if not audio_path.exists():
            continue
        tags = _get_ffprobe_tags(audio_path)
        if tags.get("voice-memo-uuid") == uuid:
            return session_dir.name
    return None


def init_session(audio_path: Path) -> tuple[str, str] | None:
    """Move audio file to new session folder.

    Returns (session_id, audio_filename) on success, None if skipped.
    """
    # Extract ffprobe tags once (used for UUID check and date)
    tags = _get_ffprobe_tags(audio_path)

    # UUID-based duplicate check (catches Voice Memos re-muxing)
    uuid = tags.get("voice-memo-uuid")
    if uuid:
        existing_session = _find_uuid_duplicate(uuid)
        if existing_session:
            DUPLICATES_DIR.mkdir(parents=True, exist_ok=True)
            dup_dest = DUPLICATES_DIR / f"{existing_session}_{audio_path.name}"
            shutil.move(audio_path, dup_dest)
            print(f"  {audio_path.name} → DUPLICATE of session {existing_session} "
                  f"(same voice-memo-uuid, moved to duplicates/)")
            return None

    creation_time = get_creation_time(audio_path)
    session_id = generate_session_id(creation_time)
    session_dir = SESSIONS_DIR / session_id

    if session_dir.exists():
        # Find existing audio file in session
        existing_audio = None
        for ext in SUPPORTED_FORMATS:
            candidate = session_dir / f"audio{ext}"
            if candidate.exists():
                existing_audio = candidate
                break

        if existing_audio and file_hash(audio_path) == file_hash(existing_audio):
            # True duplicate - move to duplicates folder
            DUPLICATES_DIR.mkdir(parents=True, exist_ok=True)
            dup_dest = DUPLICATES_DIR / f"{session_id}_{audio_path.name}"
            shutil.move(audio_path, dup_dest)
            print(f"  {audio_path.name} → DUPLICATE of session {session_id} (moved to duplicates/)")
            return None
        else:
            # Collision - different files with same timestamp
            print(f"  {audio_path.name} → COLLISION: session {session_id} exists with different content")
            print(f"              Manual resolution required - file left in inbox")
            return None

    session_dir.mkdir(parents=True)
    dest_path = session_dir / f"audio{audio_path.suffix.lower()}"
    try:
        shutil.move(audio_path, dest_path)
    except Exception:
        session_dir.rmdir()  # Clean up the empty folder
        raise  # Re-raise so caller knows it failed

    # Record the original inbox filename — the file is renamed to
    # audio.<ext>, so this provenance would otherwise be lost.
    with open(session_dir / "session-metadata.json", "w") as f:
        json.dump({"originalFilename": audio_path.name}, f, indent=2)

    print(f"  {audio_path.name} → {session_id}/{dest_path.name}")
    return session_id, dest_path.name


def main():
    """Process all audio files in inbox."""
    if not INBOX_DIR.exists():
        INBOX_DIR.mkdir(parents=True)
        print(f"Created inbox directory: {INBOX_DIR}")
        print("Drop audio files here and run again.")
        return

    audio_files = [
        f for f in INBOX_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    ]

    if not audio_files:
        print(f"No audio files found in {INBOX_DIR}")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}")
        return

    print("Processing inbox...")
    initialized = []
    skipped = 0
    errors = 0

    for audio_path in sorted(audio_files):
        try:
            result = init_session(audio_path)
            if result:
                initialized.append(result)
            else:
                skipped += 1
        except Exception as e:
            print(f"  ERROR processing {audio_path.name}: {e}")
            errors += 1

    print()
    print(f"Initialized {len(initialized)} sessions. Skipped {skipped}.", end="")
    if errors:
        print(f" Errors: {errors}.")
    else:
        print()

    if initialized:
        print("Ready for pipeline:")
        for session_id, audio_filename in initialized:
            print(f"  python src/pipeline.py sessions/{session_id}/{audio_filename}")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
