#!/usr/bin/env python3
"""The Name Book — the single store for the human's rulings about names, one file per world.

Replaces the old blessing store (``worlddict.py`` / ``data/worlds/``). Design settled
2026-07-02 (docs/name-architecture.md): every decision Saurin makes about a name lives here,
so no fixer ever re-asks a question he already answered. Three ruling types per world file
at ``data/name-book/<slug>.json``:

    {"_world": "Thomas & Friends", "_version": 2,
     "yes":  [{"heard": "Bushma", "canonical": "Bhishma", "provenance": "...", "at": "..."}],
     "no":   [{"heard_cleaned": "jammus", "provenance": "...", "at": "..."}],
     "spellings": []}

- A **yes** ("this heard form means this cast name") is recorded knowledge: during
  calibration it folds matching questions at VIEW time (covered-by-yes); at the apply pass
  it applies deterministically, forever for that world. The fixer's runs never read it.
- A **no** protects the WORD, not one suggestion (Choksi's call, 2026-07-02): "Jammus is
  her name — never auto-rewrite it to ANYTHING in this world, and never ask about it
  again." During calibration it acts at VIEW time: the review page folds matching queued
  questions into a collapsed "covered by your earlier no" section — quiet, but never
  silently swallowed, and instantly liftable (``unreject``). At the future APPLY PASS the
  no is a hard filter: a no'd word must never be rewritten, whatever canonical is proposed
  (the pass MUST implement this — nothing at run time enforces it, by design).
- A **spelling pick** rules one INVENTED-name cluster (the M9b fixer's questions — a
  character with no canon anywhere, spelled several ways in one night). ``spelling`` is
  the one spelling to unify to at the apply pass; ``None`` means "don't unify these"
  (Choksi's call, 2026-07-03: a No on a spelling card narrowly declines the unification —
  it never writes a word-level no, so it can never mute another fixer's questions about
  the same word). Keyed by ``cluster`` (the cluster's min cleaned spelling); either kind
  folds that cluster's future questions at view time, quiet-but-viewable, liftable
  (``unpick_spelling``).

A world starts EMPTY (missing file -> no rulings): the first-contact stance — no pre-seeded
rosters, only what a person ruled. ONLY explicit human gestures write this store (the bless/
reject endpoints); detectors, models, and scripts never do. The folder is gitignored (the
family's file will hold real names) and synced to the private data repo for backup.

The family is just another world here: the M9a fixer will read/write ``family.json`` with
its cast drawn from ``data/name_roster.json``.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

# Repo-root/data/name-book — src/ is one level under the root.
BOOK_DIR = Path(__file__).resolve().parents[1] / "data" / "name-book"


def slug(world: str) -> str:
    """Filesystem-safe slug for a world name. Lowercase, non-alphanumeric runs -> single
    hyphen, trimmed. "Thomas & Friends" -> "thomas-friends"; "" -> "" (caller guards)."""
    return re.sub(r"[^a-z0-9]+", "-", world.strip().lower()).strip("-")


def book_path(world: str, base: Path = BOOK_DIR) -> Path:
    """Path to a world's Name Book file (may not exist)."""
    return Path(base) / f"{slug(world)}.json"


def load_book(world: str, base: Path = BOOK_DIR) -> dict:
    """The raw per-world book, or ``{}`` when the world has no file yet (first contact).
    Fail-loud on a corrupt file — a malformed book is a real bug, not an empty world."""
    path = book_path(world, base)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_yes_map(world: str, base: Path = BOOK_DIR) -> dict[str, str]:
    """Lowered heard -> canonical for one world's yeses (empty when nothing ruled).
    The namefix gate cleans these keys itself (same treatment as the old variant map)."""
    book = load_book(world, base)
    return {y["heard"].lower(): y["canonical"] for y in book.get("yes", [])}


def load_no_words(world: str, base: Path = BOOK_DIR) -> set[str]:
    """{heard_cleaned} for one world's nos — the words the human said to leave alone.
    Callers pass already-cleaned tokens (detectors.phonetics.clean)."""
    book = load_book(world, base)
    return {n["heard_cleaned"] for n in book.get("no", [])}


def _atomic_write(path: Path, obj: dict) -> None:
    """Write JSON to ``path`` atomically — the project-wide helper (src/jsonio.py)."""
    from jsonio import atomic_write_json
    atomic_write_json(path, obj)


def _open_book(world: str, base: Path) -> dict:
    data = load_book(world, base) or {}
    data.setdefault("_world", world)
    data["_version"] = 2
    data.setdefault("yes", [])
    data.setdefault("no", [])
    data.setdefault("spellings", [])
    return data


def bless(world: str, heard: str, canonical: str, provenance: str = "human",
          base: Path = BOOK_DIR, now: str | None = None) -> dict:
    """Record a yes: ``heard`` means ``canonical`` in ``world`` — forever.

    Read-merge-write, atomic. One heard form maps to exactly one canonical (a later bless
    of the same heard form to a different canonical replaces the ruling — the human changed
    their mind). Returns the updated book.
    """
    if not world:
        raise ValueError("bless: world is required (an unrecognized world cannot be ruled on)")
    if not heard.strip() or not canonical.strip():
        raise ValueError("bless: heard and canonical must be non-empty")
    data = _open_book(world, base)
    stamp = now or datetime.now(timezone.utc).isoformat()
    entry = next((y for y in data["yes"] if y["heard"].lower() == heard.lower()), None)
    if entry is None:
        data["yes"].append({"heard": heard, "canonical": canonical,
                            "provenance": provenance, "at": stamp})
    elif entry["canonical"].lower() != canonical.lower():
        entry.update({"canonical": canonical, "provenance": provenance, "at": stamp})
    _atomic_write(book_path(world, base), data)
    return data


def reject(world: str, heard_cleaned: str, provenance: str = "human",
           base: Path = BOOK_DIR, now: str | None = None) -> dict:
    """Record a no: ``heard_cleaned`` is right as heard — never auto-rewrite it and never
    ask about it again in ``world``. Idempotent on the word. Returns the updated book."""
    if not world:
        raise ValueError("reject: world is required")
    if not heard_cleaned.strip():
        raise ValueError("reject: heard_cleaned must be non-empty")
    data = _open_book(world, base)
    if heard_cleaned not in {n["heard_cleaned"] for n in data["no"]}:
        data["no"].append({"heard_cleaned": heard_cleaned,
                           "provenance": provenance,
                           "at": now or datetime.now(timezone.utc).isoformat()})
    _atomic_write(book_path(world, base), data)
    return data


def load_spellings(world: str, base: Path = BOOK_DIR) -> dict[str, dict]:
    """cluster_cleaned -> spelling entry for one world's picks (empty when nothing ruled).
    An entry's ``spelling`` is the unify-to form, or None for "don't unify these"."""
    book = load_book(world, base)
    return {s["cluster"]: s for s in book.get("spellings", [])}


def pick_spelling(world: str, cluster_cleaned: str, spelling: str | None,
                  provenance: str = "human", base: Path = BOOK_DIR,
                  now: str | None = None) -> dict:
    """Record a spelling ruling for one invented-name cluster: unify to ``spelling`` at
    the apply pass, or — with ``spelling=None`` — "don't unify these", leave every
    occurrence as heard and never ask about this cluster again. One ruling per cluster
    (a re-pick replaces — the human changed their mind). Returns the updated book."""
    if not world:
        raise ValueError("pick_spelling: world is required")
    if not cluster_cleaned.strip():
        raise ValueError("pick_spelling: cluster_cleaned must be non-empty")
    if spelling is not None and not spelling.strip():
        raise ValueError("pick_spelling: spelling must be None (don't unify) or non-empty")
    data = _open_book(world, base)
    stamp = now or datetime.now(timezone.utc).isoformat()
    entry = next((s for s in data["spellings"] if s["cluster"] == cluster_cleaned), None)
    if entry is None:
        data["spellings"].append({"cluster": cluster_cleaned, "spelling": spelling,
                                  "provenance": provenance, "at": stamp})
    elif entry["spelling"] != spelling:
        entry.update({"spelling": spelling, "provenance": provenance, "at": stamp})
    _atomic_write(book_path(world, base), data)
    return data


def unpick_spelling(world: str, cluster_cleaned: str, base: Path = BOOK_DIR) -> bool:
    """Lift a spelling ruling (undo / the fold's lift button) — the cluster's questions
    return to the live queue on the next view. Returns True if a pick was removed."""
    data = load_book(world, base)
    if not data:
        return False
    before = len(data.get("spellings", []))
    data["spellings"] = [s for s in data.get("spellings", [])
                         if s["cluster"] != cluster_cleaned]
    if len(data["spellings"]) == before:
        return False
    _atomic_write(book_path(world, base), data)
    return True


def unbless(world: str, heard: str, base: Path = BOOK_DIR) -> bool:
    """Lift a yes (the review page's undo on a recorded verdict) — the human changed
    their mind before the apply pass. Returns True if a yes was removed."""
    data = load_book(world, base)
    if not data:
        return False
    before = len(data.get("yes", []))
    data["yes"] = [y for y in data.get("yes", []) if y["heard"].lower() != heard.lower()]
    if len(data["yes"]) == before:
        return False
    _atomic_write(book_path(world, base), data)
    return True


def unreject(world: str, heard_cleaned: str, base: Path = BOOK_DIR) -> bool:
    """Lift a no (the review page's "remove this no" button) — the future night where the
    child really does say the canon name stays rescuable. Returns True if a no was removed."""
    data = load_book(world, base)
    if not data:
        return False
    before = len(data.get("no", []))
    data["no"] = [n for n in data.get("no", []) if n["heard_cleaned"] != heard_cleaned]
    if len(data["no"]) == before:
        return False
    _atomic_write(book_path(world, base), data)
    return True


def fingerprint(world: str, base: Path = BOOK_DIR) -> str:
    """Short content stamp of one world's book ("" when the world has no file) — written
    into the pending file so a run records which rulings it saw."""
    path = book_path(world, base)
    if not path.exists():
        return ""
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
