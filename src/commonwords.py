#!/usr/bin/env python3
"""The frequency-based common-words list — namefix's answer to "is this a real word?".

Why not /usr/share/dict/words: that file is Webster's 1934 unabridged, and its noise bit
this project three separate times — it lists the archaic "jami" (which once made "Jamis"
read as a plural), the archaic "bando" (which made "Bandos" read as a plural and queued a
correct fix), and it literally contains "Kauravas" as an entry (an epic name classified as
an English word). A modern frequency list has neither failure direction: the words a person
actually says ("arrows", "beam", "father") are all high-frequency, and archaic ghosts /
canon names simply aren't in it.

`data/common-words.txt` is the top 25,000 English words by frequency (generated once from
the wordfreq corpus — regenerate with:
    python -c "from wordfreq import top_n_list; open('data/common-words.txt','w').write('\\n'.join(top_n_list('en', 25000))+'\\n')"
). Committed, so runtime needs no dependency and the behavior is deterministic/auditable.
Inflected forms are included ("arrows" is its own entry), so membership is EXACT — no
plural-stripping, which is what let the ghosts in.

Verified separation on every case that has bitten us (zipf in top-25k):
  protected:  arrows, beam, father, wars
  not falsely protected: bando, jami, kauravas, bushma, bandos, bheem

SCOPE: namefix only, for now. The validated M9b/M9c detectors keep their Webster's-based
checks until a proper re-validation — swapping a graduated detector's dictionary silently
would change its behavior unmeasured.
"""
from pathlib import Path

WORDLIST_PATH = Path(__file__).resolve().parents[1] / "data" / "common-words.txt"

_words: set | None = None


def _load() -> set:
    global _words
    if _words is None:
        _words = {line.strip().lower() for line in WORDLIST_PATH.read_text().splitlines()
                  if line.strip()}
    return _words


def is_common(cleaned: str) -> bool:
    """Exact membership of a cleaned (lowercase, letters-only) token in the top-25k list."""
    return cleaned in _load()


def wordlist_fingerprint() -> str:
    """Hash of the list file — goes into namefix's config fingerprint so a list change
    invalidates cached decisions instead of silently serving stale ones."""
    import hashlib
    return hashlib.sha256(WORDLIST_PATH.read_bytes()).hexdigest()[:16]
