"""Forced-alignment word realignment (TMAS-50 / TMAS-54).

Whisper's per-word timestamps drift — concentrated on segment-initial words,
which land too early in the pre-speech pause (the CrisperWhisper bug: the leading
space is glued onto the next word, so the silence is parked onto it). This module
re-derives word boundaries with torchaudio's MMS_FA CTC forced aligner, which the
TMAS-50 experiment validated three independent ways (seg-initial onset 22%->66%,
re-ASR word-return 70%->90%, the "Okay," anchor 1.70s->2.08s).

It aligns from the segment's trusted **text** (what forced alignment naturally
takes), which also folds in the TMAS-53 / Mode-11 fix: `clean_transcript` prunes
zero-duration words from `words[]` but never from `text`, so re-deriving the word
list from text brings the real dropped words back with proper durations.

Safety: the EMP axial-labels ground truth binds to each segment's `id`, so this
module only ever rewrites per-word `start`/`end` and the within-segment `words[]`
list, and expands the segment time envelope outward. It never touches segment
`id`, `text`, count, or order. (The id invariant is enforced by the surgical
caller, src/realign_session.py.)

Per-segment alignment in a padded window is safe and cheap: the drift makes starts
too *early*, so the true onset is always inside the window — no quadratic full-file
pass needed.
"""

from collections import namedtuple

import numpy as np

SR = 16000          # MMS_FA sample rate (16kHz mono)
FA_PAD = 0.5        # seconds of audio padded each side of a segment
CONF_MIN = 0.30     # min mean CTC confidence to trust a word-count-changing rescue
RESCUE_MAX_RATIO = 2.0   # rescue only if text has up to this many x the words

AlignerBundle = namedtuple("AlignerBundle", "model tokenizer aligner dictset device")


def load_aligner():
    """Build the MMS_FA model + tokenizer + aligner once (reusable).

    The acoustic model runs on MPS/CPU; the forced_align Viterbi op is CPU-only
    (no MPS kernel), so emissions are moved to CPU before alignment.
    """
    import torch
    import torchaudio

    bundle = torchaudio.pipelines.MMS_FA
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = bundle.get_model().to(device).eval()
    return AlignerBundle(
        model=model,
        tokenizer=bundle.get_tokenizer(),
        aligner=bundle.get_aligner(),
        dictset=set(bundle.get_dict().keys()),
        device=device,
    )


def normalize(surface, dictset):
    """Lowercase and keep only characters the CTC dictionary knows."""
    return "".join(c for c in (surface or "").lower() if c in dictset)


def text_surfaces(text):
    """Whitespace-tokenize segment text into Whisper-style word surfaces.

    Whisper word surfaces carry a leading space (" Okay,"), and for a healthy
    segment join(words) == text, so prepending a space to each whitespace token
    reproduces the exact surfaces; for an M11 segment it recovers the dropped
    words that survive only in `text`.
    """
    return [" " + tok for tok in text.split()]


def _align(audio, lo, hi, norm_tokens, bundle):
    """Return per-token (start_sec, end_sec, conf) over window [lo, hi], or None."""
    import torch

    lo = max(0.0, lo)
    clip = audio[int(lo * SR):int(hi * SR)]
    if len(clip) < int(0.1 * SR):
        return None
    wav = torch.from_numpy(np.ascontiguousarray(clip)).unsqueeze(0)
    try:
        with torch.inference_mode():
            emission, _ = bundle.model(wav.to(bundle.device))
        emission = emission.cpu()
        spans = bundle.aligner(emission[0], bundle.tokenizer(norm_tokens))
    except Exception:  # noqa: BLE001 — a bad segment must not kill the run
        return None
    ratio = wav.size(1) / emission.size(1)
    out = []
    for sp in spans:
        s = lo + ratio * sp[0].start / SR
        e = lo + ratio * sp[-1].end / SR
        dur = sum((x.end - x.start) for x in sp)
        conf = sum(x.score * (x.end - x.start) for x in sp) / dur if dur else 0.0
        out.append((s, e, float(conf)))
    return out


def realign_segment(audio, seg, bundle, next_start=None, pad=FA_PAD):
    """Realign one segment's words. Returns (new_words, status).

    status is one of:
      "realigned" — healthy segment, timestamps overwritten on existing words
      "rescued"   — word count changed (M11): words rebuilt from text
      "guarded"   — broken/loop/low-conf or text shrinks the words: left untouched
      "skipped"   — no alignable tokens / alignment error: left untouched

    next_start, when given, lets a rescue (text has more words than words[]) widen
    its window up to the next segment — heavy M11 drops collapse the segment
    envelope onto the one surviving word, so the dropped words' real audio sits
    past seg["end"] and only a wider window reaches it.
    """
    text = seg.get("text") or ""
    existing = seg.get("words") or []
    surfaces = text_surfaces(text)
    norms = [normalize(s, bundle.dictset) for s in surfaces]
    keep = [i for i, n in enumerate(norms) if n]
    if not keep:
        return existing, "skipped"

    n_text, n_words = len(surfaces), len(existing)
    lo = seg["start"] - pad
    # Tight window is the proven one (the experiment validated [start-pad, end+pad]).
    # A heavy M11 drop can leave a trailing word just past seg["end"], so allow a
    # small reach toward the next segment — but never a sprawling window (that lets
    # a low-confidence word smear across trailing silence).
    hi = seg["end"] + pad
    if n_text > n_words and next_start is not None:
        hi = min(next_start, seg["end"] + 2.0) + pad
    spans = _align(audio, lo, hi, [norms[i] for i in keep], bundle)
    if spans is None:
        return existing, "skipped"

    timing = {i: sp for i, sp in zip(keep, spans)}            # surface-idx -> (s,e,conf)
    mean_conf = float(np.mean([c for (_, _, c) in timing.values()])) if timing else 0.0

    # Healthy: same token count -> map 1:1 onto existing words, overwrite only
    # timestamps (preserves probability / _speaker / _corrections).
    if n_text == n_words and n_words > 0:
        new = []
        for i, w in enumerate(existing):
            nw = dict(w)
            if i in timing:
                s, e, c = timing[i]
                nw["start"], nw["end"] = s, e
                nw["_align_conf"] = round(c, 3)
            new.append(nw)
        return new, "realigned"

    # Rescue (M11): text has more words than words[] and the alignment is
    # trustworthy -> rebuild from text so the dropped real words come back.
    if (n_words < n_text <= max(n_words * RESCUE_MAX_RATIO, n_words + 4)
            and mean_conf >= CONF_MIN):
        dom_speaker = seg.get("_speaker")  # within-segment rescue inherits dominant
        new = []
        for i, surf in enumerate(surfaces):
            if i in timing:
                s, e, c = timing[i]
            else:  # punctuation-only token: zero-width at the previous end
                s = e = new[-1]["end"] if new else seg["start"]
                c = 0.0
            # A rescued word has no Whisper probability (it was dropped); use the
            # alignment confidence as the stand-in so `probability` stays a number
            # (the validator formats it with .toFixed and a null blanks the page).
            word = {"word": surf, "start": s, "end": e, "probability": round(c, 3),
                    "_align_conf": round(c, 3)}
            if dom_speaker is not None:
                word["_speaker"] = dom_speaker
            new.append(word)
        return new, "rescued"

    # Everything else (loop blow-up, low confidence, or text shorter than words):
    # a segment another failure mode already owns — leave it untouched.
    return existing, "guarded"


def realign_transcript(transcript, audio, bundle, pad=FA_PAD):
    """Realign every word-bearing segment in place-safe fashion.

    Returns (transcript, stats). Mutates a shallow-rebuilt segment list; preserves
    every segment id / text / count / order. Expands the segment time envelope
    outward only (never shrinks — protects story-segmentation gaps + note times).
    Skips [unintelligible]/diarization-gap and word-less segments.
    """
    stats = {"realigned": 0, "rescued": 0, "guarded": 0, "skipped": 0,
             "rescued_words": 0, "segments": 0}
    segs = transcript.get("segments", [])
    new_segments = []
    for i, seg in enumerate(segs):
        stats["segments"] += 1
        is_gap = seg.get("_source") == "diarization_gap"
        words = seg.get("words") or []
        if is_gap or not words:
            new_segments.append(seg)          # untouched (gap / word-less)
            continue
        next_start = segs[i + 1]["start"] if i + 1 < len(segs) else None
        new_words, status = realign_segment(audio, seg, bundle, next_start, pad)
        stats[status] = stats.get(status, 0) + 1
        if status in ("skipped", "guarded"):
            new_segments.append(seg)
            continue
        if status == "rescued":
            stats["rescued_words"] += len(new_words) - len(words)
        nseg = seg.copy()
        nseg["words"] = new_words
        # expand envelope outward only; never move start later than the first word
        if new_words:
            nseg["start"] = min(seg["start"], new_words[0]["start"])
            nseg["end"] = max(seg["end"], new_words[-1]["end"])
        new_segments.append(nseg)

    result = transcript.copy()
    result["segments"] = new_segments
    return result, stats
