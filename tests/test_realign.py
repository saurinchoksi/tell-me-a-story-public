"""Unit tests for src/realign.py (TMAS-54).

The acoustic alignment itself (torchaudio MMS_FA) is validated separately in the
TMAS-50 experiment; here we pin the *logic* — healthy realign vs M11 rescue vs
guard, and the structural invariants (ids/text/count/order preserved, envelope
expands outward only) — by mocking the alignment step so no model is loaded.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import realign  # noqa: E402

DICT = set("abcdefghijklmnopqrstuvwxyz'")
AUDIO = np.zeros(16000 * 600, dtype=np.float32)  # 10 min of silence; mock ignores it


def bundle():
    return realign.AlignerBundle(model=None, tokenizer=None, aligner=None,
                                 dictset=DICT, device=None)


def even_align(conf):
    """Mock _align: place N tokens evenly across [lo, hi] with a fixed confidence."""
    def _f(audio, lo, hi, norm_tokens, _bundle):
        n = len(norm_tokens)
        step = (hi - lo) / n
        return [(lo + step * i, lo + step * (i + 1), conf) for i in range(n)]
    return _f


def seg(id_, text, words, start=10.0, end=12.0, **extra):
    return {"id": id_, "text": text, "start": start, "end": end,
            "words": words, **extra}


def w(word, start, end, **extra):
    return {"word": word, "start": start, "end": end, **extra}


def test_healthy_realigns_timestamps_and_preserves_fields(monkeypatch):
    monkeypatch.setattr(realign, "_align", even_align(0.9))
    s = seg(0, " hello world",
            [w(" hello", 10.0, 11.0, probability=0.5, _speaker={"label": "S1"}),
             w(" world", 11.0, 12.0, probability=0.6, _speaker={"label": "S1"})])
    words, status = realign.realign_segment(AUDIO, s, bundle())
    assert status == "realigned"
    assert [x["word"] for x in words] == [" hello", " world"]
    assert words[0]["probability"] == 0.5 and words[0]["_speaker"] == {"label": "S1"}
    assert words[0]["start"] != 10.0 and "_align_conf" in words[0]


def test_rescue_brings_back_dropped_words(monkeypatch):
    monkeypatch.setattr(realign, "_align", even_align(0.9))
    # text has 4 words, words[] only 2 (M11 dropped "you believe")
    s = seg(1, " can you believe that",
            [w(" can", 10.0, 10.2), w(" that", 11.8, 12.0)], _speaker={"label": "S2"})
    words, status = realign.realign_segment(AUDIO, s, bundle())
    assert status == "rescued"
    assert "".join(x["word"] for x in words) == " can you believe that"
    assert all("_align_conf" in x for x in words)
    # probability must stay numeric (a null crashes the validator's .toFixed)
    assert all(isinstance(x["probability"], (int, float)) for x in words)
    assert words[1]["_speaker"] == {"label": "S2"}        # rescued word inherits dominant


def test_low_confidence_rescue_is_guarded(monkeypatch):
    monkeypatch.setattr(realign, "_align", even_align(0.1))  # below CONF_MIN
    s = seg(2, " can you believe that",
            [w(" can", 10.0, 10.2), w(" that", 11.8, 12.0)])
    words, status = realign.realign_segment(AUDIO, s, bundle())
    assert status == "guarded"
    assert [x["word"] for x in words] == [" can", " that"]  # untouched


def test_loop_blowup_is_guarded(monkeypatch):
    monkeypatch.setattr(realign, "_align", even_align(0.9))   # high conf, but...
    # 1 word vs 6 text tokens -> ratio past the rescue cap (M2/M10 territory)
    s = seg(3, " a b c d e f", [w(" hmm", 10.0, 10.5)])
    words, status = realign.realign_segment(AUDIO, s, bundle())
    assert status == "guarded"
    assert [x["word"] for x in words] == [" hmm"]


def test_realign_transcript_preserves_ids_and_skips_gaps(monkeypatch):
    monkeypatch.setattr(realign, "_align", even_align(0.9))
    t = {"segments": [
        seg(0, " hello world", [w(" hello", 10.0, 11.0), w(" world", 11.0, 12.0)]),
        {"id": "gap_5.000", "text": "[unintelligible]", "start": 5.0, "end": 6.0,
         "words": [], "_source": "diarization_gap"},
        seg(7, " can you believe that",
            [w(" can", 20.0, 20.2), w(" that", 21.8, 22.0)], start=20.0, end=22.0),
    ]}
    ids_before = [s["id"] for s in t["segments"]]
    new_t, stats = realign.realign_transcript(t, AUDIO, bundle())
    assert [s["id"] for s in new_t["segments"]] == ids_before    # ids + order + count
    assert [s["text"] for s in new_t["segments"]] == [s["text"] for s in t["segments"]]
    gap = new_t["segments"][1]
    assert gap["words"] == [] and gap["_source"] == "diarization_gap"  # untouched
    assert stats["realigned"] == 1 and stats["rescued"] == 1


def test_envelope_expands_outward_only(monkeypatch):
    # rescued first word starts before the segment start -> start widens down,
    # never up; end widens up to cover the last word.
    monkeypatch.setattr(realign, "_align", even_align(0.9))
    s = seg(0, " can you believe that",
            [w(" can", 10.5, 10.7), w(" that", 11.5, 11.6)], start=10.5, end=11.6)
    t = {"segments": [s]}
    new_t, _ = realign.realign_transcript(t, AUDIO, bundle())
    ns = new_t["segments"][0]
    assert ns["start"] <= 10.5 and ns["end"] >= 11.6
