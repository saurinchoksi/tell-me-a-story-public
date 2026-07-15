"""Unit tests for the order-robust judge (`judge_names_voted`). The model is mocked with a
canned `gen`; voting/gating arithmetic is pure and fast. The dictionary gate is isolated to
identity except where a test specifically asserts on it (it has its own tests in
test_canon_dictionary_gate.py)."""
import random

import detectors.story_names._qwen35 as q


def gen_seq(outputs):
    """A fake `gen` returning successive canned judge outputs — one per call (per round)."""
    it = iter(outputs)

    def gen(prompt, max_tokens=500):
        return next(it)

    return gen


def gen_const(text):
    def gen(prompt, max_tokens=500):
        return text

    return gen


NAMES = ["Arjun", "Bishma", "Garn", "Krishna"]


# ----------------------------- cutoff boundary ------------------------------
def test_vote_keeps_at_cutoff(monkeypatch):
    monkeypatch.setattr(q, "gate_canon_flags", lambda flags, singles: flags)
    # rounds=5, threshold=0.5 -> cutoff = ceil(2.5) = 3. Caught in exactly 3 -> kept.
    outs = ["Bishma -> Bhishma"] * 3 + [""] * 2
    flags = q.judge_names_voted(gen_seq(outs), "Mahabharata", NAMES, set(), rounds=5, threshold=0.5)
    assert [f["wrong_cleaned"] for f in flags] == [["bishma"]]
    assert flags[0]["vote_count"] == 3 and flags[0]["vote_rounds"] == 5


def test_vote_drops_below_cutoff(monkeypatch):
    monkeypatch.setattr(q, "gate_canon_flags", lambda flags, singles: flags)
    outs = ["Bishma -> Bhishma"] * 2 + [""] * 3  # caught 2/5 < cutoff 3
    flags = q.judge_names_voted(gen_seq(outs), "Mahabharata", NAMES, set(), rounds=5, threshold=0.5)
    assert flags == []


# ----------------------------- majority spelling ----------------------------
def test_vote_picks_majority_canonical(monkeypatch):
    monkeypatch.setattr(q, "gate_canon_flags", lambda flags, singles: flags)
    outs = ["Bandos -> Pandavas"] * 3 + ["Bandos -> Pandava"] * 2  # 5/5, spelling 3 vs 2
    flags = q.judge_names_voted(gen_seq(outs), "Mahabharata", NAMES, set(), rounds=5, threshold=0.5)
    assert len(flags) == 1 and flags[0]["canonical"] == "Pandavas"


def test_vote_canonical_tiebreak_is_deterministic(monkeypatch):
    monkeypatch.setattr(q, "gate_canon_flags", lambda flags, singles: flags)
    # caught 4/5, canonical tie 2-2 -> tiebreak (-count, spelling) -> "Aaaa" (alphabetical)
    outs = ["Garn -> Bbbb", "Garn -> Aaaa", "Garn -> Bbbb", "Garn -> Aaaa", ""]
    flags = q.judge_names_voted(gen_seq(outs), "Mahabharata", NAMES, set(), rounds=5, threshold=0.5)
    assert len(flags) == 1 and flags[0]["canonical"] == "Aaaa"


# ----------------------------- gate AFTER voting ----------------------------
def test_vote_gates_once_not_per_round(monkeypatch):
    """The dictionary gate must run ONCE, after the vote — not per round (gating per round lets a
    one-round canonical wobble undercount a real catch)."""
    calls = {"n": 0}

    def spy(flags, singles):
        calls["n"] += 1
        return flags

    monkeypatch.setattr(q, "gate_canon_flags", spy)
    q.judge_names_voted(gen_const("Bishma -> Bhishma"), "Mahabharata", NAMES, set(),
                        rounds=5, threshold=0.5)
    assert calls["n"] == 1


# ----------------------------- degenerate / empty ---------------------------
def test_vote_rounds1_uses_caller_order_single_call(monkeypatch):
    monkeypatch.setattr(q, "gate_canon_flags", lambda flags, singles: flags)
    seen = {"n": 0}

    def gen(prompt, max_tokens=500):
        seen["n"] += 1
        seen["prompt"] = prompt
        return "Bishma -> Bhishma"

    q.judge_names_voted(gen, "Mahabharata", NAMES, set(), rounds=1)
    assert seen["n"] == 1                        # single call, no shuffles
    assert "\n".join(NAMES) in seen["prompt"]    # caller's (sorted) order, not a random shuffle


def test_vote_empty_world_or_names_no_model_call():
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("gen must not be called"))
    assert q.judge_names_voted(boom, "", NAMES, set()) == []
    assert q.judge_names_voted(boom, "Mahabharata", [], set()) == []


# ----------------------------- seed determinism -----------------------------
def test_seed_shuffle_is_deterministic():
    # the one piece that IS reproducible under Metal non-determinism: the shuffle sequence
    a = list(NAMES); random.Random(q.JUDGE_SEED_BASE + 2).shuffle(a)
    b = list(NAMES); random.Random(q.JUDGE_SEED_BASE + 2).shuffle(b)
    assert a == b
