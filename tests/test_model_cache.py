"""Tests for the per-session model-output cache (stamp-and-skip)."""

import json

from model_cache import CACHE_FILENAME, cached_or_run, fingerprint


def test_cache_hit_skips_compute(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return ["A"]

    out1, cached1 = cached_or_run(tmp_path, "stage", "in1", "cfg1", compute)
    out2, cached2 = cached_or_run(tmp_path, "stage", "in1", "cfg1", compute)
    assert out1 == out2 == ["A"]
    assert cached1 is False and cached2 is True
    assert len(calls) == 1  # second call reused the cache, compute not re-run


def test_cache_miss_on_input_change(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return ["X"]

    cached_or_run(tmp_path, "stage", "in1", "cfg1", compute)
    _, cached = cached_or_run(tmp_path, "stage", "in2", "cfg1", compute)  # input changed
    assert cached is False
    assert len(calls) == 2


def test_cache_miss_on_config_change(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return ["X"]

    cached_or_run(tmp_path, "stage", "in1", "cfg1", compute)
    _, cached = cached_or_run(tmp_path, "stage", "in1", "cfg2", compute)  # config changed
    assert cached is False
    assert len(calls) == 2


def test_cache_separate_stages_dont_collide(tmp_path):
    cached_or_run(tmp_path, "stageA", "in", "cfg", lambda: ["a"])
    out, cached = cached_or_run(tmp_path, "stageB", "in", "cfg", lambda: ["b"])
    assert out == ["b"] and cached is False  # different stage keeps its own entry


def test_cache_persists_across_loads(tmp_path):
    cached_or_run(tmp_path, "stage", "in", "cfg", lambda: {"k": "v"})
    # A fresh read (new in-memory state) still hits the on-disk cache.
    out, cached = cached_or_run(tmp_path, "stage", "in", "cfg", lambda: {"k": "DIFFERENT"})
    assert out == {"k": "v"} and cached is True
    data = json.loads((tmp_path / CACHE_FILENAME).read_text())
    assert data["stages"]["stage"]["output"] == {"k": "v"}


def test_fingerprint_stable_and_distinct():
    assert fingerprint("hello") == fingerprint("hello")
    assert fingerprint("hello") != fingerprint("world")
    assert fingerprint("hello").startswith("sha256:")
