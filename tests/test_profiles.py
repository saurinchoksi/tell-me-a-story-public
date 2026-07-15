"""Tests for src/profiles.py — speaker profile storage."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from profiles import (
    add_embedding,
    add_voice_variant,
    compute_centroid,
    create_profile,
    load_profiles,
    save_profiles,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embedding(session_id="session-001", vector=None):
    """Build a minimal embedding_data dict matching production shape (speakers.py)."""
    if vector is None:
        vector = [1.0] * 256
    return {
        "session_id": session_id,
        "source_speaker_key": "SPEAKER_00",
        "vector": vector,
    }


def _variant(session_id="session-001", vector=None):
    """Build a minimal variant_data dict matching production shape (speakers.py)."""
    if vector is None:
        vector = [0.5] * 256
    return {
        "session_id": session_id,
        "source_speaker_key": "SPEAKER_01",
        "vector": vector,
    }


# ---------------------------------------------------------------------------
# load_profiles
# ---------------------------------------------------------------------------

def test_load_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "nonexistent.json")
        result = load_profiles(path)
        assert result == {"profiles": []}


def test_load_empty_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "empty.json")
        Path(path).write_text("")
        result = load_profiles(path)
        assert result == {"profiles": []}


def test_load_populated():
    data = {"profiles": [{"id": "spk_aaa", "name": "Test"}]}
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "profiles.json")
        Path(path).write_text(json.dumps(data))
        result = load_profiles(path)
        assert result == data


def test_load_bare_object():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "bare.json")
        Path(path).write_text("{}")
        result = load_profiles(path)
        assert result == {"profiles": []}


def test_load_malformed_json():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "bad.json")
        Path(path).write_text("{broken")
        with pytest.raises(json.JSONDecodeError):
            load_profiles(path)


# ---------------------------------------------------------------------------
# save_profiles
# ---------------------------------------------------------------------------

def test_save_round_trip():
    data = {
        "profiles": [
            {"id": "spk_abc", "name": "Alice", "embeddings": [], "centroid": None}
        ]
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "profiles.json")
        save_profiles(data, path)
        loaded = load_profiles(path)
        assert loaded == data


def test_save_deep_copy():
    data = {"profiles": [{"id": "spk_abc", "name": "Alice"}]}
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "profiles.json")
        save_profiles(data, path)
        # Mutate original after save
        data["profiles"][0]["name"] = "Bob"
        loaded = load_profiles(path)
        assert loaded["profiles"][0]["name"] == "Alice"


def test_save_creates_parent_dirs():
    data = {"profiles": []}
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "nested" / "deep" / "profiles.json")
        save_profiles(data, path)
        assert Path(path).exists()
        loaded = load_profiles(path)
        assert loaded == data


# ---------------------------------------------------------------------------
# create_profile
# ---------------------------------------------------------------------------

def test_create_basic():
    profiles = {"profiles": []}
    pid = create_profile(profiles, "Choksi", "parent")
    assert pid.startswith("spk_")
    assert len(pid) == 10  # "spk_" + 6 hex chars
    assert len(profiles["profiles"]) == 1
    assert profiles["profiles"][0]["id"] == pid


def test_create_fields():
    profiles = {"profiles": []}
    pid = create_profile(profiles, "Robin", "child")
    p = profiles["profiles"][0]
    assert p["name"] == "Robin"
    assert p["role"] == "child"
    assert p["embeddings"] == []
    assert p["centroid"] is None
    assert p["voice_variants"] == []
    # Timestamps are valid ISO format
    datetime.fromisoformat(p["created"])
    datetime.fromisoformat(p["updated"])


def test_create_unique_ids():
    profiles = {"profiles": []}
    id1 = create_profile(profiles, "A", "parent")
    id2 = create_profile(profiles, "B", "child")
    assert id1 != id2


# ---------------------------------------------------------------------------
# compute_centroid
# ---------------------------------------------------------------------------

def test_centroid_no_embeddings():
    profile = {"embeddings": []}
    assert compute_centroid(profile) is None


def test_centroid_single():
    vec = [float(i) for i in range(256)]
    profile = {"embeddings": [{"vector": vec}]}
    result = compute_centroid(profile)
    assert result == pytest.approx(vec)


# ---------------------------------------------------------------------------
# add_embedding
# ---------------------------------------------------------------------------

def test_add_embedding_first():
    profiles = {"profiles": []}
    pid = create_profile(profiles, "Test", "parent")
    vec = [2.0] * 256
    add_embedding(profiles, pid, _embedding(vector=vec))

    p = profiles["profiles"][0]
    assert len(p["embeddings"]) == 1
    assert p["centroid"] == pytest.approx(vec)


def test_add_embedding_subsequent():
    profiles = {"profiles": []}
    pid = create_profile(profiles, "Test", "parent")
    vec_a = [2.0] * 256
    vec_b = [4.0] * 256
    add_embedding(profiles, pid, _embedding(session_id="s1", vector=vec_a))
    add_embedding(profiles, pid, _embedding(session_id="s2", vector=vec_b))

    p = profiles["profiles"][0]
    assert len(p["embeddings"]) == 2
    expected = [3.0] * 256
    assert p["centroid"] == pytest.approx(expected)


def test_add_embedding_three_vectors():
    profiles = {"profiles": []}
    pid = create_profile(profiles, "Test", "parent")
    # Three orthogonal-ish vectors for clear math
    vec_a = [1.0, 0.0, 0.0] + [0.0] * 253
    vec_b = [0.0, 1.0, 0.0] + [0.0] * 253
    vec_c = [0.0, 0.0, 1.0] + [0.0] * 253
    add_embedding(profiles, pid, _embedding(session_id="s1", vector=vec_a))
    add_embedding(profiles, pid, _embedding(session_id="s2", vector=vec_b))
    add_embedding(profiles, pid, _embedding(session_id="s3", vector=vec_c))

    p = profiles["profiles"][0]
    assert len(p["embeddings"]) == 3
    expected = [1 / 3, 1 / 3, 1 / 3] + [0.0] * 253
    assert p["centroid"] == pytest.approx(expected)


def test_add_embedding_not_found():
    profiles = {"profiles": []}
    with pytest.raises(KeyError):
        add_embedding(profiles, "spk_nonexistent", _embedding())


def test_add_embedding_no_session_id():
    profiles = {"profiles": []}
    pid = create_profile(profiles, "Test", "parent")
    # Embeddings without session_id are always appended (no dedup)
    emb_a = {"vector": [1.0] * 256}
    emb_b = {"vector": [2.0] * 256}
    add_embedding(profiles, pid, emb_a)
    add_embedding(profiles, pid, emb_b)

    p = profiles["profiles"][0]
    assert len(p["embeddings"]) == 2
    expected = [1.5] * 256
    assert p["centroid"] == pytest.approx(expected)


def test_add_embedding_duplicate_session():
    profiles = {"profiles": []}
    pid = create_profile(profiles, "Test", "parent")
    vec_a = [2.0] * 256
    vec_b = [8.0] * 256
    add_embedding(profiles, pid, _embedding(session_id="same", vector=vec_a))
    add_embedding(profiles, pid, _embedding(session_id="same", vector=vec_b))

    p = profiles["profiles"][0]
    assert len(p["embeddings"]) == 1
    # Centroid reflects only the first embedding
    assert p["centroid"] == pytest.approx(vec_a)


# ---------------------------------------------------------------------------
# add_voice_variant
# ---------------------------------------------------------------------------

def test_add_variant_basic():
    profiles = {"profiles": []}
    pid = create_profile(profiles, "Test", "parent")
    created_ts = profiles["profiles"][0]["updated"]
    add_voice_variant(profiles, pid, _variant())

    p = profiles["profiles"][0]
    assert len(p["voice_variants"]) == 1
    assert p["updated"] >= created_ts
    # Auto-generated fields
    v = p["voice_variants"][0]
    assert v["id"].startswith("var_")
    assert len(v["id"]) == 10  # "var_" + 6 hex chars
    datetime.fromisoformat(v["created"])


def test_add_variant_centroid_unchanged():
    profiles = {"profiles": []}
    pid = create_profile(profiles, "Test", "parent")
    vec = [3.0] * 256
    add_embedding(profiles, pid, _embedding(vector=vec))

    centroid_before = profiles["profiles"][0]["centroid"][:]
    add_voice_variant(profiles, pid, _variant(vector=[99.0] * 256))

    assert profiles["profiles"][0]["centroid"] == pytest.approx(centroid_before)


def test_add_variant_duplicate_session():
    profiles = {"profiles": []}
    pid = create_profile(profiles, "Test", "parent")
    variant_a = {"session_id": "same-session", "vector": [1.0] * 256}
    variant_b = {"session_id": "same-session", "vector": [9.0] * 256}
    add_voice_variant(profiles, pid, variant_a)
    add_voice_variant(profiles, pid, variant_b)

    p = profiles["profiles"][0]
    assert len(p["voice_variants"]) == 1
    assert p["voice_variants"][0]["vector"][0] == 1.0


def test_add_variant_not_found():
    profiles = {"profiles": []}
    with pytest.raises(KeyError):
        add_voice_variant(profiles, "spk_nonexistent", _variant())


# ---------------------------------------------------------------------------
# Full round-trip integration
# ---------------------------------------------------------------------------

def test_full_round_trip():
    profiles = {"profiles": []}
    pid = create_profile(profiles, "Choksi", "parent")

    vec = [float(i % 7) for i in range(256)]
    add_embedding(profiles, pid, _embedding(session_id="20260207-172315", vector=vec))
    add_voice_variant(profiles, pid, _variant(session_id="20260218-185123"))

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "profiles.json")
        save_profiles(profiles, path)
        loaded = load_profiles(path)

    p = loaded["profiles"][0]
    assert p["id"] == pid
    assert p["name"] == "Choksi"
    assert p["role"] == "parent"
    assert len(p["embeddings"]) == 1
    assert p["embeddings"][0]["session_id"] == "20260207-172315"
    assert p["centroid"] == pytest.approx(vec)
    assert len(p["voice_variants"]) == 1
    assert p["voice_variants"][0]["session_id"] == "20260218-185123"
