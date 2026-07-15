"""The API must never emit a bare NaN — it is not valid JSON.

Whisper leaves NaN in a segment's `avg_logprob` when it cannot score a decode (the
repetition storms: a child shouting "No!" fifteen times). Python's json writes that as
a literal `NaN` token, which `curl` and Python both happily read back — but the
browser's JSON.parse rejects the ENTIRE response, so one unscored segment blanks a whole
page. 27 of 111 sessions carried one; it stayed hidden because every session keyed
before it happened to be clean.

These tests read the response the way a browser does — strict JSON, NaN refused.
"""

import json
import math

import pytest

from api.app import _json_safe, create_app


def strict_loads(payload):
    """Parse the way a browser's JSON.parse does: a bare NaN/Infinity is a syntax error."""
    return json.loads(payload, parse_constant=lambda c: pytest.fail(
        f"response contains the bare token {c!r} — not valid JSON, the browser will reject it"))


@pytest.fixture
def session_with_nan(tmp_path):
    """A session whose transcript carries Whisper's unscored segment."""
    sid = "20260122-203658"
    sdir = tmp_path / sid
    sdir.mkdir()
    (sdir / "transcript-rich.json").write_text(json.dumps({
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "No!", "avg_logprob": float("nan"),
             "words": [{"word": " No!", "start": 0.0, "end": 1.0, "probability": 0.9}]},
            {"id": 1, "start": 1.0, "end": 2.0, "text": "Okay.", "avg_logprob": -0.25,
             "words": [{"word": " Okay.", "start": 1.0, "end": 2.0, "probability": 0.8}]},
        ],
    }))
    return tmp_path, sid


@pytest.fixture
def client(tmp_path_factory, session_with_nan):
    sessions_dir, _ = session_with_nan
    app = create_app(sessions_dir=sessions_dir,
                     profiles_path=str(tmp_path_factory.mktemp("p") / "profiles.json"))
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_session_with_nan_logprob_is_valid_json(client, session_with_nan):
    """The bug: GET /api/sessions/<id> embeds the transcript, so one NaN broke the page."""
    _, sid = session_with_nan
    resp = client.get(f"/api/sessions/{sid}")
    assert resp.status_code == 200
    assert b"NaN" not in resp.data          # the literal token a browser chokes on
    body = strict_loads(resp.data)          # fails loudly if it slipped through anyway
    segs = body["transcript"]["segments"]
    assert segs[0]["avg_logprob"] is None   # unscored -> null, the honest JSON for "no value"
    assert segs[1]["avg_logprob"] == -0.25  # a real score is untouched


# --- _json_safe: the pure transform ---

def test_json_safe_maps_nan_and_infinity_to_none():
    out = _json_safe({"a": float("nan"), "b": float("inf"), "c": float("-inf")})
    assert out == {"a": None, "b": None, "c": None}


def test_json_safe_preserves_everything_else():
    """Only the unrepresentable floats change — no other value is touched."""
    payload = {"f": -0.25, "i": 3, "s": "NaN", "b": True, "n": None,
               "list": [1.5, {"deep": float("nan")}], "tuple": (1, 2)}
    out = _json_safe(payload)
    assert out["f"] == -0.25 and not isinstance(out["f"], bool)
    assert out["i"] == 3
    assert out["s"] == "NaN"                # the STRING "NaN" is legitimate data
    assert out["b"] is True                 # bool is an int, not a float — must survive
    assert out["n"] is None
    assert out["list"] == [1.5, {"deep": None}]
    assert out["tuple"] == [1, 2]           # tuples serialize as arrays anyway


def test_json_safe_leaves_a_clean_transcript_untouched():
    """No behaviour change for the 84 sessions that never had a NaN."""
    clean = {"segments": [{"id": 0, "avg_logprob": -0.31, "words": [{"probability": 0.97}]}]}
    assert _json_safe(clean) == clean
    assert not math.isnan(_json_safe(clean)["segments"][0]["avg_logprob"])
