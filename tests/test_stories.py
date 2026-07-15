"""Tests for enrich_with_stories — folding story regions into the transcript."""

import copy

from stories import enrich_with_stories


def _transcript():
    # mixes int ids and a gap_* string id (an injected [unintelligible] gap)
    return {"segments": [
        {"id": 0, "text": " a", "words": [{"word": " a"}]},
        {"id": 1, "text": " b", "words": [{"word": " b"}]},
        {"id": "gap_2.5", "text": "[unintelligible]", "words": []},
        {"id": 2, "text": " c", "words": [{"word": " c"}]},
        {"id": 3, "text": " d", "words": [{"word": " d"}]},
    ]}


def test_top_level_stories_written_once():
    out = enrich_with_stories(
        _transcript(), [{"start_id": 0, "end_id": 1, "title": "One", "world": "W1"}])
    assert out["_stories"] == [
        {"index": 0, "start_id": 0, "end_id": 1, "title": "One", "world": "W1"}]


def test_per_segment_story_index_spans_gaps_and_string_ids():
    stories = [
        {"start_id": 0, "end_id": "gap_2.5", "title": "A", "world": "W"},
        {"start_id": 2, "end_id": 3, "title": "B", "world": "W"},
    ]
    segs = enrich_with_stories(_transcript(), stories)["segments"]
    assert [s.get("_story") for s in segs] == [0, 0, 0, 1, 1]


def test_non_story_segments_untagged():
    segs = enrich_with_stories(
        _transcript(), [{"start_id": 0, "end_id": 1, "title": "A", "world": "W"}])["segments"]
    assert segs[0]["_story"] == 0 and segs[1]["_story"] == 0
    for i in (2, 3, 4):
        assert "_story" not in segs[i]  # outside the only story's span


def test_text_and_words_untouched_and_input_unchanged():
    t = _transcript()
    original = copy.deepcopy(t)
    out = enrich_with_stories(t, [{"start_id": 0, "end_id": 1, "title": "A", "world": "W"}])
    assert t == original  # deep copy: input transcript not mutated
    assert out["segments"][0]["text"] == " a"
    assert out["segments"][0]["words"] == [{"word": " a"}]


def test_story_with_absent_id_skipped_without_raising():
    stories = [{"start_id": 0, "end_id": 999, "title": "A", "world": "W"}]  # 999 absent
    out = enrich_with_stories(_transcript(), stories)  # must not raise
    assert out["_stories"][0]["end_id"] == 999  # still listed at top level
    assert all("_story" not in s for s in out["segments"])  # but nothing tagged


def test_reversed_story_bounds_normalized():
    # end_id earlier than start_id still tags the span (defensive swap)
    out = enrich_with_stories(
        _transcript(), [{"start_id": 1, "end_id": 0, "title": "A", "world": "W"}])
    segs = out["segments"]
    assert segs[0]["_story"] == 0 and segs[1]["_story"] == 0
