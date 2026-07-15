"""Tests for the segmenter's disk-free input builder.

The module split (segment_session -> load_segments / segment_segments) newly exposed
load_segments_from_list, which must replicate load_segments' meta-list exactly so the
in-memory pipeline path and the disk-based detector fallback segment identically. The
Gemma walk itself is ported verbatim from the sealed EMP probe and validated there.
"""
import pytest

from story_segment import full_region_lines, load_segments_from_list


def test_full_region_lines_renders_all_nonempty_with_ids():
    # PROD: pass 2 reads the whole region (no head/mid/tail thinning); empty segments
    # are skipped, gap segments (string ids) are kept, ids are preserved.
    segs = [
        {"id": 0, "text": "once there was"},
        {"id": 1, "text": ""},                       # empty -> skipped
        {"id": 2, "text": "a dragon"},
        {"id": "gap_3.5", "text": "[unintelligible]"},
        {"id": 4, "text": "the end"},
    ]
    out = full_region_lines(segs, {"start_pos": 0, "end_pos": 4})
    assert out.splitlines() == [
        '[0] "once there was"',
        '[2] "a dragon"',
        '[gap_3.5] "[unintelligible]"',
        '[4] "the end"',
    ]


def test_full_region_lines_respects_region_bounds():
    segs = [{"id": i, "text": f"line {i}"} for i in range(6)]
    out = full_region_lines(segs, {"start_pos": 2, "end_pos": 4})
    assert out.splitlines() == ['[2] "line 2"', '[3] "line 3"', '[4] "line 4"']


def test_pos_gap_flag_and_gap_before():
    segs = load_segments_from_list([
        {"id": 0, "start": 0.0, "end": 1.0, "text": " hi", "words": []},
        {"id": "gap_1.5", "start": 1.0, "end": 1.2, "text": "[unintelligible]", "words": []},
        {"id": 1, "start": 2.0, "end": 3.0, "text": " bye", "words": []},
    ])
    assert [s["pos"] for s in segs] == [0, 1, 2]
    assert [s["id"] for s in segs] == [0, "gap_1.5", 1]
    assert [s["is_gap"] for s in segs] == [False, True, False]  # string id => gap segment
    assert segs[0]["gap_before"] is None                        # first segment
    assert segs[2]["gap_before"] == pytest.approx(0.8)          # 2.0 - 1.2


def test_text_stripped_and_missing_fields_tolerated():
    segs = load_segments_from_list([{"id": 0, "text": "  spaced  "}])  # no start/end/words
    assert segs[0]["text"] == "spaced"
    assert segs[0]["start"] is None and segs[0]["gap_before"] is None
