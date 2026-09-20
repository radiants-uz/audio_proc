import numpy as np

from app.services.audio_processing.segments import (
    SegmentConfig,
    build_segments,
    drop_short,
    merge_adjacent,
    padded_bounds,
    split_long,
)
from app.services.audio_processing.types import RawTurn, Segment

CFG = SegmentConfig(merge_gap_sec=0.5, min_segment_sec=0.4, max_segment_sec=10.0, overlap_min_sec=0.3, pad_sec=0.15)


def test_merge_adjacent_same_speaker_small_gap():
    turns = [RawTurn(0, 2, "A"), RawTurn(2.3, 4, "A"), RawTurn(4.9, 6, "A"), RawTurn(6.1, 7, "B")]
    merged = merge_adjacent(turns, 0.5)
    assert [(t.start, t.end, t.speaker) for t in merged] == [(0, 4, "A"), (4.9, 6, "A"), (6.1, 7, "B")]


def test_drop_short():
    turns = [RawTurn(0, 0.3, "A"), RawTurn(1, 2, "A")]
    assert [t.start for t in drop_short(turns, 0.4)] == [1]


def test_split_long_without_audio_cuts_evenly():
    turns = [RawTurn(0, 25, "A")]
    parts = split_long(turns, 10.0)
    assert len(parts) == 3
    assert all(p.end - p.start <= 10.0 + 1e-6 for p in parts)
    assert parts[0].start == 0 and parts[-1].end == 25


def test_split_long_prefers_quiet_point():
    sr = 16000
    audio = np.ones(sr * 20, dtype=np.float32) * 0.5
    audio[sr * 9 : sr * 9 + sr // 2] = 0.0  # тишина на 9.0–9.5 с
    parts = split_long([RawTurn(0, 20, "A")], 12.0, audio)
    assert len(parts) == 2
    assert 8.9 < parts[0].end < 9.6


def test_overlap_flags():
    exclusive = [RawTurn(0, 5, "A"), RawTurn(5, 10, "B")]
    turns = [RawTurn(0, 5.5, "A"), RawTurn(4.4, 10, "B")]  # B начал на 0.6 с раньше конца A
    segs = build_segments(exclusive, turns, CFG)
    assert segs[0].overlap and segs[0].overlap_with == ["B"]
    assert segs[1].overlap and segs[1].overlap_with == ["A"]


def test_no_overlap_below_threshold():
    exclusive = [RawTurn(0, 5, "A"), RawTurn(5, 10, "B")]
    turns = [RawTurn(0, 5.1, "A"), RawTurn(5, 10, "B")]
    segs = build_segments(exclusive, turns, CFG)
    assert not segs[0].overlap and not segs[1].overlap


def test_build_segments_sorted_with_ids():
    exclusive = [RawTurn(10, 12, "B"), RawTurn(0, 3, "A"), RawTurn(3.2, 5, "A")]
    segs = build_segments(exclusive, exclusive, CFG)
    assert [s.id for s in segs] == [0, 1]
    assert [s.start for s in segs] == [0, 10]


def test_padded_bounds_respects_neighbours():
    segs = [Segment(0, "A", 0.0, 5.0), Segment(1, "B", 5.05, 10.0)]
    s, e = padded_bounds(segs[0], segs, 0.15, 100.0)
    assert s == 0.0
    assert abs(e - 5.15) < 1e-9  # заезд в чужой сегмент не больше 0.1 с
    s, e = padded_bounds(segs[1], segs, 0.15, 100.0)
    assert abs(s - 4.95) < 1e-9 or s >= 4.9
    assert abs(e - 10.15) < 1e-9
