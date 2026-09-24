"""§5.3 Segment builder: склейка → фильтр коротких → разрезание длинных →
флаги overlap → сортировка и id. Чистая функция, покрыта unit-тестами."""

from dataclasses import dataclass

import numpy as np

from app.services.audio_processing.ingest import SAMPLE_RATE
from app.services.audio_processing.types import RawTurn, Segment


@dataclass
class SegmentConfig:
    merge_gap_sec: float = 0.5
    min_segment_sec: float = 0.4
    max_segment_sec: float = 28.0
    overlap_min_sec: float = 0.3
    pad_sec: float = 0.15


def merge_adjacent(turns: list[RawTurn], merge_gap_sec: float) -> list[RawTurn]:
    out: list[RawTurn] = []
    for t in sorted(turns, key=lambda x: (x.start, x.end)):
        if out and out[-1].speaker == t.speaker and t.start - out[-1].end < merge_gap_sec:
            out[-1] = RawTurn(start=out[-1].start, end=max(out[-1].end, t.end), speaker=t.speaker)
        else:
            out.append(RawTurn(start=t.start, end=t.end, speaker=t.speaker))
    return out


def drop_short(turns: list[RawTurn], min_segment_sec: float) -> list[RawTurn]:
    return [t for t in turns if t.duration >= min_segment_sec]


def _best_cut(audio: np.ndarray | None, target: float, window: float = 3.0, frame: float = 0.05) -> float:
    """Самая тихая точка в окне ±window вокруг target (по RMS-энергии)."""
    if audio is None:
        return target
    lo = max(0.0, target - window)
    hi = min(len(audio) / SAMPLE_RATE, target + window)
    if hi - lo < frame * 2:
        return target
    best_t, best_score = target, float("inf")
    # При равной тишине (пауза, полная тишина) берём точку ближе к цели, а не
    # самую раннюю в окне — иначе куски получаются рваными и короче нужного.
    scale = float(np.mean(audio**2)) + 1e-12
    t = lo
    while t + frame <= hi:
        a, b = int(t * SAMPLE_RATE), int((t + frame) * SAMPLE_RATE)
        e = float(np.mean(audio[a:b] ** 2)) if b > a else float("inf")
        score = e / scale + 1e-3 * abs(t + frame / 2 - target) / window
        if score < best_score:
            best_score, best_t = score, t + frame / 2
        t += frame
    return best_t


def split_long(turns: list[RawTurn], max_segment_sec: float, audio: np.ndarray | None = None) -> list[RawTurn]:
    out: list[RawTurn] = []
    for t in turns:
        start = t.start
        while t.end - start > max_segment_sec:
            n_parts = int(np.ceil((t.end - start) / max_segment_sec))
            target = start + (t.end - start) / n_parts
            cut = _best_cut(audio, target)
            # не оставлять хвост/голову короче 1 с
            cut = min(max(cut, start + 1.0), t.end - 1.0)
            out.append(RawTurn(start=start, end=cut, speaker=t.speaker))
            start = cut
        out.append(RawTurn(start=start, end=t.end, speaker=t.speaker))
    return out


def mark_overlaps(segments: list[Segment], turns: list[RawTurn], overlap_min_sec: float) -> None:
    """Наложение — только с спикером, который есть в транскрипте. pyannote может
    выдать «спикера» из обрывков по 0,3–0,5 с внутри чужих реплик, не получившего
    ни одного эксклюзивного участка: в транскрипте его нет, и помечать наложение
    с ним — путать читателя (запись БЕНА, 2026-09-24)."""
    present = {seg.speaker for seg in segments}
    for seg in segments:
        others = set()
        for t in turns:
            if t.speaker == seg.speaker or t.speaker not in present:
                continue
            inter = min(seg.end, t.end) - max(seg.start, t.start)
            if inter > overlap_min_sec:
                others.add(t.speaker)
        seg.overlap = bool(others)
        seg.overlap_with = sorted(others)


def build_segments(
    exclusive_turns: list[RawTurn],
    turns: list[RawTurn],
    cfg: SegmentConfig,
    audio: np.ndarray | None = None,
) -> list[Segment]:
    merged = merge_adjacent(exclusive_turns, cfg.merge_gap_sec)
    kept = drop_short(merged, cfg.min_segment_sec)
    split = split_long(kept, cfg.max_segment_sec, audio)
    split.sort(key=lambda t: (t.start, t.end))
    segments = [Segment(id=i, speaker=t.speaker, start=t.start, end=t.end) for i, t in enumerate(split)]
    mark_overlaps(segments, turns, cfg.overlap_min_sec)
    return segments


def padded_bounds(seg: Segment, segments: list[Segment], pad_sec: float, duration: float) -> tuple[float, float]:
    """§5.3 п.5: паддинг, не заезжая в соседний сегмент другого спикера более чем на 0.1 с."""
    start = max(0.0, seg.start - pad_sec)
    end = min(duration, seg.end + pad_sec)
    for other in segments:
        if other.id == seg.id or other.speaker == seg.speaker:
            continue
        if other.end <= seg.start and other.end > start:
            start = max(start, other.end - 0.1)
        if other.start >= seg.end and other.start < end:
            end = min(end, other.start + 0.1)
    return start, end
