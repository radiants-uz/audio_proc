"""Пост-обработка спикеров после диаризации: ложные «спикеры» (обрывки шума,
эхо, реплики в 1–2 с) сливаются с ближайшим по голосу. Эмбеддинг спикера —
среднее ECAPA-эмбеддингов его сегментов.

Правила (пороги — в конфиге `SPEAKERS_*`):
1. Спикер с речью < min_speech_sec ИЛИ долей < min_share → слить с самым похожим.
2. Пары спикеров с косинусной близостью > merge_similarity → слить (итеративно).
3. Если задано expected_speakers — сливать самые похожие пары, пока не останется N.
4. Ниже min_speakers не опускаться."""

import logging
from dataclasses import dataclass

import numpy as np

from app.core.config import settings
from app.services.audio_processing.types import Segment

logger = logging.getLogger(__name__)


@dataclass
class MergeReport:
    before: int
    after: int
    merged: list[tuple[str, str, str]]  # (from, to, reason)


def _speaker_embeddings(embed_fn, audio: np.ndarray, segments: list[Segment]) -> dict[str, np.ndarray]:
    by_spk: dict[str, list[Segment]] = {}
    for s in segments:
        by_spk.setdefault(s.speaker, []).append(s)
    out = {}
    for spk, segs in by_spk.items():
        # длинные и чистые сегменты первыми, до ~30 с суммарно — этого хватает
        segs = sorted(segs, key=lambda s: (s.overlap, -s.duration))
        windows, total = [], 0.0
        for s in segs:
            windows.append((s.start, s.end))
            total += s.duration
            if total >= 30:
                break
        embs = embed_fn(audio, windows)
        if len(embs):
            v = embs.mean(axis=0)
            out[spk] = v / (np.linalg.norm(v) + 1e-8)
    return out


def merge_speakers(
    embed_fn,
    audio: np.ndarray,
    segments: list[Segment],
    expected_speakers: int | None = None,
) -> MergeReport:
    cfg = settings.speakers
    speech = {}
    for s in segments:
        speech[s.speaker] = speech.get(s.speaker, 0.0) + s.duration
    before = len(speech)
    if before <= 1:
        return MergeReport(before, before, [])
    total = sum(speech.values()) or 1.0
    floor = max(cfg.min_speakers, expected_speakers or 0, 1)

    embs = _speaker_embeddings(embed_fn, audio, segments)
    merged: list[tuple[str, str, str]] = []

    def sim(a: str, b: str) -> float:
        if a not in embs or b not in embs:
            return -1.0
        return float(np.dot(embs[a], embs[b]))

    def do_merge(src: str, dst: str, reason: str):
        for s in segments:
            if s.speaker == src:
                s.speaker = dst
            s.overlap_with = sorted({dst if o == src else o for o in s.overlap_with} - {s.speaker})
            s.overlap = bool(s.overlap_with)
        w_src, w_dst = speech.pop(src, 0.0), speech.get(dst, 0.0)
        speech[dst] = w_dst + w_src
        # эмбеддинг цели — среднее, взвешенное по времени речи
        if src in embs and dst in embs:
            v = embs[dst] * max(w_dst, 1e-3) + embs[src] * max(w_src, 1e-3)
            embs[dst] = v / (np.linalg.norm(v) + 1e-8)
        embs.pop(src, None)
        merged.append((src, dst, reason))

    def nearest(spk: str) -> str | None:
        others = [o for o in speech if o != spk]
        if not others:
            return None
        return max(others, key=lambda o: sim(spk, o))

    # 1. мелкие спикеры
    changed = True
    while changed and len(speech) > floor:
        changed = False
        for spk in sorted(speech, key=lambda k: speech[k]):
            if len(speech) <= floor:
                break
            if speech[spk] < cfg.min_speech_sec or speech[spk] / total < cfg.min_share:
                dst = nearest(spk)
                if dst is not None:
                    do_merge(spk, dst, f"small:{speech.get(dst, 0):.0f}s")
                    changed = True
                    break

    # 2. очень похожие пары
    changed = True
    while changed and len(speech) > floor:
        changed = False
        spks = sorted(speech, key=lambda k: -speech[k])
        best, best_sim = None, cfg.similarity_for_backend
        for i in range(len(spks)):
            for j in range(i + 1, len(spks)):
                v = sim(spks[i], spks[j])
                if v > best_sim:
                    best, best_sim = (spks[j], spks[i]), v
        if best:
            do_merge(best[0], best[1], f"similar:{best_sim:.2f}")
            changed = True

    # 3. явное число спикеров
    if expected_speakers:
        while len(speech) > expected_speakers:
            spks = sorted(speech, key=lambda k: -speech[k])
            pair = max(((spks[j], spks[i]) for i in range(len(spks)) for j in range(i + 1, len(spks))),
                       key=lambda p: sim(*p))
            do_merge(pair[0], pair[1], f"expected:{sim(*pair):.2f}")

    report = MergeReport(before, len(speech), merged)
    if merged:
        logger.info("speakers merged %d → %d: %s", before, len(speech), merged)
    return report
