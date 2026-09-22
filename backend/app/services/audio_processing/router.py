"""§5.6: группировка сегментов по языку, батчи, параллельность, прогресс.
Язык каждого сегмента уже известен (LID по спикеру) — ASR-API получает
готовую пару lang + аудио.

Каждый сегмент режется на куски ≤ ASR_MAX_CHUNK_SEC (в тихих точках): на дальних
записях модель на длинных кусках возвращает пустоту, на коротких — текст.
Пустой кусок при ASR_FALLBACK_LANGS повторяется на остальных языках файла."""

import asyncio
import logging
from collections.abc import Awaitable, Callable

import numpy as np

from app.core.config import settings
from app.services.audio_processing.asr.base import AsrClient
from app.services.audio_processing.ingest import SAMPLE_RATE, slice_to_wav_bytes
from app.services.audio_processing.segments import padded_bounds, split_long
from app.services.audio_processing.types import RawTurn, Segment

logger = logging.getLogger(__name__)

BATCH_SIZE = 8

_PUNCT = set(".,!?;:…-–—\"'()[]«» \t\n")


def has_words(text: str | None) -> bool:
    """ASR на шуме часто возвращает «.» — это не текст, а пустой ответ."""
    return bool(text) and any(ch not in _PUNCT for ch in text)


def chunk_bounds(
    start: float, end: float, max_sec: float, audio: np.ndarray | None, vad=None
) -> list[tuple[float, float]]:
    """Границы кусков ≤ max_sec внутри [start, end].
    С VAD — по речевым областям (тишина между репликами не попадает в ASR),
    без него — разрез в самой тихой точке."""
    regions = None
    if vad is not None and audio is not None and getattr(vad, "loaded", False):
        cfg = settings.asr
        a, b = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        try:
            regions = [
                (start + s, start + e)
                for s, e in vad.speech_regions(audio[a:b], cfg.vad_pad_sec, cfg.vad_min_silence_sec)
            ]
        except Exception as e:  # noqa: BLE001 — VAD необязателен
            logger.warning("vad failed, fixed chunking: %s", str(e)[:120])
            regions = None
    if not regions:
        regions = [(start, end)]
    out: list[tuple[float, float]] = []
    for s, e in regions:
        if max_sec <= 0 or e - s <= max_sec:
            out.append((s, e))
        else:
            out.extend((t.start, t.end) for t in split_long([RawTurn(s, e, "x")], max_sec, audio))
    return out


async def transcribe_segments(
    asr: AsrClient,
    audio: np.ndarray,
    duration: float,
    segments: list[Segment],
    on_batch_done: Callable[[int, int], Awaitable[None]] | None = None,
    fallback_langs: list[str] | None = None,
    raw_audio: np.ndarray | None = None,
    vad=None,
) -> None:
    cfg = settings.asr
    # (segment, chunk index, start, end)
    jobs: dict[str, list[tuple[Segment, int, float, float]]] = {}
    for seg in segments:
        s, e = padded_bounds(seg, segments, settings.segments.pad_sec, duration)
        for i, (a, b) in enumerate(chunk_bounds(s, e, cfg.max_chunk_sec, audio, vad if cfg.vad_chunks else None)):
            jobs.setdefault(seg.language or "ru", []).append((seg, i, a, b))

    total_batches = sum((len(v) + BATCH_SIZE - 1) // BATCH_SIZE for v in jobs.values())
    done = 0
    lock = asyncio.Lock()
    texts: dict[tuple[int, int], str] = {}
    fallback_used = 0
    raw_used = 0

    async def run_lang(lang: str, items: list[tuple[Segment, int, float, float]]):
        nonlocal done, fallback_used, raw_used
        others = [x for x in (fallback_langs or []) if x != lang]
        for i in range(0, len(items), BATCH_SIZE):
            batch = items[i : i + BATCH_SIZE]
            wavs = [slice_to_wav_bytes(audio, a, b) for _, _, a, b in batch]
            results = await asr.transcribe_batch(lang, wavs)
            retry = [(j, wavs[j]) for j, r in enumerate(results) if not has_words(r.text)]
            # 1) тот же язык, но исходный (неулучшенный) сигнал
            if retry and cfg.raw_fallback and raw_audio is not None and raw_audio is not audio:
                raw_wavs = [slice_to_wav_bytes(raw_audio, batch[j][2], batch[j][3]) for j, _ in retry]
                alt = await asr.transcribe_batch(lang, raw_wavs)
                still = []
                for (j, w), r in zip(retry, alt, strict=True):
                    if has_words(r.text):
                        results[j] = r
                        raw_used += 1
                    else:
                        still.append((j, w))
                retry = still
            # 2) другие языки файла
            if retry and others and cfg.fallback_langs:
                for other in others:
                    alt = await asr.transcribe_batch(other, [w for _, w in retry])
                    still = []
                    for (j, w), r in zip(retry, alt, strict=True):
                        if has_words(r.text):
                            results[j] = r
                            fallback_used += 1
                        else:
                            still.append((j, w))
                    retry = still
                    if not retry:
                        break
            for (seg, ci, _, _), res in zip(batch, results, strict=True):
                texts[(seg.id, ci)] = res.text
            async with lock:
                done += 1
                if on_batch_done:
                    await on_batch_done(done, total_batches)

    await asyncio.gather(*(run_lang(lang, items) for lang, items in jobs.items()))

    for seg in segments:
        parts = [texts[k] for k in sorted(k for k in texts if k[0] == seg.id)]
        seg.text = " ".join(p.strip() for p in parts if has_words(p))
    n_chunks = sum(len(v) for v in jobs.values())
    logger.info(
        "ASR done: %d segments → %d chunks (≤%.0fs) in %d batches, languages=%s, raw-fallback hits=%d, lang-fallback hits=%d",
        len(segments), n_chunks, cfg.max_chunk_sec, total_batches, list(jobs), raw_used, fallback_used,
    )
