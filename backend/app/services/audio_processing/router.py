"""§5.6: группировка сегментов по языку, батчи, параллельность, прогресс.
Язык каждого сегмента уже известен (LID по спикеру) — ASR-API получает
готовую пару lang + аудио."""

import asyncio
import logging
from collections.abc import Awaitable, Callable

import numpy as np

from app.core.config import settings
from app.services.audio_processing.asr.base import AsrClient
from app.services.audio_processing.ingest import slice_to_wav_bytes
from app.services.audio_processing.segments import padded_bounds
from app.services.audio_processing.types import Segment

logger = logging.getLogger(__name__)

BATCH_SIZE = 8


async def transcribe_segments(
    asr: AsrClient,
    audio: np.ndarray,
    duration: float,
    segments: list[Segment],
    on_batch_done: Callable[[int, int], Awaitable[None]] | None = None,
) -> None:
    by_lang: dict[str, list[Segment]] = {}
    for seg in segments:
        by_lang.setdefault(seg.language or "ru", []).append(seg)

    total_batches = sum((len(v) + BATCH_SIZE - 1) // BATCH_SIZE for v in by_lang.values())
    done = 0
    lock = asyncio.Lock()

    async def run_lang(lang: str, segs: list[Segment]):
        nonlocal done
        for i in range(0, len(segs), BATCH_SIZE):
            batch = segs[i : i + BATCH_SIZE]
            items = []
            offsets = []
            for seg in batch:
                s, e = padded_bounds(seg, segments, settings.segments.pad_sec, duration)
                items.append(slice_to_wav_bytes(audio, s, e))
                offsets.append(s)
            results = await asr.transcribe_batch(lang, items)
            for seg, res, off in zip(batch, results, offsets, strict=False):
                seg.text = res.text
                if res.words:
                    seg.words = [
                        {"word": w.word, "start": round(max(seg.start, off + w.start), 2),
                         "end": round(min(seg.end, off + w.end), 2)}
                        for w in res.words
                    ]
            async with lock:
                done += 1
                if on_batch_done:
                    await on_batch_done(done, total_batches)

    await asyncio.gather(*(run_lang(lang, segs) for lang, segs in by_lang.items()))
    logger.info("ASR done: %d segments in %d batches, languages=%s", len(segments), total_batches, list(by_lang))
