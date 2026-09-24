"""§5.6: куски → ASR по языку куска, батчи, параллельность, прогресс.

Куски (Chunk) режутся из сегментов диаризации по речевым областям silero-VAD,
не длиннее ASR_MAX_CHUNK_SEC. Язык куска уже назначен (langid.label_chunks
или, в legacy-режиме, язык спикера).

Повторы для куска:
1. пустой ответ → тот же язык на исходном (неулучшенном) сигнале;
2. пустой ответ → другие языки файла по убыванию вероятности;
3. текст есть, но textcheck говорит «не тот язык» (транслитерация) → повтор
   подсказанным языком, затем остальными; кандидат принимается, только если
   сам проходит проверку; среди прошедших выбирается самый правдоподобный.
   Исходный текст без прошедших кандидатов остаётся как есть (никогда не
   удаляем текст из-за проверки).
Кандидат из п.2 тоже проходит проверку текста: модель чужого языка на тихом
куске может «дорисовать» речь. Повторы по кускам идут параллельно.
Отвергнутый бэкендом кусок (4xx) не валит батч и не выключает язык; язык
выбывает для файла, только если бэкенд отверг сам язык."""

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable

import numpy as np

from app.core.config import settings
from app.services.audio_processing.asr.base import AsrClient, AsrResult
from app.services.audio_processing.ingest import SAMPLE_RATE, slice_to_wav_bytes
from app.services.audio_processing.segments import padded_bounds, split_long
from app.services.audio_processing.textcheck import has_words  # noqa: F401 — реэкспорт
from app.services.audio_processing.types import Chunk, RawTurn, Segment

logger = logging.getLogger(__name__)

BATCH_SIZE = 8


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


def plan_chunks(segments: list[Segment], audio: np.ndarray, duration: float, vad=None) -> list[Chunk]:
    """Куски для всех сегментов; записываются и в seg.chunks."""
    cfg = settings.asr
    chunks: list[Chunk] = []
    for seg in segments:
        s, e = padded_bounds(seg, segments, settings.segments.pad_sec, duration)
        seg.chunks = []
        for i, (a, b) in enumerate(chunk_bounds(s, e, cfg.max_chunk_sec, audio, vad if cfg.vad_chunks else None)):
            ch = Chunk(seg_id=seg.id, idx=i, start=a, end=b, speaker=seg.speaker, overlap=seg.overlap,
                       seg_start=seg.start, seg_end=seg.end, language=seg.language, lang_confidence=seg.lang_confidence)
            seg.chunks.append(ch)
            chunks.append(ch)
    return chunks


def _ranked_langs(ch: Chunk, served: list[str], first: str | None = None, exclude: set[str] | None = None) -> list[str]:
    """Языки-кандидаты для куска, кроме текущего и выбывших: подсказка, затем по убыванию вероятности."""
    skip = set(exclude or ())
    order = []
    if first and first in served and first != ch.language and first not in skip:
        order.append(first)
    probs = ch.probs or {}
    rest = sorted(
        (x for x in served if x != ch.language and x not in order and x not in skip), key=lambda x: -probs.get(x, 0.0)
    )
    return order + rest


def aggregate_segments(segments: list[Segment]) -> None:
    """Текст и язык сегмента из его кусков: язык — большинство по времени."""
    for seg in segments:
        parts = [c.text.strip() for c in seg.chunks if has_words(c.text)]
        seg.text = " ".join(parts)
        mass: dict[str, float] = {}
        conf: dict[str, float] = {}
        for c in seg.chunks:
            if not c.language:
                continue
            mass[c.language] = mass.get(c.language, 0.0) + c.duration
            conf[c.language] = conf.get(c.language, 0.0) + c.lang_confidence * c.duration
        if mass:
            lang = max(mass, key=mass.get)
            seg.language = lang
            seg.lang_confidence = round(conf[lang] / mass[lang], 4) if mass[lang] else 0.0


# Ответ 4xx, по которому видно, что бэкенд не обслуживает сам язык (а не этот кусок).
_LANG_REJECT_RE = re.compile(r"language|lang_code|язык|unsupported|not supported|not loaded", re.I)


async def transcribe_segments(
    asr: AsrClient,
    audio: np.ndarray,
    duration: float,
    segments: list[Segment],
    chunks: list[Chunk] | None = None,
    on_batch_done: Callable[[int, int], Awaitable[None]] | None = None,
    raw_audio: np.ndarray | None = None,
    vad=None,
    textcheck=None,
    languages_served: list[str] | None = None,
) -> dict:
    """Распознаёт куски и собирает seg.text/seg.language. Возвращает статистику.
    on_batch_done(готово кусков, всего кусков)."""
    cfg = settings.asr
    if chunks is None:
        chunks = plan_chunks(segments, audio, duration, vad)
    served = list(languages_served or [x for x in settings.langid.allowed if x in cfg.languages]) or list(settings.langid.allowed)
    use_check = bool(cfg.text_check and textcheck is not None and getattr(textcheck, "loaded", False))

    def check(text: str, lang: str, ch: Chunk):
        return textcheck.check(
            text, lang, ch.duration, cfg.text_check_min_tokens, cfg.text_check_min_cps,
            min_density_sec=cfg.text_check_density_min_sec,
        )

    # Язык, который бэкенд не обслуживает, заменяем лучшим обслуживаемым по вероятности.
    for ch in chunks:
        if ch.language not in served and served:
            probs = ch.probs or {}
            ch.lid_language = ch.language
            ch.language = max(served, key=lambda x: probs.get(x, 0.0))
            ch.lang_confidence = round(probs.get(ch.language, 0.0), 4)
            ch.lang_source = "served_fallback"
            ch.warning = f"lang_unserved:{ch.lid_language}"

    jobs: dict[str, list[Chunk]] = {}
    for ch in chunks:
        jobs.setdefault(ch.language or served[0], []).append(ch)

    done = 0
    lock = asyncio.Lock()
    stats = {"chunks": len(chunks), "raw_fallback": 0, "lang_fallback": 0, "text_flagged": 0,
             "text_replaced": 0, "extra_calls": 0, "rejected_chunks": 0, "unsupported": []}
    dead_langs: set[str] = set()

    def mark_dead(lang: str, why: str) -> None:
        if lang not in dead_langs:
            logger.warning("ASR does not serve language %s: %s", lang, why[:160])
            dead_langs.add(lang)
            stats["unsupported"].append(lang)

    async def safe_batch(lang: str, wavs: list[bytes]) -> list[AsrResult]:
        """Всегда len(wavs) результатов. Отвергнутый кусок → text="" + error; язык
        выбывает для всего файла, только если бэкенд отверг сам язык (или весь батч)."""
        if lang in dead_langs:
            return [AsrResult(text="", error="language not served") for _ in wavs]
        try:
            results = list(await asr.transcribe_batch(lang, wavs))
        except Exception as e:  # noqa: BLE001
            if "rejected" in str(e).lower():
                # клиент без поштучных ошибок: 4xx на весь батч = язык не обслуживается
                mark_dead(lang, str(e))
                return [AsrResult(text="", error=str(e)) for _ in wavs]
            raise
        errors = [r.error for r in results if r.error]
        if errors:
            stats["rejected_chunks"] += len(errors)
            if any(_LANG_REJECT_RE.search(x) for x in errors) or (len(errors) == len(results) and len(results) >= 2):
                mark_dead(lang, errors[0])
        return results

    def wav(ch: Chunk, src: np.ndarray) -> bytes:
        return slice_to_wav_bytes(src, ch.start, ch.end)

    async def try_lang(ch: Chunk, lang: str) -> str:
        """Один язык: улучшенный сигнал, при пустоте — исходный."""
        res = await safe_batch(lang, [wav(ch, audio)])
        stats["extra_calls"] += 1
        text = res[0].text
        if not has_words(text) and not res[0].error and cfg.raw_fallback and raw_audio is not None and raw_audio is not audio:
            res = await safe_batch(lang, [wav(ch, raw_audio)])
            stats["extra_calls"] += 1
            text = res[0].text
        return text

    async def retry_empty(ch: Chunk) -> None:
        """2) пусто → другие языки по вероятности; кандидат принимается, только если
        проходит проверку текста (иначе модель чужого языка «дорисовывает» речь)."""
        for other in _ranked_langs(ch, served, exclude=dead_langs)[: cfg.retry_langs_max]:
            t = await try_lang(ch, other)
            if not has_words(t):
                continue
            if use_check and check(t, other, ch).wrong:
                continue
            ch.text, ch.language, ch.lang_source = t, other, "asr_fallback"
            ch.lang_confidence = 1.0 if use_check else round((ch.probs or {}).get(other, 0.0), 4)
            stats["lang_fallback"] += 1
            return

    async def retry_flagged(ch: Chunk, reason: str, suggest: str | None) -> None:
        """3) текст есть, но не того языка → повтор подсказанным языком, затем остальными."""
        stats["text_flagged"] += 1
        candidates: dict[str, str] = {}
        for other in _ranked_langs(ch, served, first=suggest, exclude=dead_langs)[: cfg.retry_langs_max]:
            t = await try_lang(ch, other)
            if has_words(t) and not check(t, other, ch).wrong:
                candidates[other] = t
                break  # первый прошедший проверку кандидат — достаточно
        if candidates:
            candidates[ch.language] = ch.text
            best = textcheck.pick(candidates, original=ch.language)
            if best and best != ch.language:
                ch.text, ch.language, ch.lang_source = candidates[best], best, "asr_text"
                ch.lang_confidence = 1.0
                stats["text_replaced"] += 1
                return
        ch.warning = f"text_check:{reason}"

    async def handle(ch: Chunk, res: AsrResult) -> None:
        ch.text = res.text or ""
        if res.error and not _LANG_REJECT_RE.search(res.error) and ch.language not in dead_langs:
            # отвергнут именно кусок (слишком короткий/длинный, формат) — другие языки не помогут
            ch.warning = "asr_rejected"
            return
        if not has_words(ch.text):
            if cfg.fallback_langs:
                await retry_empty(ch)
            return
        if not use_check:
            return
        chk = check(ch.text, ch.language, ch)
        if chk.wrong:
            await retry_flagged(ch, chk.reason, chk.suggest)

    async def run_lang(lang: str, items: list[Chunk]):
        nonlocal done
        for i in range(0, len(items), BATCH_SIZE):
            batch = items[i : i + BATCH_SIZE]
            results = await safe_batch(lang, [wav(ch, audio) for ch in batch])

            # 1) пусто → тот же язык на исходном сигнале
            if cfg.raw_fallback and raw_audio is not None and raw_audio is not audio and lang not in dead_langs:
                idx = [j for j, r in enumerate(results) if not has_words(r.text) and not r.error]
                if idx:
                    alt = await safe_batch(lang, [wav(batch[j], raw_audio) for j in idx])
                    for j, r in zip(idx, alt, strict=True):
                        if has_words(r.text):
                            results[j] = r
                            stats["raw_fallback"] += 1

            # 2–3) повторы по кускам — параллельно (ограничены семафором клиента)
            await asyncio.gather(*(handle(ch, r) for ch, r in zip(batch, results, strict=True)))
            async with lock:
                done += len(batch)
                if on_batch_done:
                    await on_batch_done(done, len(chunks))

    await asyncio.gather(*(run_lang(lang, items) for lang, items in jobs.items()))
    aggregate_segments(segments)
    logger.info(
        "ASR done: %d segments → %d chunks (≤%.0fs), languages=%s, raw-fallback=%d, lang-fallback=%d, "
        "text flagged=%d replaced=%d, extra calls=%d, rejected=%d, unsupported=%s",
        len(segments), len(chunks), cfg.max_chunk_sec, sorted(jobs), stats["raw_fallback"],
        stats["lang_fallback"], stats["text_flagged"], stats["text_replaced"], stats["extra_calls"],
        stats["rejected_chunks"], stats["unsupported"],
    )
    return stats
