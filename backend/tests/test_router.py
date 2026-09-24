import asyncio

import numpy as np
import pytest

from app.core.config import settings
from app.services.audio_processing.asr.base import AsrResult
from app.services.audio_processing.router import (
    aggregate_segments,
    chunk_bounds,
    has_words,
    plan_chunks,
    transcribe_segments,
)
from app.services.audio_processing.textcheck import CheckResult
from app.services.audio_processing.types import Chunk, Segment


class FakeAsr:
    """answers: lang -> очередь текстов по порядку вызовов."""

    name = "fake"

    def __init__(self, answers):
        self.answers = {k: list(v) for k, v in answers.items()}
        self.calls = []

    async def transcribe(self, lang, wav):
        return (await self.transcribe_batch(lang, [wav]))[0]

    async def transcribe_batch(self, lang, items):
        out = []
        for _ in items:
            self.calls.append(lang)
            q = self.answers.get(lang, [])
            out.append(AsrResult(text=q.pop(0) if q else ""))
        return out

    async def aclose(self):
        pass


class RejectingAsr(FakeAsr):
    """Как remote_api при 4xx: язык не обслуживается."""

    def __init__(self, answers, rejected):
        super().__init__(answers)
        self.rejected = rejected

    async def transcribe_batch(self, lang, items):
        if lang in self.rejected:
            self.calls.append(lang)
            raise RuntimeError(f"ASR API rejected request (HTTP 400): {lang}")
        return await super().transcribe_batch(lang, items)


class FakeTextCheck:
    """wrong: set of (text) that are flagged; suggest: dict text -> lang."""

    loaded = True

    def __init__(self, wrong=(), suggest=None, scores=None):
        self.wrong, self.suggest, self.scores = set(wrong), suggest or {}, scores or {}

    def check(self, text, lang, dur, min_tokens=3, min_cps=4.0, **kw):
        if text in self.wrong:
            return CheckResult(wrong=True, reasons=["lm_margin"], suggest=self.suggest.get(text))
        return CheckResult(wrong=False)

    def pick(self, candidates, original=None, **kw):
        return max(candidates, key=lambda lang: self.scores.get(candidates[lang], 0.0))


class PerChunkRejectingAsr(FakeAsr):
    """Как новый remote_api: 4xx на кусок → AsrResult(error=...), батч не падает."""

    def __init__(self, answers, reject_calls, message):
        super().__init__(answers)
        self.reject_calls, self.message = set(reject_calls), message

    async def transcribe_batch(self, lang, items):
        out = []
        for _ in items:
            n = len(self.calls)
            self.calls.append(lang)
            if n in self.reject_calls:
                out.append(AsrResult(text="", error=self.message))
            else:
                q = self.answers.get(lang, [])
                out.append(AsrResult(text=q.pop(0) if q else ""))
        return out


def _seg(i, spk, a, b, lang="uz"):
    return Segment(i, spk, a, b, language=lang)


def _chunks_for(segs):
    out = []
    for s in segs:
        c = Chunk(seg_id=s.id, idx=0, start=s.start, end=s.end, speaker=s.speaker, language=s.language, lang_confidence=1.0, lang_source="own")
        s.chunks = [c]
        out.append(c)
    return out


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    monkeypatch.setattr(settings.asr, "raw_fallback", False)
    monkeypatch.setattr(settings.asr, "fallback_langs", True)
    monkeypatch.setattr(settings.asr, "text_check", True)
    monkeypatch.setattr(settings.asr, "retry_langs_max", 2)
    monkeypatch.setattr(settings.asr, "vad_chunks", False)
    monkeypatch.setattr(settings.asr, "max_chunk_sec", 8.0)


def test_has_words_ignores_punctuation_only():
    assert not has_words(".") and not has_words(" . ") and not has_words("") and not has_words(None)
    assert has_words("a.") and has_words("Да.")


def test_chunk_bounds_splits_long():
    parts = chunk_bounds(0.0, 20.0, 8.0, None)
    assert len(parts) == 3 and parts[0][0] == 0.0 and parts[-1][1] == 20.0
    assert all(b - a <= 8.0 + 1e-6 for a, b in parts)
    assert chunk_bounds(0.0, 5.0, 8.0, None) == [(0.0, 5.0)]


class FakeVad:
    loaded = True

    def speech_regions(self, audio, pad_sec=0.3, min_silence_sec=0.35, min_speech_sec=0.3):
        return [(0.5, 3.0), (6.0, 9.5)]


def test_chunk_bounds_uses_vad_regions():
    audio = np.zeros(16000 * 30, dtype="float32")
    assert chunk_bounds(10.0, 20.0, 8.0, audio, vad=FakeVad()) == [(10.5, 13.0), (16.0, 19.5)]


def test_plan_chunks_carries_segment_language_and_speaker(monkeypatch):
    audio = np.zeros(16000 * 30, dtype="float32")
    segs = [_seg(0, "A", 0.0, 20.0, "ru"), _seg(1, "B", 21.0, 25.0, "uz")]
    chunks = plan_chunks(segs, audio, 30.0)
    assert [c.seg_id for c in chunks] == [0, 0, 0, 1] and chunks[0].speaker == "A" and chunks[-1].language == "uz"
    assert segs[0].chunks == chunks[:3]


@pytest.mark.asyncio
async def test_chunks_grouped_by_chunk_language_not_segment():
    audio = np.zeros(16000 * 10, dtype="float32")
    seg = _seg(0, "A", 0.0, 9.0, "uz")
    c1 = Chunk(0, 0, 0.0, 4.0, "A", language="ru", lang_confidence=0.99, lang_source="own")
    c2 = Chunk(0, 1, 5.0, 9.0, "A", language="uz", lang_confidence=0.99, lang_source="own")
    seg.chunks = [c1, c2]
    asr = FakeAsr({"ru": ["привет"], "uz": ["salom"]})
    await transcribe_segments(asr, audio, 10.0, [seg], chunks=[c1, c2], textcheck=FakeTextCheck(), languages_served=["uz", "ru"])
    assert sorted(asr.calls) == ["ru", "uz"]
    assert seg.text == "привет salom" and seg.language == "ru"  # 4 с ru против 4 с uz — первый по порядку при равенстве
    assert c1.text == "привет" and c2.text == "salom"


@pytest.mark.asyncio
async def test_segment_text_is_joined_from_chunks():
    audio = np.zeros(16000 * 30, dtype="float32")
    segs = [_seg(0, "A", 0.0, 20.0), _seg(1, "A", 21.0, 25.0)]
    asr = FakeAsr({"uz": ["bir", "ikki", "uch", "to'rt"]})
    await transcribe_segments(asr, audio, 30.0, segs, textcheck=FakeTextCheck(), languages_served=["uz", "ru"])
    assert segs[0].text == "bir ikki uch" and segs[1].text == "to'rt"
    assert asr.calls == ["uz"] * 4


@pytest.mark.asyncio
async def test_empty_chunk_retried_on_other_language_by_probability():
    audio = np.zeros(16000 * 10, dtype="float32")
    seg = _seg(0, "A", 0.0, 5.0, "uz")
    ch = _chunks_for([seg])[0]
    ch.probs = {"uz": 0.6, "ru": 0.1, "en": 0.3}
    asr = FakeAsr({"uz": [""], "en": [""], "ru": ["привет"]})
    await transcribe_segments(asr, audio, 10.0, [seg], chunks=[ch], textcheck=FakeTextCheck(), languages_served=["uz", "ru", "en"])
    assert asr.calls == ["uz", "en", "ru"] and seg.text == "привет" and ch.language == "ru" and ch.lang_source == "asr_fallback"


@pytest.mark.asyncio
async def test_dot_only_answer_counts_as_empty():
    audio = np.zeros(16000 * 10, dtype="float32")
    seg = _seg(0, "A", 0.0, 5.0, "uz")
    ch = _chunks_for([seg])[0]
    asr = FakeAsr({"uz": ["."], "ru": ["привет"]})
    await transcribe_segments(asr, audio, 10.0, [seg], chunks=[ch], textcheck=FakeTextCheck(), languages_served=["uz", "ru"])
    assert seg.text == "привет"


@pytest.mark.asyncio
async def test_raw_audio_retry_before_other_language(monkeypatch):
    monkeypatch.setattr(settings.asr, "raw_fallback", True)
    enhanced = np.zeros(16000 * 10, dtype="float32")
    raw = np.ones(16000 * 10, dtype="float32") * 0.01
    seg = _seg(0, "A", 0.0, 5.0, "uz")
    ch = _chunks_for([seg])[0]
    asr = FakeAsr({"uz": ["", "salom"], "ru": ["привет"]})
    await transcribe_segments(asr, enhanced, 10.0, [seg], chunks=[ch], raw_audio=raw, textcheck=FakeTextCheck(), languages_served=["uz", "ru"])
    assert seg.text == "salom" and asr.calls == ["uz", "uz"]


@pytest.mark.asyncio
async def test_wrong_language_text_is_retried_and_replaced():
    """Русская речь через uz-модель: транслитерация помечена, повтор ru принят."""
    audio = np.zeros(16000 * 10, dtype="float32")
    seg = _seg(0, "A", 0.0, 5.0, "uz")
    ch = _chunks_for([seg])[0]
    garbage = "Zagruziti zapes Strechi"
    asr = FakeAsr({"uz": [garbage], "ru": ["Загрузите запись встречи"]})
    tc = FakeTextCheck(wrong={garbage}, suggest={garbage: "ru"}, scores={"Загрузите запись встречи": -3.5, garbage: -7.0})
    stats = await transcribe_segments(asr, audio, 10.0, [seg], chunks=[ch], textcheck=tc, languages_served=["uz", "ru", "en"])
    assert asr.calls == ["uz", "ru"]  # подсказка ru — первой, en не понадобился
    assert seg.text == "Загрузите запись встречи" and ch.language == "ru" and ch.lang_source == "asr_text"
    assert seg.language == "ru" and stats["text_replaced"] == 1 and stats["text_flagged"] == 1


@pytest.mark.asyncio
async def test_flagged_text_kept_when_no_candidate_passes():
    audio = np.zeros(16000 * 10, dtype="float32")
    seg = _seg(0, "A", 0.0, 5.0, "uz")
    ch = _chunks_for([seg])[0]
    asr = FakeAsr({"uz": ["Da."], "ru": ["Да."]})
    tc = FakeTextCheck(wrong={"Da.", "Да."})  # оба не проходят (короткие)
    await transcribe_segments(asr, audio, 10.0, [seg], chunks=[ch], textcheck=tc, languages_served=["uz", "ru"])
    assert seg.text == "Da." and ch.language == "uz" and ch.warning and ch.warning.startswith("text_check:")


@pytest.mark.asyncio
async def test_retry_limited_by_retry_langs_max(monkeypatch):
    monkeypatch.setattr(settings.asr, "retry_langs_max", 1)
    audio = np.zeros(16000 * 10, dtype="float32")
    seg = _seg(0, "A", 0.0, 5.0, "uz")
    ch = _chunks_for([seg])[0]
    ch.probs = {"uz": 0.5, "ru": 0.4, "en": 0.1}
    asr = FakeAsr({"uz": [""], "ru": [""], "en": ["hello"]})
    await transcribe_segments(asr, audio, 10.0, [seg], chunks=[ch], textcheck=FakeTextCheck(), languages_served=["uz", "ru", "en"])
    assert asr.calls == ["uz", "ru"] and seg.text == ""


@pytest.mark.asyncio
async def test_rejected_language_is_dropped_not_fatal():
    audio = np.zeros(16000 * 10, dtype="float32")
    seg = _seg(0, "A", 0.0, 5.0, "en")
    ch = _chunks_for([seg])[0]
    ch.probs = {"uz": 0.2, "ru": 0.7, "en": 0.1}
    asr = RejectingAsr({"ru": ["привет"]}, rejected={"en"})
    stats = await transcribe_segments(asr, audio, 10.0, [seg], chunks=[ch], textcheck=FakeTextCheck(), languages_served=["uz", "ru", "en"])
    assert seg.text == "привет" and ch.language == "ru" and stats["unsupported"] == ["en"]


@pytest.mark.asyncio
async def test_unserved_language_rerouted_to_best_served():
    audio = np.zeros(16000 * 10, dtype="float32")
    seg = _seg(0, "A", 0.0, 5.0, "en")
    ch = _chunks_for([seg])[0]
    ch.probs = {"uz": 0.2, "ru": 0.7, "en": 0.1}
    asr = FakeAsr({"ru": ["привет"]})
    await transcribe_segments(asr, audio, 10.0, [seg], chunks=[ch], textcheck=FakeTextCheck(), languages_served=["uz", "ru"])
    assert asr.calls == ["ru"] and ch.language == "ru"
    assert ch.lid_language == "en" and ch.lang_source == "served_fallback" and ch.warning == "lang_unserved:en"
    assert ch.lang_confidence == 0.7  # уверенность — вероятность НОВОГО языка, а не 1.0 от en


@pytest.mark.asyncio
async def test_empty_fallback_candidate_must_pass_text_gate():
    """Тихий uz-кусок: ru-модель «дорисовала» «Угу.» — без проверки текста это стало бы ru."""
    audio = np.zeros(16000 * 10, dtype="float32")
    seg = _seg(0, "A", 0.0, 5.0, "uz")
    ch = _chunks_for([seg])[0]
    ch.probs = {"uz": 0.97, "ru": 0.02, "en": 0.01}
    asr = FakeAsr({"uz": [""], "ru": ["Угу."], "en": [""]})
    tc = FakeTextCheck(wrong={"Угу."})
    stats = await transcribe_segments(asr, audio, 10.0, [seg], chunks=[ch], textcheck=tc, languages_served=["uz", "ru", "en"])
    assert seg.text == "" and ch.language == "uz" and ch.lang_source == "own" and ch.lang_confidence == 1.0
    assert stats["lang_fallback"] == 0


@pytest.mark.asyncio
async def test_empty_fallback_sets_confidence_from_text_evidence():
    audio = np.zeros(16000 * 10, dtype="float32")
    seg = _seg(0, "A", 0.0, 5.0, "uz")
    ch = _chunks_for([seg])[0]
    ch.lang_confidence = 0.97
    ch.probs = {"uz": 0.97, "ru": 0.02, "en": 0.01}
    asr = FakeAsr({"uz": [""], "ru": ["Загрузите запись встречи"]})
    await transcribe_segments(asr, audio, 10.0, [seg], chunks=[ch], textcheck=FakeTextCheck(), languages_served=["uz", "ru"])
    assert ch.language == "ru" and ch.lang_source == "asr_fallback" and ch.lang_confidence == 1.0


@pytest.mark.asyncio
async def test_dead_language_does_not_consume_retry_slot(monkeypatch):
    monkeypatch.setattr(settings.asr, "retry_langs_max", 1)
    audio = np.zeros(16000 * 20, dtype="float32")
    seg_en, seg_uz = _seg(0, "A", 0.0, 5.0, "en"), _seg(1, "A", 6.0, 11.0, "uz")
    ch_en, ch_uz = _chunks_for([seg_en, seg_uz])
    ch_uz.probs = {"uz": 0.6, "en": 0.3, "ru": 0.1}
    asr = RejectingAsr({"uz": [""], "ru": ["привет"]}, rejected={"en"})
    await transcribe_segments(asr, audio, 20.0, [seg_en, seg_uz], chunks=[ch_en, ch_uz], textcheck=FakeTextCheck(), languages_served=["uz", "ru", "en"])
    # en выбыл после первого же батча → единственный слот повтора достаётся ru, а не мёртвому en
    assert seg_uz.text == "привет" and ch_uz.language == "ru"


@pytest.mark.asyncio
async def test_single_chunk_rejection_does_not_kill_language_or_batch():
    audio = np.zeros(16000 * 20, dtype="float32")
    segs = [_seg(0, "A", 0.0, 4.0, "ru"), _seg(1, "A", 5.0, 5.4, "ru"), _seg(2, "A", 6.0, 10.0, "ru")]
    chunks = _chunks_for(segs)
    asr = PerChunkRejectingAsr({"ru": ["раз", "три"], "uz": ["bir"]}, reject_calls={1}, message="HTTP 422: audio too short")
    stats = await transcribe_segments(asr, audio, 20.0, segs, chunks=chunks, textcheck=FakeTextCheck(), languages_served=["uz", "ru"])
    assert [s.text for s in segs] == ["раз", "", "три"]
    assert chunks[1].warning == "asr_rejected" and chunks[1].language == "ru"
    assert stats["unsupported"] == [] and stats["rejected_chunks"] == 1 and "uz" not in asr.calls


@pytest.mark.asyncio
async def test_language_rejection_message_marks_language_dead():
    audio = np.zeros(16000 * 20, dtype="float32")
    seg = _seg(0, "A", 0.0, 4.0, "en")
    ch = _chunks_for([seg])[0]
    ch.probs = {"uz": 0.2, "ru": 0.7, "en": 0.1}
    asr = PerChunkRejectingAsr({"ru": ["привет"]}, reject_calls={0}, message="HTTP 400: unsupported language_code 'en'")
    stats = await transcribe_segments(asr, audio, 20.0, [seg], chunks=[ch], textcheck=FakeTextCheck(), languages_served=["uz", "ru", "en"])
    assert stats["unsupported"] == ["en"] and seg.text == "привет" and ch.language == "ru"


@pytest.mark.asyncio
async def test_progress_counts_chunks_not_batches():
    audio = np.zeros(16000 * 30, dtype="float32")
    segs = [_seg(0, "A", 0.0, 20.0), _seg(1, "A", 21.0, 25.0)]
    seen = []

    async def on_done(done, total):
        seen.append((done, total))

    asr = FakeAsr({"uz": ["bir", "ikki", "uch", "to'rt"]})
    await transcribe_segments(asr, audio, 30.0, segs, on_batch_done=on_done, textcheck=FakeTextCheck(), languages_served=["uz", "ru"])
    assert seen[-1] == (4, 4)


class SlowAsr(FakeAsr):
    """Считает одновременные запросы: повторы разных кусков должны идти параллельно."""

    def __init__(self, answers):
        super().__init__(answers)
        self.inflight = self.max_inflight = 0

    async def transcribe_batch(self, lang, items):
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        await asyncio.sleep(0.01)
        try:
            return await super().transcribe_batch(lang, items)
        finally:
            self.inflight -= 1


@pytest.mark.asyncio
async def test_per_chunk_retries_run_concurrently():
    audio = np.zeros(16000 * 30, dtype="float32")
    segs = [_seg(i, "A", i * 5.0, i * 5.0 + 4.0, "uz") for i in range(4)]
    chunks = _chunks_for(segs)
    for c in chunks:
        c.probs = {"uz": 0.5, "ru": 0.5, "en": 0.0}
    asr = SlowAsr({"uz": ["", "", "", ""], "ru": ["а", "б", "в", "г"]})
    await transcribe_segments(asr, audio, 30.0, segs, chunks=chunks, textcheck=FakeTextCheck(), languages_served=["uz", "ru"])
    assert [s.text for s in segs] == ["а", "б", "в", "г"] and asr.max_inflight >= 2


def test_aggregate_segments_majority_language_and_confidence():
    seg = _seg(0, "A", 0.0, 10.0)
    seg.chunks = [
        Chunk(0, 0, 0.0, 6.0, "A", language="ru", lang_confidence=0.9, text="а"),
        Chunk(0, 1, 6.0, 10.0, "A", language="uz", lang_confidence=1.0, text="b"),
    ]
    aggregate_segments([seg])
    assert seg.language == "ru" and seg.lang_confidence == 0.9 and seg.text == "а b"
