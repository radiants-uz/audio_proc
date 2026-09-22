import numpy as np
import pytest

from app.core.config import settings
from app.services.audio_processing.asr.base import AsrResult
from app.services.audio_processing.router import chunk_bounds, has_words, transcribe_segments
from app.services.audio_processing.types import Segment


class FakeAsr:
    name = "fake"

    def __init__(self, answers):
        self.answers = answers  # lang -> list of texts by call order
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


def test_chunk_bounds_splits_long():
    parts = chunk_bounds(0.0, 20.0, 8.0, None)
    assert len(parts) == 3 and parts[0][0] == 0.0 and parts[-1][1] == 20.0
    assert all(b - a <= 8.0 + 1e-6 for a, b in parts)
    assert chunk_bounds(0.0, 5.0, 8.0, None) == [(0.0, 5.0)]


@pytest.mark.asyncio
async def test_segment_text_is_joined_from_chunks(monkeypatch):
    monkeypatch.setattr(settings.asr, "max_chunk_sec", 8.0)
    monkeypatch.setattr(settings.asr, "fallback_langs", False)
    audio = np.zeros(16000 * 30, dtype="float32")
    segs = [Segment(0, "S0", 0.0, 20.0, language="uz"), Segment(1, "S1", 21.0, 25.0, language="uz")]
    asr = FakeAsr({"uz": ["bir", "ikki", "uch", "to'rt"]})
    await transcribe_segments(asr, audio, 30.0, segs)
    assert segs[0].text == "bir ikki uch" and segs[1].text == "to'rt"
    assert asr.calls == ["uz"] * 4


@pytest.mark.asyncio
async def test_empty_chunk_retried_on_fallback_language(monkeypatch):
    monkeypatch.setattr(settings.asr, "max_chunk_sec", 8.0)
    monkeypatch.setattr(settings.asr, "fallback_langs", True)
    audio = np.zeros(16000 * 10, dtype="float32")
    segs = [Segment(0, "S0", 0.0, 5.0, language="uz")]
    asr = FakeAsr({"uz": [""], "ru": ["привет"]})
    await transcribe_segments(asr, audio, 10.0, segs, fallback_langs=["uz", "ru"])
    assert segs[0].text == "привет" and asr.calls == ["uz", "ru"]


@pytest.mark.asyncio
async def test_no_fallback_when_disabled(monkeypatch):
    monkeypatch.setattr(settings.asr, "fallback_langs", False)
    audio = np.zeros(16000 * 10, dtype="float32")
    segs = [Segment(0, "S0", 0.0, 5.0, language="uz")]
    asr = FakeAsr({"uz": [""], "ru": ["привет"]})
    await transcribe_segments(asr, audio, 10.0, segs, fallback_langs=["uz", "ru"])
    assert segs[0].text == "" and asr.calls == ["uz"]


@pytest.mark.asyncio
async def test_empty_chunk_retried_on_raw_audio_before_other_language(monkeypatch):
    monkeypatch.setattr(settings.asr, "raw_fallback", True)
    monkeypatch.setattr(settings.asr, "fallback_langs", True)
    enhanced = np.zeros(16000 * 10, dtype="float32")
    raw = np.ones(16000 * 10, dtype="float32") * 0.01
    segs = [Segment(0, "S0", 0.0, 5.0, language="uz")]
    asr = FakeAsr({"uz": ["", "salom"], "ru": ["привет"]})
    await transcribe_segments(asr, enhanced, 10.0, segs, fallback_langs=["uz", "ru"], raw_audio=raw)
    assert segs[0].text == "salom" and asr.calls == ["uz", "uz"]


def test_has_words_ignores_punctuation_only():
    assert not has_words(".") and not has_words(" . ") and not has_words("") and not has_words(None)
    assert has_words("a.") and has_words("Да.")


@pytest.mark.asyncio
async def test_dot_only_answer_triggers_fallback(monkeypatch):
    monkeypatch.setattr(settings.asr, "raw_fallback", False)
    monkeypatch.setattr(settings.asr, "fallback_langs", True)
    audio = np.zeros(16000 * 10, dtype="float32")
    segs = [Segment(0, "S0", 0.0, 5.0, language="uz")]
    asr = FakeAsr({"uz": ["."], "ru": ["привет"]})
    await transcribe_segments(asr, audio, 10.0, segs, fallback_langs=["uz", "ru"])
    assert segs[0].text == "привет"


class FakeVad:
    loaded = True

    def speech_regions(self, audio, pad_sec=0.3, min_silence_sec=0.35, min_speech_sec=0.3):
        # два речевых куска внутри переданного окна
        return [(0.5, 3.0), (6.0, 9.5)]


def test_chunk_bounds_uses_vad_regions():
    audio = np.zeros(16000 * 30, dtype="float32")
    parts = chunk_bounds(10.0, 20.0, 8.0, audio, vad=FakeVad())
    assert parts == [(10.5, 13.0), (16.0, 19.5)]


def test_chunk_bounds_vad_region_longer_than_max_is_split():
    class LongVad(FakeVad):
        def speech_regions(self, *a, **k):
            return [(0.0, 20.0)]
    parts = chunk_bounds(0.0, 20.0, 8.0, np.zeros(16000 * 20, dtype="float32"), vad=LongVad())
    assert len(parts) == 3 and all(b - a <= 8.0 + 1e-6 for a, b in parts)


def test_chunk_bounds_without_speech_falls_back_to_whole():
    class Silent(FakeVad):
        def speech_regions(self, *a, **k):
            return []
    assert chunk_bounds(0.0, 5.0, 8.0, np.zeros(16000 * 5, dtype="float32"), vad=Silent()) == [(0.0, 5.0)]
