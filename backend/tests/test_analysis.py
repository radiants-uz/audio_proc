import pytest

from app.models.service import Service
from app.schemas.analysis import Analysis, llm_json_schema
from app.services.audio_processing.analysis.analyze import (
    analyze_transcript,
    chunk_lines,
    format_transcript_lines,
    resolve_report_lang,
)
from app.services.audio_processing.analysis.llm_base import LlmResponse, LlmUsage
from app.services.audio_processing.analysis.stats import compute_stats

TRANSCRIPT = {
    "job_id": "j", "title": None,
    "audio": {"filename": "a.wav", "duration_sec": 30, "sample_rate": 16000},
    "speakers": [
        {"id": "SPEAKER_00", "name": None, "language": "ru", "lang_confidence": 0.9, "total_speech_sec": 12, "segments_count": 2},
        {"id": "SPEAKER_01", "name": None, "language": "uz", "lang_confidence": 0.9, "total_speech_sec": 8, "segments_count": 1},
    ],
    "segments": [
        {"id": 0, "speaker": "SPEAKER_00", "language": "ru", "start": 0, "end": 6, "overlap": False, "overlap_with": [], "text": "Начнём.", "words": None},
        {"id": 1, "speaker": "SPEAKER_01", "language": "uz", "start": 6, "end": 14, "overlap": True, "overlap_with": ["SPEAKER_00"], "text": "Boshlaymiz.", "words": None},
        {"id": 2, "speaker": "SPEAKER_00", "language": "ru", "start": 14, "end": 20, "overlap": False, "overlap_with": [], "text": "", "words": None},
        {"id": 3, "speaker": "SPEAKER_00", "language": "ru", "start": 20, "end": 26, "overlap": False, "overlap_with": [], "text": "Решено: старт в понедельник.", "words": None},
    ],
    "meta": {"pipeline_version": "1", "models": {}, "timings_sec": {}, "config": {}, "warnings": []},
}


def test_stats_computed_in_code():
    s = compute_stats(TRANSCRIPT)
    assert s["total_speech_sec"] == 26 and s["speakers_count"] == 2 and s["overlap_segments"] == 1
    assert s["by_speaker"][0]["id"] == "SPEAKER_00" and s["by_speaker"][0]["share"] == round(18 / 26, 3)
    assert s["by_language"][0]["language"] == "ru"


def test_format_lines_skips_empty_and_marks_overlap():
    lines = format_transcript_lines(TRANSCRIPT, {"SPEAKER_01": "Bobur"})
    assert len(lines) == 3
    assert "[overlap]" in lines[1][1] and "Bobur" in lines[1][1]
    assert lines[0][1].startswith("[00:00:00] #0 SPEAKER_00 (ru): Начнём.")


def test_resolve_report_lang_auto_picks_dominant():
    assert resolve_report_lang(TRANSCRIPT, "auto") == "ru"
    assert resolve_report_lang(TRANSCRIPT, "en") == "en"


def test_chunk_lines_respects_budget():
    lines = [(i, "x" * 300) for i in range(20)]  # ~100 токенов каждая
    chunks = chunk_lines(lines, 350)
    assert len(chunks) == 7 and all(len(c) <= 3 for c in chunks)


def test_llm_schema_is_strict_object():
    schema = llm_json_schema()
    assert schema["type"] == "object" and schema["additionalProperties"] is False
    assert "summary" in schema["required"]


class FakeLlm:
    model_name = "fake"

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    async def complete_json(self, system, user, schema, max_tokens):
        self.calls.append(user)
        data = self.answers.pop(0)
        return LlmResponse(data=data, usage=LlmUsage(input_tokens=10, output_tokens=5, calls=1))


def _svc(**kw):
    return Service(name="glm", base_url="http://x/v1", model_name="glm-5.3", limit_requests=1, **kw)


GOOD = {"title": "Планёрка", "summary": "Обсудили.", "topics": [], "decisions": [{"text": "старт", "speaker": "SPEAKER_00", "segment_ids": [3, 99]}],
        "action_items": [], "speakers": [], "open_questions": []}


@pytest.mark.asyncio
async def test_single_call_and_segment_id_filter():
    llm = FakeLlm([GOOD])
    out = await analyze_transcript(llm, _svc(), TRANSCRIPT, "auto")
    assert out["report_lang"] == "ru" and out["meta"]["chunked"] is False
    assert out["decisions"][0]["segment_ids"] == [3]  # 99 отфильтрован
    assert out["stats"]["speakers_count"] == 2 and out["meta"]["input_tokens"] == 10
    Analysis.model_validate(out)


@pytest.mark.asyncio
async def test_invalid_json_repaired_once():
    llm = FakeLlm([{"nonsense": 1}, GOOD])
    out = await analyze_transcript(llm, _svc(), TRANSCRIPT, "ru")
    assert out is not None and len(llm.calls) == 2 and "валидацию" in llm.calls[1]


@pytest.mark.asyncio
async def test_twice_invalid_returns_none():
    llm = FakeLlm([{"nonsense": 1}, {"still": 2}])
    assert await analyze_transcript(llm, _svc(), TRANSCRIPT, "ru") is None


@pytest.mark.asyncio
async def test_map_reduce_when_over_budget():
    llm = FakeLlm([GOOD, GOOD, GOOD, GOOD])
    out = await analyze_transcript(llm, _svc(max_input_tokens=20, chunk_tokens=12), TRANSCRIPT, "en")
    assert out["meta"]["chunked"] is True and out["meta"]["chunks"] >= 2
    assert "part 1" in llm.calls[-1]  # reduce-вызов получил промежуточные


@pytest.mark.asyncio
async def test_empty_transcript_no_llm_call():
    llm = FakeLlm([])
    empty = {**TRANSCRIPT, "segments": [], "speakers": []}
    out = await analyze_transcript(llm, _svc(), empty, "uz")
    assert out["meta"]["warnings"] == ["no_speech"] and llm.calls == []
