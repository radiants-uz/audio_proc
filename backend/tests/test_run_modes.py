"""Режимы языка в PipelineRunner.transcribe (тяжёлые стадии заглушены):
lid_recheck из запроса, chunk/legacy, пересчёт языков спикеров после ASR."""

from types import SimpleNamespace

import numpy as np
import pytest

import app.services.audio_processing.run as run
from app.core.config import settings
from app.schemas.task_context import TaskParams
from app.services.audio_processing.langid import LangResult
from app.services.audio_processing.types import Chunk, Segment


def _segments():
    return [Segment(id=0, speaker="A", start=0.0, end=6.0), Segment(id=1, speaker="A", start=6.5, end=12.0)]


def _chunks(segments):
    out = []
    for s in segments:
        s.chunks = [Chunk(seg_id=s.id, idx=0, start=s.start, end=s.end, speaker=s.speaker, seg_start=s.start, seg_end=s.end)]
        out += s.chunks
    return out


def _models():
    diar = SimpleNamespace(diarize=lambda wav, mn, mx: SimpleNamespace(turns=[], exclusive_turns=[], model_name="d"), model_name="d")
    return SimpleNamespace(diarizer=diar, embedder=SimpleNamespace(model_name="e", embed=None),
                           langid=SimpleNamespace(model_name="l"), denoiser=None, vad=None, textcheck=None)


@pytest.fixture
def stubs(monkeypatch, tmp_path):
    calls: list[str] = []
    captured: dict = {}
    monkeypatch.setattr(settings.speakers, "enabled", False)
    monkeypatch.setattr(settings.enhance, "enabled", False)
    monkeypatch.setattr(run, "to_wav16k", lambda src, dst: 12.0)
    monkeypatch.setattr(run, "load_wav", lambda p: np.zeros(16000 * 12, dtype=np.float32))
    monkeypatch.setattr(run, "enhance_for_asr", lambda wav, out, den: (wav, [], SimpleNamespace(mean_db=-20.0, noise_floor_db=-50.0)))
    monkeypatch.setattr(run, "build_segments", lambda ex, turns, cfg, audio: _segments())
    monkeypatch.setattr(run, "plan_chunks", lambda segs, audio, dur, vad: _chunks(segs))

    def identify_speakers(lid, audio, segs, warnings):
        calls.append("identify_speakers")
        return {"A": LangResult(language="uz", confidence=0.95)}

    def recheck_segments(lid, audio, segs):
        calls.append("recheck_segments")
        return 0

    def label_chunks(lid, audio, chunks, warnings, on_progress=None):
        calls.append("label_chunks")
        for ch in chunks:
            ch.language, ch.lang_confidence, ch.lang_source = "uz", 0.99, "own"

    async def transcribe_segments(asr, audio, duration, segments, chunks=None, **kw):
        calls.append("transcribe_segments")
        for s in segments:
            s.text = "текст"
            s.language = "ru"
            for ch in s.chunks:
                ch.text, ch.language, ch.lang_source = "текст", "ru", "asr_fallback"
        return {}

    class FakeAsr:
        name = "mock"

        async def aclose(self): ...

    monkeypatch.setattr(run, "identify_speakers", identify_speakers)
    monkeypatch.setattr(run, "recheck_segments", recheck_segments)
    monkeypatch.setattr(run, "label_chunks", label_chunks)
    monkeypatch.setattr(run, "transcribe_segments", transcribe_segments)
    monkeypatch.setattr(run, "build_asr_client", lambda: FakeAsr())

    def assemble_transcript(**kw):
        captured.update(kw)
        return kw

    monkeypatch.setattr(run.assemble, "assemble_transcript", assemble_transcript)

    async def go(chunk_lid, per_segment_recheck, lid_recheck):
        calls.clear()
        captured.clear()
        monkeypatch.setattr(settings.langid, "chunk_lid", chunk_lid)
        monkeypatch.setattr(settings.langid, "per_segment_recheck", per_segment_recheck)
        runner = run.PipelineRunner(_models(), None, tmp_path)
        await runner.transcribe(tmp_path / "x.wav", "x.wav", "job", TaskParams(lid_recheck=lid_recheck))
        return calls, captured

    return go


@pytest.mark.asyncio
async def test_client_lid_recheck_overrides_env_in_legacy_mode(stubs):
    calls, captured = await stubs(chunk_lid=False, per_segment_recheck=True, lid_recheck=False)
    assert "recheck_segments" not in calls and "identify_speakers" in calls
    assert captured["models"]["langid_mode"] == "speaker"
    calls, _ = await stubs(chunk_lid=False, per_segment_recheck=False, lid_recheck=True)
    assert "recheck_segments" in calls and "label_chunks" not in calls


@pytest.mark.asyncio
async def test_chunk_mode_is_env_only(stubs):
    calls, captured = await stubs(chunk_lid=True, per_segment_recheck=False, lid_recheck=False)
    assert "label_chunks" in calls and "identify_speakers" not in calls
    assert captured["models"]["langid_mode"] == "chunk"


@pytest.mark.asyncio
async def test_speaker_languages_follow_asr_in_both_modes(stubs):
    """ASR переназначил все куски на ru — спикер не должен остаться «uz 0.95»."""
    for chunk_lid in (False, True):
        _, captured = await stubs(chunk_lid=chunk_lid, per_segment_recheck=False, lid_recheck=None)
        langs = captured["languages"]
        assert langs["SPEAKER_00"].language == "ru" and langs["SPEAKER_00"].languages[0]["language"] == "ru"
