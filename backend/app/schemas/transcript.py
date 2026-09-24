"""Схема транскрипта (§7 плана) — то, что лежит в transcript.json."""


from pydantic import BaseModel, Field


class AudioInfo(BaseModel):
    filename: str
    duration_sec: float
    sample_rate: int = 16000


class LanguageShare(BaseModel):
    language: str
    speech_sec: float
    share: float


class SpeakerInfo(BaseModel):
    id: str
    name: str | None = None
    # доминирующий язык спикера (большинство по времени) и его доля
    language: str
    lang_confidence: float
    languages: list[LanguageShare] | None = None
    total_speech_sec: float
    segments_count: int


class Word(BaseModel):
    word: str
    start: float
    end: float


class TranscriptChunk(BaseModel):
    """Кусок ASR внутри сегмента — язык решается на этом уровне (§5.4)."""

    start: float
    end: float
    language: str | None = None
    lang_confidence: float = 0.0
    # own | prev | next | speaker_prior | file_majority | default | asr_text | asr_fallback | speaker
    lang_source: str = ""
    text: str = ""
    warning: str | None = None
    # язык по аудио, если бэкенд ASR его не обслуживает и кусок ушёл в другую модель
    lid_language: str | None = None


class TranscriptSegment(BaseModel):
    id: int
    speaker: str
    language: str
    lang_confidence: float = 1.0
    start: float
    end: float
    overlap: bool = False
    overlap_with: list[str] = Field(default_factory=list)
    text: str = ""
    words: list[Word] | None = None
    chunks: list[TranscriptChunk] | None = None


class TranscriptMeta(BaseModel):
    pipeline_version: str
    models: dict[str, object]
    timings_sec: dict[str, float]
    config: dict[str, object]
    warnings: list[str] = Field(default_factory=list)


class Transcript(BaseModel):
    job_id: str
    title: str | None = None
    audio: AudioInfo
    speakers: list[SpeakerInfo]
    segments: list[TranscriptSegment]
    meta: TranscriptMeta
