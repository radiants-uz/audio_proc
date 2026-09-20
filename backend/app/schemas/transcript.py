"""Схема транскрипта (§7 плана) — то, что лежит в transcript.json."""


from pydantic import BaseModel, Field


class AudioInfo(BaseModel):
    filename: str
    duration_sec: float
    sample_rate: int = 16000


class SpeakerInfo(BaseModel):
    id: str
    name: str | None = None
    language: str
    lang_confidence: float
    total_speech_sec: float
    segments_count: int


class Word(BaseModel):
    word: str
    start: float
    end: float


class TranscriptSegment(BaseModel):
    id: int
    speaker: str
    language: str
    start: float
    end: float
    overlap: bool = False
    overlap_with: list[str] = Field(default_factory=list)
    text: str = ""
    words: list[Word] | None = None


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
