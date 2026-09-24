"""Внутренние структуры пайплайна (не API)."""

from dataclasses import dataclass, field


@dataclass
class RawTurn:
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Chunk:
    """Кусок речи, который уходит в ASR (≤ ASR_MAX_CHUNK_SEC, по областям VAD).
    Язык решается на этом уровне (§5.4): один спикер может переключать языки."""

    seg_id: int
    idx: int
    start: float
    end: float
    speaker: str
    overlap: bool = False
    # границы сегмента-родителя: куски режутся с паддингом и могут выходить за них
    seg_start: float | None = None
    seg_end: float | None = None
    probs: dict[str, float] | None = None   # вероятности uz/ru/en по аудио
    language: str | None = None
    # язык по аудио до подмены на обслуживаемый бэкендом (если отличается от language)
    lid_language: str | None = None
    lang_confidence: float = 0.0
    # own | prev | next | speaker_prior | file_majority | default | speaker |
    # served_fallback | asr_fallback | asr_text
    lang_source: str = ""
    text: str = ""
    warning: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def speech_sec(self) -> float:
        """Длительность куска внутри своего сегмента (без паддинга) — для долей по языкам."""
        if self.seg_start is None or self.seg_end is None:
            return self.duration
        return max(0.0, min(self.end, self.seg_end) - max(self.start, self.seg_start))


@dataclass
class Segment:
    id: int
    speaker: str
    start: float
    end: float
    overlap: bool = False
    overlap_with: list[str] = field(default_factory=list)
    language: str | None = None
    lang_confidence: float = 0.0
    text: str = ""
    words: list[dict] | None = None
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start
