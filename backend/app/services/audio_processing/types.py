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
class Segment:
    id: int
    speaker: str
    start: float
    end: float
    overlap: bool = False
    overlap_with: list[str] = field(default_factory=list)
    language: str | None = None
    text: str = ""
    words: list[dict] | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start
