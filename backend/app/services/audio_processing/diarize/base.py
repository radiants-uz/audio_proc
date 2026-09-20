from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.services.audio_processing.types import RawTurn


@dataclass
class DiarizationResult:
    turns: list[RawTurn]            # с наложениями
    exclusive_turns: list[RawTurn]  # в каждый момент один спикер
    model_name: str


class Diarizer(Protocol):
    model_name: str

    def load(self) -> None: ...

    def diarize(
        self, wav_path: Path, min_speakers: int | None, max_speakers: int | None
    ) -> DiarizationResult: ...


def make_exclusive(turns: list[RawTurn]) -> list[RawTurn]:
    """Эксклюзивная разметка из пересекающейся: на каждом участке остаётся
    спикер, начавший говорить раньше; более поздний получает остаток."""
    turns = sorted(turns, key=lambda t: (t.start, t.end))
    out: list[RawTurn] = []
    for t in turns:
        start = t.start
        for prev in out:
            if prev.end > start and prev.start <= start:
                start = max(start, prev.end)
        if t.end - start > 0.05:
            out.append(RawTurn(start=start, end=t.end, speaker=t.speaker))
    return out
