from dataclasses import dataclass
from typing import Protocol


@dataclass
class AsrWord:
    word: str
    start: float  # относительно начала переданного куска
    end: float


@dataclass
class AsrResult:
    text: str
    words: list[AsrWord] | None = None  # None — бэкенд не отдаёт слова
    # Бэкенд отверг именно этот кусок (4xx: формат, длина, язык) — текст пуст,
    # остальные куски батча не страдают. None — ответ штатный.
    error: str | None = None


class AsrClient(Protocol):
    name: str

    async def transcribe(self, lang: str, wav_bytes: bytes) -> AsrResult: ...

    async def transcribe_batch(self, lang: str, items: list[bytes]) -> list[AsrResult]: ...

    async def aclose(self) -> None: ...
