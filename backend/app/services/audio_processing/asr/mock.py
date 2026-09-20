"""Заглушка ASR для разработки без доступа к API заказчика: возвращает
детерминированный текст на нужном языке с длительностью куска."""

import asyncio

from app.services.audio_processing.asr.base import AsrResult

_SAMPLES = {
    "uz": "Bu sinov matni, avtomatik tanib olish o'rniga qo'yilgan.",
    "ru": "Это тестовый текст, подставленный вместо распознавания.",
    "en": "This is placeholder text returned instead of recognition.",
}


class MockAsrClient:
    name = "mock"

    async def transcribe(self, lang: str, wav_bytes: bytes) -> AsrResult:
        await asyncio.sleep(0.01)
        sec = max(1, (len(wav_bytes) - 44) // (16000 * 2))
        return AsrResult(text=f"[{lang} {sec}s] {_SAMPLES.get(lang, _SAMPLES['en'])}", words=None)

    async def transcribe_batch(self, lang: str, items: list[bytes]) -> list[AsrResult]:
        return [await self.transcribe(lang, b) for b in items]

    async def aclose(self) -> None:
        return None
